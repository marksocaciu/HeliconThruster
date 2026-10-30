"""Experimental fixed-system FGMRES comparison with a persistent real AMS job."""
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .ams_protocol import atomic_json, atomic_npz, wait_json, clean_worker_environment, cgroup_identity
from .ams_transfer import collective_io
from .diagnostics import IterationMonitor, linear_diagnostics
from .fingerprints import assembly_fingerprint
from .memory import cgroup_memory, memory_snapshot


class BridgeError(RuntimeError):
    pass


class AMSBridge:
    def __init__(self,P,comm,export,directory,*,rtol=0.01,max_it=100,timeout=300,
                 python='/usr/bin/python3',mpiexec='mpiexec',petsc_dir='/usr/lib/petscdir/petsc-real',cap_mb=24000):
        self.P,self.comm,self.export,self.directory=P,comm,Path(export),Path(directory)
        self.rtol,self.max_it,self.timeout=rtol,max_it,timeout
        self.python,self.mpiexec,self.petsc_dir,self.cap_mb=python,mpiexec,petsc_dir,cap_mb
        self.sequence=1;self.proc=None;self.log=None;self.ready=None;self.failure=None;self.closed=False
        self.rows=[];self.work=P.createVecLeft();self.shutdown=None

    def root_action(self,action):
        value=None;error=None
        if self.comm.rank==0:
            try:value=action()
            except Exception as exc:error=str(exc)
        error=self.comm.bcast(error,root=0)
        if error:raise BridgeError(error)
        return self.comm.bcast(value,root=0)

    def start(self):
        if not 0<self.rtol<1 or not 1<=self.max_it<=1000 or not math.isfinite(self.timeout) or not 1<=self.timeout<=3600:
            raise BridgeError('Invalid bridge bounds')
        if self.cap_mb is not None:
            cap=cgroup_memory()
            ok=cap.get('max') is not None and cap['max']<=self.cap_mb*1000000 and cap.get('swap.max')==0
            if not all(self.comm.allgather(ok)):raise BridgeError('Bridge requires a verified shared memory cap with zero swap')
        def launch():
            self.directory.mkdir(exist_ok=False)
            atomic_json(self.directory/'config.json',{'cgroup_identity':cgroup_identity()})
            self.log=(self.directory/'worker.log').open('w')
            command=[self.mpiexec,'-n',str(self.comm.size),self.python,'-m','helicon.rf.ams_worker',
                '--export',str(self.export.resolve()),'--ipc',str(self.directory.resolve()),
                '--rtol',str(self.rtol),'--max-it',str(self.max_it),'--timeout',str(self.timeout)]
            if self.cap_mb is not None:command+=['--cap-mb',str(self.cap_mb)]
            self.proc=subprocess.Popen(command,cwd=str(Path(__file__).resolve().parents[2]),
                env=clean_worker_environment(os.environ,self.petsc_dir),stdin=subprocess.DEVNULL,
                stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
            return wait_json(self.directory/'ready.json',self.timeout,
                             failed=self.directory/'failed.json',process=self.proc)
        self.ready=self.root_action(launch)

    def apply(self,pc,rhs,result):
        start=time.monotonic()
        try:
            sequence=self.sequence
            collective_io(self.comm,lambda:atomic_npz(self.directory/f'rhs-{self.comm.rank}.npz',
                sequence=sequence,values=rhs.getArray(readonly=True)))
            self.root_action(lambda:atomic_json(self.directory/'command.json',{'sequence':sequence,'command':'solve'}))
            response=self.root_action(lambda:wait_json(self.directory/'response.json',self.timeout,
                failed=self.directory/'failed.json',process=self.proc,sequence=sequence))
            # A response means the worker has advanced to the next command.
            self.sequence+=1
            def read():
                with np.load(self.directory/f'answer-{self.comm.rank}.npz') as data:
                    if int(data['sequence'])!=sequence:raise ValueError('Stale answer shard')
                    answer=data['values'].copy()
                if answer.shape!=(result.getLocalSize(),) or not np.all(np.isfinite(answer)):raise ValueError('Invalid answer shard')
                if not np.iscomplexobj(result.getArray()) and np.any(answer.imag!=0):raise ValueError('Complex output into real parent')
                result.array[:]=answer if np.iscomplexobj(result.getArray()) else answer.real
            collective_io(self.comm,read)
            self.P.mult(result,self.work);self.work.axpy(-1,rhs)
            relative=float(self.work.norm()/max(rhs.norm(),1e-300))
            row=dict(response,parent_true_relative_residual=relative if math.isfinite(relative) else None,
                     round_trip_s=time.monotonic()-start)
            self.rows.append(row)
            self.root_action(lambda:atomic_json(self.directory/'applications.json',self.rows))
            if self.comm.rank==0:
                print('RF AMS bridge: '+json.dumps({'application':sequence,'true_relative_residual':row['parent_true_relative_residual'],
                      'inner_iterations':[p['iterations'] for p in response['components']], 'round_trip_s':row['round_trip_s']}),flush=True)
            if not response['passed'] or not math.isfinite(relative) or relative>self.rtol*1.001:
                raise BridgeError('AMS application did not meet its true residual target')
            # Keep bounded IPC storage. Retain failed request/answer files for inspection.
            collective_io(self.comm,lambda:(self.directory/f'rhs-{self.comm.rank}.npz').unlink())
            collective_io(self.comm,lambda:(self.directory/f'answer-{self.comm.rank}.npz').unlink())
        except Exception as exc:
            self.failure=str(exc)
            raise BridgeError(self.failure) from exc

    def close(self):
        if self.closed:return
        self.closed=True
        def stop():
            if self.proc is None:return {'status':'not_started'}
            if self.proc.poll() is None:
                atomic_json(self.directory/'command.json',{'sequence':self.sequence,'command':'stop'})
                try:self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(self.proc.pid,signal.SIGTERM)
                    try:self.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait(timeout=5)
                    return {'status':'terminated','returncode':self.proc.returncode}
            return {'status':'exited','returncode':self.proc.returncode}
        try:self.shutdown=self.root_action(stop)
        finally:
            self.work.destroy()
            if self.log is not None:self.log.close()

    def destroy(self,pc):
        # Explicit close belongs to the comparison owner, not PETSc destruction.
        pass


def compare_ams_bridge(problem,options,PETSc,comm,out,tolerance,first,**configuration):
    from .solve import _solve_with_options
    A,P=problem.solver.getOperators()
    before=assembly_fingerprint(A,problem.b,comm,PETSc)
    p_before=assembly_fingerprint(P,problem.b,comm,PETSc)
    problem.solver.reset()  # Release baseline ASM factors before worker setup.
    context=AMSBridge(P,comm,out/'ams-export',out/'ams-worker',**configuration)
    solver=PETSc.KSP().create(comm);x=problem.x.duplicate();x.set(0);monitor=None
    report={'status':'starting','baseline':first,'inner_applications':context.rows,
            'configuration':configuration,'assembly_before':before,'P_before':p_before}
    def save():collective_io(comm,lambda:atomic_json(out/'ams_bridge_diagnostics.json',report) if comm.rank==0 else None)
    try:
        save();context.start();report['worker_ready']=context.ready
        report['memory_after_worker_setup']=memory_snapshot(comm,'outer_and_worker_ready');save()
        solver.setOptionsPrefix('rf_ams_bridge_outer_');solver.setOperators(A,P);solver.setInitialGuessNonzero(False)
        monitor=IterationMonitor(comm,context.directory);solver.setMonitor(monitor)
        trial_options={key:value for key,value in options.items() if not key.startswith(('pc_','sub_'))}
        trial_options['pc_type']='python'
        def run():
            solver.getPC().setPythonContext(context)
            solver.solve(problem.b,x)
        fixed=SimpleNamespace(A=A,b=problem.b,x=x,solver=solver,solve=run)
        try:_solve_with_options(fixed,trial_options,PETSc)
        except (BridgeError,PETSc.Error):
            if context.failure is None:raise
            report['bridge_error']=context.failure
        report['outer_result']=linear_diagnostics(fixed,PETSc,tolerance)
        if context.failure:report['outer_result']['accepted']=False
        report['status']='converged' if report['outer_result']['accepted'] else 'not_converged'
        monitor.close();report['history_write_error']=comm.bcast(monitor.error,root=0)
    except BridgeError as exc:
        report.update(status='bridge_failed',bridge_error=str(exc))
    except Exception as exc:
        report.update(status='unexpected_failure',error=str(exc));raise
    finally:
        if monitor is not None:monitor.close()
        solver.destroy();x.destroy()
        try:context.close();report['worker_shutdown']=context.shutdown
        finally:
            report['memory_final']=memory_snapshot(comm,'bridge_end')
            save()
    report['assembly_after']=assembly_fingerprint(A,problem.b,comm,PETSc)
    report['P_after']=assembly_fingerprint(P,problem.b,comm,PETSc)
    report['A_b_P_unchanged']=before==report['assembly_after'] and p_before==report['P_after']
    save()
    if not report['A_b_P_unchanged']:raise RuntimeError('AMS bridge changed A, b or P')
    if comm.rank==0:print('RF AMS bridge result: '+str(report.get('outer_result',report.get('bridge_error'))),flush=True)
    return report
