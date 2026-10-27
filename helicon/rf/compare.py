"""Compare an RF candidate with a saved direct reference; standard library only.

python3 -m helicon.rf.compare --reference RUN/direct/solution --candidate RUN/test/solution
Exit 0 = checks passed, 1 = comparison rejected, 2 = invalid input/arguments.
This is a solver comparison on one mesh, not mesh-convergence validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def compare(reference,candidate,rtol=1e-3,power_atol=1e-10,residual_tol=1e-8,job_limit_mb=24000):
    reference,candidate=Path(reference),Path(candidate)
    failures=[]; metrics={}
    for name in ('rf.msh','rf.json','background_B.npz','input_plasma_fields.npz','input_plasma_summary.json'):
        left,right=reference/name,candidate/name
        if name in ('rf.msh','rf.json') and not (left.is_file() and right.is_file()):
            failures.append('Missing required input: '+name)
        elif left.exists() != right.exists() or (left.exists() and file_hash(left)!=file_hash(right)):
            failures.append('Inputs differ: '+name)
    summaries=[]
    for label,path in (('reference',reference),('candidate',candidate)):
        if not (path/'rf_summary.json').exists():
            status=read(path/'run.json').get('status') if (path/'run.json').exists() else 'missing run'
            failures.append(f'{label}: no accepted field summary ({status})')
            summaries.append(None); continue
        summary=read(path/'rf_summary.json'); summaries.append(summary)
        if summary.get('status')!='completed' or summary.get('numerical_checks_passed') is not True:
            failures.append(label+': numerical checks did not pass')
        residual=summary.get('solver_diagnostics',{}).get('true_relative_residual')
        if residual is None or not math.isfinite(residual) or residual > residual_tol:
            failures.append(label+': true residual missing or above tolerance')
    left,right=summaries
    if left is not None and right is not None:
        def metric(name,a,b,atol):
            difference=abs(b-a)
            passed=math.isfinite(difference) and difference <= atol+rtol*abs(a)
            metrics[name]={'absolute_difference':difference if math.isfinite(difference) else None,
                           'relative_difference':difference/abs(a) if a and math.isfinite(difference) else None,
                           'passed':passed}
            if not passed: failures.append(name+': disagrees with reference')
        za,zb=left.get('ideal_gap_impedance_ohm'),right.get('ideal_gap_impedance_ohm')
        if za is None or zb is None: failures.append('Impedance missing')
        else: metric('impedance_ohm',complex(*za),complex(*zb),1e-8)
        for name in ('input_power_W','pml_absorption_W','outward_interface_flux_W'):
            metric(name,float(left[name]),float(right[name]),power_atol)
        la,ra=left['material_absorption_W'],right['material_absorption_W']
        if set(la)!=set(ra): failures.append('Material regions differ')
        else:
            for name in la: metric('absorption_'+name+'_W',float(la[name]),float(ra[name]),power_atol)
    peaks=[]; swaps=[]
    run=read(candidate/'run.json') if (candidate/'run.json').exists() else {}
    for measurement in run.get('memory',[]):
        counters=measurement.get('cgroup',{})
        if counters.get('peak') is not None: peaks.append(counters['peak'])
        if counters.get('swap.current') is not None: swaps.append(counters['swap.current'])
    launcher=candidate.parent/'launcher_status.json'
    if launcher.exists():
        status=read(launcher)
        if status.get('exit_code')!=0: failures.append('Launcher exited unsuccessfully')
        counters=status.get('cgroup_memory',{})
        if counters.get('peak') is not None: peaks.append(counters['peak'])
        if counters.get('swap.current') is not None: swaps.append(counters['swap.current'])
    peak=max(peaks) if peaks else None
    if peak is None: failures.append('No measured cgroup peak: cannot verify memory budget')
    elif peak > job_limit_mb*1e6: failures.append('Measured job memory exceeds budget')
    if swaps and max(swaps)>0: failures.append('Recorded job swap usage is nonzero')
    return {'passed':not failures,'failures':failures,'metrics':metrics,
            'candidate_peak_job_mb':peak/1e6 if peak is not None else None,
            'tolerances':{'relative':rtol,'absolute_power_W':power_atol,
                          'true_residual':residual_tol,'job_memory_mb':job_limit_mb},
            'scope':'Same-input scalar-output solver comparison; not a full field or mesh-convergence check.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',required=True); p.add_argument('--candidate',required=True)
    p.add_argument('--rtol',type=float,default=1e-3)
    p.add_argument('--power-atol',type=float,default=1e-10)
    p.add_argument('--residual-tol',type=float,default=1e-8)
    p.add_argument('--job-memory-mb',type=float,default=24000)
    a=p.parse_args()
    for name in ('rtol','power_atol','residual_tol','job_memory_mb'):
        value=getattr(a,name)
        if not math.isfinite(value) or value<=0: p.error(name+' must be positive and finite')
    try:
        result=compare(a.reference,a.candidate,a.rtol,a.power_atol,a.residual_tol,a.job_memory_mb)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result['passed'] else 1
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.exit(2,str(exc)+'\n')

if __name__=='__main__': raise SystemExit(main())
