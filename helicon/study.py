"""Plan, execute and compare convergence studies or Cartesian coil sweeps."""
import argparse
import copy
import csv
import itertools
import json
import math
import uuid
from pathlib import Path
import numpy as np
from .config import load, validate
from .diagnostics import positive
from .records import config_digest, digest, read_json, utc_now, write_json
from .run import run_case

SCALARS=('chamber_mean_Bz_T','chamber_Bz_std_T','chamber_Br_rms_T',
         'axis_exit_Bz_T','exit_area_mean_Bz_T')


def expanded_domain(config,factor):
    positive(factor,'domain factor')
    if factor<1: raise ValueError('Domain factor must be at least 1')
    c=copy.deepcopy(config); center=c['chamber']['z_start']+c['chamber']['length']/2
    c['domain']['radius']*=factor
    for key in ('z_min','z_max'):
        c['domain'][key]=center+(c['domain'][key]-center)*factor
    validate(c)
    return c


def convergence_cases(config,scales,factors):
    if len(scales)<2 or len(factors)<2: raise ValueError('Use at least two mesh scales and two domain factors')
    for value in scales: positive(value,'mesh scale')
    if any(a<=b for a,b in zip(scales,scales[1:])): raise ValueError('Mesh scales must strictly decrease')
    if any(a>=b for a,b in zip(factors,factors[1:])): raise ValueError('Domain factors must strictly increase')
    if factors[0]!=1: raise ValueError('Domain factors must start at 1')
    cases=[]
    for i,scale in enumerate(scales):
        cases.append({'id':f'mesh_{i:02d}','family':'mesh','value':scale,'mesh_scale':scale,
                      'config':copy.deepcopy(config)})
    for i,factor in enumerate(factors):
        cases.append({'id':f'domain_{i:02d}','family':'domain','value':factor,'mesh_scale':scales[-1],
                      'config':expanded_domain(config,factor)})
    return cases


def sweep_cases(config,spec,max_cases=64):
    parameters=spec.get('parameters')
    if not isinstance(parameters,list) or not parameters: raise ValueError('Sweep needs a nonempty parameters list')
    names={k['name']:i for i,k in enumerate(config['coils'])}
    seen=set(); n=1
    for item in parameters:
        if set(item)!={'coil','field','values'}: raise ValueError('Each parameter requires coil, field and values')
        if item['coil'] not in names or item['field'] not in ('z_center','ampere_turns'):
            raise ValueError('Sweep fields are z_center [m] and ampere_turns [A-turn], on a named coil')
        key=(item['coil'],item['field'])
        if key in seen: raise ValueError('Duplicate sweep parameter')
        seen.add(key)
        values=item['values']
        if not isinstance(values,list) or not values: raise ValueError('Parameter values must be a nonempty list')
        if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in values):
            raise ValueError('Sweep values must be finite numbers')
        if len(set(values))!=len(values): raise ValueError('Duplicate parameter values')
        n*=len(values)
    if n>max_cases: raise ValueError(f'{n} cases exceeds --max-cases {max_cases}')
    cases=[]
    for i,values in enumerate(itertools.product(*(item['values'] for item in parameters))):
        c=copy.deepcopy(config); settings={}
        for item,value in zip(parameters,values):
            k=c['coils'][names[item['coil']]]
            if item['field']=='ampere_turns': k['current']=value/k['turns']
            else: k[item['field']]=value
            settings[f"{item['coil']}.{item['field']}"]=value
        case={'id':f'case_{i:04d}','family':'sweep','parameters':settings,'config':c}
        try: validate(c)
        except (ValueError,KeyError) as error: case['invalid_reason']=str(error)
        cases.append(case)
    return cases


def create_study(out,kind,cases,target_gauss,tolerance_percent,spec=None):
    positive(target_gauss,'target gauss'); positive(tolerance_percent,'tolerance percent')
    out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=False)
    for case in cases:
        case['config_sha256']=config_digest(case['config'])
    plan={'schema_version':1,'kind':kind,'created_utc':utc_now(),'target_gauss':target_gauss,
          'tolerance_percent':tolerance_percent,'cases':cases,'sweep_specification':spec}
    write_json(out/'study.json',plan)
    print(f'Prepared {len(cases)} cases in {out}. Execute with: python -m helicon.study execute {out}',flush=True)
    return out


def completed_record(out,case,plan):
    path=out/case['id']/'run.json'
    if not path.exists(): return None
    record=read_json(path)
    if record['status']!='completed': return None
    if (record['config_sha256']!=case['config_sha256'] or record['mesh_scale']!=case['mesh_scale']
            or record['target_gauss']!=plan['target_gauss']):
        raise ValueError(f'{case["id"]}: saved run does not match study plan')
    for name,expected in record['artifacts_sha256'].items():
        if not (path.parent/name).exists() or digest(path.parent/name)!=expected:
            raise ValueError(f'{case["id"]}: modified or missing output {name}')
    for name,expected in record['source']['sha256'].items():
        path_to_source=path.parent/'source'/name
        if not path_to_source.exists() or digest(path_to_source)!=expected:
            raise ValueError(f'{case["id"]}: source snapshot changed: {name}')
    return record


def execute(out,retry_failed=False):
    out=Path(out).resolve(); plan=read_json(out/'study.json')
    for case in plan['cases']:
        if config_digest(case['config'])!=case['config_sha256']:
            raise ValueError('Study configuration edited after planning; create a new study')
        folder=out/case['id']
        if case.get('invalid_reason'): continue
        if completed_record(out,case,plan):
            print(f'{case["id"]}: already completed; preserved',flush=True)
            continue
        if folder.exists():
            # Preserve failed/partial runs rather than mixing their files with retries.
            record=read_json(folder/'run.json') if (folder/'run.json').exists() else {}
            if record.get('status')=='failed' and retry_failed:
                archive=out/'attempts'/f"{case['id']}-{uuid.uuid4().hex[:8]}"
                archive.parent.mkdir(exist_ok=True)
                folder.rename(archive)
                print(f'{case["id"]}: previous failure archived in {archive}',flush=True)
            else:
                print(f'{case["id"]}: existing {record.get("status","partial")} folder preserved; '
                      'use execute --retry-failed for failed cases or a new study for partial cases',flush=True)
                continue
        try:
            run_case(case['config'],folder,case['mesh_scale'],plan['target_gauss'])
        except Exception as error:
            print(f'{case["id"]}: {error}',flush=True)
    return summarize(out)


def compare_pair(out,previous,current,metrics,target_t):
    a,b=metrics[previous['id']],metrics[current['id']]
    changes={key:100*abs(b[key]-a[key])/target_t for key in SCALARS}
    relative={key:100*abs(b[key]-a[key])/max(abs(b[key]),target_t*1e-6) for key in SCALARS}
    pa=np.genfromtxt(out/previous['id']/'fem_axis.csv',delimiter=',',names=True)
    pb=np.genfromtxt(out/current['id']/'fem_axis.csv',delimiter=',',names=True)
    if not np.allclose(pa['z_m'],pb['z_m'],rtol=0,atol=1e-12):
        raise ValueError('Centerline samples must share identical coordinates')
    den=np.linalg.norm(pb['Bz_FEM_T'])
    curve=100*float(np.linalg.norm(pb['Bz_FEM_T']-pa['Bz_FEM_T'])/den) if den>1e-30 else None
    return {'from':previous['id'],'to':current['id'],'scalar_change_percent_of_target':changes,
            'axis_curve_relative_change_percent':curve,
            'scalar_relative_change_percent':relative,
            'largest_scalar_relative_change_percent':max(relative.values()),
            'largest_scalar_change_percent_of_target':max(changes.values())}


def summarize(out):
    out=Path(out); plan=read_json(out/'study.json'); rows=[]; metrics={}; sources=set(); environments=set()
    for case in plan['cases']:
        record=completed_record(out,case,plan)
        row={'case':case['id'],'family':case['family'],'value':case.get('value',''),
             'mesh_scale':case['mesh_scale'],'status':'not_run','parameters':str(case.get('parameters',{})),
             'error':case.get('invalid_reason','')}
        if case.get('invalid_reason'): row['status']='invalid'
        elif (out/case['id']/'run.json').exists():
            partial=read_json(out/case['id']/'run.json'); row['status']=partial['status']; row['error']=partial.get('error','')
        if record:
            m=record['metrics']; metrics[case['id']]=m
            sources.add(str(sorted(record['source']['sha256'].items())))
            environments.add(json.dumps({'solver':record.get('solver_environment'),
                                         'gmsh':record['mesh'].get('gmsh_version')},sort_keys=True))
            row.update(status='completed',nodes=record['mesh']['nodes'],triangles=record['mesh']['triangles'])
            for key in ('chamber_mean_Bz_G','chamber_Bz_std_G','chamber_Br_rms_G','axis_exit_Bz_G',
                        'exit_area_mean_Bz_G','chamber_Bz_nonuniformity_percent','target_mean_error_percent',
                        'axis_reference_error_percent','normalized_rms_field_error'):
                row[key]=m[key]
        rows.append(row)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    summary={'kind':plan['kind'],'complete':len(metrics)==len(plan['cases']),
             'completed_cases':len(metrics),'total_cases':len(plan['cases']),
             'same_source_snapshot':len(sources)<=1,'same_runtime_environment':len(environments)<=1,'tolerance_percent':plan['tolerance_percent']}
    if plan['kind']=='convergence':
        checks={}
        for family in ('mesh','domain'):
            cases=[c for c in plan['cases'] if c['family']==family]
            comparisons=[compare_pair(out,a,b,metrics,plan['target_gauss']/1e4)
                         for a,b in zip(cases,cases[1:]) if a['id'] in metrics and b['id'] in metrics]
            final_pair=next((p for p in comparisons if p['to']==cases[-1]['id']),None)
            passed=None
            if final_pair is not None:
                curve=final_pair['axis_curve_relative_change_percent']
                passed=(curve is not None and curve<=plan['tolerance_percent'] and
                        final_pair['largest_scalar_relative_change_percent']<=plan['tolerance_percent'])
            checks[family]={'comparisons':comparisons,'last_pair_within_tolerance':passed}
        domain_last=[c for c in plan['cases'] if c['family']=='domain'][-1]['id']
        reference_error=metrics.get(domain_last,{}).get('axis_reference_error_percent')
        summary.update(checks=checks,largest_domain_reference_error_percent=reference_error,
            sensitivity_criteria_met=bool(summary['complete'] and summary['same_source_snapshot'] and summary['same_runtime_environment'] and
                all(c['last_pair_within_tolerance'] is True for c in checks.values()) and
                reference_error is not None and reference_error<=plan['tolerance_percent']),
            criterion='Last tested mesh and domain pairs: scalar relative changes and centerline relative L2 <= tolerance; scalar denominators floored at 1e-6 * target; '
                      'largest-domain on-axis reference error <= tolerance. These are sampled numerical checks, not experimental validation.')
    else:
        ranking=sorted((r for r in rows if r['status']=='completed'),key=lambda r:r['normalized_rms_field_error'])
        summary['ranking']=[{'case':r['case'],'normalized_rms_field_error':r['normalized_rms_field_error'],
                            'target_mean_error_percent':r['target_mean_error_percent'],
                            'meets_mean_target_tolerance':abs(r['target_mean_error_percent'])<=plan['tolerance_percent']}
                           for r in ranking]
        summary['best_completed_case']=ranking[0]['case'] if ranking else None
        summary['ranking_note']='Ranking includes completed cases only; check failures and convergence before choosing a design. '
        summary['ranking_note']+='Objective = sqrt(volume_mean((Bz-target)^2+Br^2))/target. No heating, power, mass or plasma criteria.'
    write_json(out/'study_summary.json',summary)
    lines=[f'# {plan["kind"].title()} study','',f'Completed {len(metrics)} / {len(plan["cases"])} cases.',
           f'Target: {plan["target_gauss"]:g} G. Tolerance: {plan["tolerance_percent"]:g}%.','',
           '| Case | Status | Mean [G] | Axis exit [G] | Bz std [G] | Br RMS [G] | Objective |',
           '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        val=lambda k: f'{row[k]:.5g}' if k in row and row[k] is not None else '—'
        lines.append(f"| {row['case']} | {row['status']} | {val('chamber_mean_Bz_G')} | {val('axis_exit_Bz_G')} | {val('chamber_Bz_std_G')} | {val('chamber_Br_rms_G')} | {val('normalized_rms_field_error')} |")
    lines+=['','```json',json.dumps(summary,indent=2),'```']
    (out/'study_report.md').write_text('\n'.join(lines)+'\n')
    if metrics: plot_comparison(out,rows,plan)
    print(f'Summary: {out/"study_report.md"}',flush=True)
    return summary


def plot_comparison(out,rows,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    completed=[r for r in rows if r['status']=='completed']
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    fields=('chamber_mean_Bz_G','axis_exit_Bz_G','chamber_Bz_std_G','chamber_Br_rms_G')
    labels=('Chamber mean Bz [G]','Axis exit Bz [G]','Chamber Bz std [G]','Chamber Br RMS [G]')
    for ax,key,label in zip(axs.flat,fields,labels):
        for family in dict.fromkeys(r['family'] for r in completed):
            subset=[r for r in completed if r['family']==family]
            ax.plot([r['case'] for r in subset],[r[key] for r in subset],'o-',label=family)
        if key=='chamber_mean_Bz_G': ax.axhline(plan['target_gauss'],color='gray',ls=':')
        ax.set(ylabel=label); ax.tick_params(axis='x',rotation=45); ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'comparison.png',dpi=160); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    for name in ('convergence','sweep'):
        s=sub.add_parser(name); s.add_argument('--config'); s.add_argument('--out',required=True)
        s.add_argument('--target-gauss',type=float,default=140); s.add_argument('--tolerance-percent',type=float,default=1)
        s.add_argument('--prepare-only',action='store_true')
        if name=='convergence':
            s.add_argument('--mesh-scales',nargs='+',type=float,default=[1,.5,.25])
            s.add_argument('--domain-factors',nargs='+',type=float,default=[1,1.5,2])
        else:
            s.add_argument('--spec',required=True); s.add_argument('--scale',type=float,default=.5)
            s.add_argument('--max-cases',type=int,default=64)
    for name in ('execute','summarize'):
        s=sub.add_parser(name); s.add_argument('directory')
        if name=='execute': s.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if a.command in ('execute','summarize'):
        summary=execute(a.directory,a.retry_failed) if a.command=='execute' else summarize(a.directory)
        if not summary['complete']: raise SystemExit(1)
        return
    config=load(a.config); spec=None
    if a.command=='convergence': cases=convergence_cases(config,a.mesh_scales,a.domain_factors)
    else:
        positive(a.scale,'mesh scale'); spec=read_json(a.spec)
        cases=sweep_cases(config,spec,a.max_cases)
        for case in cases: case['mesh_scale']=a.scale
    out=create_study(a.out,a.command,cases,a.target_gauss,a.tolerance_percent,spec)
    if not a.prepare_only:
        summary=execute(out)
        if not summary['complete']: raise SystemExit(1)

if __name__=='__main__': main()
