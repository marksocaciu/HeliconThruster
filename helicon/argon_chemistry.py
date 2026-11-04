"""Selectable Maxwellian ground-state argon chemistry; no extrapolation."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from . import argon

LEGACY = argon.RATE_MODEL
PHELPS = 'argon_phelps_maxwellian_1_to_20_eV_v1'
TABLE = Path(__file__).with_name('data')/'argon_phelps_maxwellian.json'
COLUMNS = ('ionization_m3_s','excitation_m3_s','elastic_m3_s',
           'momentum_m3_s','elastic_energy_weighted_eVm3_s')


def model(config):
    name=config.get('chemistry_model',LEGACY)
    if name not in (LEGACY,PHELPS):raise ValueError('Unknown argon chemistry_model: '+str(name))
    return name


def bounds(config):
    return (argon.TE_MIN_EV,argon.TE_MAX_EV) if model(config)==LEGACY else tuple(table()['temperature_interval_eV'])


@lru_cache(maxsize=1)
def table():
    from scipy.interpolate import PchipInterpolator
    raw=TABLE.read_bytes()
    d=json.loads(raw)
    t=np.asarray(d['temperature_eV'],float)
    interval=d['temperature_interval_eV']
    if (d['model']!=PHELPS or d['eedf']!='Maxwellian' or t.ndim!=1 or len(t)<3
            or not np.isfinite(t).all() or np.any(np.diff(t)<=0)
            or list(interval)!=[1.,20.] or list(interval)!=[float(t[0]),float(t[-1])]):
        raise ValueError('Invalid argon rate table metadata/grid')
    interpolation={}
    for key in COLUMNS:
        values=np.asarray(d[key],float)
        if values.shape!=t.shape or not np.isfinite(values).all() or np.any(values<=0):
            raise ValueError('Invalid argon rate column: '+key)
        interpolation[key]=PchipInterpolator(np.log(t),np.log(values),extrapolate=False)
    d['_interpolation']=interpolation
    d['_sha256']=hashlib.sha256(raw).hexdigest()
    return d


def metadata(config):
    name=model(config)
    if name==LEGACY:
        return {'model':name,'temperature_interval_eV':[1.,7.],
                'eedf':'Maxwellian','source':'Lieberman & Lichtenberg (2005) fits; see docs/plasma-model.md'}
    d=table()
    if hashlib.sha256(TABLE.read_bytes()).hexdigest()!=d['_sha256']:
        raise ValueError('Argon rate table changed after loading')
    return {k:d[k] for k in ('model','temperature_interval_eV','eedf','provenance','method','limitations')} | {
        'table_sha256':d['_sha256']}


def rates(t,config):
    if model(config)==LEGACY:return argon.rates(t)
    low,high=bounds(config)
    if isinstance(t,bool) or not isinstance(t,(float,int)) or not math.isfinite(t) or not low<=t<=high:
        raise ValueError(f'Argon chemistry requires {low:g} <= Te <= {high:g} eV')
    d=table()
    out={key:float(np.exp(interp(np.log(t)))) for key,interp in d['_interpolation'].items()}
    out['ionization_energy_eVm3_s']=out['ionization_m3_s']*d['ionization_threshold_eV']
    out['excitation_energy_eVm3_s']=out['excitation_m3_s']*d['excitation_threshold_eV']
    return out


def parse_cross_sections(path):
    """Read the three-process Phelps e+Ar set, not an arbitrary LXCat mixture."""
    lines=Path(path).read_text().splitlines();result={};i=0
    while i<len(lines):
        kind=lines[i].strip()
        if kind not in ('EFFECTIVE','ELASTIC','EXCITATION','IONIZATION'):i+=1;continue
        target=lines[i+1].strip();parameter=float(lines[i+2].split()[0]);i+=3
        if target.split('->')[0].strip()!='Ar':raise ValueError('Require ground-state Ar only')
        while i<len(lines) and not lines[i].strip().startswith('-----'):i+=1
        i+=1;points=[]
        while i<len(lines) and not lines[i].strip().startswith('-----'):
            points.append([float(x) for x in lines[i].split()]);i+=1
        if kind in result:raise ValueError('Duplicate process; require the three-process Phelps set')
        a=np.asarray(points)
        if (a.ndim!=2 or a.shape[1]!=2 or len(a)<3 or not np.isfinite(a).all()
                or np.any(np.diff(a[:,0])<=0) or np.any(a<0)):
            raise ValueError('Invalid cross-section points')
        result[kind]=(parameter,a)
    if set(result)!={'EFFECTIVE','EXCITATION','IONIZATION'}:
        raise ValueError('Require EFFECTIVE, EXCITATION and IONIZATION Phelps Ar processes')
    for kind in ('EXCITATION','IONIZATION'):
        threshold,a=result[kind]
        if a[0,0]!=threshold or a[0,1]!=0:raise ValueError('Missing zero threshold point')
    if not math.isclose(result['EXCITATION'][0],11.5) or not math.isclose(result['IONIZATION'][0],15.8):
        raise ValueError('Unexpected Phelps three-process thresholds')
    if result['EFFECTIVE'][1][0,0]!=0:raise ValueError('Momentum cross sections must include zero energy')
    return result


def quadrature_grid(processes,order=16):
    """Union all linear-interpolation knots before removing inelastic contributions."""
    maximum=min(a[-1,0] for _,a in processes.values())
    if maximum<1000:raise ValueError('Insufficient collision-energy coverage for Te up to 20 eV')
    knots=np.unique(np.concatenate([a[a[:,0]<=maximum,0] for _,a in processes.values()]))
    nodes,w=np.polynomial.legendre.leggauss(order)
    energy=((knots[:-1,None]+knots[1:,None])/2+
            np.diff(knots)[:,None]/2*nodes).ravel()
    weight=(np.diff(knots)[:,None]/2*w).ravel()
    sigma={k:np.interp(energy,a[:,0],a[:,1],left=0,right=0) for k,(_,a) in processes.items()}
    elastic=sigma['EFFECTIVE']-sigma['EXCITATION']-sigma['IONIZATION']
    if np.min(elastic)<-1e-30:raise ValueError('Effective-minus-inelastic gives negative elastic momentum transfer')
    sigma['ELASTIC']=np.maximum(elastic,0.)
    return energy,weight,sigma,maximum


def integrate(t,energy,weight,sigma):
    factor=math.sqrt(8*argon.E_CHARGE/(math.pi*argon.M_E))/t**1.5
    kernel=factor*weight*energy*np.exp(-energy/t)
    return {key:float(np.dot(kernel,sigma[kind])) for key,kind in zip(COLUMNS[:4],
        ('IONIZATION','EXCITATION','ELASTIC','EFFECTIVE'))} | {
        COLUMNS[4]:float(np.dot(kernel*energy,sigma['ELASTIC']))}


def build(source,out):
    """Generate derived calculation results; retain no redistributed raw cross sections."""
    text = Path(source).read_text()
    if 'Phelps database' not in text: raise ValueError('Missing Phelps database attribution')
    generated = re.search(r'Generated on (\d{2} \w{3} \d{4})', text)
    if generated is None: raise ValueError('Missing LXCat retrieval date')
    retrieved = datetime.strptime(generated[1], '%d %b %Y').date().isoformat()
    processes=parse_cross_sections(source)
    e,w,sigma,maximum=quadrature_grid(processes)
    temperatures=np.geomspace(1.,20.,401)
    values=[integrate(float(t),e,w,sigma) for t in temperatures]
    result={'model':PHELPS,'eedf':'Maxwellian','temperature_interval_eV':[1.,20.],
        'temperature_eV':temperatures.tolist(),
        'excitation_threshold_eV':processes['EXCITATION'][0],
        'ionization_threshold_eV':processes['IONIZATION'][0],
        'provenance':{'database':'Phelps, LXCat','retrieved_date':retrieved,
            'url':'https://www.lxcat.net/Phelps',
            'cross_sections_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'references':['Yamabe, Buckman and Phelps, Phys. Rev. A 27, 1345 (1983), with October 1997 revisions',
                'Phelps compilation process comments: excitation Schapert-Scheibner at low energy; ionization Rapp-Schram',
                'LXCat reference: Pitchford et al., Plasma Process. Polym. 14, 1600098 (2017)']},
        'method':{'rate':'sqrt(8 e/(pi me)) T^-1.5 integral sigma(E) E exp(-E/T) dE; E,T in eV',
            'cross_section_interpolation':'piecewise linear, common knot union',
            'integration':'16-point Gauss-Legendre on each knot interval',
            'collision_energy_max_eV':float(maximum),'cross_section_extrapolation':False,
            'rate_interpolation':'PCHIP of log(rate) versus log(Te), no extrapolation',
            'momentum':'effective total for mobility; elastic = effective - excitation - ionization',
            'elastic_loss':'(2 me/MAr) <sigma_el v E> (1 - Tg_eV/Te)',
            'excitation_loss':'effective total excitation rate times the dataset 11.5 eV loss'},
        'limitations':['Ground-state Ar only; effective excitation is an energy sink, not a metastable population.',
            'Maxwellian EEDF assumed; no Boltzmann solve or kinetic validation.',
            '1-20 eV is the selected numerical table domain, not an experimental validity claim.',
            'No stepwise ionization, superelastic reactions, multiple charge states or recombination.',
            'Legacy collision floor remains prescribed; no added floor-related energy sink.']}
    result.update({key:[v[key] for v in values] for key in COLUMNS})
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    with Path(out).open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    return result


def verify(source, out):
    """Independent adaptive quadrature and overlap comparison, using original data."""
    import csv
    from scipy.integrate import quad
    processes = parse_cross_sections(source)
    if hashlib.sha256(Path(source).read_bytes()).hexdigest() != table()['provenance']['cross_sections_sha256']:
        raise ValueError('Reference cross sections differ from the bundled calculation provenance')
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    knots = np.unique(np.concatenate([a[:,0] for _,a in processes.values()]))
    def sigma(kind, energy):
        _, a = processes[kind]
        return float(np.interp(energy, a[:,0], a[:,1], left=0., right=0.))
    def cross(key, energy):
        if key == COLUMNS[0]: return sigma('IONIZATION',energy)
        if key == COLUMNS[1]: return sigma('EXCITATION',energy)
        if key == COLUMNS[3]: return sigma('EFFECTIVE',energy)
        elastic = sigma('EFFECTIVE',energy)-sigma('EXCITATION',energy)-sigma('IONIZATION',energy)
        return elastic*energy if key == COLUMNS[4] else elastic
    reference = []; maximum_error = 0.
    for t in (1.,1.7,3.,5.3,7.,10.,13.7,20.):
        row = {'temperature_eV':t}
        for key in COLUMNS:
            factor = math.sqrt(8*argon.E_CHARGE/(math.pi*argon.M_E))/t**1.5
            expected = factor*sum(quad(lambda energy:cross(key,energy)*energy*math.exp(-energy/t),
                lo,hi,epsabs=1e-40,epsrel=1e-11)[0] for lo,hi in zip(knots[:-1],knots[1:]))
            actual = rates(t,{'chemistry_model':PHELPS})[key]
            error = abs(actual-expected)/expected
            maximum_error = max(maximum_error,error)
            row[key] = expected; row[key+'_interpolation_relative_error'] = error
        reference.append(row)
    result = {'source_cross_sections_sha256':table()['provenance']['cross_sections_sha256'],
        'independent_method':'scipy QUADPACK adaptive integration on each cross-section knot interval',
        'max_relative_rate_error':maximum_error,'reference':reference}
    (out/'argon-phelps-reference.json').write_text(json.dumps(result,indent=2)+'\n')
    rows = []
    for t in np.linspace(1.,7.,61):
        new = rates(float(t),{'chemistry_model':PHELPS}); old = argon.rates(float(t))
        row = {'Te_eV':float(t)}
        for key in old:
            row[key+'_legacy']=old[key];row[key+'_phelps']=new[key];row[key+'_ratio']=new[key]/old[key]
        row['excitation_loss_ratio']=new['excitation_energy_eVm3_s']/(old['excitation_m3_s']*argon.EXCITATION_EV)
        rows.append(row)
    with (out/'argon-chemistry-overlap.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if maximum_error > 1e-5: raise ValueError('Rate interpolation/integration verification failed')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cross-sections',required=True);p.add_argument('--out',required=True)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    if args.verify:
        result=verify(args.cross_sections,args.out)
        print('Independent quadrature maximum relative error: '+str(result['max_relative_rate_error']))
    else:
        build(args.cross_sections,args.out)
        print('Generated Maxwellian calculation table: '+args.out)


if __name__=='__main__':main()
