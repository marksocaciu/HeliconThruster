"""Configuration and geometry validation; all lengths are metres."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path=None):
    c = json.loads(Path(path or ROOT / 'config/geometry.json').read_text())
    validate(c)
    return c


def rectangles(c):
    h, d = c['chamber'], c['downstream']
    end = h['z_start'] + h['length']
    result = [
        ('PLASMA_CHAMBER', 1, (0, h['z_start'], h['radius'], h['length'])),
        ('CHAMBER_WALL', 2, (h['radius'], h['z_start'], h['wall_thickness'], h['length'])),
        ('DOWNSTREAM_REGION', 3, (0, end, d['radius'], d['length'])),
    ]
    for i, coil in enumerate(c['coils']):
        result.append((coil['name'].upper(), 101 + i, (
            coil['radius'] - coil['radial_thickness']/2,
            coil['z_center'] - coil['axial_width']/2,
            coil['radial_thickness'], coil['axial_width'])))
    return result


def validate(c):
    def number(obj, key, positive=False):
        x = obj[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or (positive and x <= 0):
            raise ValueError(f'{key} must be a finite {"positive " if positive else ""}number')
    for section, keys in [('chamber', ['radius','length','wall_thickness']),
                          ('domain', ['radius']), ('downstream', ['radius','length']),
                          ('mesh', ['chamber','wall','coil','downstream','far'])]:
        for key in keys:
            number(c[section], key, True)
    for section, keys in [('chamber',['z_start']), ('domain',['z_min','z_max'])]:
        for key in keys:
            number(c[section], key)
    if not c['coils']:
        raise ValueError('At least one coil is required')
    names = {'PLASMA_CHAMBER','CHAMBER_WALL','DOWNSTREAM_REGION','SURROUNDING_AIR','AXIS','FAR_BOUNDARY'}
    for coil in c['coils']:
        import re
        if not isinstance(coil['name'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', coil['name']):
            raise ValueError('Coil names must be identifiers')
        name = coil['name'].upper()
        if name in names:
            raise ValueError('Physical names must be unique (case insensitive)')
        names.add(name)
        for key in ['radius','radial_thickness','axial_width','turns']:
            number(coil, key, True)
        if int(coil['turns']) != coil['turns']:
            raise ValueError('turns must be an integer')
        for key in ['z_center','current']:
            number(coil, key)
        if coil['radius'] - coil['radial_thickness']/2 <= 0:
            raise ValueError('Coil must stay away from the axis')
    d = c['domain']
    rects = rectangles(c)
    for name, _, (r,z,w,h) in rects:
        if r < 0 or r+w >= d['radius'] or z <= d['z_min'] or z+h >= d['z_max']:
            raise ValueError(f'{name} must lie inside the outer domain')
    for i, (a,_,(r,z,w,h)) in enumerate(rects):
        for b,_,(s,t,v,k) in rects[i+1:]:
            if min(r+w,s+v)-max(r,s) > 1e-12 and min(z+h,t+k)-max(z,t) > 1e-12:
                raise ValueError(f'Overlapping regions: {a} and {b}')
