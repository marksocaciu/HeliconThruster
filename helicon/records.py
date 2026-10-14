"""Portable, strict JSON records and integrity checks for scientific runs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from .config import ROOT


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_record(out):
    """Snapshot only this program, its dependency recipes and Git identity."""
    destination = Path(out) / 'source'
    destination.mkdir()
    files = list((ROOT / 'helicon').glob('*.py'))
    files += [ROOT / name for name in ('requirements.txt', 'environment.yml')]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(target)
    def git(*args):
        try:
            return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    dirty = git('status', '--porcelain')
    return {'sha256': hashes, 'git_commit': git('rev-parse', 'HEAD'),
            'git_dirty': bool(dirty) if dirty is not None else None,
            'python': sys.version, 'platform': platform.platform()}
