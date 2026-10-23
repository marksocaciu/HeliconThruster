"""Durable diagnostics; checkpoints are evidence, not automatic restart files."""
from pathlib import Path
from ..records import write_json, read_json, utc_now


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, data)
    temporary.replace(path)


def save_accepted(out, entry):
    atomic_json(Path(out)/'last_accepted.json', {
        'status': 'accepted_iteration', 'saved_utc': utc_now(),
        'note': 'Accepted numerical checks, not proof of mesh or physical convergence.',
        **entry,
    })


def save_failure(out, record, history, exc, stage):
    """Keep accepted state separate from the rejected candidate and its diagnostics."""
    out = Path(out)
    status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else (
        'not_converged' if record.get('status') == 'not_converged' else 'failed')
    accepted = out/'last_accepted.json'
    result = {'model': record['model'], 'status': status, 'failed_stage': stage,
              'error': f'{type(exc).__name__}: {exc}', 'iteration_count': len(history),
              'history': history, 'last_accepted': read_json(accepted) if accepted.exists() else None,
              'finished_utc': utc_now(),
              'note': 'Failed candidate was not accepted. Diagnostics remain in iterations/. No automatic retry or restart was performed.'}
    atomic_json(out/'failure.json', result)
    atomic_json(out/'coupling_summary.json', result)
    return result
