"""RF memory measurements and conservative admission checks (decimal MB)."""
import os
from pathlib import Path


def process_rss_bytes():
    """Current resident memory; never mislabel a per-process high-water mark."""
    try:
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return None


def cgroup_memory():
    """Current cgroup-v2 counters, including children and charged file cache."""
    try:
        entry = next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
        directory = Path('/sys/fs/cgroup')/entry[3:].lstrip('/')
        result = {}
        for name in ('current', 'peak', 'max', 'swap.current', 'swap.max'):
            path = directory/('memory.'+name)
            if path.exists():
                value = path.read_text().strip()
                result[name] = None if value == 'max' else int(value)
        return result
    except (OSError, ValueError, StopIteration):
        return {}


def memory_snapshot(comm, stage):
    local = process_rss_bytes()
    ranks = comm.allgather(local)
    return {'stage': stage, 'rank_rss_bytes': ranks,
            'sum_rank_rss_bytes': sum(ranks) if all(x is not None for x in ranks) else None,
            'cgroup': cgroup_memory() if comm.rank == 0 else {}}


def admission(analysis, baseline_bytes, job_memory_mb, reserve_mb, workspace_mb=None):
    """Estimates are admission criteria, not guarantees of peak consumption."""
    reasons = []
    required = analysis['estimated_total_mb']
    if required <= 0:
        reasons.append('MUMPS did not return a positive memory estimate')
    if workspace_mb is not None:
        for rank, value in enumerate(analysis['rank_estimated_mb']):
            if value > workspace_mb:
                reasons.append(f'rank {rank} estimate {value} MB exceeds workspace {workspace_mb} MB')
    projected = None
    if job_memory_mb is not None:
        if baseline_bytes is None:
            reasons.append('Current RSS unavailable: cannot enforce the estimate gate')
        else:
            projected = baseline_bytes/1e6 + required + reserve_mb
            if projected > job_memory_mb:
                reasons.append(f'projected {projected:.1f} MB exceeds job budget {job_memory_mb} MB')
    return {'accepted': not reasons, 'reasons': reasons,
            'projected_job_mb': projected, 'reserve_mb': reserve_mb,
            'job_memory_mb': job_memory_mb}
