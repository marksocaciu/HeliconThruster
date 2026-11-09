"""Retry the saved alpha=1 iterate with a fresh nonlinear evaluation budget."""
import argparse
from pathlib import Path
import sys

import numpy as np

from .. import spatial
from ..records import digest, read_json, write_json
from ..spatial_budget import reconstruct
from .prescribed import metrics, validate_response


def retry(source, out, max_nfev):
    source, out = Path(source).resolve(), Path(out)
    if type(max_nfev) is not int or max_nfev <= 0:
        raise ValueError("max_nfev must be a positive integer")
    parent = read_json(source / "continuation_summary.json")
    candidate = parent["history"][-1]
    if parent["status"] != "not_converged" or candidate["alpha"] != 1 or candidate["accepted"]:
        raise ValueError("Require a continuation with a rejected alpha=1 endpoint")
    folder = (source / candidate["directory"]).resolve()
    if source not in folder.parents:
        raise ValueError("Saved endpoint directory must be inside the continuation")
    paths = [source / name for name in ("geometry.json", "plasma_config.json", "heating_endpoints.npz")]
    summary_path = folder / "spatial_summary.json"
    fields_path = folder / "plasma_fields.npz"
    paths.extend((summary_path, fields_path))
    hashes = {str(path.relative_to(source)): digest(path) for path in paths}
    if any(parent["artifacts_sha256"].get(name) != sha for name, sha in hashes.items()):
        raise ValueError("Saved endpoint artifacts differ from the continuation manifest")
    geometry = read_json(paths[0])
    config = spatial.load(paths[1])
    summary = read_json(summary_path)
    if summary["status"] != "not_converged":
        raise ValueError("Saved transport status differs from continuation history")
    with np.load(fields_path, allow_pickle=False) as data:
        fields = {key: data[key] for key in data.files}
    grid, _, checks, _ = reconstruct(geometry, config, summary, fields)
    with np.load(paths[2], allow_pickle=False) as data:
        heating = data["target_rf_q_W_m3"]
    if not np.array_equal(heating, fields["q_abs_W_m3"]):
        raise ValueError("Saved iterate heating differs from the alpha=1 target")
    if not np.isclose(np.sum(heating * grid.volumes), config["absorbed_power_W"],
                      rtol=1e-12, atol=0):
        raise ValueError("Saved endpoint heating differs from the prescribed power")
    config["max_nfev"] = max_nfev

    out.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "alpha": 1.0, "source": str(source),
        "initialization": "saved rejected iterate at the same alpha",
        "input_sha256": hashes, "input": config,
        "previous_solver": summary["solver"],
        "previous_local_residuals": checks["reconstructed_max_local_residuals"],
    }
    write_json(out / "retry_summary.json", record)
    try:
        result, solved = spatial.solve(geometry, config, heating, initial=fields)
        spatial.save(out / "endpoint", result, solved)
        record.update(solver=result["solver"], balances=result["balances"],
                      metrics=metrics(result, solved, grid))
        try:
            validate_response(result, solved)
        except RuntimeError as exc:
            record.update(status="not_converged", error=str(exc))
        else:
            record["status"] = "completed"
        if any(digest(source / name) != sha for name, sha in hashes.items()):
            raise ValueError("Saved endpoint inputs changed during retry")
    except (Exception, KeyboardInterrupt) as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      error=str(exc))
        raise
    finally:
        write_json(out / "retry_summary.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-nfev", type=int, required=True)
    args = parser.parse_args()
    try:
        result = retry(args.continuation, args.out, args.max_nfev)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Endpoint retry stopped: {exc}", file=sys.stderr)
        return 2
    print(f"Endpoint {result['status']} after {result['solver']['nfev']} evaluations")
    print(result["balances"]["max_local_relative_residuals"])
    print(f"See {args.out}/retry_summary.json")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
