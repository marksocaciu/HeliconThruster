"""A retry must solve the archived endpoint and reject modified inputs."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helicon import spatial
from helicon.config import load as load_geometry
from helicon.records import digest, write_json
from helicon.rf.retry_endpoint import retry


class RetryEndpoint(unittest.TestCase):
    def test_reuses_saved_target_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "continuation"
            step = source / "steps" / "step-040"
            geometry = load_geometry()
            config = spatial.load()
            config.update(nr=3, nz=4, max_nfev=1)
            grid = spatial.Grid(geometry, 3, 4)
            heating = np.broadcast_to(np.linspace(.8, 1.2, 4)[:, None], grid.shape).copy()
            heating *= config["absorbed_power_W"] / np.sum(heating * grid.volumes)
            summary, fields = spatial.solve(geometry, config, heating)
            self.assertEqual(summary["status"], "not_converged")
            spatial.save(step, summary, fields)
            write_json(source / "geometry.json", geometry)
            write_json(source / "plasma_config.json", config)
            np.savez(source / "heating_endpoints.npz", target_rf_q_W_m3=heating)
            manifest = {str(path.relative_to(source)): digest(path)
                        for path in source.rglob("*") if path.is_file()}
            write_json(source / "continuation_summary.json", {
                "status": "not_converged",
                "history": [{"alpha": 1.0, "accepted": False, "directory": "steps/step-040"}],
                "artifacts_sha256": manifest,
            })
            result = retry(source, root / "retry", 400)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["input"], dict(config, max_nfev=400))
            with np.load(root / "retry" / "endpoint" / "plasma_fields.npz") as data:
                np.testing.assert_array_equal(data["q_abs_W_m3"], heating)
            for name, sha in manifest.items():
                self.assertEqual(digest(source / name), sha)
            np.savez(source / "heating_endpoints.npz", target_rf_q_W_m3=heating * 2)
            with self.assertRaisesRegex(ValueError, "manifest"):
                retry(source, root / "changed", 400)
            self.assertFalse((root / "changed").exists())
