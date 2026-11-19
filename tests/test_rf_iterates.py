import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

from helicon.rf.iterates import IterateCheckpoint

class Comm:
    rank = 0
    def allgather(self, x): return [x]
    def bcast(self, x, root=0): return x
class Vec:
    def __init__(self, values): self.values = np.array(values, dtype=complex)
    def getArray(self, readonly=False): return self.values
class KSP:
    def getOperators(self): return None, None
    def setInitialGuessNonzero(self, value): self.nonzero = value
    def setTolerances(self, **kw): self.tolerances = kw

class Checkpoints(unittest.TestCase):
    @patch('helicon.rf.iterates.assembly_fingerprint', return_value={'system': 1})
    def test_resume_retention_corruption_and_mismatch(self, fingerprint):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            ksp = KSP()
            checkpoint = IterateCheckpoint(Comm(), path, None, total_limit=10000)
            checkpoint.prepare(ksp, None, Vec([0, 0]))
            checkpoint.save(Vec([1, 2]), 9900, 0.1)
            checkpoint.save(Vec([3, 4]), 10000, 0.2)
            checkpoint.save(Vec([float('nan'), 4]), 10001, 1)
            self.assertEqual(json.loads((path/'latest.json').read_text())['iteration'], 10000)
            self.assertEqual(json.loads((path/'best.json').read_text())['iteration'], 9900)
            self.assertEqual(len(list(path.glob('*.npy'))), 2)
            restarted = IterateCheckpoint(Comm(), path, None, resume=path/'latest.json', total_limit=13000)
            x = Vec([0, 0])
            restarted.prepare(ksp, None, x)
            np.testing.assert_array_equal(x.values, [3, 4])
            self.assertEqual(ksp.tolerances['max_it'], 3000)
            restarted.save(Vec([5, 6]), 100, .05)
            self.assertEqual(json.loads((path/'latest.json').read_text())['iteration'], 10100)
            fingerprint.return_value = {'system': 2}
            with self.assertRaises(ValueError): restarted.prepare(ksp, None, x)
            fingerprint.return_value = {'system': 1}
            manifest = json.loads((path/'latest.json').read_text())
            np.save(path/manifest['files'][0], np.array([9, 9], dtype=complex))
            with self.assertRaises(ValueError): restarted.prepare(ksp, None, x)

if __name__ == '__main__': unittest.main()
