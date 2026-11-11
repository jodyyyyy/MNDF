"""Scene regeneration checks; run with python -m unittest discover -s tests -v."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mndf.scenes import ROOT, generate


class SceneGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / "inputs"
        cls.manifest = generate(cls.output, examples=True, samples=2)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def load(self, kind, task, index=1):
        path = self.output / kind / f"{task}_heter_1_rc60_ra10_rs30_{index}.json"
        return json.loads(path.read_text())

    def test_bundled_examples_reproduced(self):
        for path in (ROOT / "examples").rglob("*.json"):
            if path.parent.name not in ("NodeInitialLocationData", "TaskInfo"):
                continue
            actual = json.loads((self.output / path.relative_to(ROOT / "examples")).read_text())
            expected = json.loads(path.read_text())
            if path.parent.name == "NodeInitialLocationData":
                self.assertEqual(actual, expected)
            else:
                self.assertEqual(actual.keys(), expected.keys())
                for key in expected:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-10)

    def test_fixed_initial_search_robots_and_changing_tasks(self):
        first = self.load("NodeInitialLocationData", "searchandrescue", 1)
        second = self.load("NodeInitialLocationData", "searchandrescue", 2)
        robots = lambda nodes: [node for node in nodes if "environment" not in node["type"]]
        tasks = lambda nodes: [node for node in nodes if node["type"] == "environmentTask"]
        self.assertEqual(robots(first), robots(second))
        self.assertEqual(len(robots(first)), 63)
        self.assertEqual(len(tasks(first)), 60)
        self.assertNotEqual(tasks(first), tasks(second))
        self.assertEqual([n["id"] for n in first], list(range(len(first))))

    def test_seed_reproducibility_and_overwrite_protection(self):
        repeat = Path(self.temporary.name) / "repeat"
        generate(repeat, examples=True, samples=2)
        for path in self.output.rglob("*.json"):
            self.assertEqual(path.read_bytes(), (repeat / path.relative_to(self.output)).read_bytes())
        with self.assertRaises(FileExistsError):
            generate(repeat, examples=True, samples=2)

    def test_changed_seed_and_relay_only_communication_range(self):
        other = Path(self.temporary.name) / "other"
        generate(other, config_name="searchandrescue_heter_1_rc90_ra5_rs15", seed=43)
        path = other / "NodeInitialLocationData/searchandrescue_heter_1_rc90_ra5_rs15_1.json"
        nodes = json.loads(path.read_text())
        robots = [n for n in nodes if "environment" not in n["type"]]
        for node in robots:
            self.assertEqual(node["r_c"], 90 if node["type"] == "relay" else 30)
            if node["type"] == "action":
                self.assertEqual(node["r_a"], 5)
            if node["type"] == "sensing":
                self.assertEqual(node["r_s"], 15)
        original_tasks = [n["pos"] for n in self.load("NodeInitialLocationData", "searchandrescue") if n["type"] == "environmentTask"]
        self.assertNotEqual(original_tasks, [n["pos"] for n in nodes if n["type"] == "environmentTask"])

    def test_invalid_requests(self):
        for options in ({"samples": 0}, {"seed": -1}, {"task": "invalid"}, {"config_name": "missing"}, {"config_name": "crossing_heter_1_rc60_ra10_rs30", "task": "searchandrescue"}):
            with self.assertRaises(ValueError):
                generate(Path(self.temporary.name) / "invalid", **options)


if __name__ == "__main__":
    unittest.main()
