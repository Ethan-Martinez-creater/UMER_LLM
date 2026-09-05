import importlib.util
import unittest
from pathlib import Path


RUNNER = (
    Path(__file__).resolve().parents[1]
    / "round_013_joint_trifusion"
    / "run_fold.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("round13_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FourMetricMeanSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_runner()

    def test_uses_arithmetic_mean_of_exactly_four_reported_metrics(self):
        metrics = {
            "accuracy": 0.91,
            "weighted_f1": 0.92,
            "macro_f1": 0.89,
            "rumor_f1": 0.86,
            "loss": 99.0,
        }
        self.assertEqual(
            self.runner.checkpoint_key(metrics, "four_metric_mean"),
            ((0.91 + 0.92 + 0.89 + 0.86) / 4,),
        )

    def test_epoch17_beats_epoch11_for_round32_fold2(self):
        epoch11 = {
            "accuracy": 0.9085603112840467,
            "weighted_f1": 0.909389851531241,
            "macro_f1": 0.9042546023819432,
            "rumor_f1": 0.8839506172839506,
        }
        epoch17 = {
            "accuracy": 0.9124513618677043,
            "weighted_f1": 0.9124969831410338,
            "macro_f1": 0.9065662176082083,
            "rumor_f1": 0.8831168831168831,
        }
        self.assertGreater(
            self.runner.checkpoint_key(epoch17, "four_metric_mean"),
            self.runner.checkpoint_key(epoch11, "four_metric_mean"),
        )


if __name__ == "__main__":
    unittest.main()
