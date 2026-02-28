import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluate import compute_accuracy, compute_mae, compute_correlation, compute_mistake_metrics


class TestComputeAccuracy:
    def test_perfect(self):
        assert compute_accuracy(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_none_correct(self):
        assert compute_accuracy(["a", "b"], ["c", "d"]) == 0.0

    def test_partial(self):
        assert compute_accuracy(["a", "b", "c", "d"], ["a", "b", "x", "x"]) == 0.5

    def test_empty(self):
        assert compute_accuracy([], []) == 0


class TestComputeMAE:
    def test_perfect(self):
        assert compute_mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_off_by_one(self):
        assert compute_mae([1, 2, 3], [2, 3, 4]) == 1.0

    def test_mixed(self):
        assert compute_mae([5, 5], [3, 5]) == 1.0

    def test_empty(self):
        assert compute_mae([], []) == 0


class TestComputeCorrelation:
    def test_perfect_positive(self):
        r = compute_correlation([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
        assert abs(r - 1.0) < 0.001

    def test_perfect_negative(self):
        r = compute_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
        assert abs(r - (-1.0)) < 0.001

    def test_no_variance(self):
        r = compute_correlation([3, 3, 3], [1, 2, 3])
        assert r == 0

    def test_single_element(self):
        assert compute_correlation([1], [1]) == 0


class TestComputeMistakeMetrics:
    def test_perfect_detection(self):
        dataset = [
            {"id": 1, "ground_truth": {"agent_mistakes": ["rude_tone"]}},
            {"id": 2, "ground_truth": {"agent_mistakes": []}},
        ]
        analysis_map = {
            1: {"agent_mistakes": ["rude_tone"]},
            2: {"agent_mistakes": []},
        }
        metrics = compute_mistake_metrics(dataset, analysis_map)
        assert metrics["rude_tone"]["tp"] == 1
        assert metrics["rude_tone"]["fp"] == 0
        assert metrics["rude_tone"]["fn"] == 0
        assert metrics["rude_tone"]["f1"] == 1.0

    def test_false_positive(self):
        dataset = [
            {"id": 1, "ground_truth": {"agent_mistakes": []}},
        ]
        analysis_map = {
            1: {"agent_mistakes": ["no_resolution"]},
        }
        metrics = compute_mistake_metrics(dataset, analysis_map)
        assert metrics["no_resolution"]["fp"] == 1
        assert metrics["no_resolution"]["precision"] == 0

    def test_false_negative(self):
        dataset = [
            {"id": 1, "ground_truth": {"agent_mistakes": ["ignored_question"]}},
        ]
        analysis_map = {
            1: {"agent_mistakes": []},
        }
        metrics = compute_mistake_metrics(dataset, analysis_map)
        assert metrics["ignored_question"]["fn"] == 1
        assert metrics["ignored_question"]["recall"] == 0
