import pytest
from deepforest import main, get_data

def test_benchmark_release():
    """
    Benchmark test to ensure the specific release version of the model
    produces consistent results.
    """
    # Load the model using a SPECIFIC revision (Commit SHA)
    release_sha = "cc21436bc5d572dde8ff5f93c1e71a32f563cace"

    m = main.deepforest()
    m.load_model("weecology/deepforest-tree", revision=release_sha)

    csv_file = get_data("OSBS_029.csv")
    results = m.evaluate(csv_file, iou_threshold=0.4)

    # Strict Assertions (for The "Benchmark")
    assert results["box_precision"] == pytest.approx(0.8, abs=0.01)
    assert results["box_recall"] == pytest.approx(0.7213, abs=0.01)
