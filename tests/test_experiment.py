import cv2
import numpy as np

from eggvision.experiment import run_classical_experiment


def test_experiment_selects_on_dev_and_writes_eval_artifacts(tmp_path):
    data = tmp_path / "dataset"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir()
    for index in range(5):
        image = np.full((80, 100, 3), 25, dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=np.uint8)
        center = (25 + index * 10, 40)
        cv2.ellipse(image, center, (13, 9), index * 3, 0, 360, (235, 235, 235), -1)
        cv2.ellipse(mask, center, (13, 9), index * 3, 0, 360, 255, -1)
        assert cv2.imwrite(str(data / "images" / f"egg_{index}.png"), image)
        assert cv2.imwrite(str(data / "masks" / f"egg_{index}.png"), mask)

    output = tmp_path / "run"
    result = run_classical_experiment(data, output, seed=19)
    assert result["status"] == "completed"
    assert result["split"]["development_images"] == 4
    assert result["split"]["evaluation_images"] == 1
    assert result["protocol"]["evaluation_used_for_selection"] is False
    candidates = result["development_candidate_results"]
    assert sum(candidate["selected"] for candidate in candidates) == 1
    assert all(candidate["rationale"] and candidate["selection_rationale"] for candidate in candidates)
    assert sum(candidate["decision"] == "KEEP_SELECTED" for candidate in candidates) == 1
    assert sum(candidate["decision"] == "DISCARD" for candidate in candidates) == 8
    assert result["evaluation"]["summary"]["image_count"] == 1
    assert (output / "experiment.json").is_file()
    assert (output / "development_candidates.csv").is_file()
    assert (output / "evaluation_per_image.csv").is_file()
    assert (output / "selected_config.json").is_file()
    assert list((output / "qualitative").glob("*_errors.png"))
