import json

import cv2
import numpy as np

from eggvision.classical import ClassicalConfig, segment_classical
from eggvision.experiment import run_classical_experiment
from eggvision.pseudo_agreement import run_pseudo_agreement


def test_synthetic_manifest_comparison_is_labeled_pseudo_agreement(tmp_path):
    """Exercise file wiring only; this fixture is not a real SAM inference."""
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    for index in range(5):
        image = np.full((80, 100, 3), 25, dtype=np.uint8)
        mask = np.zeros((80, 100), dtype=np.uint8)
        center = (25 + index * 10, 40)
        cv2.ellipse(image, center, (13, 9), index * 3, 0, 360, (235, 235, 235), -1)
        cv2.ellipse(mask, center, (13, 9), index * 3, 0, 360, 255, -1)
        assert cv2.imwrite(str(dataset / "images" / f"egg_{index}.png"), image)
        assert cv2.imwrite(str(dataset / "masks" / f"egg_{index}.png"), mask)

    experiment_dir = tmp_path / "classical"
    experiment = run_classical_experiment(dataset, experiment_dir, seed=19)
    assert experiment["status"] == "completed"
    eval_key = experiment["evaluation"]["per_image"][0]["image"]
    image = cv2.imread(str(dataset / eval_key))
    pseudo_mask = segment_classical(image, ClassicalConfig(**experiment["selected_config"]))

    sam_dir = tmp_path / "sam"
    union_relative = f"masks/{eval_key.replace('/', '__')}/union.png"
    union_path = sam_dir / union_relative
    union_path.parent.mkdir(parents=True)
    assert cv2.imwrite(str(union_path), pseudo_mask)
    (sam_dir / "sam_manifest.json").write_text(json.dumps({
        "status": "test_fixture_only",
        "image_outputs": [{"image": eval_key, "mode": "synthetic test stub", "union_mask_file": union_relative}],
    }), encoding="utf-8")

    result = run_pseudo_agreement(
        dataset,
        experiment_dir / "experiment.json",
        sam_dir / "sam_manifest.json",
        tmp_path / "agreement",
    )
    assert result["status"] == "completed"
    assert result["compared_images"] == 1
    assert "unverified SAM pseudo-labels" in result["interpretation_warning"]
    assert (tmp_path / "agreement" / "qualitative" / f"{eval_key.replace('/', '__').removesuffix('.png')}_sam_agreement.png").is_file()
