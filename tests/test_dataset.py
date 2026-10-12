"""Tests for the TB-Sentry dataset."""

from pathlib import Path

import torch

from tb_sentry.data.dataset import TBX11KDataset


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "interim" / "tbx11k_manifest.csv"


def test_training_dataset_counts():
    dataset = TBX11KDataset(MANIFEST, split="train")

    assert len(dataset) == 6600


def test_validation_dataset_counts():
    dataset = TBX11KDataset(MANIFEST, split="val")

    assert len(dataset) == 1800


def test_training_sample():
    dataset = TBX11KDataset(MANIFEST, split="train")
    sample = dataset[0]

    assert sample["image"].shape == (3, 224, 224)
    assert sample["label"].item() in (0, 1)
    assert isinstance(sample["image_path"], str)
    assert isinstance(sample["boxes"], list)
    assert isinstance(sample["localization_categories"], list)
    assert sample["annotation_status"] in {
        "boxes_present",
        "no_boxes",
    }


def test_validation_sample():
    dataset = TBX11KDataset(MANIFEST, split="val")
    sample = dataset[0]

    assert sample["image"].shape == (3, 224, 224)
    assert sample["label"].item() in (0, 1)


def test_test_split_is_rejected():
    try:
        TBX11KDataset(MANIFEST, split="test")
    except ValueError:
        return

    raise AssertionError("The test split should be rejected.")