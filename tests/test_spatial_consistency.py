"""Independent tests for spatial-consistency metric calculations."""

import numpy as np

from scripts.evaluate_spatial_consistency import (
    boxes_to_mask,
    calculate_metrics,
)


def test_perfect_overlap():
    """A heatmap exactly matching the box should score 1.0."""
    heatmap = np.zeros((224, 224), dtype=float)
    heatmap[10:20, 10:20] = 1.0

    box_mask = np.zeros((224, 224), dtype=bool)
    box_mask[10:20, 10:20] = True

    metrics = calculate_metrics(heatmap, box_mask)

    assert metrics["box_coverage"] == 1.0
    assert metrics["heatmap_precision"] == 1.0


def test_no_overlap():
    """Disjoint highlighted and annotated regions should score zero."""
    heatmap = np.zeros((224, 224), dtype=float)
    heatmap[10:20, 10:20] = 1.0

    box_mask = np.zeros((224, 224), dtype=bool)
    box_mask[30:40, 30:40] = True

    metrics = calculate_metrics(heatmap, box_mask)

    assert metrics["box_coverage"] == 0.0
    assert metrics["heatmap_precision"] == 0.0


def test_partial_overlap():
    """Known overlap should produce predictable metrics."""
    heatmap = np.zeros((224, 224), dtype=float)
    heatmap[10:20, 10:20] = 1.0

    box_mask = np.zeros((224, 224), dtype=bool)
    box_mask[15:25, 10:20] = True

    metrics = calculate_metrics(heatmap, box_mask)

    # 50 of the 100 highlighted pixels overlap the box.
    assert metrics["box_coverage"] == 0.5
    assert metrics["heatmap_precision"] == 0.5


def test_box_scaling():
    """A 512x512 box should scale correctly to a 224x224 mask."""
    boxes = [[128.0, 128.0, 128.0, 128.0]]

    mask, transformed_boxes = boxes_to_mask(
        boxes,
        original_width=512,
        original_height=512,
    )

    # Coordinates scale by 224/512 = 0.4375.
    expected = {
        "x1": 56,
        "y1": 56,
        "x2": 112,
        "y2": 112,
    }

    assert transformed_boxes[0] == expected
    assert mask.sum() == 56 * 56


def test_union_of_multiple_boxes():
    """Multiple boxes should be combined without double-counting pixels."""
    boxes = [
        [0, 0, 100, 100],
        [50, 50, 100, 100],
    ]

    mask, _ = boxes_to_mask(
        boxes,
        original_width=224,
        original_height=224,
    )

    expected_area = 100 * 100 + 100 * 100 - 50 * 50

    assert mask.sum() == expected_area