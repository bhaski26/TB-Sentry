import numpy as np


def analyze_heatmap(heatmap: np.ndarray) -> dict:
    """
    Analyze the distribution of Grad-CAM activation.

    This function measures activation concentration using normalized entropy.
    It does NOT measure anatomical correctness or lesion localization accuracy.

    Returns:
        dict containing:
            - normalized_entropy: 0 to 1
            - concentration_score: 0 to 1
            - activated_area_ratio: 0 to 1
    """

    heatmap = np.asarray(heatmap, dtype=np.float32)

    if heatmap.size == 0:
        raise ValueError("Heatmap cannot be empty.")

    if not np.isfinite(heatmap).all():
        raise ValueError("Heatmap contains non-finite values.")

    heatmap = np.clip(heatmap, 0.0, None)

    total_activation = float(heatmap.sum())

    if total_activation <= 1e-8:
        return {
            "normalized_entropy": 1.0,
            "concentration_score": 0.0,
            "activated_area_ratio": 0.0,
        }

    probabilities = heatmap.flatten() / total_activation
    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12))
    max_entropy = np.log(heatmap.size)

    normalized_entropy = (
        float(entropy / max_entropy)
        if max_entropy > 0
        else 0.0
    )

    concentration_score = float(
        np.clip(1.0 - normalized_entropy, 0.0, 1.0)
    )

    threshold = 0.5 * float(heatmap.max())
    activated_area_ratio = float(
        np.mean(heatmap >= threshold)
    )

    return {
        "normalized_entropy": float(normalized_entropy),
        "concentration_score": concentration_score,
        "activated_area_ratio": activated_area_ratio,
    }