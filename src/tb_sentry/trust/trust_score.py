from typing import Dict, Any


def calculate_trust(
    confidence: float,
    concentration_score: float,
    normalized_entropy: float,
    activated_area_ratio: float,
) -> Dict[str, Any]:
    """
    Calculate an experimental Trust Score for TB-Sentry.

    This score is a heuristic combining classifier confidence and
    Grad-CAM activation distribution characteristics.

    It is NOT a clinical reliability score.
    """

    confidence = max(0.0, min(1.0, confidence))
    concentration_score = max(0.0, min(1.0, concentration_score))

    # Experimental weighted score
    score = (0.70 * confidence + 0.30 * concentration_score) * 100

    if score >= 80:
        category = "High"
    elif score >= 60:
        category = "Moderate"
    else:
        category = "Low"

    reasons = []

    if confidence < 0.60:
        reasons.append("Low model confidence")

    if normalized_entropy > 0.85:
        reasons.append("Highly diffuse Grad-CAM activation")

    if activated_area_ratio > 0.70:
        reasons.append("Activation covers a large portion of the image")

    if concentration_score < 0.20:
        reasons.append("Grad-CAM activation is weakly concentrated")

    if not reasons:
        reasons.append("No predefined heuristic warning was triggered")

    review_required = (
        confidence < 0.60
        or normalized_entropy > 0.85
        or activated_area_ratio > 0.70
        or concentration_score < 0.20
    )

    return {
        "trust_score": round(score, 2),
        "trust_category": category,
        "review_required": review_required,
        "reasons": reasons,
        "components": {
            "confidence": round(confidence, 4),
            "concentration_score": round(concentration_score, 4),
            "normalized_entropy": round(normalized_entropy, 4),
            "activated_area_ratio": round(activated_area_ratio, 4),
        },
    }