"""
TB-Sentry: Explainable Tuberculosis Screening Research Dashboard
UI 2.0 Redesign

Academic research prototype only.
Not intended for clinical diagnosis or medical decision-making.
"""

import sys
from pathlib import Path

# Ensure project root and src directory are in Python path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from src.tb_sentry.classification.densenet import build_densenet121
from src.tb_sentry.data.transforms import build_evaluation_transform
from tb_sentry.trust import analyze_heatmap, calculate_trust


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "checkpoints"
    / "densenet121_baseline.pt"
)

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

CLASS_NAMES = {
    0: "Non-TB",
    1: "TB",
}


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TB-Sentry — Medical AI Research Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS — PREMIUM DARK MEDICAL AI INTERFACE
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Base application styling — targeted without breaking Streamlit text children */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background-color: #0B1120 !important;
    color: #F1F5F9 !important;
}

/* Ensure markdown containers do NOT paint opaque background boxes over dark cards */
div[data-testid="stMarkdownContainer"],
div[data-testid="stMarkdownContainer"] *,
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] span,
div[data-testid="stMarkdownContainer"] div {
    background-color: transparent !important;
}

/* Restrain main container width for optimal editorial balance */
.main .block-container {
    max-width: 1140px !important;
    padding-top: 1.75rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0D1526 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem !important;
}

/* Header container */
.tbs-header-wrapper {
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.tbs-header-top {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.tbs-badge-prototype {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #53D8D0;
    background: rgba(83, 216, 208, 0.1);
    border: 1px solid rgba(83, 216, 208, 0.25);
    border-radius: 4px;
}

.tbs-badge-meta {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    color: #94A3B8;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 4px;
}

.tbs-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #F1F5F9;
    margin: 0 0 0.25rem 0;
    line-height: 1.2;
}

.tbs-subtitle {
    font-size: 0.95rem;
    font-weight: 400;
    color: #94A3B8;
    margin: 0 0 0.85rem 0;
    line-height: 1.4;
}

.tbs-metadata-strip {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.5rem 1rem;
    font-size: 0.78rem;
    color: #64748B;
}

.tbs-meta-item {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    color: #94A3B8;
}

.tbs-meta-dot {
    color: rgba(255, 255, 255, 0.15);
}

/* Disclaimer panel */
.tbs-disclaimer-panel {
    background: #131C2E;
    border: 1px solid rgba(83, 216, 208, 0.2);
    border-left: 3px solid #53D8D0;
    border-radius: 8px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 2rem;
    display: flex;
    gap: 0.85rem;
    align-items: flex-start;
}

.tbs-disclaimer-icon {
    flex-shrink: 0;
    margin-top: 0.15rem;
}

.tbs-disclaimer-content {
    font-size: 0.82rem;
    line-height: 1.5;
    color: #CBD5E1;
}

.tbs-disclaimer-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #53D8D0;
    margin-bottom: 0.2rem;
}

/* Surface cards */
.tbs-card {
    background: #131C2E;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
}

.tbs-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.tbs-section-title {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94A3B8;
    margin: 0;
}

.tbs-section-badge {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: #53D8D0;
    background: rgba(83, 216, 208, 0.08);
    border: 1px solid rgba(83, 216, 208, 0.18);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}

/* File Uploader integration */
div[data-testid="stFileUploader"] {
    width: 100% !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
    background-color: #131C2E !important;
    border: 1px dashed rgba(83, 216, 208, 0.35) !important;
    border-radius: 8px !important;
    padding: 1.25rem 1rem !important;
    transition: border-color 0.2s ease, background-color 0.2s ease;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #53D8D0 !important;
    background-color: #162238 !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] * {
    background-color: transparent !important;
}

/* Browse files button inside uploader */
div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button {
    background-color: #182338 !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 6px !important;
    color: #F1F5F9 !important;
    padding: 0.4rem 0.9rem !important;
    font-weight: 500 !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: #53D8D0 !important;
    color: #53D8D0 !important;
    background-color: #1D2B42 !important;
}

div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] button * {
    color: inherit !important;
    background-color: transparent !important;
}

/* Uploaded file chip in file uploader */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background-color: #182338 !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 6px !important;
    padding: 0.5rem 0.75rem !important;
    margin-top: 0.6rem !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    background-color: transparent !important;
}

/* Primary buttons — solid cyan with dark text, no inner black boxes */
div[data-testid="stButton"] {
    width: 100% !important;
    margin-top: 0.75rem !important;
}

div[data-testid="stButton"] > button {
    width: 100% !important;
    background-color: #53D8D0 !important;
    border: 1px solid #53D8D0 !important;
    border-radius: 6px !important;
    padding: 0.65rem 1.25rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 10px rgba(83, 216, 208, 0.2) !important;
    cursor: pointer !important;
}

div[data-testid="stButton"] > button * {
    background-color: transparent !important;
    color: #0B1120 !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.02em !important;
}

div[data-testid="stButton"] > button:hover {
    background-color: #6fe2db !important;
    border-color: #6fe2db !important;
    box-shadow: 0 4px 16px rgba(83, 216, 208, 0.35) !important;
    transform: translateY(-1px);
}

div[data-testid="stButton"] > button:hover * {
    color: #0B1120 !important;
}

div[data-testid="stButton"] > button:active {
    transform: translateY(0);
}

/* Results panel */
.tbs-result-card {
    background: #131C2E;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.tbs-result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.25rem;
    margin-bottom: 1rem;
}

.tbs-result-well {
    background: #182338 !important;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 1.15rem;
}

.tbs-result-well * {
    background-color: transparent !important;
}

.tbs-result-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}

.tbs-result-value-tb {
    font-size: 1.75rem;
    font-weight: 700;
    color: #FB7185;
    letter-spacing: -0.01em;
}

.tbs-result-value-nontb {
    font-size: 1.75rem;
    font-weight: 700;
    color: #53D8D0;
    letter-spacing: -0.01em;
}

.tbs-result-sub {
    font-size: 0.78rem;
    color: #64748B;
    margin-top: 0.35rem;
}

/* Class probabilities progress bars */
.tbs-prob-card {
    background: #182338 !important;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 1.15rem;
    margin-top: 0.5rem;
}

.tbs-prob-card * {
    background-color: transparent !important;
}

.tbs-prob-row {
    margin-bottom: 0.85rem;
}

.tbs-prob-row:last-child {
    margin-bottom: 0;
}

.tbs-prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}

.tbs-prob-name {
    font-size: 0.82rem;
    font-weight: 600;
    color: #E2E8F0;
    letter-spacing: 0.02em;
}

.tbs-prob-val {
    font-size: 0.85rem;
    font-weight: 700;
    font-feature-settings: "tnum";
    color: #F1F5F9;
}

.tbs-prob-track {
    width: 100%;
    height: 7px;
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 4px;
    overflow: hidden;
}

.tbs-prob-fill-nontb {
    height: 100%;
    background: #53D8D0 !important;
    border-radius: 4px;
}

.tbs-prob-fill-tb {
    height: 100%;
    background: #FB7185 !important;
    border-radius: 4px;
}

/* Visualization grid */
.tbs-viz-header {
    background: #182338;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 0.65rem 0.9rem;
}

.tbs-viz-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #F1F5F9;
    letter-spacing: 0.02em;
}

.tbs-viz-desc {
    font-size: 0.7rem;
    color: #94A3B8;
    margin-top: 0.15rem;
}

/* Trust layer analytical section */
.tbs-trust-container {
    background: #131C2E;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.tbs-trust-kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.tbs-trust-kpi-card {
    background: #182338 !important;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 1rem 1.15rem;
}

.tbs-trust-kpi-card * {
    background-color: transparent !important;
}

.tbs-trust-kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 0.35rem;
}

.tbs-trust-kpi-val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #F1F5F9;
    letter-spacing: -0.01em;
}

.tbs-trust-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    border-radius: 4px;
    text-transform: uppercase;
    margin-left: 0.4rem;
    vertical-align: middle;
}

.tbs-badge-high {
    background: rgba(83, 216, 208, 0.15) !important;
    color: #53D8D0;
    border: 1px solid rgba(83, 216, 208, 0.3);
}

.tbs-badge-moderate {
    background: rgba(245, 158, 11, 0.15) !important;
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.tbs-badge-low {
    background: rgba(239, 68, 68, 0.15) !important;
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.tbs-review-alert-warn {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 0.82rem;
    color: #FDE68A;
}

.tbs-review-alert-ok {
    background: rgba(83, 216, 208, 0.08);
    border: 1px solid rgba(83, 216, 208, 0.25);
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.65rem;
    font-size: 0.82rem;
    color: #A7F3D0;
}

.tbs-comp-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1.25rem;
}

.tbs-comp-card {
    background: #182338 !important;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
}

.tbs-comp-card * {
    background-color: transparent !important;
}

.tbs-comp-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.4rem;
}

.tbs-comp-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: #CBD5E1;
}

.tbs-comp-val {
    font-size: 0.82rem;
    font-weight: 700;
    color: #F1F5F9;
    font-feature-settings: "tnum";
}

.tbs-comp-track {
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 3px;
    overflow: hidden;
}

.tbs-comp-fill {
    height: 100%;
    background: #53D8D0 !important;
    border-radius: 3px;
}

/* Assessment notes */
.tbs-notes-card {
    background: #182338 !important;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 1rem 1.15rem;
    margin-bottom: 1rem;
}

.tbs-notes-card * {
    background-color: transparent !important;
}

.tbs-notes-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 0.6rem;
}

.tbs-note-item {
    font-size: 0.82rem;
    color: #CBD5E1;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}

.tbs-note-bullet {
    color: #53D8D0;
    font-size: 0.75rem;
}

/* Subtle notice / footnote */
.tbs-subtle-notice {
    font-size: 0.76rem;
    line-height: 1.5;
    color: #64748B;
    background: rgba(255, 255, 255, 0.02);
    border-left: 2px solid rgba(255, 255, 255, 0.1);
    padding: 0.5rem 0.85rem;
    border-radius: 0 4px 4px 0;
    margin-top: 0.75rem;
}

/* Sidebar elements */
.tbs-sidebar-brand {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: #F1F5F9;
    margin-bottom: 0.2rem;
}

.tbs-sidebar-tagline {
    font-size: 0.74rem;
    color: #94A3B8;
    margin-bottom: 1.25rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.tbs-sidebar-section {
    margin-bottom: 1.25rem;
}

.tbs-sidebar-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748B;
    margin-bottom: 0.4rem;
}

.tbs-sidebar-val {
    font-size: 0.82rem;
    font-weight: 500;
    color: #E2E8F0;
    margin-bottom: 0.6rem;
}

.tbs-sidebar-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 6px;
    padding: 0.75rem 0.85rem;
    font-size: 0.76rem;
    line-height: 1.5;
    color: #94A3B8;
    margin-top: 0.5rem;
}

/* Footer */
.tbs-footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.75rem;
    font-size: 0.75rem;
    color: #64748B;
}

.tbs-footer-links {
    display: flex;
    gap: 1rem;
}

/* Hide default streamlit header decoration */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .tbs-result-grid, .tbs-trust-kpi-grid, .tbs-comp-grid {
        grid-template-columns: 1fr !important;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# MODEL LOADING (PRESERVED ARCHITECTURE & BEHAVIOR)
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the trained DenseNet-121 checkpoint.

    The checkpoint is expected to have been generated by this
    project's own training scripts.
    """

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT_PATH}"
        )

    model = build_densenet121(
        pretrained=False,
        freeze_backbone=True,
    )

    # DenseNet's forward() contains a functional in-place ReLU
    # operation. Replace it with an equivalent non-in-place
    # operation to avoid modifying tensors used by Grad-CAM.
    #
    # This changes the forward operation, not the learned weights.
    original_forward = model.forward

    # Disable in-place ReLU modules as an additional precaution.
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    # Patch the model's forward method to avoid DenseNet's
    # functional F.relu(features, inplace=True).
    def safe_forward(input_tensor):
        features = model.features(input_tensor)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = model.classifier(out)
        return out

    model.forward = safe_forward

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location=DEVICE,
            weights_only=False,
        )
    except Exception:
        model.forward = original_forward
        raise

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected checkpoint format: expected a dictionary."
        )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Also allow a checkpoint containing only model weights.
        state_dict = checkpoint

    # Support checkpoints saved using DataParallel.
    if state_dict and all(
        key.startswith("module.") for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE PREPROCESSING (PRESERVED)
# ============================================================

IMAGE_TRANSFORM = build_evaluation_transform(
    image_size=IMAGE_SIZE
)


def prepare_image(image):
    """Convert an uploaded image to the model's input format."""

    rgb_image = image.convert("RGB")

    input_tensor = IMAGE_TRANSFORM(
        rgb_image
    ).unsqueeze(0)

    return rgb_image, input_tensor.to(DEVICE)


# ============================================================
# GRAD-CAM (PRESERVED FORWARD/BACKWARD HOOK LOGIC)
# ============================================================

def generate_gradcam(model, input_tensor, target_class):
    """
    Generate Grad-CAM using a forward hook and a tensor gradient hook.

    A tensor gradient hook is used instead of a module full backward
    hook to avoid the view/in-place interaction encountered earlier.
    """

    target_layer = model.features.denseblock4

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        # Keep the activation connected to the computation graph.
        activations.append(output)

        # Capture the gradient of the selected feature map directly.
        output.register_hook(
            lambda gradient: gradients.append(gradient)
        )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        # Gradients must be enabled for Grad-CAM.
        input_for_gradcam = input_tensor.detach().requires_grad_(True)

        with torch.enable_grad():
            logits = model(input_for_gradcam)

            if logits.shape != (1, 2):
                raise RuntimeError(
                    "Expected model output shape (1, 2), "
                    f"received {tuple(logits.shape)}."
                )

            score = logits[0, int(target_class)]
            score.backward()

        if not activations or not gradients:
            raise RuntimeError(
                "Grad-CAM did not capture feature activations "
                "and gradients."
            )

        feature_maps = activations[0].detach()
        feature_gradients = gradients[0].detach()

        # Average gradients over the spatial dimensions.
        weights = feature_gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted sum of feature maps.
        cam = (
            weights * feature_maps
        ).sum(dim=1, keepdim=True)

        # Retain positive evidence.
        cam = F.relu(cam)

        # Resize to the original model input dimensions.
        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]

        # Normalize the heatmap to [0, 1].
        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min).item() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy()

    finally:
        handle.remove()


# ============================================================
# POLISHED VISUALIZATION FIGURES
# ============================================================

def create_original_figure(image):
    """Create a dark-framed figure for the original input radiograph."""
    image_array = np.asarray(
        image.resize((IMAGE_SIZE, IMAGE_SIZE))
    )

    figure, axis = plt.subplots(figsize=(5, 5), facecolor="#131C2E")
    axis.imshow(image_array)
    axis.axis("off")
    figure.tight_layout(pad=0.2)
    return figure


def create_heatmap_figure(heatmap):
    """Create a standalone Grad-CAM heatmap figure with matching dark frame."""
    figure, axis = plt.subplots(figsize=(5, 5), facecolor="#131C2E")

    axis.imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axis.axis("off")
    figure.tight_layout(pad=0.2)
    return figure


def create_overlay_figure(image, heatmap):
    """Overlay the Grad-CAM heatmap on the chest X-ray with matching dark frame."""
    image_array = np.asarray(
        image.resize((IMAGE_SIZE, IMAGE_SIZE))
    )

    figure, axis = plt.subplots(figsize=(5, 5), facecolor="#131C2E")

    axis.imshow(image_array)
    axis.imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )
    axis.axis("off")
    figure.tight_layout(pad=0.2)
    return figure


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="tbs-sidebar-brand">TB-SENTRY</div>
        <div class="tbs-sidebar-tagline">Explainable Medical AI System</div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tbs-sidebar-section">
            <div class="tbs-sidebar-label">MODEL</div>
            <div class="tbs-sidebar-val">DenseNet-121 (Huang et al.)</div>
            <div class="tbs-sidebar-label">EXPLAINABILITY</div>
            <div class="tbs-sidebar-val">Grad-CAM (DenseBlock 4)</div>
            <div class="tbs-sidebar-label">CLASSIFICATION</div>
            <div class="tbs-sidebar-val">Binary (Non-TB / TB)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tbs-sidebar-section">
            <div class="tbs-sidebar-label">ABOUT</div>
            <div class="tbs-sidebar-card">
                TB-Sentry investigates explainable deep learning and
                inference-time reliability heuristics for tuberculosis
                screening from chest radiographs.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="tbs-sidebar-section">
            <div class="tbs-sidebar-label">DISCLAIMER</div>
            <div class="tbs-sidebar-card" style="border-left: 2px solid #53D8D0;">
                Heatmaps are visual explanations of model activation—not verified lesion localization.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 1. APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="tbs-header-wrapper">
        <div class="tbs-header-top">
            <span class="tbs-badge-prototype">RESEARCH PROTOTYPE</span>
            <span class="tbs-badge-meta">DenseNet-121 · Grad-CAM · Trust Assessment</span>
        </div>
        <h1 class="tbs-title">TB-SENTRY</h1>
        <p class="tbs-subtitle">Explainable AI for Chest X-Ray Analysis</p>
        <div class="tbs-metadata-strip">
            <span class="tbs-meta-item">Architecture: DenseNet-121</span>
            <span class="tbs-meta-dot">·</span>
            <span class="tbs-meta-item">Resolution: 224×224 RGB</span>
            <span class="tbs-meta-dot">·</span>
            <span class="tbs-meta-item">Dataset: TBX11K Benchmark</span>
            <span class="tbs-meta-dot">·</span>
            <span class="tbs-meta-item">Compute: CPU Compatible</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. RESEARCH DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="tbs-disclaimer-panel">
        <div class="tbs-disclaimer-icon">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#53D8D0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
        </div>
        <div class="tbs-disclaimer-content">
            <div class="tbs-disclaimer-title">Research Prototype Notice</div>
            This application is an academic research prototype. It has not been validated
            for clinical use. Its predictions must not be used to diagnose, exclude,
            or treat tuberculosis.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. IMAGE UPLOAD & ANALYSIS WORKSPACE
# ============================================================

st.markdown(
    """
    <div class="tbs-card-header" style="margin-bottom: 0.75rem;">
        <span class="tbs-section-title">Radiograph Ingestion Workspace</span>
        <span class="tbs-section-badge">SUPPORTED: PNG, JPG, JPEG</span>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, preview_col = st.columns([1.1, 0.9], gap="large")

with upload_col:
    st.markdown(
        """
        <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.75rem; line-height: 1.5;">
            Upload an anonymized standard posteroanterior (PA) chest radiograph.
            The image will be resized to 224×224 and normalized for model inference.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["png", "jpg", "jpeg"],
        help="Select a supported chest X-ray image for screening analysis.",
        label_visibility="collapsed",
    )

    analyze_clicked = False
    if uploaded_file is not None:
        analyze_clicked = st.button(
            "Analyze Chest X-Ray",
            type="primary",
            use_container_width=True,
        )

# Reset session cache when file changes
if "current_file_id" not in st.session_state:
    st.session_state["current_file_id"] = None
    st.session_state["analysis_cache"] = None

if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state["current_file_id"] != file_id:
        st.session_state["current_file_id"] = file_id
        st.session_state["analysis_cache"] = None

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        st.error(f"Could not open the uploaded image: {error}")
        st.stop()

    with preview_col:
        st.markdown(
            f"""
            <div class="tbs-viz-header" style="border-radius: 8px 8px 0 0;">
                <div class="tbs-viz-title">Radiograph Preview</div>
                <div class="tbs-viz-desc">{uploaded_file.name} · {image.size[0]}×{image.size[1]} px · {image.mode}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.image(
            image,
            use_container_width=True,
        )
else:
    with preview_col:
        st.markdown(
            """
            <div style="height: 180px; display: flex; flex-direction: column; align-items: center; justify-content: center; background: #131C2E; border: 1px dashed rgba(255,255,255,0.08); border-radius: 8px; color: #64748B; font-size: 0.82rem;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 0.5rem; opacity: 0.5;">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                No radiograph loaded
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ANALYSIS PIPELINE EXECUTION
# ============================================================

if uploaded_file is not None and (analyze_clicked or st.session_state.get("analysis_cache") is not None):

    if analyze_clicked or st.session_state.get("analysis_cache") is None:
        try:
            with st.spinner("Executing DenseNet-121 inference & Grad-CAM analysis..."):
                model = load_model()
                rgb_image, input_tensor = prepare_image(image)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)[0]

                predicted_class = int(torch.argmax(probabilities).item())
                predicted_label = CLASS_NAMES[predicted_class]
                confidence = float(probabilities[predicted_class].item())

                # Explain predicted class via Grad-CAM
                heatmap = generate_gradcam(
                    model,
                    input_tensor,
                    target_class=predicted_class,
                )

                # Heuristic inference reliability analysis
                heatmap_analysis = analyze_heatmap(heatmap)

                trust_result = calculate_trust(
                    confidence=confidence,
                    concentration_score=heatmap_analysis["concentration_score"],
                    normalized_entropy=heatmap_analysis["normalized_entropy"],
                    activated_area_ratio=heatmap_analysis["activated_area_ratio"],
                )

                # Cache results for interactive view stability
                st.session_state["analysis_cache"] = {
                    "predicted_class": predicted_class,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "probabilities": probabilities.cpu().numpy(),
                    "heatmap": heatmap,
                    "trust_result": trust_result,
                }

        except Exception as error:
            st.error("The analysis could not be completed.")
            st.exception(error)
            st.markdown(
                """
                <div class="tbs-subtle-notice">
                    <strong>Troubleshooting:</strong> Verify that the model checkpoint exists at
                    <code>models/checkpoints/densenet121_baseline.pt</code>, the architecture matches
                    the saved weights, and dependencies are correctly installed.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.stop()

    cache = st.session_state["analysis_cache"]
    predicted_class = cache["predicted_class"]
    predicted_label = cache["predicted_label"]
    confidence = cache["confidence"]
    probs = cache["probabilities"]
    heatmap = cache["heatmap"]
    trust_result = cache["trust_result"]

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)


    # ============================================================
    # 4. PREDICTION SUMMARY
    # ============================================================

    is_tb = (predicted_class == 1)
    status_color = "#FB7185" if is_tb else "#53D8D0"
    status_title = "SCREENING PREDICTION: TB" if is_tb else "SCREENING PREDICTION: NON-TB"
    val_class = "tbs-result-value-tb" if is_tb else "tbs-result-value-nontb"

    st.markdown(
        f"""
        <div class="tbs-result-card">
            <div class="tbs-card-header">
                <span class="tbs-section-title">Model Classification Output</span>
                <span class="tbs-section-badge" style="color: {status_color}; border-color: {status_color}; background: rgba(255,255,255,0.03);">
                    {status_title}
                </span>
            </div>
            <div class="tbs-result-grid">
                <div class="tbs-result-well">
                    <div class="tbs-result-label">Predicted Class</div>
                    <div class="{val_class}">{predicted_label}</div>
                    <div class="tbs-result-sub">Binary categorization based on learned radiographic features</div>
                </div>
                <div class="tbs-result-well">
                    <div class="tbs-result-label">Model Softmax Confidence</div>
                    <div class="tbs-trust-kpi-val" style="color: #F1F5F9;">{confidence * 100:.2f}%</div>
                    <div class="tbs-result-sub">Relative probability assigned by model output layer</div>
                </div>
            </div>
            <div class="tbs-subtle-notice">
                The displayed score is the model's softmax output, not a clinically validated probability of disease.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ============================================================
    # 5. CLASS PROBABILITIES
    # ============================================================

    nontb_prob = float(probs[0]) * 100
    tb_prob = float(probs[1]) * 100

    st.markdown(
        f"""
        <div class="tbs-card">
            <div class="tbs-card-header">
                <span class="tbs-section-title">Class Distribution Probabilities</span>
                <span class="tbs-section-badge">TOTAL: 100.0%</span>
            </div>
            <div class="tbs-prob-card">
                <div class="tbs-prob-row">
                    <div class="tbs-prob-header">
                        <span class="tbs-prob-name">Non-TB (Negative Screen)</span>
                        <span class="tbs-prob-val">{nontb_prob:.2f}%</span>
                    </div>
                    <div class="tbs-prob-track">
                        <div class="tbs-prob-fill-nontb" style="width: {max(1.0, nontb_prob)}%;"></div>
                    </div>
                </div>
                <div class="tbs-prob-row">
                    <div class="tbs-prob-header">
                        <span class="tbs-prob-name">TB (Positive Screen)</span>
                        <span class="tbs-prob-val">{tb_prob:.2f}%</span>
                    </div>
                    <div class="tbs-prob-track">
                        <div class="tbs-prob-fill-tb" style="width: {max(1.0, tb_prob)}%;"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ============================================================
    # 6. GRAD-CAM EXPLAINABILITY VISUALIZATIONS
    # ============================================================

    st.markdown(
        """
        <div class="tbs-card-header" style="margin-top: 1.5rem;">
            <span class="tbs-section-title">Grad-CAM Explainability Triad</span>
            <span class="tbs-section-badge">TARGET: DENSEBLOCK 4</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gcol1, gcol2, gcol3 = st.columns(3, gap="medium")

    with gcol1:
        st.markdown(
            """
            <div class="tbs-viz-header">
                <div class="tbs-viz-title">1. Original Radiograph</div>
                <div class="tbs-viz-desc">Preprocessed 224×224 input image</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        orig_fig = create_original_figure(image)
        st.pyplot(orig_fig, use_container_width=True)
        plt.close(orig_fig)

    with gcol2:
        st.markdown(
            """
            <div class="tbs-viz-header">
                <div class="tbs-viz-title">2. Grad-CAM Heatmap</div>
                <div class="tbs-viz-desc">Normalized spatial gradient activation [0, 1]</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        heat_fig = create_heatmap_figure(heatmap)
        st.pyplot(heat_fig, use_container_width=True)
        plt.close(heat_fig)

    with gcol3:
        st.markdown(
            """
            <div class="tbs-viz-header">
                <div class="tbs-viz-title">3. Blended Overlay</div>
                <div class="tbs-viz-desc">Chest X-ray with Jet colormap (α = 0.40)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        overlay_fig = create_overlay_figure(image, heatmap)
        st.pyplot(overlay_fig, use_container_width=True)
        plt.close(overlay_fig)

    st.markdown(
        """
        <div class="tbs-subtle-notice" style="margin-bottom: 2rem;">
            <strong>Explainability Notice:</strong> Grad-CAM highlights image regions that influence the model's prediction.
            It does not prove the presence or absence of tuberculosis, and unhighlighted regions do not rule it out.
        </div>
        """,
        unsafe_allow_html=True,
    )


    # ============================================================
    # 7. EXPERIMENTAL TRUST LAYER
    # ============================================================

    cat = trust_result["trust_category"].lower()
    badge_style = "tbs-badge-high" if cat == "high" else ("tbs-badge-moderate" if cat == "moderate" else "tbs-badge-low")
    comps = trust_result["components"]

    st.markdown(
        f"""
        <div class="tbs-trust-container">
            <div class="tbs-card-header">
                <span class="tbs-section-title">Experimental Trust Layer</span>
                <span class="tbs-section-badge">HEURISTIC RELIABILITY ASSESSMENT</span>
            </div>
            <div class="tbs-trust-kpi-grid">
                <div class="tbs-trust-kpi-card">
                    <div class="tbs-trust-kpi-label">Experimental Trust Score</div>
                    <div class="tbs-trust-kpi-val">
                        {trust_result['trust_score']}<span style="font-size: 0.95rem; color: #64748B;">/100</span>
                        <span class="tbs-trust-badge {badge_style}">{trust_result['trust_category']}</span>
                    </div>
                </div>
                <div class="tbs-trust-kpi-card">
                    <div class="tbs-trust-kpi-label">Model Confidence</div>
                    <div class="tbs-trust-kpi-val">{comps['confidence'] * 100:.2f}%</div>
                </div>
                <div class="tbs-trust-kpi-card">
                    <div class="tbs-trust-kpi-label">Heatmap Concentration</div>
                    <div class="tbs-trust-kpi-val">{comps['concentration_score'] * 100:.2f}%</div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if trust_result["review_required"]:
        st.markdown(
            """
            <div class="tbs-review-alert-warn">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                    <line x1="12" y1="9" x2="12" y2="13"></line>
                    <line x1="12" y1="17" x2="12.01" y2="17"></line>
                </svg>
                <span><strong>Human Review Recommended:</strong> One or more heuristic warning triggers were activated for this screening instance.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="tbs-review-alert-ok">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                <span><strong>Standard Heuristic Met:</strong> No predefined heuristic review triggers were activated.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    dist_score = 1.0 - comps["normalized_entropy"]

    st.markdown(
        f"""
            <div class="tbs-comp-grid">
                <div class="tbs-comp-card">
                    <div class="tbs-comp-top">
                        <span class="tbs-comp-label">Model Confidence</span>
                        <span class="tbs-comp-val">{comps['confidence']:.2%}</span>
                    </div>
                    <div class="tbs-comp-track">
                        <div class="tbs-comp-fill" style="width: {comps['confidence'] * 100:.1f}%;"></div>
                    </div>
                </div>
                <div class="tbs-comp-card">
                    <div class="tbs-comp-top">
                        <span class="tbs-comp-label">Heatmap Concentration</span>
                        <span class="tbs-comp-val">{comps['concentration_score']:.2%}</span>
                    </div>
                    <div class="tbs-comp-track">
                        <div class="tbs-comp-fill" style="width: {comps['concentration_score'] * 100:.1f}%;"></div>
                    </div>
                </div>
                <div class="tbs-comp-card">
                    <div class="tbs-comp-top">
                        <span class="tbs-comp-label">Heatmap Distribution Score</span>
                        <span class="tbs-comp-val">{dist_score:.2%}</span>
                    </div>
                    <div class="tbs-comp-track">
                        <div class="tbs-comp-fill" style="width: {dist_score * 100:.1f}%;"></div>
                    </div>
                </div>
                <div class="tbs-comp-card">
                    <div class="tbs-comp-top">
                        <span class="tbs-comp-label">Activated Area Ratio</span>
                        <span class="tbs-comp-val">{comps['activated_area_ratio']:.2%}</span>
                    </div>
                    <div class="tbs-comp-track">
                        <div class="tbs-comp-fill" style="width: {comps['activated_area_ratio'] * 100:.1f}%;"></div>
                    </div>
                </div>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # 8. ASSESSMENT NOTES
    notes_html = "".join([
        f'<div class="tbs-note-item"><span class="tbs-note-bullet">▪</span><span>{reason}</span></div>'
        for reason in trust_result["reasons"]
    ])

    st.markdown(
        f"""
            <div class="tbs-notes-card">
                <div class="tbs-notes-title">Assessment Notes & Heuristic Traces</div>
                {notes_html}
            </div>
            <div class="tbs-subtle-notice">
                This Trust Layer is an experimental heuristic combining model confidence and Grad-CAM activation distribution.
                It is not clinically validated and does not establish diagnostic correctness, lesion localization accuracy, or disease probability.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 9. ABOUT / METHODOLOGY SPECIFICATIONS
# ============================================================

with st.expander("Technical Architecture & Methodology Details", expanded=False):
    st.markdown(
        """
        <div style="font-size: 0.82rem; line-height: 1.6; color: #CBD5E1;">
            <p><strong>Backbone Network:</strong> DenseNet-121 architecture (Huang et al., 2017) with 4 dense connectivity blocks,
            global average pooling, and a linear 2-class projection head.</p>
            <p><strong>Explainability Engine:</strong> Gradient-weighted Class Activation Mapping (Grad-CAM) computed on the final
            convolutional features in <code>features.denseblock4</code>. Feature activations and tensor gradient hooks isolate class-discriminative
            spatial evidence without modifying the underlying weights.</p>
            <p><strong>Experimental Trust Layer:</strong> An inference-time heuristic assessment formulating a normalized composite score
            from classifier softmax confidence (0.70 weight) and activation concentration score (0.30 weight), gated by entropy and area ratio thresholds.</p>
            <p><strong>Dataset Benchmark:</strong> TBX11K Tuberculosis Chest X-Ray Benchmark (11,200 chest radiographs with healthy, sick, and active tuberculosis manifestations).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown(
    """
    <div class="tbs-footer">
        <div>TB-SENTRY · Explainable AI for Chest X-Ray Analysis</div>
        <div>Academic research prototype · Not validated for clinical use</div>
    </div>
    """,
    unsafe_allow_html=True,
)
