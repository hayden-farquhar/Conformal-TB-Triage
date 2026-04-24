"""
Interactive Conformal TB Triage Demo

Demonstrates the conformal prediction pipeline from the accompanying manuscript.
Upload a chest X-ray → see the conformal prediction set, triage tier, and
calibrated probability interval.

This is a DEMONSTRATION of the method, not a validated clinical tool.
Deploying this in a clinical setting would require site-level recalibration
with local data (see §5.3.11 of the manuscript).

Usage:
    pip install streamlit torch torchvision transformers Pillow
    streamlit run app/app.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent


# ── Load pipeline components (cached) ────────────────────────────────

@st.cache_resource
def load_pipeline():
    """Load probe, isotonic calibrator, and conformal thresholds."""
    with open(APP_DIR / "probe.pkl", "rb") as f:
        probe = pickle.load(f)
    with open(APP_DIR / "isotonic.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(APP_DIR / "conformal_thresholds.json") as f:
        conf = json.load(f)
    thresholds = {int(k): v for k, v in conf["thresholds"].items()}
    return probe, iso, thresholds, conf["alpha"]


@st.cache_resource
def load_rad_dino():
    """Load RAD-DINO model and processor from HuggingFace."""
    import torch
    from transformers import AutoModel, AutoImageProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("microsoft/rad-dino", use_fast=False)
    model = AutoModel.from_pretrained("microsoft/rad-dino").to(device)
    model.eval()
    return model, processor, device


# ── Inference ────────────────────────────────────────────────────────

def extract_embedding(image, model, processor, device):
    """Extract RAD-DINO CLS embedding from a PIL image."""
    import torch
    from sklearn.preprocessing import normalize

    inputs = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        emb = model(pixel_values=inputs).last_hidden_state[:, 0, :]
    emb_np = emb.cpu().float().numpy()
    return normalize(emb_np, norm="l2")


def predict(emb, probe, iso, thresholds):
    """Run probe + isotonic + Mondrian conformal on an embedding."""
    raw_prob = probe.predict_proba(emb)[:, 1][0]
    cal_prob = iso.predict([raw_prob])[0]

    # Mondrian conformal prediction set
    pred_set = set()
    if (1 - cal_prob) <= thresholds.get(1, 1.0):
        pred_set.add("TB")
    if cal_prob <= thresholds.get(0, 1.0):
        pred_set.add("non-TB")

    # Triage tier
    if pred_set == {"non-TB"}:
        tier = 1
        tier_label = "CLEAR"
        tier_action = "Discharge (low TB probability)"
        tier_color = "#28a745"
    elif pred_set == {"TB"}:
        tier = 2
        tier_label = "REFER"
        tier_action = "Refer for Xpert MTB/RIF confirmatory testing"
        tier_color = "#dc3545"
    elif pred_set == {"TB", "non-TB"}:
        tier = 3
        tier_label = "UNCERTAIN"
        tier_action = "Clinical review or repeat imaging required"
        tier_color = "#ffc107"
    else:
        tier = 3
        tier_label = "UNCERTAIN"
        tier_action = "Empty prediction set — defer to clinician"
        tier_color = "#ffc107"

    return {
        "raw_prob": float(raw_prob),
        "cal_prob": float(cal_prob),
        "pred_set": pred_set,
        "tier": tier,
        "tier_label": tier_label,
        "tier_action": tier_action,
        "tier_color": tier_color,
    }


# ── Streamlit UI ─────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Conformal TB Triage Demo",
        page_icon="🫁",
        layout="wide",
    )

    st.title("Conformal TB Triage Demo")
    st.markdown(
        "Upload a chest X-ray to see the conformal prediction set and triage tier. "
        "This demonstrates the methodology from the accompanying manuscript — "
        "it is **not a validated clinical tool**."
    )

    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown(
            "**Method:** Frozen RAD-DINO embeddings + linear probe + "
            "Mondrian class-conditional conformal prediction.\n\n"
            "**Coverage guarantee:** 90% TB-class coverage "
            "(calibrated on Shenzhen + Montgomery, n = 800).\n\n"
            "**Pre-registration:** "
            "[OSF](https://doi.org/10.17605/OSF.IO/KBAMC)"
        )
        st.divider()
        st.markdown(
            "**Hayden Farquhar** MBBS MPHTM\n\n"
            "ORCID: [0009-0002-6226-440X]"
            "(https://orcid.org/0009-0002-6226-440X)"
        )
        st.divider()
        st.caption(
            "This tool requires site-level recalibration before clinical use. "
            "See manuscript §5.3.11 for the recalibration protocol."
        )

    # File upload
    uploaded = st.file_uploader(
        "Upload a PA/AP chest X-ray (PNG, JPEG, or DICOM)",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)

        with col2:
            with st.spinner("Loading RAD-DINO model..."):
                model, processor, device = load_rad_dino()
            probe, iso, thresholds, alpha = load_pipeline()

            with st.spinner("Extracting embedding and running inference..."):
                emb = extract_embedding(image, model, processor, device)
                result = predict(emb, probe, iso, thresholds)

            st.subheader("Results")

            # Triage tier
            st.markdown(
                f"### Tier {result['tier']}: "
                f"<span style='color:{result['tier_color']}'>"
                f"{result['tier_label']}</span>",
                unsafe_allow_html=True,
            )
            st.info(result["tier_action"])

            # Conformal prediction set
            set_str = ", ".join(sorted(result["pred_set"])) if result["pred_set"] else "empty"
            st.metric("Conformal Prediction Set", f"{{{set_str}}}")

            # Probabilities
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.metric("Calibrated P(TB)", f"{result['cal_prob']:.3f}")
            with pcol2:
                st.metric("Raw P(TB)", f"{result['raw_prob']:.3f}")

            # Coverage guarantee
            st.caption(
                f"Mondrian conformal at α = {alpha:.2f}. "
                f"TB-class coverage guarantee: ≥{(1-alpha)*100:.0f}% "
                f"(conditional on calibration data exchangeability)."
            )

    else:
        st.info("Upload a chest X-ray image to begin.")

        # Show example output
        with st.expander("Example output (pre-computed)"):
            st.markdown(
                "**Tier 2: REFER**\n\n"
                "- Conformal prediction set: {TB}\n"
                "- Calibrated P(TB): 0.847\n"
                "- Action: Refer for Xpert MTB/RIF confirmatory testing\n\n"
                "**Tier 1: CLEAR**\n\n"
                "- Conformal prediction set: {non-TB}\n"
                "- Calibrated P(TB): 0.023\n"
                "- Action: Discharge (low TB probability)\n\n"
                "**Tier 3: UNCERTAIN**\n\n"
                "- Conformal prediction set: {TB, non-TB}\n"
                "- Calibrated P(TB): 0.412\n"
                "- Action: Clinical review or repeat imaging required"
            )


if __name__ == "__main__":
    main()
