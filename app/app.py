"""TrustXRay — uncertainty-aware pneumonia detection on chest X-rays (research demo)."""
import glob
import json
import os
import threading
import urllib.request

import matplotlib.cm as cm
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_URL = "https://github.com/Adib008/TrustXRay/releases/download/v1.0/resnet18_baseline.pt"
MODEL_PATH = os.environ.get("TRUSTXRAY_MODEL", os.path.join(APP_DIR, "resnet18_baseline.pt"))
with open(os.path.join(APP_DIR, "trustxray_config.json")) as f:
    CONFIG = json.load(f)

THR_LOGIT = CONFIG["threshold_logit"]
OOD_HARD_FACTOR = 1.2  # distance between limit and 1.2 x limit -> referred; above -> refused
Z_CUTOFF = CONFIG["z_cutoff"]
T = CONFIG["mc_passes"]
SIZE = 224
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

st.set_page_config(page_title="TrustXRay", page_icon="🫁", layout="wide")

# The model is shared by all visitors of the app: one prediction at a time.
MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------------- model
def build_model(dropout):
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(model.fc.in_features, 1))
    return model


@st.cache_resource(show_spinner="Loading the model (first launch only)...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    model = build_model(CONFIG["dropout"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource(show_spinner=False)
def load_ood_stats():
    path = os.path.join(APP_DIR, "ood_stats.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["mu"].astype(np.float64), d["precision"].astype(np.float64), float(d["threshold"])


def mahalanobis(feats, stats):
    mu, precision, _ = stats
    diff = feats.astype(np.float64) - mu
    return float(np.sqrt(diff @ precision @ diff))


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def prepare_image(img):
    """Grayscale -> center square crop -> 224x224 (same as MedMNIST)."""
    img = img.convert("L")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return img.resize((SIZE, SIZE), Image.BILINEAR)


# ---------------------------------------------------------------- inference
@torch.no_grad()
def mc_dropout(model, x):
    """T stochastic passes with dropout active (dropout is only in the head)."""
    torch.manual_seed(0)  # reproducible result for the same image
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feats = feature_extractor(x).flatten(1)
    model.fc.train()
    logits = torch.stack([model.fc(feats).squeeze() for _ in range(T)]).numpy()
    model.fc.eval()
    return logits, feats.squeeze(0).numpy()


def grad_cam(model, x):
    store = {}

    def fwd_hook(module, inp, out):
        store["act"] = out
        if out.requires_grad:  # never fails, even if called without gradients
            out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.layer4.register_forward_hook(fwd_hook)
    try:
        model.zero_grad()
        score = model(x.requires_grad_(True)).squeeze()
        score.backward()
    finally:
        handle.remove()  # always remove the hook, even after an error
    weights = store["grad"].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().numpy()
    return cam / (cam.max() + 1e-8)


def is_almost_uniform(img):
    """Blank or nearly blank image (all white, all black, uniform gray)."""
    return float(np.asarray(img.convert("L"), dtype=np.float32).std()) < 8.0


def looks_like_xray(img):
    """X-rays are grayscale: reject clearly colored images (photos, screenshots...)."""
    rgb = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    saturation = (rgb.max(axis=2) - rgb.min(axis=2)).mean()
    return saturation < 0.08


def overlay(gray_img, cam, alpha=0.45):
    base = np.asarray(gray_img.convert("RGB"), dtype=np.float32) / 255.0
    heat = cm.jet(cam)[..., :3]
    return np.clip((1 - alpha) * base + alpha * heat, 0, 1)


# ---------------------------------------------------------------- UI
st.title("🫁 TrustXRay")
st.markdown("**Uncertainty-aware pneumonia detection on pediatric chest X-rays** — "
            "the AI tells you what it predicts, *how sure it is*, *where it looked*, "
            "and refers uncertain cases to a radiologist.")
st.warning("⚠️ **Research demo — not a medical device.** Not clinically validated. Do not use for diagnosis.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        f"1. **ResNet18** (transfer learning) trained on PneumoniaMNIST (pediatric X-rays).\n"
        f"2. **Monte Carlo Dropout**: the X-ray is analysed **{T} times** with random dropout.\n"
        f"3. **Uncertainty z** = distance to the decision boundary, in standard deviations.\n"
        f"4. If z < {Z_CUTOFF:.1f} → **referred to a radiologist**.\n"
        f"5. **Grad-CAM** shows the regions that pushed toward *pneumonia*.\n"
        f"6. **Out-of-distribution check** (Mahalanobis distance to the training X-rays): "
        f"slightly unusual images are **referred**, very different images are **refused**."
    )
    st.header("Test-set results")
    st.markdown(
        "- AUC **0.990**\n"
        "- Sensitivity **99.5%** on all cases\n"
        "- With referral (16.8% of cases): **100%** sensitivity on the cases kept by the AI, "
        "**2/2** missed pneumonias caught"
    )
    st.header("Limitations")
    st.markdown("Children aged 1–5 only, one hospital, low-resolution images. "
                "Works best on frontal pediatric chest X-rays similar to the training data.")
    st.markdown("[GitHub repository](https://github.com/Adib008/TrustXRay)")

model = load_model()
ood_stats = load_ood_stats()

examples = sorted(glob.glob(os.path.join(APP_DIR, "examples", "*.png")))
source = st.radio("Choose an X-ray", ["Example from the test set", "Upload my own image"], horizontal=True)
img = None
if source == "Example from the test set" and examples:
    names = [os.path.splitext(os.path.basename(p))[0] for p in examples]
    choice = st.selectbox("Example", names)
    img = Image.open(examples[names.index(choice)])
elif source == "Upload my own image":
    st.info("ℹ️ This model was trained **only on frontal chest X-rays of children aged 1–5** "
            "(one hospital, low resolution). Results on other images are not meaningful.")
    up = st.file_uploader("Frontal chest X-ray (PNG or JPG)", type=["png", "jpg", "jpeg"])
    if up is not None:
        img = Image.open(up)
elif not examples:
    st.info("No example images found — upload an image instead.")

if img is not None and source == "Upload my own image" and not looks_like_xray(img):
    st.error("🚫 **This does not look like a chest X-ray** (the image contains colors). "
             "Please upload a grayscale frontal chest X-ray.")
    img = None

if img is not None and is_almost_uniform(img):
    st.error("🚫 **This image is almost uniform** (blank, white or black). Please upload a chest X-ray.")
    img = None

if img is not None:
    gray = prepare_image(img)
    x = preprocess(gray.convert("RGB")).unsqueeze(0)

    with MODEL_LOCK:
        model.layer4._forward_hooks.clear()  # safety: remove any leftover hook
        logits, feats = mc_dropout(model, x)

    ood_distance = mahalanobis(feats, ood_stats) if ood_stats else None
    ood_limit = ood_stats[2] if ood_stats else None
    if ood_distance is not None and ood_distance > OOD_HARD_FACTOR * ood_limit:
        st.error(f"🚫 **Out of distribution — the model refuses to answer.**\n\n"
                 f"This image is too different from the chest X-rays used for training "
                 f"(distance {ood_distance:.1f} > refusal limit {OOD_HARD_FACTOR * ood_limit:.1f}). "
                 f"The model is only valid on frontal pediatric chest X-rays of good quality.")
        st.image(gray, width=300)
        st.stop()
    unusual = ood_distance is not None and ood_distance > ood_limit

    with MODEL_LOCK:
        model.layer4._forward_hooks.clear()
        cam = grad_cam(model, preprocess(gray.convert("RGB")).unsqueeze(0))
    mean, std = float(logits.mean()), float(logits.std() + 1e-6)
    votes = int((logits >= THR_LOGIT).sum())
    pred_pneumonia = mean >= THR_LOGIT
    z = abs(mean - THR_LOGIT) / std
    referred = (z < Z_CUTOFF) or unusual

    if source == "Upload my own image":
        st.warning("⚠️ **Out-of-distribution warning:** this image does not come from the training dataset. "
                   "The model has not been validated on external images — interpret with caution.")

    col1, col2, col3 = st.columns([1, 1, 1.1])
    with col1:
        st.subheader("X-ray")
        st.image(gray, width="stretch")
    with col2:
        st.subheader("Grad-CAM")
        st.image(overlay(gray, cam), width="stretch",
                 caption="Red = regions pushing toward 'pneumonia'")
    with col3:
        st.subheader("Decision")
        label = "PNEUMONIA" if pred_pneumonia else "NORMAL"
        if unusual:
            st.warning(f"### ⚠️ Unusual image — refer to a radiologist\nThe model leans toward **{label}**, "
                       "but this image is less typical than 99% of the training X-rays, so the AI does not decide.")
        elif referred:
            st.warning(f"### ⚠️ Uncertain — refer to a radiologist\nThe model leans toward **{label}** but is not sure.")
        elif pred_pneumonia:
            st.error("### 🔴 Pneumonia detected\nConfident prediction.")
        else:
            st.success("### 🟢 Normal\nConfident prediction.")

        st.metric("Votes for pneumonia", f"{votes} / {T}")
        st.progress(votes / T)
        st.metric("Uncertainty score z", f"{z:.1f}")
        st.caption(f"Referral cut-off: z < {Z_CUTOFF:.1f} → "
                   + ("**below the cut-off → referred**" if z < Z_CUTOFF else "above the cut-off"))
        if ood_distance is not None:
            if unusual:
                st.caption(f"⚠️ Familiarity check: distance {ood_distance:.1f} > {ood_limit:.1f} "
                           f"(unusual) but ≤ {OOD_HARD_FACTOR * ood_limit:.1f} (not refused) → referred")
            else:
                st.caption(f"✅ Familiarity check passed: distance {ood_distance:.1f} ≤ {ood_limit:.1f}")

    with st.expander("How to read these results"):
        st.markdown(
            f"- **Votes**: out of {T} analyses with random dropout, how many say *pneumonia*. "
            f"{T}/{T} or 0/{T} = all analyses agree.\n"
            f"- **z**: how far the average answer is from the decision boundary, measured in standard "
            f"deviations of the {T} answers. Small z = the answers are close to the boundary = uncertain.\n"
            f"- The referral cut-off (z < {Z_CUTOFF:.1f}) was chosen on the validation set (≈10% of cases referred).\n"
            "- We do not display a probability: the model's probabilities are not calibrated "
            "(class weighting shifts them), so a number like '92%' would be misleading.\n"
            "- On normal X-rays the Grad-CAM map is often nearly empty: there is no evidence for pneumonia to show."
        )
