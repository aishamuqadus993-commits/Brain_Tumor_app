import streamlit as st
import numpy as np
import cv2
import pickle
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
from datetime import datetime

# ── Supabase ──────────────────────────────────────────────────────────────────
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

@st.cache_resource
def get_supabase() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_config():
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model

    model = load_model("brain_tumor_model.h5", compile=False)
    feat_extractor = Model(
        inputs=model.input,
        outputs=model.get_layer("dense_1").output,
    )
    feat_extractor.trainable = False

    with open("mri_prototype.pkl", "rb") as f:
        prototype = pickle.load(f)

    with open("config.json", "r") as f:
        config = json.load(f)

    return model, feat_extractor, prototype, config

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_INFO = {
    "glioma": {
        "label":   "Glioma Tumor",
        "color":   "#E74C3C",
        "icon":    "🔴",
        "desc":    "Gliomas arise from glial cells in the brain or spine. They are the most common primary brain tumors and vary widely in grade and prognosis.",
        "severity":"High — Immediate medical consultation required",
    },
    "meningioma": {
        "label":   "Meningioma Tumor",
        "color":   "#9B59B6",
        "icon":    "🟣",
        "desc":    "Meningiomas arise from the meninges surrounding the brain and spinal cord. Most are benign and slow-growing.",
        "severity":"Moderate — Medical follow-up recommended",
    },
    "no_tumor": {
        "label":   "No Tumor Detected",
        "color":   "#27AE60",
        "icon":    "🟢",
        "desc":    "No tumor was detected in this MRI scan. The brain tissue appears within normal limits.",
        "severity":"Normal — No immediate concern",
    },
    "pituitary": {
        "label":   "Pituitary Tumor",
        "color":   "#F39C12",
        "icon":    "🟡",
        "desc":    "Pituitary tumors form in the pituitary gland at the base of the brain. Most are benign (adenomas) and treatable.",
        "severity":"Moderate — Endocrinology consultation recommended",
    },
}

# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp { background: #080c14; }

    /* Hide default streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Page title ── */
    .app-title {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .app-title h1 {
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-title p {
        color: #4a6080;
        font-size: 0.9rem;
        margin: 0.4rem 0 0;
    }

    /* ── Cards ── */
    .card {
        background: #0d1117;
        border: 1px solid #1e2535;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Prediction result card ── */
    .result-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
    }
    .result-accept {
        background: #0a1a0f;
        border: 2px solid #27AE60;
    }
    .result-reject {
        background: #1a0000;
        border: 2px solid #e74c3c;
    }

    /* ── Confidence bar ── */
    .conf-bar-wrap {
        background: #1a2030;
        border-radius: 6px;
        height: 10px;
        overflow: hidden;
        margin: 4px 0 12px;
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.6s ease;
    }

    /* ── Buttons ── */
    .stButton > button {
        font-family: 'Space Mono', monospace !important;
        border-radius: 8px !important;
        font-size: 13px !important;
        letter-spacing: 0.5px !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a6fff, #0d4bcc) !important;
        border: none !important;
        color: white !important;
    }

    /* ── Input fields ── */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background: #0d1117 !important;
        border: 1px solid #1e2535 !important;
        color: #e0e6f0 !important;
        border-radius: 8px !important;
    }
    label { color: #8892a4 !important; font-size: 13px !important; }

    /* ── Tag pills ── */
    .pill {
        display: inline-block;
        background: #1a2535;
        color: #4a9eff;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-family: monospace;
        margin: 3px 4px 3px 0;
    }

    /* ── History table ── */
    .hist-row {
        background: #0d1117;
        border: 1px solid #1e2535;
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        color: #2a3040;
        font-size: 11px;
        text-align: center;
        margin-top: 1.5rem;
        font-family: monospace;
    }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def pixel_rejection_check(img_bgr, image_size):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    hue_std = float(hsv[:, :, 0].std())
    if hue_std > 25.0:
        return False, "Natural photo detected (too much color variety)"

    lap_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, bm      = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    bright_pct = float(np.sum(bm > 0) / gray.size)
    if lap_var > 4500 and bright_pct > 0.12:
        return False, "CT scan detected — please upload an MRI scan"

    dark_ratio = float(np.sum(gray < 30) / gray.size)
    if dark_ratio < 0.03:
        return False, "Insufficient dark background for a brain MRI"

    edges  = cv2.Canny(gray, 50, 150)
    lines  = cv2.HoughLinesP(edges, 1, np.pi / 180,
                              threshold=80, minLineLength=60, maxLineGap=10)
    n_lines      = len(lines) if lines is not None else 0
    edge_density = float(edges.sum() / (gray.size * 255))
    if edge_density > 0.12 and n_lines > 35:
        return False, "X-ray pattern detected"

    return True, "passed"


def compute_gradcam(model, arr_pre, class_idx, image_size, layer_name="block5_conv3"):
    import tensorflow as tf
    from tensorflow.keras.models import Model

    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        inp             = tf.cast(np.expand_dims(arr_pre, 0), tf.float32)
        conv_out, preds = grad_model(inp)
        loss            = preds[:, class_idx]
    grads  = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam    = tf.reduce_sum(tf.multiply(pooled, conv_out[0]), axis=-1).numpy()
    cam    = np.maximum(cam, 0)
    cam    = cv2.resize(cam, (image_size, image_size))
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def run_inference(pil_image):
    """
    Returns dict with keys:
      accepted (bool), reason (str), pred_class, confidence, probs, gradcam_img (PIL)
    """
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import preprocess_input
    from scipy.spatial.distance import mahalanobis

    model, feat_extractor, prototype, config = load_model_and_config()
    CLASS_NAMES  = config["class_names"]
    IMAGE_SIZE   = config["image_size"]
    INFER_CONFIG = config["infer_config"]

    img_rgb   = np.array(pil_image.convert("RGB"))
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr_r = cv2.resize(img_bgr, (IMAGE_SIZE, IMAGE_SIZE))

    # Layer 1 — pixel check
    ok, reason = pixel_rejection_check(img_bgr_r, IMAGE_SIZE)
    if not ok:
        return {"accepted": False, "reason": reason}

    # Preprocess
    img_res = pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr     = img_to_array(img_res).astype(np.float32)
    arr_pre = preprocess_input(arr.copy())

    # Layer 2 — feature distance
    feat = feat_extractor.predict(np.expand_dims(arr_pre, 0), verbose=0).flatten()
    dist = float(mahalanobis(feat, prototype["mean"], prototype["cov_inv"]))
    if dist >= INFER_CONFIG["mahalanobis_threshold"]:
        return {"accepted": False,
                "reason": "Image does not match brain MRI characteristics"}

    # Prediction
    preds      = model.predict(np.expand_dims(arr_pre, 0), verbose=0)[0]
    pred_idx   = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(preds[pred_idx])

    # Layer 3 — confidence gate
    if confidence < INFER_CONFIG["confidence_threshold"]:
        return {"accepted": False,
                "reason": f"Low model confidence ({confidence:.1%}) — image too unclear"}

    # Grad-CAM
    try:
        cam = compute_gradcam(model, arr_pre, pred_idx, IMAGE_SIZE)
        base    = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = (base * 0.55 + heatmap * 0.45).astype(np.uint8)
        gradcam_pil = Image.fromarray(overlay)
    except Exception:
        gradcam_pil = None

    return {
        "accepted":    True,
        "pred_class":  pred_class,
        "confidence":  confidence,
        "probs":       {CLASS_NAMES[i]: float(preds[i]) for i in range(len(CLASS_NAMES))},
        "gradcam_img": gradcam_pil,
        "dist":        dist,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  SUPABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def register_user(username, age, sex, password):
    sb = get_supabase()
    # Use a fake email derived from username for Supabase auth
    email = f"{username.lower().replace(' ', '_')}@neuroscan.app"
    try:
        res = sb.auth.sign_up({"email": email, "password": password})
        user_id = res.user.id
        sb.table("profiles").insert({
            "id":       user_id,
            "username": username,
            "age":      int(age),
            "sex":      sex,
        }).execute()
        return True, user_id, username, age, sex
    except Exception as e:
        return False, str(e), None, None, None


def login_user(username, password):
    sb = get_supabase()
    email = f"{username.lower().replace(' ', '_')}@neuroscan.app"
    try:
        res  = sb.auth.sign_in_with_password({"email": email, "password": password})
        uid  = res.user.id
        prof = sb.table("profiles").select("*").eq("id", uid).single().execute()
        d    = prof.data
        return True, uid, d["username"], d["age"], d["sex"]
    except Exception as e:
        return False, str(e), None, None, None


def save_prediction(user_id, username, age, sex, pred_class, confidence, status, description):
    sb = get_supabase()
    sb.table("predictions").insert({
        "user_id":    user_id,
        "username":   username,
        "age":        int(age),
        "sex":        sex,
        "tumor_class": pred_class,
        "confidence": round(confidence, 4),
        "status":     status,
        "description": description,
    }).execute()


def get_history(user_id):
    sb = get_supabase()
    res = (sb.table("predictions")
             .select("*")
             .eq("user_id", user_id)
             .order("created_at", desc=True)
             .execute())
    return res.data or []


def delete_prediction(record_id):
    sb = get_supabase()
    sb.table("predictions").delete().eq("id", record_id).execute()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: LOGIN / REGISTER
# ═══════════════════════════════════════════════════════════════════════════════

def page_login():
    st.markdown("""
    <div class="app-title">
        <h1>🧠 NeuroScan AI</h1>
        <p>Brain Tumor MRI Detection — powered by VGG16 deep learning</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab_login, tab_reg = st.tabs(["Login", "Register"])

        # ── LOGIN ──
        with tab_login:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            username = st.text_input("Username", key="li_user")
            password = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login", use_container_width=True, key="btn_login"):
                if username and password:
                    ok, uid_or_err, uname, age, sex = login_user(username, password)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user_id   = uid_or_err
                        st.session_state.username  = uname
                        st.session_state.age       = age
                        st.session_state.sex       = sex
                        st.session_state.page      = "upload"
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please fill in all fields.")
            st.markdown("</div>", unsafe_allow_html=True)

        # ── REGISTER ──
        with tab_reg:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            r_name = st.text_input("Full Name",   key="rg_name")
            r_age  = st.number_input("Age", min_value=1, max_value=120,
                                     value=25, key="rg_age")
            r_sex  = st.selectbox("Sex", ["Male", "Female", "Other"], key="rg_sex")
            r_pass = st.text_input("Password", type="password", key="rg_pass")
            r_pass2= st.text_input("Confirm Password", type="password", key="rg_pass2")
            if st.button("Create Account", use_container_width=True, key="btn_reg"):
                if not all([r_name, r_pass, r_pass2]):
                    st.warning("Please fill in all fields.")
                elif r_pass != r_pass2:
                    st.error("Passwords do not match.")
                else:
                    ok, uid_or_err, uname, age, sex = register_user(
                        r_name, r_age, r_sex, r_pass
                    )
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user_id   = uid_or_err
                        st.session_state.username  = uname
                        st.session_state.age       = age
                        st.session_state.sex       = sex
                        st.session_state.page      = "upload"
                        st.rerun()
                    else:
                        st.error(f"Registration failed: {uid_or_err}")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ FOR EDUCATIONAL PURPOSES ONLY — NOT A MEDICAL DEVICE — ALWAYS CONSULT A QUALIFIED PHYSICIAN
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: UPLOAD + RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def page_upload():
    navbar()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;margin-bottom:1.5rem'>
        <span class='pill'>VGG16 Transfer Learning</span>
        <span class='pill' style='color:#4aff8a'>Grad-CAM Visualization</span>
        <span class='pill' style='color:#ffa04a'>OOD Rejection</span>
        <span class='pill' style='color:#ff4a9e'>4-Class Detection</span>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_res = st.columns([1, 1.2], gap="large")

    with col_up:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**UPLOAD BRAIN MRI**")
        uploaded = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded image", use_column_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            analyze = st.button("🔍 Analyze MRI", use_container_width=True,
                                 type="primary", disabled=uploaded is None)
        with col_b:
            clear = st.button("Clear", use_container_width=True)
            if clear:
                st.session_state.pop("result", None)
                st.rerun()

        st.markdown("""
        <div style='margin-top:1rem;font-size:12px;color:#4a5568'>
            <div style='color:#4aff8a'>✅ Brain MRI scans (axial / coronal / sagittal)</div>
            <div style='color:#ff4a4a'>❌ CT scans, X-rays, natural photos</div>
            <div style='color:#ff4a4a'>❌ Non-brain MRIs (knee, spine, etc.)</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        if analyze and uploaded:
            with st.spinner("Analyzing MRI…"):
                result = run_inference(pil_img)
            st.session_state.result = result

            # Save to Supabase
            if result["accepted"]:
                info = CLASS_INFO.get(result["pred_class"], {})
                save_prediction(
                    user_id    = st.session_state.user_id,
                    username   = st.session_state.username,
                    age        = st.session_state.age,
                    sex        = st.session_state.sex,
                    pred_class = result["pred_class"],
                    confidence = result["confidence"],
                    status     = "Accepted",
                    description= info.get("desc", ""),
                )
            else:
                save_prediction(
                    user_id    = st.session_state.user_id,
                    username   = st.session_state.username,
                    age        = st.session_state.age,
                    sex        = st.session_state.sex,
                    pred_class = "rejected",
                    confidence = 0.0,
                    status     = "Rejected",
                    description= result.get("reason", ""),
                )

        result = st.session_state.get("result")

        if result is None:
            st.markdown("""
            <div class='card' style='text-align:center;padding:3rem 1rem;'>
                <div style='font-size:3rem'>🧠</div>
                <p style='color:#4a5568;margin-top:0.5rem'>
                    Upload an MRI and click Analyze to see results
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif not result["accepted"]:
            st.markdown(f"""
            <div class='result-card result-reject'>
                <h3 style='color:#e74c3c;margin:0 0 8px'>⛔ Rejected — Not a Brain MRI</h3>
                <p style='color:#ff8888;margin:0 0 6px'>
                    Upload a clear Brain MRI image.
                </p>
                <p style='color:#666;font-size:13px;margin:0'>
                    Reason: {result.get('reason', 'Unknown')}
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            info  = CLASS_INFO.get(result["pred_class"], {})
            color = info.get("color", "#fff")
            conf  = result["confidence"]
            bar_w = int(conf * 100)

            st.markdown(f"""
            <div class='result-card result-accept'>
                <h3 style='color:{color};margin:0 0 4px'>
                    {info.get('icon','')} {info.get('label', result['pred_class'])}
                </h3>
                <div style='font-size:13px;color:#888;margin-bottom:8px'>
                    Confidence: <b style='color:{color}'>{conf:.1%}</b>
                </div>
                <div class='conf-bar-wrap'>
                    <div class='conf-bar-fill'
                         style='width:{bar_w}%;background:{color}'></div>
                </div>
                <p style='color:#ccc;font-size:13px;line-height:1.6;
                           border-left:3px solid {color};padding-left:10px;
                           margin:0 0 10px'>
                    {info.get('desc','')}
                </p>
                <div style='background:#111;border-radius:6px;padding:8px 12px;
                             font-size:13px'>
                    <span style='color:#666'>STATUS &nbsp;</span>
                    <span style='color:{color};font-weight:600'>
                        {info.get('severity','')}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Class probabilities
            st.markdown("**All class probabilities**")
            for cls, prob in result["probs"].items():
                ci     = CLASS_INFO.get(cls, {})
                c      = ci.get("color", "#888")
                w      = int(prob * 100)
                lbl    = ci.get("label", cls)
                st.markdown(f"""
                <div style='margin:5px 0'>
                    <div style='display:flex;justify-content:space-between;
                                font-size:13px;color:#ccc;margin-bottom:3px'>
                        <span>{lbl}</span>
                        <span style='color:{c};font-weight:600'>{prob:.1%}</span>
                    </div>
                    <div class='conf-bar-wrap'>
                        <div class='conf-bar-fill' style='width:{w}%;background:{c}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Grad-CAM
            if result.get("gradcam_img"):
                st.markdown("**Grad-CAM Heatmap**")
                st.image(result["gradcam_img"],
                         caption="Attention regions highlighted",
                         use_column_width=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ FOR EDUCATIONAL PURPOSES ONLY — NOT A MEDICAL DEVICE — CONSULT A QUALIFIED PHYSICIAN
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

def page_history():
    navbar()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Scan History")

    records = get_history(st.session_state.user_id)

    if not records:
        st.markdown("""
        <div class='card' style='text-align:center;padding:2.5rem'>
            <p style='color:#4a5568'>No scan history yet. Upload your first MRI!</p>
        </div>
        """, unsafe_allow_html=True)
        return

    for rec in records:
        info   = CLASS_INFO.get(rec["tumor_class"], {})
        color  = info.get("color", "#888")
        date   = rec["created_at"][:10] if rec.get("created_at") else "—"
        status = rec.get("status", "—")

        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"""
            <div class='card' style='margin-bottom:0.5rem'>
                <div style='display:flex;gap:1.5rem;align-items:center;flex-wrap:wrap'>
                    <div>
                        <div style='color:#888;font-size:11px'>DATE</div>
                        <div style='color:#e0e6f0;font-size:14px'>{date}</div>
                    </div>
                    <div>
                        <div style='color:#888;font-size:11px'>RESULT</div>
                        <div style='color:{color};font-size:14px;font-weight:600'>
                            {info.get('icon','') + ' ' + info.get('label', rec['tumor_class'])}
                        </div>
                    </div>
                    <div>
                        <div style='color:#888;font-size:11px'>CONFIDENCE</div>
                        <div style='color:{color};font-size:14px'>
                            {f"{rec['confidence']:.1%}" if rec.get('confidence') else '—'}
                        </div>
                    </div>
                    <div>
                        <div style='color:#888;font-size:11px'>STATUS</div>
                        <div style='color:#ccc;font-size:13px'>{status}</div>
                    </div>
                    <div style='flex:1;min-width:160px'>
                        <div style='color:#888;font-size:11px'>PATIENT</div>
                        <div style='color:#ccc;font-size:13px'>
                            {rec.get('username','—')} · {rec.get('age','—')} yrs · {rec.get('sex','—')}
                        </div>
                    </div>
                </div>
                {f"<p style='color:#555;font-size:12px;margin:8px 0 0;'>{rec.get('description','')[:120]}</p>" if rec.get('description') else ''}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("🗑 Delete", key=f"del_{rec['id']}"):
                delete_prediction(rec["id"])
                st.success("Deleted.")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  NAVBAR
# ═══════════════════════════════════════════════════════════════════════════════

def navbar():
    page = st.session_state.get("page", "upload")
    uname = st.session_state.get("username", "")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        st.markdown(
            "<span style='font-family:monospace;font-size:1.1rem;"
            "color:#fff;font-weight:700'>🧠 NeuroScan AI</span>",
            unsafe_allow_html=True,
        )
    with col2:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Upload", use_container_width=True,
                          type="primary" if page == "upload" else "secondary"):
                st.session_state.page = "upload"
                st.rerun()
        with c2:
            if st.button("History", use_container_width=True,
                          type="primary" if page == "history" else "secondary"):
                st.session_state.page = "history"
                st.rerun()
    with col3:
        st.markdown(
            f"<div style='text-align:right;color:#4a9eff;font-size:13px'>"
            f"👤 {uname}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Logout", use_container_width=True):
            for key in ["logged_in", "user_id", "username",
                        "age", "sex", "page", "result"]:
                st.session_state.pop(key, None)
            st.rerun()

    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="NeuroScan AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "page" not in st.session_state:
        st.session_state.page = "upload"

    if not st.session_state.logged_in:
        page_login()
    elif st.session_state.page == "history":
        page_history()
    else:
        page_upload()


if __name__ == "__main__":
    main()
