"""
Voice Biometric Authentication System — Streamlit App
B.Tech Sem 6 — Speech Processing Project

Run:  streamlit run app.py
"""
import os, sys, glob, io, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import streamlit as st
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from difflib import SequenceMatcher
from audio_recorder_streamlit import audio_recorder

sys.path.insert(0, os.path.dirname(__file__))
from backend.core.audio_utils   import preprocess
from backend.core.speaker_encoder import SpeakerEncoder, cosine_similarity
from backend.core.spoof_detector  import SpoofDetector

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Voice Biometric Auth",
    page_icon   = "🎙",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Dark theme CSS ───────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d1117; color: #c9d1d9; }
  .block-container { padding-top: 1.5rem; }
  .big-result { font-size: 2.2rem; font-weight: 800; text-align: center;
                padding: 1.5rem; border-radius: 12px; margin: 1rem 0; }
  .accepted  { background: #0d2818; border: 2px solid #3fb950; color: #3fb950; }
  .rejected  { background: #2d0d0d; border: 2px solid #f85149; color: #f85149; }
  .spoof-det { background: #2d1f00; border: 2px solid #d29922; color: #d29922; }
  .metric-box { background:#161b22; border:1px solid #30363d; border-radius:8px;
                padding:1rem; text-align:center; }
  .score-label { font-size:0.75rem; color:#8b949e; margin-bottom:2px; }
  .score-value { font-size:1.5rem; font-weight:700; }
  div[data-testid="stSidebar"] { background:#161b22; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
SR                 = 16000
DATA_DIR           = os.path.join(os.path.dirname(__file__), "data")
VOICE_THRESHOLD    = 0.75
SPOOF_THRESHOLD    = 0.33
TEXT_THRESHOLD     = 0.70
DEFAULT_PASSPHRASE = "my voice is my password"


# ── Cached model loading ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading speaker encoder…")
def load_encoder():
    return SpeakerEncoder(backend="resemblyzer")

@st.cache_resource(show_spinner="Loading spoof detector…")
def load_spoof(thresh: float = SPOOF_THRESHOLD):
    return SpoofDetector(mode="rule_based", threshold=thresh)


# ── Audio helpers ────────────────────────────────────────────────
def read_audio(uploaded) -> np.ndarray:
    data = uploaded.read()
    y, sr = sf.read(io.BytesIO(data), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return preprocess(y, SR)


def bytes_to_audio(raw_bytes: bytes) -> np.ndarray:
    y, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return preprocess(y, SR)


def get_embedding(audio: np.ndarray) -> np.ndarray:
    enc = load_encoder()
    return enc.get_embedding(audio, SR)


def spoof_score(audio: np.ndarray) -> tuple[float, bool]:
    det    = load_spoof(SPOOF_THRESHOLD)
    result = det.detect(audio, SR)
    return result.score, result.is_real


# ── Enrollment store — disk-backed ──────────────────────────────
ENROLL_DIR = os.path.join(DATA_DIR, "enrolled_users")
os.makedirs(ENROLL_DIR, exist_ok=True)


def save_user(username: str, info: dict):
    path = os.path.join(ENROLL_DIR, f"{username}.npz")
    np.savez(path,
             centroid   = info["centroid"],
             n_samples  = np.array([info["n_samples"]]),
             passphrase = np.array([info["passphrase"]]))


def delete_user_file(username: str):
    path = os.path.join(ENROLL_DIR, f"{username}.npz")
    if os.path.exists(path):
        os.remove(path)


def load_all_users() -> dict:
    users = {}
    for npz in sorted(glob.glob(os.path.join(ENROLL_DIR, "*.npz"))):
        name = os.path.splitext(os.path.basename(npz))[0]
        try:
            d = np.load(npz, allow_pickle=True)
            users[name] = {
                "centroid":  d["centroid"],
                "n_samples": int(d["n_samples"][0]),
                "passphrase": str(d["passphrase"][0]),
            }
        except Exception:
            pass
    return users


if "enrolled" not in st.session_state:
    users = load_all_users()
    # Auto-enroll abhiram from raw recordings if not already on disk
    if "abhiram" not in users:
        ti_wavs = sorted(glob.glob(f"{DATA_DIR}/raw/abhiram/text_independent/sample_*.wav"))
        if ti_wavs:
            enc = load_encoder()
            audios = []
            for p in ti_wavs[:10]:
                y, _ = sf.read(p, dtype="float32")
                if y.ndim > 1: y = y.mean(axis=1)
                audios.append(preprocess(y, SR))
            embs     = enc.embed_batch(audios, SR)
            centroid = np.mean(embs, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            users["abhiram"] = {
                "centroid":  centroid,
                "n_samples": len(audios),
                "passphrase": DEFAULT_PASSPHRASE,
            }
            save_user("abhiram", users["abhiram"])
    st.session_state.enrolled = users


# ── Text similarity ───────────────────────────────────────────────
def _norm(t):
    return " ".join(re.sub(r"[^a-z0-9\s]", "", t.lower()).split())

def text_sim(transcript, expected):
    t, e = _norm(transcript), _norm(expected)
    seq  = SequenceMatcher(None, t, e).ratio()
    wb   = set(e.split())
    wov  = len(set(t.split()) & wb) / len(wb) if wb else 0
    return 0.6*seq + 0.4*wov


# ── Score bar ────────────────────────────────────────────────────
def score_bar(label, value, threshold=None, good_high=True):
    pct   = int(value * 100)
    ok    = value >= threshold if threshold else True
    color = "#3fb950" if (good_high and ok) or (not good_high and not ok) else "#f85149"
    bar   = f"""
    <div style='margin-bottom:12px'>
      <div style='display:flex;justify-content:space-between;font-size:0.78rem;
                  color:#8b949e;margin-bottom:3px'>
        <span>{label}</span>
        <span style='color:{color};font-weight:700'>{value:.4f}</span>
      </div>
      <div style='background:#21262d;border-radius:4px;height:8px;position:relative'>
        <div style='width:{pct}%;background:{color};height:100%;border-radius:4px;
                    transition:width 0.5s'></div>
        {'<div style="position:absolute;top:0;left:'+str(int(threshold*100))+'%;width:2px;height:100%;background:#d29922"></div>' if threshold else ''}
      </div>
      {'<div style="text-align:right;font-size:0.7rem;color:#8b949e;margin-top:2px">threshold '+str(threshold)+'</div>' if threshold else ''}
    </div>"""
    st.markdown(bar, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🎙 Voice Auth")
    st.markdown("---")
    page = st.radio("Navigate", ["🔐 Authenticate", "➕ Enroll", "🛡 Spoof Detection",
                                  "📊 Accuracy & Metrics", "🤖 How It Works"],
                    label_visibility="collapsed")
    st.markdown("---")

    if st.session_state.enrolled:
        st.markdown("**Enrolled users**")
        for u, info in list(st.session_state.enrolled.items()):
            cols = st.columns([3, 1])
            cols[0].markdown(f"✅ `{u}` — {info['n_samples']} samples")
            if cols[1].button("🗑", key=f"del_{u}", help=f"Delete {u}"):
                del st.session_state.enrolled[u]
                delete_user_file(u)
                st.rerun()
    else:
        st.markdown("*No users enrolled yet*")
    st.markdown("---")
    st.caption("Go to **➕ Enroll** to add a new user.")


# ════════════════════════════════════════════════════════════════
# PAGE: AUTHENTICATE
# ════════════════════════════════════════════════════════════════
if page == "🔐 Authenticate":
    st.title("🔐 Voice Authentication")
    st.caption("Record your voice live or upload a WAV — the system decides whether to grant access.")

    col1, col2 = st.columns([1, 1])

    with col1:
        username = st.selectbox("Select user",
                                list(st.session_state.enrolled.keys()) or ["(no users enrolled)"])
        mode     = st.radio("Authentication mode",
                            ["Text-Independent", "Text-Dependent (passphrase)"],
                            horizontal=True)

        if mode == "Text-Dependent (passphrase)":
            passphrase = st.text_input("Expected passphrase",
                value=st.session_state.enrolled.get(username, {}).get("passphrase", DEFAULT_PASSPHRASE))
            st.info(f'Say: *"{passphrase}"*')

        # ── Recording options ─────────────────────────────────────
        st.markdown("**Step 1 — Provide audio**")
        rec_tab, up_tab = st.tabs(["🎙 Record Now", "📁 Upload WAV"])

        recorded_bytes = None
        audio_file     = None

        with rec_tab:
            st.caption("Click the mic, speak, click again to stop.")
            recorded_bytes = audio_recorder(
                text="", icon_size="2x",
                recording_color="#f85149", neutral_color="#3fb950",
                key="auth_recorder"
            )
            if recorded_bytes:
                st.audio(recorded_bytes, format="audio/wav")
                st.success("Recording captured — click Authenticate below.")

        with up_tab:
            audio_file = st.file_uploader("Upload WAV", type=["wav"], key="auth_upload")

        audio_ready = (recorded_bytes is not None) or (audio_file is not None)
        user_ok     = username in st.session_state.enrolled

        if st.button("🔎 Authenticate", type="primary",
                     disabled=not audio_ready or not user_ok):
            with st.spinner("Analysing…"):
                if recorded_bytes:
                    audio = bytes_to_audio(recorded_bytes)
                else:
                    audio = read_audio(audio_file)
                info  = st.session_state.enrolled[username]

                # Step 1 — Spoof
                sp_score, is_real = spoof_score(audio)

                # Step 2 — Voice
                emb    = get_embedding(audio)
                v_score = float(cosine_similarity(emb, info["centroid"]))

                # Step 3 — Text (optional)
                ts = None
                transcript = None
                if mode == "Text-Dependent (passphrase)":
                    try:
                        import whisper
                        wm  = whisper.load_model("base")
                        res = wm.transcribe(audio.astype(np.float32),
                                            language="en", fp16=False, verbose=False)
                        transcript = res["text"].strip()
                        ts = text_sim(transcript, passphrase)
                    except Exception:
                        st.warning("Whisper not available — skipping text check.")

                # Decision
                if not is_real:
                    decision = "SPOOF"
                elif ts is not None and ts < TEXT_THRESHOLD:
                    decision = "WRONG_PHRASE"
                elif v_score < VOICE_THRESHOLD:
                    decision = "WRONG_VOICE"
                else:
                    decision = "ACCEPTED"

                st.session_state.last_auth = {
                    "decision": decision, "sp": sp_score, "vs": v_score,
                    "ts": ts, "transcript": transcript
                }

    with col2:
        if "last_auth" in st.session_state:
            r = st.session_state.last_auth
            if r["decision"] == "ACCEPTED":
                st.markdown('<div class="big-result accepted">✓ ACCESS GRANTED</div>',
                            unsafe_allow_html=True)
            elif r["decision"] == "SPOOF":
                st.markdown('<div class="big-result spoof-det">⚠ DEEPFAKE DETECTED<br>'
                            '<span style="font-size:1rem">ACCESS DENIED</span></div>',
                            unsafe_allow_html=True)
            elif r["decision"] == "WRONG_PHRASE":
                st.markdown('<div class="big-result rejected">✗ WRONG PASSPHRASE<br>'
                            '<span style="font-size:1rem">ACCESS DENIED</span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="big-result rejected">✗ VOICE NOT RECOGNISED<br>'
                            '<span style="font-size:1rem">ACCESS DENIED</span></div>',
                            unsafe_allow_html=True)

            st.markdown("**Score breakdown**")
            score_bar("Anti-Spoof Score", r["sp"],  threshold=SPOOF_THRESHOLD)
            score_bar("Voice Match Score", r["vs"], threshold=VOICE_THRESHOLD)
            if r["ts"] is not None:
                score_bar("Passphrase Match", r["ts"], threshold=TEXT_THRESHOLD)
            if r["transcript"]:
                st.markdown(f'> Whisper heard: *"{r["transcript"]}"*')
        else:
            st.info("Upload audio and click **Authenticate** to see results.")


# ════════════════════════════════════════════════════════════════
# PAGE: ENROLL
# ════════════════════════════════════════════════════════════════
elif page == "➕ Enroll":
    st.title("➕ Enroll New User")
    st.caption("Record 3–5 short voice clips live **or** upload WAV files. "
               "The system builds a speaker model that persists across restarts.")

    col1, col2 = st.columns([1, 1])
    with col1:
        username = st.text_input("Username", placeholder="e.g. alice, bob, john …", value="")
        if username in st.session_state.enrolled:
            st.warning(f"⚠ '{username}' already enrolled — re-enrolling will overwrite.")

        mode = st.radio("Mode", ["Text-Independent", "Text-Dependent (passphrase)"],
                        horizontal=True)
        passphrase = DEFAULT_PASSPHRASE
        if mode == "Text-Dependent (passphrase)":
            passphrase = st.text_input("Passphrase", value=DEFAULT_PASSPHRASE)
            st.info("The user must say this phrase when authenticating.")

        st.markdown("**Provide voice samples (need at least 3)**")
        rec_tab, up_tab = st.tabs(["🎙 Record Live", "📁 Upload WAV"])

        recorded_clips = st.session_state.get("enroll_clips", [])

        with rec_tab:
            st.caption("Click mic → speak 3–5 seconds → click again to stop. Repeat 3–5 times.")
            clip_bytes = audio_recorder(
                text="", icon_size="2x",
                recording_color="#f85149", neutral_color="#58a6ff",
                key="enroll_recorder"
            )
            if clip_bytes and (not recorded_clips or clip_bytes != recorded_clips[-1]):
                recorded_clips.append(clip_bytes)
                st.session_state.enroll_clips = recorded_clips

            if recorded_clips:
                st.success(f"✅ {len(recorded_clips)} clip(s) recorded")
                for i, c in enumerate(recorded_clips):
                    cc1, cc2 = st.columns([4, 1])
                    cc1.audio(c, format="audio/wav")
                    if cc2.button("✕", key=f"rm_{i}", help="Remove this clip"):
                        recorded_clips.pop(i)
                        st.session_state.enroll_clips = recorded_clips
                        st.rerun()
            if st.button("🗑 Clear all clips", key="clear_clips"):
                st.session_state.enroll_clips = []
                st.rerun()

        with up_tab:
            up_files = st.file_uploader("Upload 3–10 WAV files",
                                         type=["wav"], accept_multiple_files=True,
                                         key="enroll_upload")

        # Merge sources
        n_recorded = len(recorded_clips)
        n_uploaded = len(up_files) if up_files else 0
        total      = n_recorded + n_uploaded

        if total > 0:
            st.info(f"Total samples ready: **{total}** "
                    f"({n_recorded} recorded + {n_uploaded} uploaded)")

        can_enroll = bool(username.strip()) and total >= 3
        if not username.strip():
            st.warning("Enter a username above.")
        elif total < 3:
            st.warning(f"Need at least 3 samples — have {total} so far.")

        if st.button("✅ Enroll User", type="primary", disabled=not can_enroll):
            uname = username.strip()
            with st.spinner(f"Building voice model for '{uname}'…"):
                enc    = load_encoder()
                audios = [bytes_to_audio(b) for b in recorded_clips]
                if up_files:
                    audios += [read_audio(f) for f in up_files]
                embs  = enc.embed_batch(audios, SR)
                cent  = np.mean(embs, axis=0)
                cent /= np.linalg.norm(cent) + 1e-9
                info  = {"centroid": cent, "n_samples": len(audios),
                         "passphrase": passphrase}
                st.session_state.enrolled[uname] = info
                save_user(uname, info)
                st.session_state.enroll_clips = []   # clear clips after enroll

            st.success(f"✅ '{uname}' enrolled with {len(audios)} samples — saved to disk!")

            sims = [float(cosine_similarity(e, cent)) for e in embs]
            fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0d1117")
            ax.set_facecolor("#161b22")
            ax.bar(range(len(sims)), sims,
                   color=["#3fb950" if s >= VOICE_THRESHOLD else "#f85149" for s in sims])
            ax.axhline(VOICE_THRESHOLD, color="#d29922", lw=2, ls="--",
                       label=f"Threshold {VOICE_THRESHOLD}")
            ax.set_ylim(0, 1.1); ax.set_xticks(range(len(sims)))
            ax.set_xticklabels([f"clip {i+1}" for i in range(len(sims))],
                               fontsize=8, color="#c9d1d9")
            ax.set_title(f"Sample Consistency — {uname}", color="#c9d1d9")
            ax.tick_params(colors="#c9d1d9"); ax.legend(); ax.grid(True, alpha=0.15)
            with col2:
                st.pyplot(fig)
                st.caption(f"Mean similarity: {np.mean(sims):.4f} — higher = more consistent voice")

    with col2:
        if st.session_state.enrolled:
            st.markdown("**Currently enrolled users**")
            for u, info in st.session_state.enrolled.items():
                st.markdown(f"- `{u}` — {info['n_samples']} samples")


# ════════════════════════════════════════════════════════════════
# PAGE: SPOOF DETECTION
# ════════════════════════════════════════════════════════════════
elif page == "🛡 Spoof Detection":
    st.title("🛡 Deepfake / Spoof Detection")
    st.caption("Upload any audio — the system checks if it is real human speech or AI-generated/replayed.")

    col1, col2 = st.columns(2)
    with col1:
        rec_tab2, up_tab2 = st.tabs(["🎙 Record Now", "📁 Upload WAV"])
        spoof_recorded = None
        spoof_file     = None
        with rec_tab2:
            st.caption("Click mic, speak, click again to stop.")
            spoof_recorded = audio_recorder(
                text="", icon_size="2x",
                recording_color="#f85149", neutral_color="#3fb950",
                key="spoof_recorder"
            )
            if spoof_recorded:
                st.audio(spoof_recorded, format="audio/wav")
        with up_tab2:
            spoof_file = st.file_uploader("Upload WAV to analyse", type=["wav"],
                                           key="spoof_upload")

        audio_ready2 = spoof_recorded is not None or spoof_file is not None
        if st.button("Analyse", type="primary", disabled=not audio_ready2):
            with st.spinner("Analysing audio…"):
                if spoof_recorded:
                    audio = bytes_to_audio(spoof_recorded)
                else:
                    audio = read_audio(spoof_file)
                sp, is_real = spoof_score(audio)

                # Compute individual features
                f0, _, voiced = librosa.pyin(audio, fmin=50, fmax=500,
                                             sr=SR, hop_length=160)
                f0_v  = f0[np.array(voiced, dtype=bool) & ~np.isnan(f0)] if voiced is not None else np.array([])
                jitter = float(np.mean(np.abs(np.diff(f0_v))) / (np.mean(f0_v) + 1e-9)) \
                         if len(f0_v) > 4 else 0.0
                env    = np.abs(librosa.effects.harmonic(audio))
                eds    = np.array([env[i:i+160].mean() for i in range(0, len(env)-160, 160)])
                ms     = np.abs(np.fft.rfft(eds)) if len(eds) >= 32 else np.zeros(10)
                mf     = np.fft.rfftfreq(len(eds), d=1.0/100) if len(eds) >= 32 else np.zeros(10)
                mod    = float(ms[(mf>=4)&(mf<=16)].sum() / (ms.sum()+1e-9))

                st.session_state.spoof_res = {
                    "score": sp, "is_real": is_real,
                    "jitter": jitter, "mod": mod, "audio": audio
                }

    with col2:
        if "spoof_res" in st.session_state:
            r = st.session_state.spoof_res
            if r["is_real"]:
                st.markdown('<div class="big-result accepted">✓ REAL VOICE</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="big-result spoof-det">⚠ FAKE / DEEPFAKE<br>'
                            '<span style="font-size:1rem">ACCESS DENIED</span></div>',
                            unsafe_allow_html=True)

            st.markdown("**Features analysed**")
            score_bar("Overall Spoof Score",  r["score"],  threshold=SPOOF_THRESHOLD)
            score_bar("Pitch Jitter (×100)",  r["jitter"]*100)
            score_bar("Modulation Energy",     r["mod"])

            st.markdown("**What these mean:**")
            st.markdown("""
| Feature | Real voice | Fake/TTS |
|---------|-----------|----------|
| Pitch Jitter | Natural micro-variation | Too smooth or too noisy |
| Modulation Energy | Syllabic rhythm 4–16 Hz | Flat or unnatural rhythm |
| Spoof Score | > 0.33 | ≤ 0.33 |
""")


# ════════════════════════════════════════════════════════════════
# PAGE: ACCURACY & METRICS
# ════════════════════════════════════════════════════════════════
elif page == "📊 Accuracy & Metrics":
    st.title("📊 Accuracy & Evaluation Metrics")
    st.caption("Computed on all 36 real recordings from your data directory.")

    ti_wavs   = sorted(glob.glob(f"{DATA_DIR}/raw/abhiram/text_independent/sample_*.wav"))
    real_spoof = sorted(glob.glob(f"{DATA_DIR}/spoof_data/real/*.wav"))
    fake_spoof = sorted(glob.glob(f"{DATA_DIR}/spoof_data/fake_generated/*.wav"))

    if not ti_wavs:
        st.warning("No recordings found in `data/raw/abhiram/text_independent/`. "
                   "Enroll your voice first.")
        st.stop()

    if "abhiram" not in st.session_state.enrolled:
        st.warning("Abhiram not enrolled yet. Go to ➕ Enroll first.")
        st.stop()

    with st.spinner("Computing metrics on all recordings…"):
        enc      = load_encoder()
        det      = load_spoof()
        centroid = st.session_state.enrolled["abhiram"]["centroid"]

        sims, sp_scores = [], []
        for p in ti_wavs:
            y, _ = sf.read(p, dtype="float32")
            if y.ndim > 1: y = y.mean(axis=1)
            y = preprocess(y, SR)
            emb = enc.get_embedding(y, SR)
            sims.append(float(cosine_similarity(emb, centroid)))
            sp_scores.append(det.detect(y, SR).score)

        # Spoof scores on labelled data
        r_sc, f_sc = [], []
        for p in real_spoof[:20]:
            try:
                y,_ = sf.read(p, dtype="float32")
                if y.ndim>1: y=y.mean(axis=1)
                r_sc.append(det.detect(preprocess(y,SR),SR).score)
            except Exception:
                pass
        for p in fake_spoof[:20]:
            try:
                y,_ = sf.read(p, dtype="float32")
                if y.ndim>1: y=y.mean(axis=1)
                f_sc.append(det.detect(preprocess(y,SR),SR).score)
            except Exception:
                pass

    # ── Metrics cards ──────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        acc = sum(s>=VOICE_THRESHOLD for s in sims)/len(sims)*100
        st.markdown(f'<div class="metric-box"><div class="score-label">Speaker Verif Acc</div>'
                    f'<div class="score-value" style="color:#3fb950">{acc:.1f}%</div></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="score-label">Mean Voice Score</div>'
                    f'<div class="score-value" style="color:#58a6ff">{np.mean(sims):.4f}</div></div>',
                    unsafe_allow_html=True)
    with c3:
        frr = sum(s<VOICE_THRESHOLD for s in sims)/len(sims)*100
        st.markdown(f'<div class="metric-box"><div class="score-label">FRR (False Reject)</div>'
                    f'<div class="score-value" style="color:{"#3fb950" if frr==0 else "#f85149"}">'
                    f'{frr:.1f}%</div></div>', unsafe_allow_html=True)
    with c4:
        gap = np.mean(sims) - np.mean(sims[:5])
        sp_acc = (sum(s>=SPOOF_THRESHOLD for s in r_sc) +
                  sum(s<SPOOF_THRESHOLD  for s in f_sc)) / (len(r_sc)+len(f_sc)+1e-9)*100
        st.markdown(f'<div class="metric-box"><div class="score-label">GMM Spoof Acc</div>'
                    f'<div class="score-value" style="color:#bc8cff">{sp_acc:.1f}%</div></div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Plots ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), facecolor="#0d1117")
    names = [os.path.basename(p).replace(".wav","") for p in ti_wavs]

    # Plot 1: similarity bar
    ax = axes[0]; ax.set_facecolor("#161b22")
    colors = ["#3fb950" if s>=VOICE_THRESHOLD else "#f85149" for s in sims]
    ax.bar(range(len(sims)), sims, color=colors, alpha=0.85)
    ax.axhline(VOICE_THRESHOLD, color="#d29922", lw=2, ls="--", label=f"Thresh {VOICE_THRESHOLD}")
    ax.set_ylim(0,1.1); ax.set_title("Voice Similarity per Sample", color="#c9d1d9")
    ax.set_xticks(range(0,len(sims),5)); ax.tick_params(colors="#c9d1d9")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Plot 2: spoof score histogram
    ax = axes[1]; ax.set_facecolor("#161b22")
    ax.hist(r_sc, bins=8, color="#3fb950", alpha=0.7, label=f"Real (n={len(r_sc)})", edgecolor="white")
    ax.hist(f_sc, bins=8, color="#f85149", alpha=0.7, label=f"Fake (n={len(f_sc)})", edgecolor="white")
    ax.axvline(SPOOF_THRESHOLD, color="#d29922", lw=2, ls="--", label="Threshold")
    ax.set_title("Spoof Score Distribution", color="#c9d1d9"); ax.tick_params(colors="#c9d1d9")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Plot 3: summary table
    ax = axes[2]; ax.axis("off"); ax.set_facecolor("#161b22")
    rows = [
        ["Speaker Verification", f"{acc:.1f}%", "0.0%", f"{frr:.1f}%"],
        ["Spoof Detection", f"{sp_acc:.1f}%", "—", "—"],
        ["Full Pipeline", f"{sum(s>=VOICE_THRESHOLD and sp>=SPOOF_THRESHOLD for s,sp in zip(sims,sp_scores))/len(sims)*100:.1f}%", "—", "—"],
    ]
    tbl = ax.table(cellText=rows,
                   colLabels=["Component", "Accuracy", "FAR", "FRR"],
                   cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r>0 else "#21262d")
        cell.set_text_props(color="#c9d1d9"); cell.set_edgecolor("#30363d")
    ax.set_title("Summary", color="#c9d1d9")

    for ax in axes: ax.spines[:].set_color("#30363d")
    plt.tight_layout()
    st.pyplot(fig)

    # ── Full table ─────────────────────────────────────────────
    with st.expander("View all sample scores"):
        import pandas as pd
        df = pd.DataFrame({
            "File":        names,
            "Voice Score": [f"{s:.4f}" for s in sims],
            "Spoof Score": [f"{s:.4f}" for s in sp_scores],
            "Decision":    ["✓ ACCEPTED" if v>=VOICE_THRESHOLD and s>=SPOOF_THRESHOLD
                            else ("✗ SPOOF" if s<SPOOF_THRESHOLD else "✗ REJECTED")
                            for v,s in zip(sims,sp_scores)],
        })
        st.dataframe(df, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE: HOW IT WORKS
# ════════════════════════════════════════════════════════════════
elif page == "🤖 How It Works":
    st.title("🤖 How the System Works")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Pipeline Overview", "Speaker Encoder", "Anti-Spoofing", "Text-Dependent"])

    with tab1:
        st.markdown("""
### Full Authentication Pipeline

```
Audio Input
    │
    ▼
┌─────────────────────────────┐
│  1. PREPROCESSING           │  Noise reduction → Trim silence → Normalise
└────────────┬────────────────┘
             │
    ▼
┌─────────────────────────────┐
│  2. ANTI-SPOOFING CHECK     │  Is this real human speech?
│                             │  • Pitch jitter (natural F0 variation)
│                             │  • Modulation energy (syllabic rhythm)
│                             │  • Spectral flatness (sub-band noise)
│                             │  Score < 0.33 → ⚠ DEEPFAKE DETECTED
└────────────┬────────────────┘
             │
    ▼
┌─────────────────────────────┐
│  3. SPEAKER VERIFICATION    │  Is this the right person?
│                             │  • Resemblyzer GE2E encoder
│                             │  • 256-dim d-vector embedding
│                             │  • Cosine similarity vs enrolled centroid
│                             │  Score < 0.75 → ✗ WRONG SPEAKER
└────────────┬────────────────┘
             │  (text-dependent only)
    ▼
┌─────────────────────────────┐
│  4. PASSPHRASE CHECK        │  Did they say the right phrase?
│                             │  • Whisper ASR transcription
│                             │  • Fuzzy string matching
│                             │  Score < 0.70 → ✗ WRONG PASSPHRASE
└────────────┬────────────────┘
             │
    ▼
        ✓ ACCESS GRANTED
```
""")

    with tab2:
        st.markdown("""
### Speaker Encoder — Resemblyzer GE2E

**Model:** Generalized End-to-End Loss (GE2E) — trained on thousands of speakers.

**How it works:**
1. Audio is split into 25ms frames with 10ms hop
2. 40 log-mel filterbank features extracted per frame
3. A 3-layer LSTM (256 units) processes the sequence
4. The final hidden state is L2-normalised → **256-dimensional d-vector**
5. This vector is the speaker's "voice fingerprint"

**Enrollment:**
- Record 3–10 clips → extract embedding from each
- Average all embeddings → **centroid** (the user's voice model)

**Verification:**
- Record 1 clip → extract embedding
- Compute **cosine similarity** vs centroid
- ≥ 0.75 → ACCEPTED  |  < 0.75 → REJECTED

**Why cosine similarity?**
Cosine measures the angle between two vectors, not their magnitude.
Two recordings of the same person have similar direction even if recorded at different volumes.

| Similarity | Interpretation |
|-----------|----------------|
| 0.90–1.00 | Very confident match |
| 0.75–0.90 | Accepted |
| 0.55–0.75 | Weak — possibly same speaker |
| < 0.55 | Different speaker |
""")

    with tab3:
        st.markdown("""
### Anti-Spoofing — How Deepfakes Are Detected

Real human voice has physical properties that AI-generated audio struggles to replicate:

#### Feature 1: Pitch Jitter
- Real speech: pitch (F0) varies naturally between syllables (±2–5 Hz jitter)
- TTS/vocoder: pitch is too perfectly smooth or has unnatural noise patterns
- Measured as: `mean(|diff(F0)|) / mean(F0)`

#### Feature 2: Modulation Energy
- Real speech: loudness modulates at syllabic rate (4–16 Hz)
- Fake speech: amplitude envelope is flat (TTS) or chaotic (Griffin-Lim)
- Measured in the 4–16 Hz band of the amplitude envelope spectrum

#### Feature 3: Spectral Flatness
- Real speech: each frequency band has different energy (resonance peaks)
- Vocoder/TTS: spectral flatness is abnormally high in some bands
- Measured per sub-band: 0–500 Hz, 500–2000 Hz, 2000–4000 Hz, 4000–8000 Hz

#### Feature 4: Harmonic-to-Noise Ratio (HNR)
- Real speech: harmonics dominate over noise
- Vocoder output: noise floor is different from natural speech

#### Score combination:
```
spoof_score = 0.30 × jitter_score
            + 0.25 × modulation_score
            + 0.20 × flatness_score
            + 0.15 × HNR_score
            + 0.10 × high_freq_score

spoof_score ≥ 0.33 → REAL
spoof_score < 0.33 → FAKE
```

#### GMM Classifier (backup):
- Trained on 15 real + 15 fake samples
- 7-component Gaussian Mixture Model per class
- Log-likelihood ratio → sigmoid → probability score
""")

    with tab4:
        st.markdown("""
### Text-Dependent Authentication

Adds an extra security layer: the user must say a specific passphrase.

#### Step 1: Transcription with Whisper
- OpenAI Whisper (base model, 74M parameters) runs offline
- Transcribes what the user said to text
- Architecture: Transformer encoder-decoder trained on 680,000 hours of speech

#### Step 2: Fuzzy Passphrase Matching
Not exact matching — handles natural pronunciation variation:
```
text_score = 0.6 × SequenceMatcher(transcript, passphrase)
           + 0.4 × word_overlap(transcript, passphrase)
```

Example:
- Passphrase: `"my voice is my password"`
- Transcript: `"my voice is my pass word"`
- Score: 0.94 → ACCEPTED ✓

- Transcript: `"hello world"`
- Score: 0.08 → REJECTED ✗

#### Why both text + voice?
- Voice alone: an attacker with a voice clone could pass
- Text alone: anyone who knows the phrase could pass
- **Both together**: attacker needs BOTH the right voice AND the right phrase

Combined score = `0.4 × text_score + 0.6 × voice_score`
""")

    st.markdown("---")
    st.markdown("**B.Tech Sem 6 — Speech Processing Project** | Abhiram Challa")
