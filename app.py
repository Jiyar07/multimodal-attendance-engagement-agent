"""
Multimodal Attendance & Engagement Agent
Privacy-first local classroom analytics prototype.

Run:
    pip install -r requirements.txt
    streamlit run app.py

Important:
- Uploaded media is processed by this Python process; this app does not intentionally
  transmit classroom media to an external service.
- Face detections are anonymous temporary tracks, NOT verified student identities.
- Visual/audio heuristics are approximate and must not be used for grading, discipline,
  or high-stakes decisions. Review outputs with an educator.
"""
from __future__ import annotations

import io
import json
import math
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

APP_NAME = "Multimodal Attendance & Engagement Agent"
st.set_page_config(page_title=APP_NAME, page_icon="🎓", layout="wide")

# ---------- Styling ----------
st.markdown("""
<style>
    .main {background: #f7f9fc;}
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .hero {padding: 1.25rem 1.5rem; border-radius: 16px;
           background: linear-gradient(120deg,#172554,#2563eb); color: white;
           margin-bottom: 1rem;}
    .hero h1 {color:white; margin:0;}
    .hero p {color:#dbeafe; margin:.4rem 0 0 0;}
    div[data-testid="stMetric"] {background:white; border:1px solid #e5eaf2;
        padding:14px; border-radius:12px;}
    .note {padding: .8rem 1rem; border-left: 4px solid #2563eb;
           background: #eff6ff; border-radius: 6px;}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <h1>🎓 {APP_NAME}</h1>
  <p>Offline-first classroom signals • aggregate engagement insights • educator-led decisions</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "Privacy & responsible use: this is a prototype, not a validated attendance or "
    "emotion-recognition system. Face tracks are anonymous and can be wrong; audio labels "
    "are heuristic. Obtain appropriate consent, follow institutional rules, and do not "
    "use the scores for grading, discipline, or decisions about individual students."
)

# ---------- Session state ----------
defaults = {
    "attendance_df": pd.DataFrame(),
    "audio_df": pd.DataFrame(),
    "fusion_df": pd.DataFrame(),
    "suggestions": [],
    "session_report": {},
    "face_counts": {},
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------- Utilities ----------
def safe_name(name: str) -> str:
    return Path(name).name.replace(" ", "_")


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def load_image(upload) -> np.ndarray:
    data = np.frombuffer(upload.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode this image file.")
    return image


def make_face_detector():
    # MediaPipe is preferred; Haar cascade provides a fallback if unavailable.
    try:
        import mediapipe as mp
        detector = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5
        )
        return ("mediapipe", detector)
    except Exception:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError("Neither MediaPipe nor OpenCV face detector could be loaded.")
        return ("haar", cascade)


def detect_faces(frame: np.ndarray, detector_pack) -> list[tuple[int, int, int, int]]:
    kind, detector = detector_pack
    h, w = frame.shape[:2]
    if kind == "mediapipe":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.process(rgb)
        boxes = []
        if result.detections:
            for det in result.detections:
                box = det.location_data.relative_bounding_box
                x = max(0, int(box.xmin * w))
                y = max(0, int(box.ymin * h))
                bw = max(1, int(box.width * w))
                bh = max(1, int(box.height * h))
                boxes.append((x, y, min(bw, w - x), min(bh, h - y)))
        return boxes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return [tuple(map(int, b)) for b in detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )]


def process_video(video_bytes: bytes, filename: str, sample_every: int,
                  presence_threshold: float) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Sample video frames and assign anonymous centroid-based temporary track IDs."""
    suffix = Path(filename).suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        temp_path = tmp.name
    cap = None
    detector_pack = None
    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Could not open video. Try MP4/MOV with a supported codec.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / fps if frame_count else 0
        step = max(1, int(sample_every))
        detector_pack = make_face_detector()
        tracks: dict[int, dict[str, Any]] = {}
        next_id = 1
        preview = None
        frame_index = 0
        last_centers: dict[int, tuple[float, float]] = {}

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % step == 0:
                if preview is None:
                    preview = frame.copy()
                boxes = detect_faces(frame, detector_pack)
                timestamp = frame_index / fps
                centers = [(x + bw / 2, y + bh / 2) for x, y, bw, bh in boxes]
                unmatched = set(last_centers.keys())
                current_centers = {}
                # Greedy nearest-centroid matching; IDs are temporary and not identity.
                for center, box in zip(centers, boxes):
                    best_id, best_dist = None, float("inf")
                    for track_id in unmatched:
                        old = last_centers[track_id]
                        dist = math.dist(center, old)
                        if dist < best_dist:
                            best_id, best_dist = track_id, dist
                    if best_id is not None and best_dist < max(frame.shape[:2]) * 0.16:
                        track_id = best_id
                        unmatched.discard(track_id)
                    else:
                        track_id = next_id
                        next_id += 1
                        tracks[track_id] = {
                            "first": timestamp, "last": timestamp,
                            "observations": 0, "box": box,
                        }
                    rec = tracks[track_id]
                    rec["last"] = timestamp
                    rec["observations"] += 1
                    rec["box"] = box
                    current_centers[track_id] = center
                last_centers = current_centers
            frame_index += 1

        if detector_pack and detector_pack[0] == "mediapipe":
            detector_pack[1].close()
        if cap:
            cap.release()

        sampled_frames = max(1, math.ceil(max(frame_index, 1) / step))
        if duration <= 0:
            duration = max(0.0, frame_index / fps)
        rows = []
        for track_id, rec in tracks.items():
            observed = rec["observations"] * step / fps
            # Approximate presence from sampled detections; capped at session duration.
            estimated = min(duration or observed, max(observed, rec["last"] - rec["first"] + step / fps))
            ratio = rec["observations"] / sampled_frames
            rows.append({
                "Student ID": f"Track-{track_id:03d}",
                "First Seen": format_seconds(rec["first"]),
                "Last Seen": format_seconds(rec["last"]),
                "Total Duration": format_seconds(estimated),
                "Presence Ratio (%)": round(ratio * 100, 1),
                "Status": "Present" if ratio >= presence_threshold else "Review",
                "Observations": rec["observations"],
            })
        df = pd.DataFrame(rows, columns=[
            "Student ID", "First Seen", "Last Seen", "Total Duration",
            "Presence Ratio (%)", "Status", "Observations"
        ])
        if preview is None:
            preview = np.zeros((360, 640, 3), dtype=np.uint8)
        metadata = {"fps": fps, "frames": frame_index, "duration": duration,
                    "sampled_frames": sampled_frames, "tracks": len(tracks)}
        return df, preview, metadata
    finally:
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        try:
            if detector_pack and detector_pack[0] == "mediapipe":
                detector_pack[1].close()
        except Exception:
            pass
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def process_photo(image: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    detector_pack = make_face_detector()
    try:
        boxes = detect_faces(image, detector_pack)
    finally:
        if detector_pack[0] == "mediapipe":
            detector_pack[1].close()
    rows = []
    for i, (x, y, w, h) in enumerate(boxes, start=1):
        rows.append({
            "Student ID": f"Track-{i:03d}",
            "First Seen": "00:00",
            "Last Seen": "00:00",
            "Total Duration": "N/A (photo)",
            "Presence Ratio (%)": 0.0,
            "Status": "Review (photo only)",
            "Observations": 1,
        })
        cv2.rectangle(image, (x, y), (x + w, y + h), (40, 200, 100), 2)
        cv2.putText(image, f"Track-{i:03d}", (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 200, 100), 2)
    return pd.DataFrame(rows), image


def audio_segments(audio_bytes: bytes, filename: str, window_s: float = 3.0) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower() or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        temp_path = tmp.name
    try:
        y, sr = librosa.load(temp_path, sr=22050, mono=True)
        if y.size == 0:
            raise ValueError("Audio file contains no readable samples.")
        rows = []
        win = max(1, int(window_s * sr))
        rms_all = librosa.feature.rms(y=y)[0]
        noise_floor = float(np.percentile(rms_all, 25)) if len(rms_all) else 0.0
        speech_floor = max(0.012, noise_floor * 2.2)
        for start in range(0, len(y), win):
            segment = y[start:start + win]
            if len(segment) < int(0.15 * sr):
                continue
            rms = float(np.sqrt(np.mean(segment ** 2) + 1e-12))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(segment)[0]))
            centroid = float(np.mean(librosa.feature.spectral_centroid(
                y=segment, sr=sr
            )[0]))
            duration = len(segment) / sr
            # Heuristic energy-based label; it cannot reliably identify who is speaking.
            if rms < max(0.006, noise_floor * 1.25):
                label = "Silence"
            elif rms >= max(0.09, speech_floor * 5.0) and centroid > 1800:
                label = "High Ambient Classroom Distraction"
            elif rms >= speech_floor and zcr < 0.22:
                label = "Instructor Lecture / Speech"
            elif rms >= speech_floor:
                label = "Student Discussion / Verbal Interaction"
            else:
                label = "Low-level ambient sound"
            rows.append({
                "Start (s)": round(start / sr, 2),
                "End (s)": round((start / sr) + duration, 2),
                "RMS Energy": round(rms, 4),
                "Spectral Centroid (Hz)": round(centroid, 1),
                "Zero Crossing Rate": round(zcr, 4),
                "Audio Class": label,
            })
        return pd.DataFrame(rows)
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def build_fusion(audio_df: pd.DataFrame, attendance_df: pd.DataFrame) -> pd.DataFrame:
    if audio_df.empty:
        return pd.DataFrame(columns=["Time (s)", "Audio Class", "Audio Activity", "Visual Presence", "Engagement Index"])
    out = audio_df.copy()
    active_labels = {
        "Instructor Lecture / Speech": 0.65,
        "Student Discussion / Verbal Interaction": 0.9,
        "High Ambient Classroom Distraction": 0.2,
        "Silence": 0.45,
        "Low-level ambient sound": 0.4,
    }
    out["Audio Activity"] = out["Audio Class"].map(active_labels).fillna(0.4)
    # Attendance tracks are a coarse room-level presence proxy, not attention.
    present_count = len(attendance_df[attendance_df["Status"] == "Present"]) if not attendance_df.empty and "Status" in attendance_df else 0
    track_count = len(attendance_df)
    visual_presence = min(1.0, present_count / max(track_count, 1)) if track_count else 0.5
    out["Visual Presence"] = visual_presence
    # Avoid claiming gaze/posture detection: this prototype only uses detected face presence.
    out["Engagement Index"] = (
        100 * (0.65 * out["Audio Activity"] + 0.35 * out["Visual Presence"])
    ).round(1)
    out["Time (s)"] = out["Start (s)"]
    return out


def teaching_suggestions(fusion_df: pd.DataFrame) -> list[str]:
    if fusion_df.empty:
        return ["Upload an audio clip to generate a preliminary advisory."]
    suggestions = []
    mean_score = float(fusion_df["Engagement Index"].mean())
    distraction = float((fusion_df["Audio Class"] == "High Ambient Classroom Distraction").mean())
    discussion = float((fusion_df["Audio Class"] == "Student Discussion / Verbal Interaction").mean())
    silence = float((fusion_df["Audio Class"] == "Silence").mean())
    if mean_score < 45:
        suggestions.append("Collective signal score is low: try a quick poll, a short question, or a two-minute reset break.")
    elif mean_score < 65:
        suggestions.append("Signals are mixed: use a one-minute recap or ask learners to explain the key idea to a partner.")
    else:
        suggestions.append("Signals look relatively active: continue, and check understanding with a brief formative question.")
    if distraction > 0.25:
        suggestions.append("Elevated high-energy/noisy segments detected: pause briefly, clarify the activity, and check room acoustics.")
    if discussion > 0.3:
        suggestions.append("Verbal interaction segments are frequent: consider a structured think-pair-share or summarize learner questions.")
    if silence > 0.5:
        suggestions.append("Many quiet segments: silence can mean focused work, so use a low-pressure comprehension check rather than assuming disengagement.")
    return suggestions


def make_report(attendance_df: pd.DataFrame, audio_df: pd.DataFrame,
                fusion_df: pd.DataFrame) -> dict:
    present = int((attendance_df["Status"] == "Present").sum()) if not attendance_df.empty and "Status" in attendance_df else 0
    tracks = int(len(attendance_df))
    mean_engagement = round(float(fusion_df["Engagement Index"].mean()), 1) if not fusion_df.empty else None
    audio_counts = audio_df["Audio Class"].value_counts().to_dict() if not audio_df.empty else {}
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "attendance_tracks_detected": tracks,
        "tracks_meeting_presence_threshold": present,
        "audio_windows_analysed": int(len(audio_df)),
        "mean_engagement_index": mean_engagement,
        "audio_class_window_counts": audio_counts,
        "limitations": [
            "Anonymous face tracks are not verified student identities.",
            "Audio classes are energy/spectral heuristics, not reliable speaker identification.",
            "Engagement Index is a prototype signal summary, not a validated measure of learning or attention.",
            "No external database is used; session data is held in Streamlit session memory.",
        ],
    }


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Session settings")
    session_name = st.text_input("Session name", "Classroom Session 01")
    presence_threshold = st.slider(
        "Sustained presence threshold", 0.50, 0.95, 0.70, 0.05,
        help="Prototype uses sampled-frame detection ratio. It does not identify real student names."
    )
    sample_every = st.slider("Video sampling interval (frames)", 3, 60, 12, 3)
    audio_window = st.slider("Audio window (seconds)", 1.0, 8.0, 3.0, 0.5)
    st.caption("All processing runs locally in this app process. Avoid public deployment with real classroom media.")

tabs = st.tabs([
    "1 · Attendance", "2 · Audio Dynamics", "3 · Signal Fusion",
    "4 · Teaching Advisories", "5 · Session Report"
])

# ---------- Tab 1 ----------
with tabs[0]:
    st.subheader("Automated presence-based attendance")
    st.markdown(
        "Upload a classroom video for sampled face detection and temporary track IDs. "
        "A single photo can count visible faces, but cannot establish sustained attendance."
    )
    media_type = st.radio("Input type", ["Video", "Photo"], horizontal=True)
    if media_type == "Video":
        video_file = st.file_uploader("Upload classroom video", type=["mp4", "mov", "avi", "mkv"], key="video_upload")
        if video_file and st.button("Process video locally", type="primary"):
            with st.spinner("Sampling video and detecting faces locally…"):
                try:
                    df, preview, meta = process_video(
                        video_file.getvalue(), video_file.name, sample_every, presence_threshold
                    )
                    st.session_state.attendance_df = df
                    st.session_state.video_meta = meta
                    st.session_state.video_preview = preview
                    st.success("Video processing finished.")
                except Exception as exc:
                    st.error(f"Video processing failed: {exc}")
        if "video_meta" in st.session_state:
            meta = st.session_state.video_meta
            c1, c2, c3 = st.columns(3)
            c1.metric("Video duration", format_seconds(meta["duration"]))
            c2.metric("Temporary tracks", meta["tracks"])
            c3.metric("Frames sampled", meta["sampled_frames"])
            st.caption(f"Decoder FPS: {meta['fps']:.1f} · Processed frames: {meta['frames']}")
            if "video_preview" in st.session_state:
                st.image(cv2.cvtColor(st.session_state.video_preview, cv2.COLOR_BGR2RGB),
                         caption="First readable frame (preview only)", use_container_width=True)
    else:
        photo_file = st.file_uploader("Upload classroom photo", type=["jpg", "jpeg", "png", "webp"], key="photo_upload")
        if photo_file and st.button("Detect faces in photo", type="primary"):
            try:
                image = load_image(photo_file)
                df, annotated = process_photo(image)
                st.session_state.attendance_df = df
                st.session_state.photo_preview = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.success(f"Detected {len(df)} face(s). A photo cannot establish attendance duration.")
            except Exception as exc:
                st.error(f"Photo processing failed: {exc}")
        if "photo_preview" in st.session_state:
            st.image(st.session_state.photo_preview, caption="Anonymous face boxes", use_container_width=True)

    attendance_df = st.session_state.attendance_df
    if not attendance_df.empty:
        st.markdown("#### Attendance register")
        st.dataframe(attendance_df, use_container_width=True, hide_index=True)
        st.download_button("Download attendance CSV", dataframe_csv(attendance_df),
                           file_name=f"{safe_name(session_name)}_attendance.csv",
                           mime="text/csv")
        st.caption("Track IDs are temporary detections, not student identity. Review all statuses manually.")
    else:
        st.info("Upload and process a video or photo to populate this table.")

# ---------- Tab 2 ----------
with tabs[1]:
    st.subheader("Acoustic interaction & noise dynamics")
    st.markdown("Audio is divided into windows and labelled using RMS energy, spectral centroid, and zero-crossing rate.")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "ogg", "flac", "m4a"], key="audio_upload")
    if audio_file and st.button("Analyse audio locally", type="primary"):
        with st.spinner("Analysing audio windows locally…"):
            try:
                st.session_state.audio_df = audio_segments(
                    audio_file.getvalue(), audio_file.name, audio_window
                )
                st.success("Audio analysis finished.")
            except Exception as exc:
                st.error(f"Audio processing failed: {exc}")
    audio_df = st.session_state.audio_df
    if not audio_df.empty:
        st.dataframe(audio_df, use_container_width=True, hide_index=True)
        st.download_button("Download audio analysis CSV", dataframe_csv(audio_df),
                           file_name=f"{safe_name(session_name)}_audio.csv", mime="text/csv")
        fig, ax = plt.subplots(figsize=(10, 3.2))
        ax.plot(audio_df["Start (s)"], audio_df["RMS Energy"], marker="o", linewidth=1.5)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("RMS energy")
        ax.set_title("Audio energy by window")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        counts = audio_df["Audio Class"].value_counts().rename_axis("Audio Class").reset_index(name="Windows")
        st.bar_chart(counts.set_index("Audio Class"))
    else:
        st.info("Upload an audio clip and run analysis to see acoustic windows.")

# ---------- Tab 3 ----------
with tabs[2]:
    st.subheader("Multimodal signal fusion")
    st.markdown(
        "The prototype combines audio activity with a coarse room-level face-presence proxy. "
        "It does **not** estimate gaze, emotion, or posture; those require additional validated models."
    )
    if st.button("Build fusion timeline", type="primary"):
        st.session_state.fusion_df = build_fusion(
            st.session_state.audio_df, st.session_state.attendance_df
        )
        st.session_state.suggestions = teaching_suggestions(st.session_state.fusion_df)
    fusion_df = st.session_state.fusion_df
    if not fusion_df.empty:
        mean_score = float(fusion_df["Engagement Index"].mean())
        low_windows = int((fusion_df["Engagement Index"] < 45).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean prototype index", f"{mean_score:.1f}/100")
        c2.metric("Windows below 45", low_windows)
        c3.metric("Windows analysed", len(fusion_df))
        fig, ax = plt.subplots(figsize=(10, 3.8))
        ax.plot(fusion_df["Time (s)"], fusion_df["Engagement Index"], marker="o", linewidth=2)
        ax.axhline(45, linestyle="--", linewidth=1, label="Advisory threshold: 45")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Prototype index (0–100)")
        ax.set_title("Preliminary multimodal signal timeline")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.dataframe(fusion_df, use_container_width=True, hide_index=True)
        st.download_button("Download fusion CSV", dataframe_csv(fusion_df),
                           file_name=f"{safe_name(session_name)}_fusion.csv", mime="text/csv")
    else:
        st.info("Analyse audio first, then select “Build fusion timeline”.")

# ---------- Tab 4 ----------
with tabs[3]:
    st.subheader("Teaching suggestion engine")
    st.markdown("Advisories are gentle prompts for the educator, not judgements about students.")
    if st.button("Generate teaching advisories", type="primary"):
        st.session_state.suggestions = teaching_suggestions(st.session_state.fusion_df)
    if st.session_state.suggestions:
        for i, suggestion in enumerate(st.session_state.suggestions, start=1):
            st.markdown(f'<div class="note"><b>Suggestion {i}</b><br>{suggestion}</div>', unsafe_allow_html=True)
    else:
        st.info("Build the fusion timeline to generate context-aware suggestions.")
    st.caption("A quiet room may indicate focused work; loudness does not necessarily mean distraction.")

# ---------- Tab 5 ----------
with tabs[4]:
    st.subheader("Session analytics & educator report")
    if st.button("Generate session report", type="primary"):
        st.session_state.session_report = make_report(
            st.session_state.attendance_df,
            st.session_state.audio_df,
            st.session_state.fusion_df,
        )
    report = st.session_state.session_report
    if report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Anonymous tracks", report["attendance_tracks_detected"])
        c2.metric("Tracks over threshold", report["tracks_meeting_presence_threshold"])
        c3.metric("Audio windows", report["audio_windows_analysed"])
        c4.metric("Mean index", f'{report["mean_engagement_index"]}/100' if report["mean_engagement_index"] is not None else "N/A")
        st.markdown("#### Audio window breakdown")
        if report["audio_class_window_counts"]:
            breakdown = pd.DataFrame(
                [{"Audio Class": k, "Windows": v} for k, v in report["audio_class_window_counts"].items()]
            )
            st.dataframe(breakdown, use_container_width=True, hide_index=True)
        st.markdown("#### Limitations & review notes")
        for item in report["limitations"]:
            st.write(f"• {item}")
        st.download_button(
            "Download session report (JSON)",
            data=json.dumps({"session_name": session_name, **report}, indent=2).encode("utf-8"),
            file_name=f"{safe_name(session_name)}_report.json",
            mime="application/json",
        )
    else:
        st.info("Generate a report to view the session KPIs and download a JSON summary.")

st.divider()
st.caption("Local prototype • no external database • outputs are approximate and should be reviewed by an educator.")
