import os
import tempfile
import subprocess
import time
import glob
import hashlib
import io
import zipfile
import yt_dlp
import whisper
import streamlit as st
import datetime
import re
import base64
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =========================
# FFmpeg PATH (AUTO-DETECT)
# =========================
import shutil
import platform

IS_WINDOWS = platform.system() == "Windows"
EXE_SUFFIX = ".exe" if IS_WINDOWS else ""

def find_ffmpeg_path():
    """Auto-detect ffmpeg location — works on any machine."""
    # 1. Check if ffmpeg is already on PATH
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return os.path.dirname(os.path.abspath(ffmpeg_bin))

    # 2. Common Windows install locations
    if IS_WINDOWS:
        common_paths = [
            r"C:\ffmpeg\bin",
            r"C:\ffmpeg\ffmpeg-8.1.1-full_build\bin",
            r"C:\Program Files\ffmpeg\bin",
            r"C:\Program Files (x86)\ffmpeg\bin",
            os.path.expanduser(r"~\ffmpeg\bin"),
        ]
        # Also check C:\ffmpeg\*\bin pattern
        ffmpeg_root = r"C:\ffmpeg"
        if os.path.isdir(ffmpeg_root):
            for entry in os.listdir(ffmpeg_root):
                candidate = os.path.join(ffmpeg_root, entry, "bin")
                if os.path.isdir(candidate):
                    common_paths.insert(0, candidate)

        for p in common_paths:
            if os.path.isfile(os.path.join(p, "ffmpeg.exe")):
                return p

    # 3. Common Linux / macOS locations
    for p in ["/usr/bin", "/usr/local/bin", "/opt/homebrew/bin"]:
        if os.path.isfile(os.path.join(p, "ffmpeg")):
            return p

    return ""

FFMPEG_PATH = find_ffmpeg_path()
if FFMPEG_PATH:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH
    os.environ["FFMPEG_BINARY"] = os.path.join(FFMPEG_PATH, f"ffmpeg{EXE_SUFFIX}")
    os.environ["FFPROBE_BINARY"] = os.path.join(FFMPEG_PATH, f"ffprobe{EXE_SUFFIX}")
else:
    # ffmpeg might still work if it's on the system PATH already
    pass


# =========================
# CACHED MODEL LOADERS
# =========================
@st.cache_resource
def load_summarizer():
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")


# =========================
# DETAILED SUMMARIZER (CHUNKED)
# =========================
def generate_detailed_summary(text, tokenizer, model):
    words = text.split()
    if len(words) < 40:
        return text

    chunk_size = 900
    overlap = 150
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap

    chunk_summaries = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", max_length=1024, truncation=True)
        wc = len(chunk.split())
        safe_max = min(300, max(100, int(wc * 0.65)))
        safe_min = min(60, int(safe_max * 0.4))
        ids = model.generate(inputs["input_ids"], max_length=safe_max, min_length=safe_min,
                             length_penalty=1.0, num_beams=5, early_stopping=True,
                             no_repeat_ngram_size=3)
        chunk_summaries.append(tokenizer.decode(ids[0], skip_special_tokens=True))

    if len(chunk_summaries) > 1:
        combined = " ".join(chunk_summaries)
        if len(combined.split()) > 120:
            inputs = tokenizer(combined, return_tensors="pt", max_length=1024, truncation=True)
            ids = model.generate(inputs["input_ids"], max_length=400, min_length=120,
                                 length_penalty=1.0, num_beams=5, early_stopping=True,
                                 no_repeat_ngram_size=3)
            overview = tokenizer.decode(ids[0], skip_special_tokens=True)
            return f"**Overview:**\n{overview}\n\n**Detailed Breakdown:**\n" + "\n\n".join(
                [f"**Section {i+1}:**\n• {s}" for i, s in enumerate(chunk_summaries)])
        return "\n\n".join([f"**Section {i+1}:**\n• {s}" for i, s in enumerate(chunk_summaries)])
    return chunk_summaries[0]


# =========================
# AUDIO DOWNLOADER
# =========================
def download_audio(youtube_url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, "%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best", "outtmpl": output_template,
            "quiet": True, "no_warnings": True,
            "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        }
        if FFMPEG_PATH:
            ydl_opts["ffmpeg_location"] = FFMPEG_PATH
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_id = info["id"]

            # Try the expected .mp3 path first
            expected = os.path.join(temp_dir, video_id + ".mp3")
            if os.path.exists(expected):
                return expected, info

            # Fallback: find whatever file yt-dlp actually produced
            candidates = glob.glob(os.path.join(temp_dir, f"{video_id}.*"))
            if candidates:
                return candidates[0], info

            # Last resort: any audio file in the temp dir
            all_files = glob.glob(os.path.join(temp_dir, "*"))
            audio_files = [f for f in all_files if not f.endswith('.part') and os.path.getsize(f) > 1000]
            if audio_files:
                return audio_files[0], info

            return f"ERROR_DOWNLOAD: Audio file was not created in {temp_dir}. Files found: {all_files}", None
    except Exception as e:
        return f"ERROR_DOWNLOAD: {str(e)}", None


# =========================
# SMART KEYFRAME EXTRACTION
# =========================
def download_video_for_frames(youtube_url):
    try:
        temp_dir = tempfile.mkdtemp()
        output_template = os.path.join(temp_dir, "video.%(ext)s")
        ydl_opts = {
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]/best",
            "outtmpl": output_template,
            "quiet": True, "no_warnings": True,
            "merge_output_format": "mp4",
        }
        if FFMPEG_PATH:
            ydl_opts["ffmpeg_location"] = FFMPEG_PATH
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(youtube_url, download=True)
            files = glob.glob(os.path.join(temp_dir, "video.*"))
            return files[0] if files else None, temp_dir
    except Exception:
        return None, None


def get_video_duration(video_path):
    ffprobe_exe = shutil.which("ffprobe") or os.path.join(FFMPEG_PATH, f"ffprobe{EXE_SUFFIX}")
    try:
        cmd = [ffprobe_exe, "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0


def file_hash(path):
    """Quick hash of file for deduplication — uses file size + first 4KB."""
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            data = f.read(4096)
        return hashlib.md5(data + str(size).encode()).hexdigest()
    except Exception:
        return None


def remove_duplicate_frames(frame_files, min_size_kb=3):
    """Remove near-duplicate and blank/tiny frames."""
    seen_hashes = set()
    unique = []
    for f in frame_files:
        # Skip frames that are too small (solid color / blank)
        if os.path.getsize(f) < min_size_kb * 1024:
            continue
        h = file_hash(f)
        if h and h not in seen_hashes:
            seen_hashes.add(h)
            unique.append(f)
    return unique


def extract_keyframes_smart(video_path, output_dir):
    """
    Multi-pass keyframe extraction:
    Pass 1: Scene detection (threshold 0.25) — catches major visual changes
    Pass 2: Interval capture (every 10s) — catches slides/static content missed by scene detection
    Then: merge, deduplicate, and return ALL important frames (no hard cap).
    """
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    ffmpeg_exe = shutil.which("ffmpeg") or os.path.join(FFMPEG_PATH, f"ffmpeg{EXE_SUFFIX}")
    duration = get_video_duration(video_path)

    all_frames = []

    # ── Pass 1: Scene change detection (catches transitions, new slides, camera changes) ──
    scene_dir = os.path.join(frames_dir, "scene")
    os.makedirs(scene_dir, exist_ok=True)
    try:
        cmd = [ffmpeg_exe, "-y", "-i", video_path,
               "-vf", "select='gt(scene,0.25)',showinfo",
               "-vsync", "vfr", "-frame_pts", "1", "-q:v", "1",
               os.path.join(scene_dir, "s_%04d.jpg"),
               "-loglevel", "info"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        scene_frames = sorted(glob.glob(os.path.join(scene_dir, "s_*.jpg")))

        # Parse timestamps from ffmpeg showinfo output
        pts_times = []
        for line in result.stderr.split('\n'):
            if 'pts_time:' in line:
                match = re.search(r'pts_time:\s*([\d.]+)', line)
                if match:
                    pts_times.append(float(match.group(1)))

        for i, f in enumerate(scene_frames):
            ts = pts_times[i] if i < len(pts_times) else (duration / max(len(scene_frames), 1)) * i
            all_frames.append({"path": f, "timestamp": ts, "source": "scene"})
    except Exception:
        pass

    # ── Pass 2: Interval capture (every 10s — catches static slides, PPTs, diagrams) ──
    interval_dir = os.path.join(frames_dir, "interval")
    os.makedirs(interval_dir, exist_ok=True)

    # Adjust interval based on video length
    if duration <= 120:
        interval = 8
    elif duration <= 600:
        interval = 10
    elif duration <= 1800:
        interval = 15
    else:
        interval = 20

    try:
        cmd = [ffmpeg_exe, "-y", "-i", video_path,
               "-vf", f"fps=1/{interval}",
               "-q:v", "1",
               os.path.join(interval_dir, "i_%04d.jpg"),
               "-loglevel", "error"]
        subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        interval_frames = sorted(glob.glob(os.path.join(interval_dir, "i_*.jpg")))
        for i, f in enumerate(interval_frames):
            ts = interval * i
            all_frames.append({"path": f, "timestamp": ts, "source": "interval"})
    except Exception:
        pass

    if not all_frames:
        return []

    # ── Merge & Deduplicate ──
    # Sort all frames by timestamp
    all_frames.sort(key=lambda x: x["timestamp"])

    # Remove frames that are too close in time (within 3 seconds of each other, keep scene over interval)
    merged = []
    for frame in all_frames:
        if not merged:
            merged.append(frame)
            continue
        time_gap = abs(frame["timestamp"] - merged[-1]["timestamp"])
        if time_gap < 3:
            # Keep the scene-detected one if there's a conflict
            if frame["source"] == "scene" and merged[-1]["source"] == "interval":
                merged[-1] = frame
            continue
        merged.append(frame)

    # Remove duplicates by image content (blank frames, identical slides)
    paths = [f["path"] for f in merged]
    unique_paths = set(remove_duplicate_frames(paths, min_size_kb=3))
    final = [f for f in merged if f["path"] in unique_paths]

    return final


def image_to_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


# =========================
# TRANSCRIPTION
# =========================
def transcribe_audio(audio_path):
    try:
        if not audio_path or not os.path.exists(audio_path):
            return "ERROR_TRANSCRIBE: Audio file not found"
        model = load_whisper_model()
        return model.transcribe(audio_path, fp16=False, verbose=False)
    except Exception as e:
        return f"ERROR_TRANSCRIBE: {str(e)}"


# =========================
# LIVE STREAM FUNCTIONS
# =========================
def get_live_stream_url(youtube_url):
    try:
        ydl_opts = {"format": "bestaudio/best", "quiet": True, "no_warnings": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            if not info.get("is_live", False):
                return None, info.get("title", ""), "Not a live stream. Use an active live YouTube URL."
            return info.get("url", ""), info.get("title", "Unknown"), None
    except Exception as e:
        return None, "", f"Stream error: {str(e)}"


def record_audio_chunk(stream_url, duration=30):
    try:
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"chunk_{int(time.time())}.mp3")
        ffmpeg_exe = shutil.which("ffmpeg") or os.path.join(FFMPEG_PATH, f"ffmpeg{EXE_SUFFIX}")
        cmd = [ffmpeg_exe, "-y", "-i", stream_url, "-t", str(duration), "-vn",
               "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-q:a", "4",
               "-loglevel", "error", output_path]
        subprocess.run(cmd, capture_output=True, timeout=duration + 45)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path, None
        return None, "Recorded file too small"
    except subprocess.TimeoutExpired:
        return None, "Timed out"
    except Exception as e:
        return None, str(e)


def transcribe_chunk(audio_path, w_model):
    try:
        if not audio_path or not os.path.exists(audio_path):
            return None
        return w_model.transcribe(audio_path, fp16=False, verbose=False)
    except Exception:
        return None


# =========================
# HELPERS
# =========================
def format_time(seconds): return str(datetime.timedelta(seconds=int(seconds)))


def extract_events(segments, keywords):
    events = []
    if not keywords or not keywords.strip():
        keywords = "important, conclusion, summary, step, remember, finally, key, result, therefore, announce, launch, introduce, reveal, update, change, problem, solution, recommend, warning, breaking"
    kw_list = [k.strip().lower() for k in keywords.split(',') if k.strip()]

    # Structural markers that indicate important content
    structural_markers = {
        "Announcement": ["we are announcing", "we're announcing", "introducing", "we've launched", "we are launching",
                         "i'm excited to", "we're excited to", "proud to announce", "pleased to announce",
                         "happy to share", "big news", "breaking news", "just released", "now available"],
        "Key Point": ["the key takeaway", "the main point", "most importantly", "what matters is",
                      "the bottom line", "in summary", "to summarize", "in conclusion", "the takeaway",
                      "here's the thing", "the important thing", "let me emphasize", "pay attention to",
                      "crucial", "critical point", "fundamental"],
        "Topic Shift": ["moving on to", "let's talk about", "next up", "now let's", "switching to",
                        "another topic", "on the topic of", "speaking of", "turning to", "let's move to",
                        "the next thing", "now i want to", "let me now"],
        "Definition": ["what this means is", "in other words", "basically", "essentially",
                       "to put it simply", "what i mean is", "this refers to", "defined as"],
        "Statistic": ["percent", "million", "billion", "thousand", "doubled", "tripled", "increased by",
                      "decreased by", "grew by", "dropped by", "according to", "data shows", "research shows",
                      "studies show", "survey found", "report says"],
        "Warning": ["be careful", "watch out", "don't forget", "keep in mind", "be aware",
                    "common mistake", "pitfall", "avoid", "danger", "risk", "warning"],
        "Recommendation": ["i recommend", "i suggest", "you should", "my advice", "best practice",
                           "pro tip", "the best way", "i'd recommend", "make sure you", "always remember to"],
        "Step/Process": ["step one", "step two", "step three", "first step", "second step", "the first thing",
                         "the second thing", "number one", "number two", "number three", "firstly", "secondly",
                         "thirdly", "the process is", "here's how"],
    }

    seen_texts = set()

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text or text in seen_texts:
            continue
        low = text.lower()
        ts = format_time(seg.get("start", 0))

        # 1. User keyword matches
        matched_kws = set()
        for kw in kw_list:
            if kw in low and kw not in matched_kws:
                matched_kws.add(kw)
                events.append({"time": ts, "type": f"Keyword: '{kw}'", "text": text})
                seen_texts.add(text)

        # 2. Structural marker matches
        if text not in seen_texts:
            for event_type, markers in structural_markers.items():
                if any(m in low for m in markers):
                    events.append({"time": ts, "type": event_type, "text": text})
                    seen_texts.add(text)
                    break

        # 3. Questions (audience or rhetorical)
        if "?" in text and text not in seen_texts:
            events.append({"time": ts, "type": "Question", "text": text})
            seen_texts.add(text)

        # 4. Sentences with numbers/stats (likely data points)
        if text not in seen_texts and re.search(r'\b\d{2,}\b|\b\d+\.\d+\b|\$\d+|\d+%', text):
            events.append({"time": ts, "type": "Data Point", "text": text})
            seen_texts.add(text)

    return events


def get_speech_pace(segments):
    if not segments or len(segments) < 2:
        return None
    data = []
    for seg in segments:
        dur = seg.get("end", 0) - seg.get("start", 0)
        if dur > 0:
            wpm = (len(seg.get("text", "").split()) / dur) * 60
            data.append({"time": seg.get("start", 0), "wpm": min(wpm, 300)})
    return data


def build_transcript_download(transcript_text, segments, video_info=None):
    """Build a formatted transcript string for download."""
    lines = []
    if video_info:
        lines.append(f"Title: {video_info.get('title', 'N/A')}")
        lines.append(f"Channel: {video_info.get('channel', video_info.get('uploader', 'N/A'))}")
        lines.append(f"URL: {video_info.get('webpage_url', 'N/A')}")
        lines.append(f"Duration: {format_time(video_info.get('duration', 0))}")
        lines.append("=" * 60)
        lines.append("")

    lines.append("FULL TRANSCRIPT")
    lines.append("-" * 40)
    lines.append(transcript_text)
    lines.append("")

    if segments:
        lines.append("TIMESTAMPED TRANSCRIPT")
        lines.append("-" * 40)
        for seg in segments:
            ts = format_time(seg.get("start", 0))
            lines.append(f"[{ts}] {seg.get('text', '').strip()}")

    return "\n".join(lines)


STOPWORDS = {"i", "me", "my", "we", "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
             "them", "their", "what", "which", "who", "this", "that", "these", "those", "am", "is", "are", "was",
             "were", "be", "been", "have", "has", "had", "do", "does", "did", "a", "an", "the", "and", "but", "if",
             "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "into", "through",
             "during", "before", "after", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
             "again", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
             "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
             "too", "very", "can", "will", "just", "don", "should", "now", "like", "know", "think", "going", "want",
             "would", "could", "really", "also", "get", "got", "one", "two", "way", "thing", "things", "right",
             "well", "back", "people", "make", "said", "say", "see", "come", "much", "let", "yeah", "okay"}

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Voxel", page_icon="◈", layout="centered", initial_sidebar_state="collapsed")

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

themes = {
    "Dark": {
        "bg": "#08080d", "card": "rgba(16,16,26,0.85)", "border": "rgba(255,255,255,0.06)",
        "text": "#e4e4ec", "subtext": "#7a7a94", "accent": "#7c6aef", "accent2": "#e06a8c",
        "accent_grad": "linear-gradient(135deg, #7c6aef 0%, #e06a8c 100%)",
        "input_bg": "rgba(255,255,255,0.03)", "glow": "rgba(124,106,239,0.12)",
        "shadow": "0 16px 56px -16px rgba(0,0,0,0.95)",
        "positive": "#34d399", "negative": "#f87171", "chart_bg": "rgba(0,0,0,0)",
        "text_surface": "rgba(20,20,36,0.95)", "text_surface_border": "rgba(124,106,239,0.12)",
    },
    "Light": {
        "bg": "#f4f2ee", "card": "rgba(255,255,255,0.92)", "border": "rgba(0,0,0,0.07)",
        "text": "#16162a", "subtext": "#6b6b80", "accent": "#5b4ec4", "accent2": "#c4507a",
        "accent_grad": "linear-gradient(135deg, #5b4ec4 0%, #c4507a 100%)",
        "input_bg": "rgba(91,78,196,0.04)", "glow": "rgba(91,78,196,0.08)",
        "shadow": "0 16px 56px -16px rgba(0,0,0,0.06)",
        "positive": "#059669", "negative": "#dc2626", "chart_bg": "rgba(0,0,0,0)",
        "text_surface": "rgba(255,255,255,0.97)", "text_surface_border": "rgba(91,78,196,0.10)",
    }
}

t = themes[st.session_state.theme]

# =========================
# CSS
# =========================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  :root {{
    --bg:{t["bg"]}; --card:{t["card"]}; --border:{t["border"]};
    --text:{t["text"]}; --subtext:{t["subtext"]}; --accent:{t["accent"]};
    --accent2:{t["accent2"]}; --accent-grad:{t["accent_grad"]};
    --input-bg:{t["input_bg"]}; --glow:{t["glow"]}; --shadow:{t["shadow"]};
    --text-surface:{t["text_surface"]}; --text-surface-border:{t["text_surface_border"]};
  }}

  .stApp {{ background:var(--bg); font-family:'Outfit',sans-serif; color:var(--text); }}
  .block-container {{ padding-top:0.5rem !important; padding-bottom:3rem !important; max-width:800px !important; }}
  h1,h2,h3,h4,h5,h6,p,span,div {{ color:var(--text); }}

  /* Ambient glow */
  .stApp::before {{ content:''; position:fixed; top:-25%; left:-15%; width:55%; height:55%;
    background:radial-gradient(ellipse,var(--glow) 0%,transparent 50%);
    filter:blur(120px); opacity:0.7; z-index:-1; pointer-events:none;
    animation:drift 25s ease-in-out infinite; }}
  .stApp::after {{ content:''; position:fixed; bottom:-25%; right:-15%; width:55%; height:55%;
    background:radial-gradient(ellipse,rgba(224,106,140,0.06) 0%,transparent 50%);
    filter:blur(120px); z-index:-1; pointer-events:none;
    animation:drift 30s ease-in-out infinite reverse; }}
  @keyframes drift {{
    0%,100%{{transform:translate(0,0) scale(1)}}
    33%{{transform:translate(25px,-18px) scale(1.03)}}
    66%{{transform:translate(-18px,12px) scale(0.97)}}
  }}

  /* Hero */
  .hero {{ text-align:center; padding:2.2rem 1rem 1.6rem; }}
  .hero-icon {{ font-size:2.2rem; background:var(--accent-grad); -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    filter:drop-shadow(0 0 28px var(--glow)); }}
  .hero h1 {{ font-size:2.8rem; font-weight:800; letter-spacing:-0.05em; margin:0.2rem 0 0;
    background:var(--accent-grad); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .hero p {{ font-size:0.88rem; color:var(--subtext); margin-top:0.3rem; font-weight:400; letter-spacing:0.03em; }}

  /* Cards */
  .card {{ background:var(--card); backdrop-filter:blur(24px) saturate(1.4); border:1px solid var(--border);
    border-radius:16px; padding:1.6rem; margin-bottom:1rem; box-shadow:var(--shadow);
    transition:transform 0.25s cubic-bezier(0.4,0,0.2,1); }}
  .card:hover {{ transform:translateY(-1px); }}
  .card-label {{ font-size:0.6rem; font-weight:700; letter-spacing:0.22em; text-transform:uppercase;
    color:var(--accent); margin-bottom:1rem; display:flex; align-items:center; gap:7px; }}
  .card-label::before {{ content:''; width:5px; height:5px; border-radius:1.5px; background:var(--accent-grad); display:block; }}

  /* Inputs */
  .stTextInput>div>div>input {{ background:var(--input-bg) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; color:var(--text) !important; font-family:'Outfit',sans-serif !important;
    font-size:0.9rem !important; padding:0.85rem 1rem !important; }}
  .stTextInput>div>div>input:focus {{ border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--glow) !important; }}
  .stTextInput label {{ display:none !important; }}
  .stTextInput>div>div>input::placeholder {{ color:var(--subtext) !important; opacity:0.6 !important; }}

  /* Buttons */
  .stButton>button {{ width:100%; background:var(--accent-grad) !important; color:#fff !important;
    border:none !important; border-radius:10px !important; padding:0.8rem 1.3rem !important;
    font-family:'Outfit',sans-serif !important; font-size:0.88rem !important; font-weight:600 !important;
    transition:all 0.2s cubic-bezier(0.4,0,0.2,1) !important; margin-top:0.2rem;
    letter-spacing:0.01em !important; }}
  .stButton>button:hover {{ transform:translateY(-2px) !important; box-shadow:0 8px 28px var(--glow) !important; }}

  /* Tabs */
  button[data-baseweb="tab"] {{ background:transparent !important; border:none !important; color:var(--subtext) !important;
    font-family:'Outfit',sans-serif !important; font-weight:500 !important; font-size:0.82rem !important;
    padding:0.4rem 0.8rem 0.75rem !important; margin-right:0.15rem; border-bottom:2px solid transparent !important;
    transition:color 0.2s ease !important; }}
  button[data-baseweb="tab"]:hover {{ color:var(--text) !important; }}
  button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--accent) !important; border-bottom:2px solid var(--accent) !important; font-weight:600 !important; }}
  div[data-baseweb="tab-list"] {{ border-bottom:1px solid var(--border); margin-bottom:1.2rem; gap:0 !important; }}
  div[data-baseweb="tab-panel"] {{ background:transparent !important; }}

  /* TextArea — transcript readability fix */
  .stTextArea textarea {{ background:var(--text-surface) !important; border:1px solid var(--text-surface-border) !important;
    border-radius:12px !important; color:var(--text) !important; font-family:'JetBrains Mono',monospace !important;
    font-size:0.8rem !important; line-height:1.85 !important; padding:1.3rem 1.4rem !important;
    box-shadow:inset 0 2px 8px rgba(0,0,0,0.04) !important; }}
  .stTextArea label {{ display:none !important; }}

  /* Download button */
  .stDownloadButton>button {{ background:transparent !important; color:var(--accent) !important;
    border:1px solid var(--text-surface-border) !important; font-weight:600 !important;
    border-radius:10px !important; backdrop-filter:blur(8px) !important;
    transition:all 0.2s ease !important; }}
  .stDownloadButton>button:hover {{ background:var(--accent) !important; color:#fff !important;
    border-color:var(--accent) !important; transform:translateY(-1px) !important;
    box-shadow:0 4px 16px var(--glow) !important; }}

  /* Stats */
  .stat-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:0.7rem; margin-top:0.4rem; }}
  .stat-pill {{ background:var(--input-bg); border:1px solid var(--border); border-radius:12px;
    padding:0.9rem 0.6rem; text-align:center; transition:all 0.2s ease; }}
  .stat-pill:hover {{ border-color:var(--accent); transform:translateY(-2px); box-shadow:0 4px 16px var(--glow); }}
  .stat-pill .val {{ font-size:1.2rem; font-weight:700; font-family:'JetBrains Mono',monospace;
    background:var(--accent-grad); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .stat-pill .lbl {{ font-size:0.58rem; text-transform:uppercase; letter-spacing:0.12em; color:var(--subtext); margin-top:0.25rem; font-weight:600; }}

  /* Scroll containers — timeline & events readability */
  .custom-scroll {{ height:420px; overflow-y:auto; padding-right:10px; font-size:0.86rem; line-height:1.7; color:var(--text); }}
  .custom-scroll::-webkit-scrollbar {{ width:4px; }}
  .custom-scroll::-webkit-scrollbar-track {{ background:transparent; }}
  .custom-scroll::-webkit-scrollbar-thumb {{ background:var(--accent); border-radius:6px; opacity:0.5; }}

  /* Timeline segment rows */
  .tl-row {{ margin-bottom:0; padding:12px 14px; border-bottom:1px solid var(--border);
    transition:background 0.15s ease; }}
  .tl-row:hover {{ background:var(--input-bg); }}
  .tl-row:last-child {{ border-bottom:none; }}
  .tl-ts {{ color:var(--accent); font-family:'JetBrains Mono',monospace; font-size:0.73rem;
    margin-right:10px; font-weight:500; white-space:nowrap; }}
  .tl-text {{ color:var(--text); font-size:0.86rem; line-height:1.65; }}

  /* Event boxes */
  .event-box {{ margin-bottom:8px; padding:13px 16px; background:var(--text-surface); border-radius:10px;
    border:1px solid var(--border); color:var(--text); transition:all 0.15s ease; }}
  .event-box:hover {{ transform:translateX(3px); border-color:var(--text-surface-border); }}
  .event-box p {{ color:var(--text) !important; margin:5px 0 0 !important; font-size:0.86rem !important; line-height:1.6 !important; }}

  /* Frames */
  .frame-card {{ border-radius:10px; overflow:hidden; border:1px solid var(--border); background:var(--input-bg); transition:all 0.2s; margin-bottom:8px; }}
  .frame-card:hover {{ border-color:var(--accent); transform:scale(1.01); }}
  .frame-card img {{ width:100%; display:block; }}
  .frame-card .fm {{ padding:7px 9px; font-size:0.68rem; color:var(--subtext); font-family:'JetBrains Mono',monospace;
    display:flex; justify-content:space-between; align-items:center; }}
  .frame-card .fm .badge {{ background:var(--accent); color:#fff; padding:1px 6px; border-radius:4px; font-size:0.6rem; font-weight:600; }}

  /* Summary box — KEY readability fix */
  .summary-box {{ background:var(--text-surface); border:1px solid var(--text-surface-border); border-radius:14px;
    padding:1.6rem 1.8rem; line-height:1.95; font-size:0.9rem; color:var(--text);
    border-left:3px solid var(--accent); }}
  .summary-box strong {{ color:var(--accent); font-weight:700; }}
  .summary-box br + br {{ display:block; content:''; margin-top:0.3rem; }}

  /* Live */
  @keyframes livePulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.25}} }}
  .live-dot {{ width:7px; height:7px; border-radius:50%; background:#ef4444; display:inline-block; margin-right:7px; animation:livePulse 1s ease-in-out infinite; }}
  .live-badge {{ display:inline-flex; align-items:center; background:rgba(239,68,68,0.07); border:1px solid rgba(239,68,68,0.2);
    border-radius:14px; padding:4px 11px; font-size:0.68rem; font-weight:700; color:#ef4444; letter-spacing:0.08em; }}

  .stSlider label {{ display:none !important; }}
  .video-meta {{ display:flex; gap:1.2rem; flex-wrap:wrap; margin-top:0.5rem; font-size:0.8rem; color:var(--subtext); }}
  .video-meta span {{ display:inline-flex; align-items:center; gap:4px; }}

  /* Checkbox styling */
  .stCheckbox label span {{ color:var(--text) !important; font-size:0.82rem !important; }}

  /* Selectbox / Slider */
  .stSelectbox label {{ color:var(--subtext) !important; }}
  .stSelectbox>div>div {{ background:var(--input-bg) !important; border-color:var(--border) !important; color:var(--text) !important; }}

  /* Alert / Info boxes */
  div[data-testid="stAlert"] {{ background:var(--text-surface) !important; border:1px solid var(--text-surface-border) !important;
    border-radius:10px !important; }}
  div[data-testid="stAlert"] p {{ color:var(--text) !important; }}

  /* Streamlit element backgrounds forced transparent */
  .stTabs, div[data-testid="stVerticalBlockBorderWrapper"] {{ background:transparent !important; }}
  .stMarkdown p, .stMarkdown li, .stMarkdown span {{ color:var(--text) !important; }}

  /* Spinner text */
  .stSpinner>div>span {{ color:var(--subtext) !important; }}

  /* Image captions */
  .stImage>div>div>p {{ color:var(--subtext) !important; font-size:0.72rem !important; font-family:'JetBrains Mono',monospace !important; }}

  /* Toggle */
  .stToggle label span {{ color:var(--subtext) !important; }}

  /* Success/Error boxes */
  .stSuccess, .stError, .stWarning, .stInfo {{
    background:var(--text-surface) !important; border-radius:10px !important; }}

  #MainMenu, footer, header {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# =========================
# THEME TOGGLE
# =========================
col1, col2 = st.columns([8.5, 1.5])
with col2:
    toggle_val = st.toggle("☀/🌙", value=(st.session_state.theme == "Dark"))
    new_theme = "Dark" if toggle_val else "Light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()

# HERO
st.markdown("""<div class="hero">
  <div class="hero-icon">◈</div>
  <h1>Voxel</h1>
  <p>Intelligence extraction from any YouTube content</p>
</div>""", unsafe_allow_html=True)

# MODE SELECTOR
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Video Analysis"

col_m1, col_m2 = st.columns(2)
with col_m1:
    if st.button("🎬  Video Analysis", key="mode_video", use_container_width=True):
        st.session_state.app_mode = "Video Analysis"; st.rerun()
with col_m2:
    if st.button("🔴  Live Stream", key="mode_live", use_container_width=True):
        st.session_state.app_mode = "Live Stream"; st.rerun()

st.markdown(f"""<div style="text-align:center; margin:-0.2rem 0 1rem;">
  <span style="font-size:0.65rem; color:var(--accent); font-weight:700; letter-spacing:0.18em; text-transform:uppercase;">● {st.session_state.app_mode}</span>
</div>""", unsafe_allow_html=True)

app_mode = st.session_state.app_mode

# ╔══════════════════════════════════════════════════════════╗
# ║              VIDEO ANALYSIS                              ║
# ╚══════════════════════════════════════════════════════════╝
if app_mode == "Video Analysis":

    # Initialize video results in session state
    if "vid_results" not in st.session_state:
        st.session_state.vid_results = None

    st.markdown('<div class="card"><div class="card-label">Data Source</div>', unsafe_allow_html=True)
    url = st.text_input("URL", placeholder="https://www.youtube.com/watch?v=...", key="video_url")
    st.markdown(f"<p style='color:{t['subtext']}; font-size:0.76rem; margin:8px 0 3px 3px;'>Event Keywords (comma separated)</p>", unsafe_allow_html=True)
    keywords_input = st.text_input("KW", placeholder="e.g. AI, framework, architecture, pricing", key="video_keywords")
    col_btn, col_frames = st.columns([3, 2])
    with col_btn:
        analyze = st.button("Analyze Video")
    with col_frames:
        extract_frames = st.checkbox("Extract Visual Frames", value=True, help="Smart keyframe capture via scene detection")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Run analysis and STORE results in session state ──
    if analyze:
        if not url: st.error("Enter a valid YouTube URL"); st.stop()

        st.markdown('<div class="card"><div class="card-label">Processing Pipeline</div>', unsafe_allow_html=True)

        with st.spinner("Downloading audio..."):
            audio_result = download_audio(url)
            if isinstance(audio_result[0], str) and audio_result[0].startswith("ERROR"):
                st.error(audio_result[0]); st.markdown('</div>', unsafe_allow_html=True); st.stop()
            audio_file, video_info = audio_result

        with st.spinner("Running Whisper transcription..."):
            transcript_data = transcribe_audio(audio_file)
        if isinstance(transcript_data, str) and "ERROR" in transcript_data:
            st.error(transcript_data); st.markdown('</div>', unsafe_allow_html=True); st.stop()

        keyframes_data = []
        if extract_frames:
            with st.spinner("Smart keyframe extraction (scene detection + interval scan)..."):
                video_path, temp_dir = download_video_for_frames(url)
                if video_path:
                    raw_keyframes = extract_keyframes_smart(video_path, temp_dir)
                    # Convert images to base64 NOW so they survive temp dir cleanup
                    for kf in raw_keyframes:
                        b64 = image_to_base64(kf["path"])
                        if b64:
                            keyframes_data.append({
                                "b64": b64,
                                "timestamp": kf.get("timestamp", 0),
                                "source": kf.get("source", "")
                            })

        # Generate summary once and cache it
        transcript_text = transcript_data["text"]
        summary_text = ""
        word_count = len(transcript_text.split())
        if word_count >= 40:
            with st.spinner("Generating detailed summary..."):
                try:
                    tokenizer, model = load_summarizer()
                    summary_text = generate_detailed_summary(transcript_text, tokenizer, model)
                except Exception:
                    summary_text = ""

        # Save everything to session state
        st.session_state.vid_results = {
            "transcript_text": transcript_text,
            "segments": transcript_data.get("segments", []),
            "video_info": video_info,
            "keyframes": keyframes_data,
            "summary": summary_text,
        }

        st.success(f"Done — {len(keyframes_data)} keyframes captured" if keyframes_data else "Processing complete")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── DISPLAY results from session state (persists across reruns) ──
    if st.session_state.vid_results:
        r = st.session_state.vid_results
        transcript_text = r["transcript_text"]
        segments = r["segments"]
        video_info = r["video_info"]
        keyframes = r["keyframes"]
        summary = r["summary"]

        # Video metadata
        if video_info:
            v_title = video_info.get("title", "Unknown")
            v_channel = video_info.get("channel", video_info.get("uploader", "Unknown"))
            v_duration = video_info.get("duration", 0)
            v_views = video_info.get("view_count", 0)
            st.markdown(f"""<div class="card" style="padding:1.2rem 1.5rem;">
              <p style="margin:0; font-weight:700; font-size:1.05rem; line-height:1.4;">{v_title}</p>
              <div class="video-meta">
                <span>📺 {v_channel}</span>
                <span>⏱ {format_time(v_duration) if v_duration else 'N/A'}</span>
                <span>👁 {v_views:,}</span>
                <span>🖼 {len(keyframes)} frames</span>
              </div></div>""", unsafe_allow_html=True)

        word_count = len(transcript_text.split())
        char_count = len(transcript_text)
        est_minutes = max(round(word_count / 150, 1), 0.1)
        unique_words = len(set(re.findall(r'\b[a-z]{3,}\b', transcript_text.lower())) - STOPWORDS)

        st.markdown(f"""<div class="card"><div class="card-label">Telemetry</div>
          <div class="stat-row">
            <div class="stat-pill"><div class="val">{word_count:,}</div><div class="lbl">Words</div></div>
            <div class="stat-pill"><div class="val">{char_count:,}</div><div class="lbl">Chars</div></div>
            <div class="stat-pill"><div class="val">{est_minutes}m</div><div class="lbl">Read Time</div></div>
            <div class="stat-pill"><div class="val">{unique_words}</div><div class="lbl">Unique</div></div>
          </div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Intelligence Report</div>', unsafe_allow_html=True)
        tab_names = ["Transcript", "Timeline", "AI Summary", "Key Events", "Analytics"]
        if keyframes:
            tab_names.append(f"Frames ({len(keyframes)})")
        tabs = st.tabs(tab_names)

        # TAB: Transcript
        with tabs[0]:
            st.text_area("T", transcript_text, height=400)
            dl_text = build_transcript_download(transcript_text, segments, video_info)
            st.download_button("Download Transcript (.txt)", dl_text, file_name="voxel_transcript.txt", mime="text/plain")

        # TAB: Timeline
        with tabs[1]:
            if not segments: st.info("No timeline data.")
            else:
                h = "<div class='custom-scroll'>"
                for seg in segments:
                    h += f"<div class='tl-row'><span class='tl-ts'>[{format_time(seg.get('start',0))}]</span><span class='tl-text'>{seg.get('text','')}</span></div>"
                h += "</div>"; st.markdown(h, unsafe_allow_html=True)

        # TAB: AI Summary (pre-generated, just display)
        with tabs[2]:
            if not summary:
                if word_count < 40: st.info("Transcript too short."); st.write(transcript_text)
                else: st.info("Summary generation failed. Try analyzing again.")
            else:
                st.markdown(f"<div class='summary-box'>{summary.replace(chr(10), '<br>')}</div>", unsafe_allow_html=True)
                st.download_button("Download Summary (.txt)", summary, file_name="voxel_summary.txt", mime="text/plain")

        # TAB: Key Events
        with tabs[3]:
            events = extract_events(segments, keywords_input)
            if not events: st.info("No events detected.")
            else:
                # Color map for different event types
                event_colors = {
                    "Question": t["negative"],
                    "Announcement": "#f59e0b",
                    "Key Point": t["positive"],
                    "Topic Shift": t["accent"],
                    "Definition": "#06b6d4",
                    "Statistic": "#8b5cf6",
                    "Data Point": "#8b5cf6",
                    "Warning": t["negative"],
                    "Recommendation": t["positive"],
                    "Step/Process": t["accent2"],
                }

                # Count by type
                from collections import Counter as _C
                type_counts = _C(ev["type"].split(":")[0].strip() for ev in events)
                type_summary = " · ".join(f"{k}: {v}" for k, v in type_counts.most_common())

                st.markdown(f"<p style='color:{t['subtext']}; font-size:0.78rem; margin-bottom:12px;'>{len(events)} events detected — {type_summary}</p>", unsafe_allow_html=True)
                h = "<div class='custom-scroll'>"
                for ev in events:
                    etype = ev["type"].split(":")[0].strip() if ":" in ev["type"] else ev["type"]
                    c = event_colors.get(etype, t["accent"])
                    h += f"<div class='event-box' style='border-left:3px solid {c};'><span style='color:{c}; font-family:\"JetBrains Mono\",monospace; font-size:0.72rem; text-transform:uppercase; font-weight:600;'>{ev['time']} · {ev['type']}</span><p>{ev['text']}</p></div>"
                h += "</div>"; st.markdown(h, unsafe_allow_html=True)

        # TAB: Analytics
        with tabs[4]:
            if not transcript_text.strip() or not segments: st.info("Not enough data.")
            else:
                cfg = {'displayModeBar': False}
                sents = {"Positive": 0, "Neutral": 0, "Negative": 0}
                sot = []
                for seg in segments:
                    p = TextBlob(seg["text"]).sentiment.polarity
                    sents["Positive" if p > 0.1 else ("Negative" if p < -0.1 else "Neutral")] += 1
                    sot.append({"time": seg.get("start", 0), "polarity": p})

                df = pd.DataFrame(list(sents.items()), columns=["Sentiment", "Count"])
                fig = px.pie(df, values="Count", names="Sentiment", title="Tone Distribution", color="Sentiment",
                             color_discrete_map={"Positive": t["positive"], "Neutral": t["accent"], "Negative": t["negative"]}, hole=0.65)
                fig.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color=t["card"], width=2)))
                fig.update_layout(plot_bgcolor=t["chart_bg"], paper_bgcolor=t["chart_bg"], font_color=t["text"],
                                  font_family="Outfit", title_font_size=15, margin=dict(t=45, b=15, l=0, r=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config=cfg)

                st.markdown(f"<div style='height:1px; background:{t['border']}; margin:1.3rem 0;'></div>", unsafe_allow_html=True)

                if sot:
                    df_s = pd.DataFrame(sot)
                    fig_l = px.line(df_s, x="time", y="polarity", title="Sentiment Flow", color_discrete_sequence=[t["accent"]])
                    fig_l.add_hline(y=0, line_dash="dash", line_color=t["subtext"], opacity=0.3)
                    fig_l.update_layout(plot_bgcolor=t["chart_bg"], paper_bgcolor=t["chart_bg"], font_color=t["text"],
                                        font_family="Outfit", title_font_size=15, margin=dict(t=45, b=35, l=35, r=15),
                                        xaxis_title="Time (s)", yaxis_title="Polarity",
                                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=t["border"]))
                    st.plotly_chart(fig_l, use_container_width=True, config=cfg)

                st.markdown(f"<div style='height:1px; background:{t['border']}; margin:1.3rem 0;'></div>", unsafe_allow_html=True)

                wds = re.findall(r'\b[a-z]{3,}\b', transcript_text.lower())
                top = Counter([w for w in wds if w not in STOPWORDS]).most_common(15)
                if top:
                    df_w = pd.DataFrame(top, columns=["Word", "Freq"]).sort_values("Freq", ascending=True)
                    fig_b = px.bar(df_w, x="Freq", y="Word", orientation='h', title="Top Keywords", color_discrete_sequence=[t["accent"]])
                    fig_b.update_layout(plot_bgcolor=t["chart_bg"], paper_bgcolor=t["chart_bg"], font_color=t["text"],
                                        font_family="Outfit", title_font_size=15, margin=dict(t=45, b=15, l=0, r=0),
                                        xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False))
                    st.plotly_chart(fig_b, use_container_width=True, config=cfg)

                st.markdown(f"<div style='height:1px; background:{t['border']}; margin:1.3rem 0;'></div>", unsafe_allow_html=True)

                pace = get_speech_pace(segments)
                if pace and len(pace) > 3:
                    df_p = pd.DataFrame(pace)
                    fig_p = px.area(df_p, x="time", y="wpm", title="Speaking Pace", color_discrete_sequence=[t["accent2"]])
                    fig_p.update_layout(plot_bgcolor=t["chart_bg"], paper_bgcolor=t["chart_bg"], font_color=t["text"],
                                        font_family="Outfit", title_font_size=15, margin=dict(t=45, b=35, l=35, r=15),
                                        xaxis_title="Time (s)", yaxis_title="WPM",
                                        xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor=t["border"]))
                    fig_p.update_traces(fill='tozeroy', fillcolor="rgba(224,106,140,0.08)")
                    st.plotly_chart(fig_p, use_container_width=True, config=cfg)

        # TAB: Frames (using stored base64 data — survives reruns)
        if keyframes:
            with tabs[5]:
                scene_count = sum(1 for f in keyframes if f.get("source") == "scene")
                interval_count = sum(1 for f in keyframes if f.get("source") == "interval")
                st.markdown(f"<p style='color:{t['subtext']}; font-size:0.78rem; margin-bottom:10px;'>{len(keyframes)} unique frames — {scene_count} from scene detection, {interval_count} from interval scan. Hover over any frame and click the ↔ icon to view full size.</p>", unsafe_allow_html=True)

                # ── Select / Download All controls ──
                col_sel, col_dl_sel, col_dl_all = st.columns([2, 2, 2])

                # Initialize selection state
                if "selected_frames" not in st.session_state:
                    st.session_state.selected_frames = set()

                with col_sel:
                    if st.button("Select All", key="frames_select_all"):
                        st.session_state.selected_frames = set(range(len(keyframes)))
                        st.rerun()
                with col_dl_sel:
                    sel_count = len(st.session_state.selected_frames)
                    if sel_count > 0:
                        zip_buf = io.BytesIO()
                        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                            for idx in sorted(st.session_state.selected_frames):
                                kf = keyframes[idx]
                                ts_str = format_time(kf.get("timestamp", 0)).replace(":", "-")
                                fname = f"frame_{idx+1}_{ts_str}_{kf.get('source','')}.jpg"
                                zf.writestr(fname, base64.b64decode(kf["b64"]))
                        st.download_button(
                            f"Download Selected ({sel_count})",
                            zip_buf.getvalue(),
                            file_name="voxel_selected_frames.zip",
                            mime="application/zip",
                            key="dl_selected_frames"
                        )
                    else:
                        st.markdown(f"<p style='color:{t['subtext']}; font-size:0.75rem; padding-top:0.5rem;'>Tick frames below to select</p>", unsafe_allow_html=True)
                with col_dl_all:
                    zip_buf_all = io.BytesIO()
                    with zipfile.ZipFile(zip_buf_all, "w", zipfile.ZIP_DEFLATED) as zf:
                        for idx, kf in enumerate(keyframes):
                            ts_str = format_time(kf.get("timestamp", 0)).replace(":", "-")
                            fname = f"frame_{idx+1}_{ts_str}_{kf.get('source','')}.jpg"
                            zf.writestr(fname, base64.b64decode(kf["b64"]))
                    st.download_button(
                        f"Download All ({len(keyframes)})",
                        zip_buf_all.getvalue(),
                        file_name="voxel_all_frames.zip",
                        mime="application/zip",
                        key="dl_all_frames"
                    )

                st.markdown(f"<div style='height:1px; background:{t['border']}; margin:0.6rem 0 1rem;'></div>", unsafe_allow_html=True)

                # ── Frame grid with checkboxes and individual download ──
                for row_start in range(0, len(keyframes), 3):
                    cols = st.columns(3)
                    for j in range(3):
                        idx = row_start + j
                        if idx < len(keyframes):
                            kf = keyframes[idx]
                            ts = format_time(kf.get("timestamp", 0))
                            src_type = kf.get("source", "")
                            badge = "SCENE" if src_type == "scene" else "INTERVAL"
                            with cols[j]:
                                img_bytes = base64.b64decode(kf['b64'])
                                st.image(img_bytes, caption=f"#{idx+1} · {ts} · {badge}", use_container_width=True)

                                cb_col, dl_col = st.columns(2)
                                with cb_col:
                                    checked = st.checkbox("Select", value=(idx in st.session_state.selected_frames), key=f"sel_frame_{idx}", label_visibility="visible")
                                    if checked:
                                        st.session_state.selected_frames.add(idx)
                                    else:
                                        st.session_state.selected_frames.discard(idx)
                                with dl_col:
                                    ts_safe = ts.replace(":", "-")
                                    st.download_button(
                                        "⬇",
                                        img_bytes,
                                        file_name=f"frame_{idx+1}_{ts_safe}_{src_type}.jpg",
                                        mime="image/jpeg",
                                        key=f"dl_frame_{idx}"
                                    )

        st.markdown('</div>', unsafe_allow_html=True)

# ╔══════════════════════════════════════════════════════════╗
# ║              LIVE STREAM                                 ║
# ╚══════════════════════════════════════════════════════════╝
elif app_mode == "Live Stream":

    for key, default in [("live_running", False), ("live_transcript", ""), ("live_segments", []),
                         ("live_chunks", 0), ("live_elapsed", 0.0), ("live_stream_url", ""),
                         ("live_stream_title", ""), ("live_errors", 0), ("live_youtube_url", "")]:
        if key not in st.session_state: st.session_state[key] = default

    st.markdown('<div class="card"><div class="card-label">Live Stream Source</div>', unsafe_allow_html=True)
    live_url = st.text_input("Live URL", placeholder="https://www.youtube.com/watch?v=... (must be LIVE)", key="live_url")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"<p style='color:{t['subtext']}; font-size:0.76rem; margin:3px 0 3px 3px;'>Chunk Duration</p>", unsafe_allow_html=True)
        chunk_duration = st.select_slider("CD", options=[15, 20, 25, 30, 45, 60], value=30, label_visibility="collapsed")
    with col_b:
        st.markdown(f"<p style='color:{t['subtext']}; font-size:0.76rem; margin:3px 0 3px 3px;'>Keywords</p>", unsafe_allow_html=True)
        live_keywords = st.text_input("LKW", placeholder="e.g. breaking, alert", key="live_keywords", label_visibility="collapsed")

    col_start, col_stop, col_reset = st.columns(3)
    if not st.session_state.live_running:
        with col_start:
            if st.button("Start Capture", key="btn_start"):
                if not live_url: st.error("Enter a live YouTube URL.")
                else:
                    st.session_state.update(live_running=True, live_transcript="", live_segments=[],
                                            live_chunks=0, live_elapsed=0.0, live_stream_url="",
                                            live_stream_title="", live_errors=0, live_youtube_url=live_url)
                    st.rerun()
    else:
        with col_stop:
            if st.button("Stop Capture", key="btn_stop"): st.session_state.live_running = False; st.rerun()
    with col_reset:
        if st.session_state.live_chunks > 0 and not st.session_state.live_running:
            if st.button("Clear All", key="btn_reset"):
                st.session_state.update(live_transcript="", live_segments=[], live_chunks=0, live_elapsed=0.0,
                                        live_stream_url="", live_stream_title="", live_errors=0, live_youtube_url="")
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.live_running:
        st.markdown(f"""<div style="text-align:center; margin:0.8rem 0;">
          <span class="live-badge"><span class="live-dot"></span>CAPTURING — Chunk #{st.session_state.live_chunks + 1}</span>
        </div>""", unsafe_allow_html=True)

        yt_url = st.session_state.live_youtube_url
        with st.spinner("Connecting..."):
            stream_url, title, error = get_live_stream_url(yt_url)
            if error:
                st.session_state.live_errors += 1
                if st.session_state.live_errors >= 5: st.error(error); st.session_state.live_running = False; st.stop()
                st.warning(f"Retry ({st.session_state.live_errors}/5)..."); time.sleep(3); st.rerun()
            st.session_state.live_stream_title = title

        if st.session_state.live_stream_title:
            st.markdown(f"""<div class="card" style="padding:0.9rem 1.3rem;">
              <p style="margin:0; font-size:0.75rem; color:{t['subtext']};">Connected:</p>
              <p style="margin:2px 0 0; font-weight:600;">{st.session_state.live_stream_title}</p>
            </div>""", unsafe_allow_html=True)

        with st.spinner(f"Recording {chunk_duration}s..."):
            audio_path, rec_error = record_audio_chunk(stream_url, duration=chunk_duration)
        if rec_error or not audio_path:
            st.session_state.live_errors += 1
            if st.session_state.live_errors >= 5: st.error("Stream may have ended."); st.session_state.live_running = False; st.stop()
            st.warning(f"Retry ({st.session_state.live_errors}/5)..."); time.sleep(2); st.rerun()

        with st.spinner("Transcribing..."):
            w_model = load_whisper_model(); chunk_result = transcribe_chunk(audio_path, w_model)

        try:
            if audio_path and os.path.exists(audio_path): os.remove(audio_path); os.rmdir(os.path.dirname(audio_path))
        except Exception: pass

        if chunk_result and chunk_result.get("text", "").strip():
            txt = chunk_result["text"].strip()
            offset = st.session_state.live_elapsed
            adj = [dict(seg, start=seg.get("start", 0) + offset, end=seg.get("end", 0) + offset) for seg in chunk_result.get("segments", [])]
            st.session_state.live_transcript += (" " + txt if st.session_state.live_transcript else txt)
            st.session_state.live_segments.extend(adj)
            st.session_state.live_chunks += 1; st.session_state.live_elapsed += chunk_duration; st.session_state.live_errors = 0
        else:
            st.session_state.live_chunks += 1; st.session_state.live_elapsed += chunk_duration

    if st.session_state.live_transcript:
        transcript_text = st.session_state.live_transcript
        segments = st.session_state.live_segments
        wc = len(transcript_text.split())

        st.markdown(f"""<div class="card"><div class="card-label">Live Telemetry</div>
          <div class="stat-row">
            <div class="stat-pill"><div class="val">{st.session_state.live_chunks}</div><div class="lbl">Chunks</div></div>
            <div class="stat-pill"><div class="val">{wc:,}</div><div class="lbl">Words</div></div>
            <div class="stat-pill"><div class="val">{format_time(st.session_state.live_elapsed)}</div><div class="lbl">Duration</div></div>
            <div class="stat-pill"><div class="val">{len(segments)}</div><div class="lbl">Segments</div></div>
          </div></div>""", unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-label">Live Intelligence</div>', unsafe_allow_html=True)
        lt1, lt2, lt3, lt4, lt5 = st.tabs(["Transcript", "Timeline", "Summary", "Events", "Analytics"])

        with lt1:
            st.text_area("L", transcript_text, height=400, key="lta")
            st.download_button("Download Live Transcript", transcript_text, file_name="voxel_live_transcript.txt", mime="text/plain")
        with lt2:
            if segments:
                h = "<div class='custom-scroll'>"
                for seg in segments:
                    h += f"<div class='tl-row'><span class='tl-ts'>[{format_time(seg.get('start',0))}]</span><span class='tl-text'>{seg.get('text','')}</span></div>"
                h += "</div>"; st.markdown(h, unsafe_allow_html=True)
        with lt3:
            if wc < 40: st.info("Need more text...")
            else:
                try:
                    tok, mdl = load_summarizer()
                    s = generate_detailed_summary(transcript_text, tok, mdl)
                    st.markdown(f"<div class='summary-box'>{s.replace(chr(10),'<br>')}</div>", unsafe_allow_html=True)
                except Exception as e: st.error(str(e))
        with lt4:
            evts = extract_events(segments, live_keywords)
            if not evts: st.info("No events yet.")
            else:
                h = "<div class='custom-scroll'>"
                for ev in evts:
                    c = t["negative"] if "Question" in ev["type"] else t["positive"]
                    h += f"<div class='event-box' style='border-left:3px solid {c};'><span style='color:{c}; font-family:\"JetBrains Mono\",monospace; font-size:0.72rem; text-transform:uppercase; font-weight:600;'>{ev['time']} · {ev['type']}</span><p>{ev['text']}</p></div>"
                h += "</div>"; st.markdown(h, unsafe_allow_html=True)
        with lt5:
            if segments:
                cfg = {'displayModeBar': False}
                ss = {"Positive": 0, "Neutral": 0, "Negative": 0}
                for seg in segments:
                    p = TextBlob(seg["text"]).sentiment.polarity
                    ss["Positive" if p > 0.1 else ("Negative" if p < -0.1 else "Neutral")] += 1
                df = pd.DataFrame(list(ss.items()), columns=["Sentiment", "Count"])
                fig = px.pie(df, values="Count", names="Sentiment", title="Live Tone", color="Sentiment",
                             color_discrete_map={"Positive": t["positive"], "Neutral": t["accent"], "Negative": t["negative"]}, hole=0.65)
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(plot_bgcolor=t["chart_bg"], paper_bgcolor=t["chart_bg"], font_color=t["text"],
                                  font_family="Outfit", margin=dict(t=45, b=15, l=0, r=0), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config=cfg)

                wl = re.findall(r'\b[a-z]{3,}\b', transcript_text.lower())
                tw = Counter([w for w in wl if w not in STOPWORDS]).most_common(15)
                if tw:
                    dw = pd.DataFrame(tw, columns=["Word", "Freq"]).sort_values("Freq", ascending=True)
                    fb = px.bar(dw, x="Freq", y="Word", orientation='h', title="Keywords", color_discrete_sequence=[t["accent"]])
                    fb.update_layout(plot_bgcolor=t["chart_bg"], paper_bgcolor=t["chart_bg"], font_color=t["text"],
                                     font_family="Outfit", margin=dict(t=45, b=15, l=0, r=0),
                                     xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False))
                    st.plotly_chart(fb, use_container_width=True, config=cfg)
        st.markdown('</div>', unsafe_allow_html=True)

    elif not st.session_state.live_running:
        st.markdown(f"""<div class="card" style="text-align:center; padding:2.2rem;">
          <p style="font-size:1.05rem; color:{t['subtext']}; margin:0;">Paste a live YouTube URL and hit <strong>Start Capture</strong></p>
          <p style="font-size:0.78rem; color:{t['subtext']}; margin-top:0.3rem; opacity:0.5;">Captured in {chunk_duration}s chunks · Zero paid APIs</p>
        </div>""", unsafe_allow_html=True)

    if st.session_state.live_running: time.sleep(1); st.rerun()

# FOOTER
st.markdown(f"""<div style="text-align:center; padding:2.2rem 0 1rem; border-top:1px solid var(--border); margin-top:1rem;">
  <span style="font-size:0.63rem; color:var(--subtext); letter-spacing:0.08em; font-weight:500;">
    Whisper · yt-dlp · Transformers · Streamlit — Zero paid APIs
  </span>
</div>""", unsafe_allow_html=True)