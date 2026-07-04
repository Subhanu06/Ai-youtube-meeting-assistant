import os
import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
import traceback

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Pipeline Definition ─────────────────────────────────────────────────────────
STEPS = [
    ("audio",      "🔊", "Audio Processing",  "Extracting & chunking audio"),
    ("transcript", "📝", "Transcription",      "Converting speech to text"),
    ("title",      "🏷️", "Title Generation",   "Naming this session"),
    ("summary",    "📋", "Summarisation",      "Distilling key points"),
    ("extract",    "🔍", "Extraction",         "Pulling items & decisions"),
    ("rag",        "🧠", "RAG Engine",         "Indexing for Q&A"),
]

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background: var(--bg) !important; }

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; color: var(--text) !important; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.card:hover { border-color: var(--accent); transform: translateY(-2px); }
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content { font-size: 0.875rem; line-height: 1.7; color: var(--text); }

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }
.badge-red    { background: rgba(239,68,68,0.15);  color: var(--danger);     border: 1px solid rgba(239,68,68,0.3); }

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important; }
.stButton > button:disabled { opacity: 0.5 !important; transform: none !important; box-shadow: none !important; }
.stButton > button[kind="secondary"] { background: var(--surface-2) !important; border: 1px solid var(--border) !important; }

/* ── Pipeline Progress (sidebar, live) ── */
.pipeline-wrap { margin-top: 0.25rem; }
.pipeline-progress-track {
    width: 100%;
    height: 6px;
    background: var(--surface-2);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.9rem;
    border: 1px solid var(--border);
}
.pipeline-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    border-radius: 4px;
    transition: width 0.4s ease;
}
.pipeline-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0.85rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.35rem 0;
    border: 1px solid var(--border);
    font-size: 0.78rem;
    transition: border-color 0.3s, background 0.3s;
}
.status-bar.is-active { border-color: rgba(159,103,255,0.5); background: rgba(124,58,237,0.08); }
.status-bar.is-done   { border-color: rgba(16,185,129,0.3); }
.status-bar.is-error  { border-color: rgba(239,68,68,0.5); background: rgba(239,68,68,0.08); }

.status-label { flex: 1; display: flex; flex-direction: column; }
.status-label .name { color: var(--text); }
.status-label .desc { color: var(--text-muted); font-size: 0.65rem; margin-top: 0.1rem; }
.status-time { font-size: 0.65rem; color: var(--text-muted); white-space: nowrap; }

.status-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.3s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }
.dot-error    { background: var(--danger); box-shadow: 0 0 8px var(--danger); }

@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }

/* Compact top-bar mirror of pipeline status for mobile / collapsed sidebar */
.top-status-strip {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    padding: 0.75rem 1rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 1.2rem;
}
.top-chip {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.7rem;
    border-radius: 20px;
    font-size: 0.68rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text-muted);
}
.top-chip.is-active { color: var(--accent-glow); border-color: rgba(159,103,255,0.5); }
.top-chip.is-done { color: var(--success); border-color: rgba(16,185,129,0.3); }

.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 420px;
    overflow-y: auto;
    margin-bottom: 1rem;
}
.chat-msg { margin-bottom: 1rem; display: flex; flex-direction: column; gap: 0.2rem; }
.chat-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.15em; text-transform: uppercase; }
.chat-bubble { display: inline-block; padding: 0.6rem 1rem; border-radius: 10px; font-size: 0.85rem; line-height: 1.6; max-width: 90%; }
.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }
.user-bubble { background: rgba(124,58,237,0.15); border: 1px solid rgba(124,58,237,0.25); align-self: flex-end; }
.bot-bubble  { background: rgba(6,182,212,0.1);  border: 1px solid rgba(6,182,212,0.2);   align-self: flex-start; }

hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.8rem !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
    "pipeline_error_step": None,
    "step_durations": {},
    "pipeline_t0": None,
    "pipeline_total_time": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Rendering Helpers ────────────────────────────────────────────────────────────
def _clean_html(html: str) -> str:
    """Strip leading whitespace on every line so Markdown doesn't treat
    indented HTML as a code block (4+ leading spaces = code block)."""
    return "\n".join(line.lstrip() for line in html.split("\n"))


def _fmt_secs(s):
    if s is None:
        return ""
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m {int(s % 60)}s"


def pipeline_fraction(steps_state):
    if not steps_state:
        return 0.0
    done = sum(1 for k, *_ in STEPS if steps_state.get(k) == "done")
    active = sum(1 for k, *_ in STEPS if steps_state.get(k) == "active")
    return (done + 0.5 * active) / len(STEPS)


def render_pipeline_html(steps_state, durations, t0, has_started):
    """Builds the full live pipeline status block (progress bar + step rows)."""
    if not has_started:
        rows = "".join(
            f"""<div class="status-bar">
                    <div class="status-dot dot-pending"></div>
                    <div class="status-label"><span class="name">{icon} {label}</span>
                        <span class="desc">{desc}</span></div>
                </div>""" for _, icon, label, desc in STEPS
        )
        return _clean_html(f"""<div class="pipeline-wrap">
            <div class="pipeline-meta"><span>Waiting to start</span><span>0%</span></div>
            <div class="pipeline-progress-track"><div class="pipeline-progress-fill" style="width:0%"></div></div>
            {rows}
        </div>""")

    frac = pipeline_fraction(steps_state)
    pct = int(round(frac * 100))
    elapsed = time.time() - t0 if t0 and not st.session_state.pipeline_total_time else st.session_state.pipeline_total_time
    elapsed_str = _fmt_secs(elapsed) if elapsed is not None else ""
    status_word = "Complete" if pct == 100 else ("Failed" if st.session_state.pipeline_error_step else "Processing…")

    rows_html = []
    for key, icon, label, desc in STEPS:
        state = steps_state.get(key, "pending")
        css_class = {"active": "is-active", "done": "is-done", "error": "is-error"}.get(state, "")
        dot_class = {"active": "dot-active", "done": "dot-done", "error": "dot-error"}.get(state, "dot-pending")
        dur = durations.get(key)
        time_str = _fmt_secs(dur) if dur is not None else ("running…" if state == "active" else "")
        display_label = f"{icon} {label}" if state != "done" else f"✅ {label}"
        rows_html.append(
            f'<div class="status-bar {css_class}">'
            f'<div class="status-dot {dot_class}"></div>'
            f'<div class="status-label"><span class="name">{display_label}</span>'
            f'<span class="desc">{desc}</span></div>'
            f'<div class="status-time">{time_str}</div>'
            f'</div>'
        )

    return _clean_html(f"""<div class="pipeline-wrap">
        <div class="pipeline-meta"><span>{status_word}</span><span>{pct}% · {elapsed_str}</span></div>
        <div class="pipeline-progress-track"><div class="pipeline-progress-fill" style="width:{pct}%"></div></div>
        {''.join(rows_html)}
    </div>""")


def render_top_strip_html(steps_state, has_started):
    if not has_started:
        return ""
    chips = []
    for key, icon, label, _ in STEPS:
        state = steps_state.get(key, "pending")
        cls = {"active": "is-active", "done": "is-done"}.get(state, "")
        mark = "✅" if state == "done" else ("⏳" if state == "active" else icon)
        chips.append(f'<div class="top-chip {cls}">{mark} {label}</div>')
    return f'<div class="top-status-strip">{"".join(chips)}</div>'


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI<br>Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input</span>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a video/audio file",
        type=["mp4", "mov", "mkv", "avi", "mp3", "wav", "m4a"],
    )
    source_url = st.text_input(
        "…or paste a YouTube URL / file path",
        placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4",
    )
    language = st.selectbox("Language", ["english", "hinglish"], index=0)
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    st.markdown("---")
    st.markdown('<span class="badge badge-cyan">Pipeline Status</span>', unsafe_allow_html=True)
    sidebar_status_slot = st.empty()

# Show whatever state we have right now (pending / mid-run / done) on every render
has_started = bool(st.session_state.pipeline_steps)
sidebar_status_slot.markdown(
    render_pipeline_html(
        st.session_state.pipeline_steps,
        st.session_state.step_durations,
        st.session_state.pipeline_t0,
        has_started,
    ),
    unsafe_allow_html=True,
)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)

top_strip_slot = st.empty()
top_strip_slot.markdown(render_top_strip_html(st.session_state.pipeline_steps, has_started), unsafe_allow_html=True)

st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    source = None

    if uploaded_file is not None:
        downloads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        saved_path = os.path.join(downloads_dir, uploaded_file.name)
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source = saved_path
    elif source_url.strip():
        source = source_url.strip()

    if not source:
        st.error("Please upload a file or paste a YouTube URL / file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}
        st.session_state.step_durations = {}
        st.session_state.pipeline_error_step = None
        st.session_state.pipeline_total_time = None
        st.session_state.pipeline_t0 = time.time()

        step_start_times = {}

        def refresh_status():
            html = render_pipeline_html(
                st.session_state.pipeline_steps,
                st.session_state.step_durations,
                st.session_state.pipeline_t0,
                True,
            )
            sidebar_status_slot.markdown(html, unsafe_allow_html=True)
            top_strip_slot.markdown(render_top_strip_html(st.session_state.pipeline_steps, True), unsafe_allow_html=True)

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            if state == "active":
                step_start_times[key] = time.time()
            elif state == "done" and key in step_start_times:
                st.session_state.step_durations[key] = time.time() - step_start_times[key]
            elif state == "error":
                st.session_state.pipeline_error_step = key
            refresh_status()

        try:
            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.pipeline_total_time = time.time() - st.session_state.pipeline_t0
            refresh_status()

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            st.toast("✅ Analysis complete!")
            st.rerun()

        except Exception as e:
            failed_key = next((k for k, _, _, _ in STEPS if st.session_state.pipeline_steps.get(k) == "active"), None)
            if failed_key:
                update_step(failed_key, "error")
            st.session_state.pipeline_total_time = time.time() - st.session_state.pipeline_t0
            st.error(f"❌ Pipeline failed during **{failed_key or 'processing'}**: {e}")
            st.code(traceback.format_exc())
# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)
        st.download_button(
            "⬇ Download Transcript",
            data=r["transcript"],
            file_name="transcript.txt",
            use_container_width=True,
        )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    report_md = (
        f"# {r['title']}\n\n"
        f"## Summary\n{r['summary']}\n\n"
        f"## Action Items\n{r['action_items']}\n\n"
        f"## Key Decisions\n{r['key_decisions']}\n\n"
        f"## Open Questions\n{r['open_questions']}\n"
    )
    st.download_button("⬇ Download Full Report (.md)", data=report_md, file_name=f"{r['title']}.md", use_container_width=False)

    st.markdown("---")

    # ── RAG Chat ──────────────────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask anything about your meeting transcript</div>
        </div>""", unsafe_allow_html=True)

    user_input = st.chat_input("Ask about this meeting… e.g. What were the main decisions?")
    if user_input:
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:380px;line-height:1.7">
            Upload a video/audio file or paste a YouTube URL in the sidebar, choose your language, and hit <strong>Analyse</strong> to get started.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Transcription</span>
            <span class="badge badge-cyan">Summarisation</span>
            <span class="badge badge-green">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)

  # <-- add this line temporarily