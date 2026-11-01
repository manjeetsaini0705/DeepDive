"""
Streamlit UI for the Multi-Agent Research System.
Place this file next to pipeline.py and run:  streamlit run app_v7.py
"""
import json
import os
import uuid
from html import escape
from datetime import datetime

import streamlit as st

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain

HISTORY_FILE = "research_history.json"

st.set_page_config(page_title="DeepDive · Research Agents", page_icon="◆", layout="wide", initial_sidebar_state="auto")

# ───────────────────────── STYLES ─────────────────────────
st.markdown(
    """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Work+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
/* Research desk design tokens */
:root {
  --bg:#080A0D; --panel:#14191D; --panel2:#14191D;
  --line:#222A30; --text:#E4E9ED; --muted:#8A949E; --faint:#5C6873;
  --acc:#42E4BF; --acc2:#69B7F4; --ink:#080A0D;
  --grid:rgba(105,183,244,.05); --focus:rgba(66,228,191,.22);
  --accent-line:rgba(66,228,191,.30); --wash:rgba(66,228,191,.06);
  --panel-wash:rgba(20,25,29,.80); --input-wash:rgba(8,10,13,.80);
  --shadow:0 12px 36px rgba(0,0,0,.18);
  --font-heading:'Instrument Serif',serif; --font-body:'Work Sans',sans-serif;
}
html,body,.stApp,input,textarea,button,[data-testid="stMarkdownContainer"] {
  font-family:var(--font-body);color:var(--text);letter-spacing:0;
}
.stApp {background:var(--bg);}
[data-testid="stMain"] {
  background-color:var(--bg);
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:30px 30px;
}
#MainMenu,footer {visibility:hidden;}
header[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:900px;padding:6rem 3.5rem 4rem;}
h1,h2,h3,h4 {font-family:var(--font-heading)!important;font-weight:400!important;letter-spacing:0!important;}
section[data-testid="stSidebar"] {background:var(--panel);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {padding-top:2.5rem;}
.brand {display:flex;align-items:center;gap:12px;margin-bottom:12px;}
.brand-mark {width:14px;height:14px;flex:0 0 14px;border-radius:50%;background:var(--acc);font-size:0;}
.brand-name {font-family:var(--font-heading);font-size:46px;line-height:1.2;}
.brand-name span {color:var(--text);}
.brand-caption {font-size:11px;color:var(--muted);margin-bottom:28px;}
.sidebar-heading {display:flex;justify-content:space-between;margin:34px 0 16px;font-size:12px;color:var(--muted);}
.history-count {font-family:var(--font-body);font-size:12px;color:var(--acc);}
.hist-empty {font-size:14px;color:var(--muted);}
section[data-testid="stSidebar"] .stButton>button {
  width:100%;background:transparent;color:var(--text);border:1px solid var(--line);
  border-radius:8px;min-height:44px;padding:10px 14px;transition:border-color .2s,background .2s;
}
section[data-testid="stSidebar"] .stButton>button p {font-size:13px;text-align:left;white-space:pre-line;overflow-wrap:anywhere;}
section[data-testid="stSidebar"] .stButton>button:hover {border-color:var(--accent-line);background:var(--wash);}
section[data-testid="stSidebar"] .st-key-new button {background:var(--acc);border-color:var(--acc);color:var(--ink);}
section[data-testid="stSidebar"] .st-key-new button p {color:var(--ink);font-weight:600;}
.hero {padding:0;margin-bottom:36px;}
.hero .tag {font-size:11px;color:var(--acc);margin-bottom:22px;}
.hero h1 {font-size:56px;line-height:1.12;margin:0 0 20px;padding:0!important;text-wrap:balance;}
.hero h1 .headline-line {display:block;line-height:1.12;}
.hero h1 em {font-style:italic;color:var(--acc);}
.hero p {font-size:17px;line-height:1.7;color:var(--muted);margin:0;max-width:52ch;}
/* One balanced native search row; the action stays aligned with the input. */
[data-testid="stForm"] {
  padding:16px!important;border:1px solid var(--line)!important;border-radius:8px!important;
  background:var(--panel-wash);box-shadow:var(--shadow);
  transition:border-color .25s,box-shadow .25s;
}
[data-testid="stForm"]:focus-within {border-color:var(--accent-line)!important;box-shadow:0 0 0 1px var(--accent-line),var(--shadow);}
[data-testid="stForm"] [data-testid="stVerticalBlock"] {gap:14px;}
[data-testid="stForm"] [data-testid="stTextInput"] label p {font-size:12px!important;color:var(--muted)!important;margin-bottom:4px;}
[data-testid="stForm"] [data-baseweb="input"],
[data-testid="stForm"] [data-testid="stTextInputRootElement"] {
  min-height:60px!important;background:var(--input-wash)!important;
  border:1px solid transparent!important;border-radius:6px!important;
}
[data-testid="stForm"] [data-baseweb="base-input"] {background:transparent!important;}
[data-testid="stForm"] input {
  min-height:58px!important;padding:0 18px!important;font-family:var(--font-body)!important;
  font-size:16px!important;background:transparent!important;color:var(--text)!important;
  caret-color:var(--acc);border:0!important;box-shadow:none!important;
}
[data-testid="stForm"] input::placeholder {color:var(--muted)!important;opacity:1!important;}
[data-testid="stForm"] [data-baseweb="input"]:focus-within,
[data-testid="stForm"] [data-testid="stTextInputRootElement"]:focus-within {border-color:var(--acc)!important;box-shadow:0 0 0 2px var(--focus)!important;}
[data-testid="stForm"] [data-testid="stHorizontalBlock"] {align-items:center;gap:12px;}
.composer-label {font-size:12px;font-weight:500;color:var(--text);margin:0 2px 2px;}
.composer-status {font-size:12px;color:var(--muted);line-height:1.6;}
.composer-status::before {content:"";display:inline-block;width:5px;height:5px;background:var(--acc);border-radius:50%;margin-right:8px;vertical-align:middle;}
[data-testid="stFormSubmitButton"]>button {
  width:100%!important;height:60px!important;min-height:60px!important;padding:10px 18px!important;
  background:var(--acc)!important;color:var(--ink)!important;border:1px solid var(--acc)!important;
  border-radius:6px!important;transition:transform .2s,filter .2s,box-shadow .2s;
}
[data-testid="stFormSubmitButton"]>button p {font-family:var(--font-body)!important;font-size:14px!important;font-weight:600!important;color:var(--ink)!important;white-space:nowrap;}
[data-testid="stFormSubmitButton"]>button:hover {transform:translateY(-2px);filter:brightness(1.06);box-shadow:0 5px 18px var(--focus);}
[data-testid="stFormSubmitButton"]>button:active {transform:translateY(0);}
[data-testid="stFormSubmitButton"]>button:focus-visible {outline:2px solid var(--acc2);outline-offset:3px;}
.pipeline-label {font-size:11px;color:var(--muted);margin:36px 0 12px;}
.steps {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:0 0 30px;}
.step {position:relative;overflow:hidden;min-height:96px;border:1px solid var(--line);border-radius:8px;padding:16px 12px;background:var(--panel-wash);transition:border-color .3s,background .3s;}
.step .n {font-size:10px;color:var(--muted);}
.step .t {font-size:13px;font-weight:500;margin-top:8px;overflow-wrap:anywhere;}
.step .s {font-size:11px;color:var(--muted);margin-top:7px;min-height:18px;}
.step.active {border-color:var(--accent-line);background:var(--panel);animation:stage-enter .35s ease-out;}
.step.active::before {content:"";position:absolute;top:0;left:0;width:35%;height:2px;background:var(--acc);animation:agent-scan 2s ease-in-out infinite;}
.step.active .t,.step.active .s {color:var(--acc);}
.step.done {border-color:var(--accent-line);}
.step.done .s {color:var(--acc);}
.dot {display:inline-block;width:5px;height:5px;border-radius:50%;margin-right:6px;background:var(--muted);}
.active .dot {background:var(--acc);animation:pulse 1.2s infinite;}
.done .dot {background:var(--acc);}
.stTabs [role="tablist"],.stTabs [data-baseweb="tab-list"] {gap:24px;background:transparent;border-bottom:1px solid var(--line);}
.stTabs [role="tab"],.stTabs [data-baseweb="tab"] {padding:10px 2px;color:var(--muted);font-weight:500;transition:color .2s;}
.stTabs [aria-selected="true"],.stTabs [aria-selected="true"] p {color:var(--acc)!important;}
.stTabs .react-aria-SelectionIndicator {background:var(--acc)!important;}
.stTabs [data-baseweb="tab-highlight"] {background:var(--acc);transition:left .3s,width .3s;}
.stTabs [data-baseweb="tab-border"] {display:none;}
.stTabs [role="tabpanel"],.stTabs [data-baseweb="tab-panel"] {padding-top:22px;animation:tab-reveal .3s ease-out;}
.meta {color:var(--muted);font-size:11px;}
.result-title h2 {font-size:32px;line-height:1.3;overflow-wrap:anywhere;}
.empty-state {border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px;}
.empty-state b {color:var(--text);font-weight:400;}
.stDownloadButton>button {background:var(--panel);color:var(--acc);border:1px solid var(--accent-line);border-radius:6px;}
.stDownloadButton>button:hover {background:var(--wash);color:var(--acc);border-color:var(--acc);}
.st-key-critic_review {border-left:2px solid var(--accent-line);padding:4px 0 4px 24px;margin-top:8px;}
.review-header {padding-bottom:20px;border-bottom:1px solid var(--line);}
.review-eyebrow {font-size:11px;color:var(--acc);margin-bottom:9px;}
.review-header h3 {font-size:30px!important;margin:0 0 8px!important;}
.review-header p {font-size:14px;color:var(--muted);line-height:1.6;}
.st-key-critic_review [data-testid="stMarkdownContainer"] p,
.st-key-critic_review [data-testid="stMarkdownContainer"] li {font-size:15px;line-height:1.85;overflow-wrap:anywhere;}
.st-key-critic_review [data-testid="stMarkdownContainer"] strong {color:var(--acc);font-weight:500;}
.st-key-critic_review [data-testid="stMarkdownContainer"] li {margin-bottom:10px;}
.st-key-critic_review [data-testid="stMarkdownContainer"] blockquote {background:var(--wash);border-left:2px solid var(--acc);padding:12px 18px;margin:18px 0;}
/* Report layout is unframed; hierarchy and typography do the work. */
.result-heading {padding:24px 0 26px;border-top:1px solid var(--line);}
.result-eyebrow {font-size:11px;color:var(--acc);margin-bottom:14px;}
.result-heading h2 {font-size:40px;line-height:1.18;margin:0 0 18px;overflow-wrap:anywhere;text-wrap:balance;}
.result-meta {display:flex;flex-wrap:wrap;gap:12px 24px;font-size:12px;color:var(--muted);}
.st-key-research_report {max-width:100%;padding:6px 0 24px;}
.st-key-research_report [data-testid="stMarkdownContainer"] {line-height:1.85;}
.st-key-research_report [data-testid="stMarkdownContainer"] p,
.st-key-research_report [data-testid="stMarkdownContainer"] li {font-size:16px;line-height:1.85;color:var(--text);overflow-wrap:anywhere;}
.st-key-research_report [data-testid="stMarkdownContainer"] h1 {font-size:34px;line-height:1.25;margin:10px 0 24px;}
.st-key-research_report [data-testid="stMarkdownContainer"] h2 {font-size:29px;line-height:1.3;margin:32px 0 16px;padding-top:18px;border-top:1px solid var(--line);}
.st-key-research_report [data-testid="stMarkdownContainer"] h3 {font-family:var(--font-body)!important;font-size:18px;font-weight:500!important;line-height:1.5;margin:24px 0 12px;}
.st-key-research_report [data-testid="stMarkdownContainer"] strong {color:var(--text);font-weight:600;}
.st-key-research_report [data-testid="stMarkdownContainer"] li {margin-bottom:10px;}
.st-key-research_report [data-testid="stMarkdownContainer"] a {color:var(--acc2);text-underline-offset:4px;}
.st-key-research_report [data-testid="stMarkdownContainer"] blockquote {border-left:2px solid var(--acc);padding:6px 20px;margin:24px 0;background:var(--wash);}
.st-key-research_report [data-testid="stMarkdownContainer"] blockquote p {color:var(--muted);}
.st-key-research_report [data-testid="stMarkdownContainer"] table {width:100%;font-size:14px;border-collapse:collapse;}
.st-key-research_report [data-testid="stMarkdownContainer"] th {background:var(--panel);color:var(--acc);font-weight:500;}
.st-key-research_report [data-testid="stMarkdownContainer"] th,
.st-key-research_report [data-testid="stMarkdownContainer"] td {padding:12px 14px;border-bottom:1px solid var(--line);}
.st-key-research_report [data-testid="stMarkdownContainer"] pre {background:var(--panel);border:1px solid var(--line);border-radius:6px;overflow-x:auto;}
@media(max-width:760px) {.result-heading h2 {font-size:32px;}.st-key-research_report [data-testid="stMarkdownContainer"] p,.st-key-research_report [data-testid="stMarkdownContainer"] li {font-size:15px;}}
@keyframes agent-scan {from {transform:translateX(-110%);}to {transform:translateX(390%);}}
@keyframes pulse {0%,100% {opacity:1;}50% {opacity:.35;}}
@keyframes stage-enter {from {opacity:.5;transform:translateY(3px);}to {opacity:1;transform:translateY(0);}}
@keyframes tab-reveal {from {opacity:0;transform:translateY(5px);}to {opacity:1;transform:translateY(0);}}
@media(max-width:760px) {
  .block-container {padding:3.5rem 1rem 2rem;}
  .hero h1 {font-size:38px;}
  .hero p {font-size:15px;}
  [data-testid="stForm"] {padding:14px!important;}
  [data-testid="stForm"] [data-testid="stHorizontalBlock"] {flex-direction:column;align-items:stretch;}
  [data-testid="stForm"] [data-testid="stColumn"] {width:100%!important;flex:1 1 100%!important;min-width:0!important;}
  .steps {grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;}
  .st-key-critic_review {padding-left:16px;}
}
@media(prefers-reduced-motion:reduce) {* {animation:none!important;transition:none!important;}}
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────── HISTORY ─────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def to_text(x):
    return x.content if hasattr(x, "content") else str(x)


if "history" not in st.session_state:
    st.session_state.history = load_history()
if "current" not in st.session_state:
    st.session_state.current = None

STEPS = [
    ("01", "Search Agent", "Finding sources"),
    ("02", "Reader Agent", "Scraping content"),
    ("03", "Writer", "Drafting report"),
    ("04", "Critic", "Reviewing draft"),
]


def render_steps(slot, active=-1, done=0):
    html = '<div class="steps">'
    for i, (n, t, s) in enumerate(STEPS):
        cls = "done" if i < done else ("active" if i == active else "")
        label = "Complete" if i < done else (s + "…" if i == active else "Waiting")
        html += f'<div class="step {cls}"><div class="n">STEP {n}</div><div class="t">{t}</div><div class="s"><span class="dot"></span>{label}</div></div>'
    slot.markdown(html + "</div>", unsafe_allow_html=True)


# ───────────────────────── PIPELINE (with live progress) ─────────────────────────
def run_pipeline_ui(topic, slot):
    state = {}

    render_steps(slot, active=0)
    search_result = build_search_agent().invoke(
        {"messages": [("user", f"find recent, reliable and detailed information about:{topic}")]}
    )
    msgs = search_result["messages"]
    tool_msgs = [m for m in msgs if m.__class__.__name__ == "ToolMessage"]
    state["search_results"] = (
        "\n\n".join(str(m.content) for m in tool_msgs) if tool_msgs else msgs[-1].content
    )

    render_steps(slot, active=1, done=1)
    reader_result = build_reader_agent().invoke(
        {
            "messages": [
                (
                    "user",
                    f"""You are given search results for the topic:
{topic}

Search Results:
{state["search_results"]}

Identify the most relevant URL from the search results.
Then use the scrape_url tool with that URL.
Do NOT simply summarize the search results.
Actually call scrape_url on one of the URLs.
Return the scraped content.""",
                )
            ]
        }
    )
    state["scraped_content"] = reader_result["messages"][-1].content

    render_steps(slot, active=2, done=2)
    research = (
        f"SEARCH RESULTS: \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT: \n {state['scraped_content']}"
    )
    state["report"] = to_text(writer_chain.invoke({"topic": topic, "research": research}))

    render_steps(slot, active=3, done=3)
    state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))

    render_steps(slot, done=4)
    return {k: to_text(v) for k, v in state.items()}


# ───────────────────────── SIDEBAR / HISTORY ─────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">◆</div>'
        '<div class="brand-name">Deep<span>Dive</span></div></div>'
        '<div class="brand-caption">MULTI-AGENT RESEARCH</div>',
        unsafe_allow_html=True,
    )
    if st.button("＋  New research", key="new", use_container_width=True):
        st.session_state.current = None
        st.rerun()

    st.markdown(
        '<div class="sidebar-heading">History'
        f'<span class="history-count">{len(st.session_state.history):02d}</span></div>',
        unsafe_allow_html=True,
    )
    items = list(reversed(st.session_state.history))
    if not items:
        st.markdown('<div class="hist-empty">No research yet.</div>', unsafe_allow_html=True)
    for h in items:
        label = h["topic"][:38] + ("…" if len(h["topic"]) > 38 else "")
        if st.button(f"{label}\n{h['time']}", key=f"h_{h['id']}", use_container_width=True):
            st.session_state.current = h
            st.rerun()

    if st.session_state.history:
        st.write("")
        if st.button("Clear history", key="clear", use_container_width=True):
            st.session_state.history = []
            st.session_state.current = None
            save_history([])
            st.rerun()

# ───────────────────────── MAIN ─────────────────────────
st.markdown(
    """
<div class="hero">
  <div class="tag">01 · RESEARCH DESK</div>
  <h1><span class="headline-line">Explore a topic.</span><span class="headline-line">Discover <em>what matters.</em></span></h1>
  <p>From reliable sources to a thoughtfully reviewed report.</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.form("research_form", clear_on_submit=False):
    st.markdown('<div class="composer-label">Research topic</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1.5], gap="small", vertical_alignment="center")
    with c1:
        topic = st.text_input(
            "Research topic",
            placeholder="What would you like to explore?",
            label_visibility="collapsed",
        )
    with c2:
        go = st.form_submit_button("Research →", use_container_width=True)

st.markdown('<div class="pipeline-label">PIPELINE · FOUR STAGES</div>', unsafe_allow_html=True)
progress_slot = st.empty()

if go and topic.strip():
    try:
        with st.spinner("Agents are working…"):
            result = run_pipeline_ui(topic.strip(), progress_slot)
        entry = {
            "id": uuid.uuid4().hex[:8],
            "topic": topic.strip(),
            "time": datetime.now().strftime("%d %b %Y · %H:%M"),
            **result,
        }
        st.session_state.history.append(entry)
        save_history(st.session_state.history)
        st.session_state.current = entry
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
elif go:
    st.warning("Please enter a research topic.")

cur = st.session_state.current
if cur:
    if not go:
        render_steps(progress_slot, done=4)
    report_words = len(cur["report"].split())
    st.markdown(
        '<div class="result-heading">'
        '<div class="result-eyebrow">RESEARCH REPORT</div>'
        f'<h2>{escape(cur["topic"])}</h2>'
        '<div class="result-meta">'
        f'<span>{escape(cur["time"])}</span>'
        f'<span>{report_words:,} words</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    t1, t2 = st.tabs(["Report", "Critic review"])
    with t1:
        with st.container(key="research_report"):
            # Native Markdown preserves lists, tables, citations, and code.
            st.markdown(cur["report"])
        st.download_button(
            "Download report (.md)",
            data=f"# {cur['topic']}\n\n{cur['report']}\n\n---\n\n## Critic Feedback\n\n{cur['feedback']}",
            file_name=f"report_{cur['id']}.md",
            mime="text/markdown",
        )
    with t2:
        with st.container(key="critic_review"):
            st.markdown(
                '<div class="review-header">'
                '<div class="review-eyebrow">CRITIC REVIEW</div>'
                '<h3>A closer look at your research</h3>'
                '<p>An independent review of the report’s strengths, gaps, '
                'and opportunities for improvement.</p></div>',
                unsafe_allow_html=True,
            )
            feedback = cur.get("feedback", "")
            if feedback and feedback.strip():
                # Preserve the agent's original Markdown and wording.
                st.markdown(feedback)
            else:
                st.info("No critic feedback was returned for this report.")
elif not go:
    render_steps(progress_slot)
    st.markdown(
        '<div class="empty-state"><b>Your next discovery starts here.</b> '
        '<em>What are you curious about?</em></div>', 
        unsafe_allow_html=True,
    )