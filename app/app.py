import os
import sys
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from fpdf import FPDF

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

SERVER_URL = "https://cyberguard-api.onrender.com"

def apply_ui() -> None:
    style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');

    /* 1. Remove Link Icons/Anchors next to headers */
    .viewerBadge_container__1QSob, .stHeaderActionElements, a.tw-7ebm6u { display: none !important; }
    button[kind="header"] { display: none !important; }
    .st-emotion-cache-15zrgzn e1nzilvr4 { display: none !important; }

    /* 3. Remove gap above logo/header and fill browser */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        max-width: 100% !important;
    }

    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}

    /* UPDATED: Greener and deeper smooth animated background */
    .stApp {
        background-color: #0b0f1a;
        background-image: 
            radial-gradient(circle at center, rgba(32, 201, 151, 0.15) 0%, transparent 75%),
            linear-gradient(90deg, transparent 0%, rgba(32, 201, 151, 0.12) 50%, transparent 100%),
            linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
            linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
        background-size: 100% 100%, 200% 100%, 50px 50px, 50px 50px;
        background-repeat: no-repeat, no-repeat, repeat, repeat;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
        animation: smoothGlow 14s ease-in-out infinite;
    }

    @keyframes smoothGlow {
        0% { background-position: 0% 0%, -100% 0%, 0% 0%, 0% 0%; }
        50% { background-position: 0% 0%, 100% 0%, 0% 0%, 0% 0%; }
        100% { background-position: 0% 0%, -100% 0%, 0% 0%, 0% 0%; }
    }

    /* Brand Header */
    .brand-header {
        position: relative;
        padding-left: 2rem;
        margin-top: 1rem;
        margin-bottom: 2rem;
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .brand-header span { color: #20C997; }

    /* Hero Section */
    .hero-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding-top: 1rem;
    }

    .hero-badge {
        background: rgba(32, 201, 151, 0.1);
        color: #20C997;
        padding: 8px 24px;
        border-radius: 100px;
        font-size: 0.95rem;
        font-weight: 600;
        border: 1px solid rgba(32, 201, 151, 0.3);
        margin-bottom: 2rem;
        display: inline-block;
    }

    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -2px;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    .hero-title span {
        background: linear-gradient(90deg, #37f5c7, #20C997, #17a589);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        max-width: 700px;
        line-height: 1.6;
        margin-bottom: 3rem;
    }

    /* Risk badges */
    .risk-critical { 
        background: rgba(239, 68, 68, 0.2); 
        color: #f87171; 
        padding: 8px 16px; 
        border-radius: 100px;
        font-weight: 600;
    }
    .risk-high { 
        background: rgba(249, 115, 22, 0.2); 
        color: #fb923c; 
        padding: 8px 16px; 
        border-radius: 100px;
        font-weight: 600;
    }
    .risk-low { 
        background: rgba(32, 201, 151, 0.2); 
        color: #20C997; 
        padding: 8px 16px; 
        border-radius: 100px;
        font-weight: 600;
    }

    /* Code highlighting */
    .code-line { font-family: monospace; font-size: 12px; padding: 2px 8px; margin: 2px 0; }
    .red-bg { background: rgba(239, 68, 68, 0.15); border-left: 3px solid #f87171; }
    .green-bg { background: rgba(32, 201, 151, 0.15); border-left: 3px solid #20C997; }

    /* Tabs & Buttons */
    .stTabs [data-baseweb="tab-list"] { justify-content: center; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        font-weight: 600;
        padding: 10px 25px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #20C997 !important;
        background-color: rgba(32, 201, 151, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(32, 201, 151, 0.1) !important;
        color: #20C997 !important;
        border-bottom: 2px solid #20C997 !important;
    }

    /* Inputs & Buttons */
    .stTextArea textarea, .stTextInput input {
        background-color: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f8fafc !important;
    }
    
    div.stButton > button {
        background: #20C997 !important;
        color: #0b0f1a !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        height: 50px !important;
        width: 100% !important;
        border: none !important;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    div.stButton > button:hover {
        box-shadow: 0 0 25px rgba(32, 201, 151, 0.6) !important;
        transform: translateY(-2px);
        background: #26e0a9 !important;
    }
    </style>

    <div class="brand-header">Cyber<span>Guard</span> 🛡️</div>

    <div class="hero-wrapper">
        <div class="hero-badge">Zero-Day Predictor</div>
        <h1 class="hero-title">Securing Code with<br><span>AI-Powered</span> Risk Analysis</h1>
        <p class="hero-subtitle">
            Democratizing security analysis to help developers worldwide build safer applications by identifying vulnerabilities early.
        </p>
    </div>
    """
    st.markdown(style, unsafe_allow_html=True)

def render_risk_chip(risk: str) -> None:
    risk_upper = (risk or "LOW").upper()
    if "CRITICAL" in risk_upper:
        cls = "risk-critical"
    elif "HIGH" in risk_upper:
        cls = "risk-high"
    else:
        cls = "risk-low"
    st.markdown(f"<span class='risk-chip {cls}'>{risk_upper}</span>", unsafe_allow_html=True)

def call_scan_endpoint(mode: str, text: str | None = None, repo_url: str | None = None, uploaded_file=None):
    if mode == "manual":
        return requests.post(f"{SERVER_URL}/scan", json={"code": text}, timeout=90)
    if mode == "github":
        return requests.post(f"{SERVER_URL}/scan-url", json={"repo_url": repo_url}, timeout=120)
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "text/plain")}
    return requests.post(f"{SERVER_URL}/scan-file", files=files, timeout=120)

def run_scan(mode: str, text: str | None = None, repo_url: str | None = None, uploaded_file=None) -> None:
    if mode == "manual" and text:
        with st.spinner("CyberGuard: Quick Risk Assessment..."):
            try:
                # 1. Quick ML Scan (Instant)
                resp = requests.post(f"{SERVER_URL}/scan/quick", json={"code": text}, timeout=250)
                if resp.status_code == 200:
                    quick_data = resp.json()
                    st.session_state.report = quick_data
                    st.session_state.current_code = text
                    st.session_state.source_meta = {}
                    st.session_state.solution = None
                    
                    # Force a temporary display by updating state
                    st.toast("⚡ Quick scan complete! fetching deep analysis...")
                else:
                    st.error("Quick scan failed.")
                    return
            except Exception as e:
                st.error(f"Connection Error: {e}")
                return

        # 2. Deep AI Analysis (Step 2 - Patch)
        with st.status("Deep AI Security Analysis...", expanded=False) as status:
            try:
                deep_resp = requests.post(f"{SERVER_URL}/scan/deep", json={"code": text}, timeout=120)
                if deep_resp.status_code == 200:
                    deep_data = deep_resp.json()
                    # Merge deep data (Expert Analysis, Worst Case, etc.) into the report
                    st.session_state.report = {**st.session_state.report, **deep_data}
                    status.update(label="✅ Full Analysis Complete!", state="complete", expanded=False)
                    st.rerun()
            except Exception:
                status.update(label="⚠️ AI Analysis partially unavailable.", state="error")
    else:
        # For GitHub/File, stick to full scan for accuracy
        with st.spinner("Scanning repository/file..."):
            response = call_scan_endpoint(mode, text=text, repo_url=repo_url, uploaded_file=uploaded_file)
            if response.status_code == 200:
                payload = response.json()
                st.session_state.report = payload.get("report", {})
                st.session_state.source_meta = payload.get("source_meta", {})
                st.session_state.solution = None
            else:
                st.error(f"Scan failed: {response.text}")

def generate_pdf_report(report: dict) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    
    def clean_text(text: str) -> str:
        if not text:
            return "N/A"
        # Safely encode to latin-1, replacing unsupported characters with '?'
        return str(text).encode('latin-1', 'replace').decode('latin-1')

    # Header
    pdf.set_font("helvetica", "B", 24)
    pdf.set_text_color(32, 201, 151) # CyberGuard Green
    pdf.cell(0, 20, "CyberGuard Security Report", ln=True, align="C")
    
    pdf.set_font("helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)

    # Risk Summary Section
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", "B", 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 12, " 1. Risk Summary", ln=True, fill=True)
    pdf.ln(5)
    
    pdf.set_font("helvetica", "", 12)
    pdf.cell(0, 8, f"Final Risk Level: {clean_text(report.get('final_risk'))}", ln=True)
    pdf.cell(0, 8, f"Risk Score: {report.get('score', 0)} / 10.0", ln=True)
    pdf.cell(0, 8, f"Confidence: {clean_text(report.get('confidence'))}", ln=True)
    if report.get("is_zero_day"):
        pdf.set_text_color(200, 0, 0)
        pdf.cell(0, 8, "Zero-Day Behavioral Pattern: Detected!", ln=True)
        pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # AI Analysis Section
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 12, " 2. AI Security Analysis", ln=True, fill=True)
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, "Technical DNA (Exact Analysis):", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, clean_text(report.get("exact_analysis", "No analysis provided.")))
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, "Forecast (Worst Case Scenario):", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, clean_text(report.get("worst_case", "No analysis provided.")))
    pdf.ln(5)

    pdf.set_font("helvetica", "B", 13)
    pdf.cell(0, 8, "Analogy (Attack Path Story):", ln=True)
    pdf.set_font("helvetica", "", 11)
    pdf.multi_cell(0, 6, clean_text(report.get("story", "No analysis provided.")))
    pdf.ln(10)

    # Code Context Section
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 12, " 3. Code Evidence", ln=True, fill=True)
    pdf.ln(5)
    
    pdf.set_font("courier", "", 9)
    # Clean up code text for PDF encoding
    code_text = clean_text(report.get("original_code", ""))
    pdf.multi_cell(0, 5, code_text)
    
    return bytes(pdf.output())

def render_step_flow() -> None:
    report = st.session_state.get("report")
    if not report:
        return

    st.subheader("Step 1: Risk Score and Code Highlights")
    col_chip, col_download = st.columns([1, 1])
    with col_chip:
        render_risk_chip(report.get("final_risk", "LOW"))
        st.write(f"Risk Score: **{report.get('score', 0):.2f} / 10.0** | Confidence: **{report.get('confidence', 'N/A')}**")
    
    with col_download:
        pdf_bytes = generate_pdf_report(report)
        st.download_button(
            label="📄 Download Security Report (PDF)",
            data=pdf_bytes,
            file_name=f"CyberGuard_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

    with st.expander("🚨 Worst Case Scenario", expanded=True):
        st.write(report.get("worst_case", "Analysis pending..."))
        
    with st.expander("📖 The Story / Attack Path"):
        st.write(report.get("story", "Analysis pending..."))
        
    with st.expander("🦠 Zero-Day Potential"):
        if report.get("is_zero_day"):
            st.error(f"WARNING: Highly deceptive zero-day pattern detected! {report.get('zero_day_reason', '')}")
        else:
            st.success("No immediate zero-day behavioral patterns found. Standard analysis applies.")

    st.markdown("<br>", unsafe_allow_html=True)

    highlighted = report.get("highlighted_code", [])
    if highlighted:
        code_html = "".join(
            [
                f"<div class='code-line {line.get('style', 'normal')}'>"
                f"Line {line.get('line_number', '')}: {line.get('content', '')}</div>"
                for line in highlighted
            ]
        )
        st.markdown(f"<div style='max-height: 300px; overflow-y: auto;'>{code_html}</div>", unsafe_allow_html=True)

    flagged = report.get("flagged_lines", [])
    if flagged:
        st.dataframe(pd.DataFrame(flagged), use_container_width=True)

    st.markdown("---")
    st.subheader("Step 2: AI Solution")
    if st.button("Get AI Solution?", use_container_width=True):
        with st.spinner("Generating secure solution..."):
            solution_response = requests.post(
                f"{SERVER_URL}/scan-solution",
                json={
                    "code_context": report.get("original_code", st.session_state.get("current_code", "")),
                    "risk_score": report.get("score", 0.5),
                },
                timeout=180,
            )
        if solution_response.status_code == 200:
            st.session_state.solution = solution_response.json().get("solution")
        else:
            st.error(f"Solution generation failed: {solution_response.text}")

    solution = st.session_state.get("solution")
    if not solution:
        return

    st.markdown("---")
    st.subheader("Step 3: Review and Decision")
    st.write(solution.get("brief", ""))
    st.write(solution.get("worst_case", ""))
    st.code(solution.get("secure_code", "No patch generated."), language="python")

    col_allow, col_deny = st.columns(2)
    with col_allow:
        if st.button("Allow Edit", use_container_width=True):
            st.session_state.solution_decision = "accept"
            requests.post(
                f"{SERVER_URL}/feedback",
                json={"action": "accept", "code_context": report.get("original_code", "")},
                timeout=30,
            )
    with col_deny:
        if st.button("Deny", use_container_width=True):
            st.session_state.solution_decision = "decline"
            requests.post(
                f"{SERVER_URL}/feedback",
                json={"action": "decline", "code_context": report.get("original_code", "")},
                timeout=30,
            )

    decision = st.session_state.get("solution_decision")
    if decision == "accept":
        st.success("Edit approved and saved.")
    elif decision == "decline":
        st.info("Edit denied. Original scan result is retained.")

def render_floating_chatbot() -> None:
    st.markdown("""
    <span id="chat-marker"></span>
    <style>
    div[data-testid="stVerticalBlock"] > div:has(#chat-marker) {
        display: none !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(#chat-marker) + div {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        width: auto !important;
        z-index: 9999 !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(#chat-marker) + div button {
        border-radius: 50px !important;
        box-shadow: 0 4px 15px rgba(32, 201, 151, 0.5) !important;
        padding: 0 24px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.button("🛡️ Chat with CyberGuard", key="toggle_chat", help="Click on the icon"):
        st.session_state.chat_open = not st.session_state.get("chat_open", False)
        st.rerun()

    if not st.session_state.get("chat_open", False):
        return

    st.markdown("""
    <span id="chat-window-marker"></span>
    <style>
    div[data-testid="stVerticalBlock"] > div:has(#chat-window-marker) {
        display: none !important;
    }
    div[data-testid="stVerticalBlock"] > div:has(#chat-window-marker) + div {
        position: fixed !important;
        bottom: 80px !important;
        right: 20px !important;
        width: 440px !important;
        max-height: 800px !important;
        height: 700px !important;
        background-color: #0b0f1a !important;
        border: 1px solid #20C997 !important;
        border-radius: 15px !important;
        padding: 20px !important;
        z-index: 9999 !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.7) !important;
        overflow-y: auto !important;
    }
    div:has(#close-btn-marker) + div button {
        background: transparent !important;
        color: #ff4b4b !important;
        box-shadow: none !important;
        min-height: 20px !important;
        height: auto !important;
        padding: 5px !important;
        font-size: 16px !important;
        margin-top: -5px !important;
    }
    div:has(#close-btn-marker) + div button:hover {
        background: rgba(255, 75, 75, 0.1) !important;
        transform: scale(1.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        col_title, col_close = st.columns([8, 1])
        with col_title:
            st.markdown("""
                <strong style='font-size: 1.1rem; color: #ffffff;'>CyberGuard Assistant</strong>
                <p style="font-size: 12px; color: #6C757D; margin-bottom: 5px;">CyberGuard is AI and can make mistakes. Always verify security findings with manual review.</p>
            """, unsafe_allow_html=True)
        with col_close:
            st.markdown('<span id="close-btn-marker"></span>', unsafe_allow_html=True)
            if st.button("❌", key="btn_close_chat", help="Close window"):
                st.session_state.chat_open = False
                st.rerun()
        
        st.markdown('<hr style="border-color: rgba(255,255,255,0.1); margin-top: 0px; margin-bottom: 15px;">', unsafe_allow_html=True)

        for item in st.session_state.get("chat_messages", []):
            role = "You" if item["role"] == "user" else "CyberGuard"
            st.markdown(f"**{role}:** {item['content']}")

        query = st.text_input("Ask a follow-up question", key="chat_query")
        if st.button("Send", key="chat_send", use_container_width=True):
            if not query.strip():
                st.warning("Please enter a question.")
            else:
                current_code = ""
                if st.session_state.get("report"):
                    current_code = st.session_state["report"].get("original_code", "")
                if not current_code:
                    current_code = st.session_state.get("current_code", "")
                if not current_code.strip():
                    active_tab = st.session_state.get("active_tab", "Description/Code")
                    if active_tab == "Description/Code":
                        msg = "Please paste your code in the description to start the chat."
                    elif active_tab == "GitHub Repo":
                        msg = "Please paste a GitHub repository URL to start the chat."
                    else:
                        msg = "Please upload a file to start the chat."
                    st.warning(f"⚠️ {msg}")
                else:
                    st.session_state.chat_messages.append({"role": "user", "content": query})
                    
                    try:
                        with st.spinner("CyberGuard is thinking..."):
                            response = requests.post(
                                f"{SERVER_URL}/chat",
                                json={"user_query": query, "code_context": current_code},
                                timeout=60,
                            )
                        if response.status_code == 200:
                            data = response.json()
                            answer = data.get("response", "No response")
                            if not answer.strip():
                                answer = "I'm processing your request. Could you please rephrase or ask something else?"
                            
                            if "OLLAMA_NOT_FOUND" in answer:
                                st.error("🚨 **Ollama Not Found!**")
                                st.info("Please download and install Ollama from [ollama.com](https://ollama.com) to enable the AI chatbot.")
                                st.session_state.chat_messages.pop()
                                return
                        else:
                            answer = f"⚠️ CyberGuard is currently busy. Please try again in a few seconds. (Error: {response.text[:50]})"
                    except Exception as e:
                        answer = "⚠️ Connection to CyberGuard lost. Please ensure Ollama is running."
                        
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})
                    st.rerun()

def main() -> None:
    st.set_page_config(page_title="CyberGuard Website", page_icon="🛡️", layout="wide")
    apply_ui()

    if "report" not in st.session_state:
        st.session_state.report = None
    if "source_meta" not in st.session_state:
        st.session_state.source_meta = {}
    if "solution" not in st.session_state:
        st.session_state.solution = None
    if "solution_decision" not in st.session_state:
        st.session_state.solution_decision = None
    if "current_code" not in st.session_state:
        st.session_state.current_code = ""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Description/Code"

    _, col_center, _ = st.columns([1, 5, 1])

    with col_center:
        st.markdown(
            "<h3 style='text-align: center; color: white; margin-bottom: 30px;'>Choose Analysis Mode</h3>",
            unsafe_allow_html=True
        )

        tabs = st.tabs(["📝 Description/Code", "🔗 GitHub Repo", "📁 Upload File"])

        with tabs[0]:
            st.session_state.active_tab = "Description/Code"
            st.markdown("<br>", unsafe_allow_html=True)
            code_input = st.text_area("Code", height=220, placeholder="Paste code here...", label_visibility="collapsed")
            if st.button("Predict Risk ➔", key="text_scan"):
                if code_input.strip():
                    run_scan("manual", text=code_input)
                else:
                    st.warning("Please provide code text.")

        with tabs[1]:
            st.session_state.active_tab = "GitHub Repo"
            st.markdown("<br>", unsafe_allow_html=True)
            repo_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo", label_visibility="collapsed")
            if st.button("Scan Repository ➔", key="repo_scan"):
                if repo_url.strip():
                    run_scan("github", repo_url=repo_url)
                else:
                    st.warning("Please provide a GitHub URL.")

        with tabs[2]:
            st.session_state.active_tab = "Upload File"
            st.markdown("<br>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload File", type=["py", "js", "java", "c", "cpp", "go", "sh", "txt"], label_visibility="collapsed")
            if st.button("Scan File ➔", key="file_scan"):
                if uploaded_file:
                    run_scan("file", uploaded_file=uploaded_file)
                else:
                    st.warning("Please upload a file.")

    st.markdown("---")
    render_step_flow()
    render_floating_chatbot()

    st.caption(f"CyberGuard is AI and can make mistakes. Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()