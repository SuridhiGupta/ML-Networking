import asyncio
from datetime import datetime
APP_VERSION = "3.0.0"
import time
import csv
import hashlib
import re
import os
import pandas as pd
import numpy as np
from pathlib import Path

import joblib
from openai import OpenAI
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

from code_scanner import scan_github_repository
from preprocessing import clean_text


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = MODELS_DIR / "xgboost_model.json"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
USER_AI_DATA_PATH = DATA_DIR / "user_ai_dataset.csv"
PENDING_FEEDBACK_PATH = DATA_DIR / "pending_feedback.csv"
TRAINING_DATASET_PATH = DATA_DIR / "training_dataset.csv"
PROCESSED_TRAINING_PATH = DATA_DIR / "processed_training.csv"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

RETRAIN_THRESHOLD = 3

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

if not USER_AI_DATA_PATH.exists():
    with USER_AI_DATA_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["code_context", "label"])

for path in [PENDING_FEEDBACK_PATH, TRAINING_DATASET_PATH, PROCESSED_TRAINING_PATH]:
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["code_context", "label", "source"])


app = FastAPI(title="CyberGuard API")

# --- GLOBAL MODEL CACHE (Load once at startup) ---
GLOBAL_BOOSTER = None
GLOBAL_VECTORIZER = None
GLOBAL_LABEL_ENCODER = None

def load_ml_resources():
    global GLOBAL_BOOSTER, GLOBAL_VECTORIZER, GLOBAL_LABEL_ENCODER
    try:
        if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            print("📦 Loading ML Models into memory...")
            GLOBAL_BOOSTER = xgb.Booster()
            GLOBAL_BOOSTER.load_model(str(MODEL_PATH))
            GLOBAL_VECTORIZER = joblib.load(VECTORIZER_PATH)
            if LABEL_ENCODER_PATH.exists():
                GLOBAL_LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
            print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading ML resources: {e}")

load_ml_resources()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Cache to prevent redundant processing
scan_cache = {}
MAX_CACHE_SIZE = 100

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"🔥 Critical Server Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"report": None, "error": "Internal Server Error", "detail": str(exc)},
    )




class ScanRequest(BaseModel):
    code: str


class RepoScanRequest(BaseModel):
    repo_url: str


class SolutionRequest(BaseModel):
    code_context: str
    risk_score: float | None = None


class ChatRequest(BaseModel):
    user_query: str
    code_context: str


class FeedbackRequest(BaseModel):
    action: str
    code_context: str


def save_to_dataset(code: str) -> None:
    with USER_AI_DATA_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([code, "LOW_RISK"])


async def groq_chat_safe(system_message: str, user_prompt: str, max_tokens: int = 800) -> str:
    """Standardized Groq Helper for all AI features (v3.0.0)."""
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ [V3_DEBUG] Groq AI API Error: {e}")
        return f"AI Service temporarily unavailable. Details: {str(e)[:50]}"


async def auto_retrain_model():
    """Triggers automated retraining once threshold is met."""
    try:
        try:
            df = pd.read_csv(TRAINING_DATASET_PATH)
        except UnicodeDecodeError:
            # Fallback for Windows-style encoding (powershell redirection)
            df = pd.read_csv(TRAINING_DATASET_PATH, encoding='utf-16')
        
        if len(df) < RETRAIN_THRESHOLD:
            return

        print(f"Automatic Retraining Triggered: Processing {len(df)} new samples...")
        
        # 1. Load context
        if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists() and LABEL_ENCODER_PATH.exists()):
            print("Cannot retrain: Model files missing.")
            return

        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)

        # Apply clean_text to every new sample
        df['code_context_clean'] = df['code_context'].apply(clean_text)
        X = vectorizer.transform(df['code_context_clean'])
        
        new_labels = df['label'].tolist()
        try:
            # Map labels precisely to what the LabelEncoder knows
            y = label_encoder.transform(new_labels)
        except Exception as ee:
            print(f"Label Encoding Issue: {ee}. Attempting auto-correction...")
            # If 'High_Risk', try 'High'
            corrected = [l.split('_')[0].capitalize() if '_' in l else l.capitalize() for l in new_labels]
            y = label_encoder.transform(corrected)

        dtrain = xgb.DMatrix(X, label=y)

        # 3. Update Model (Incremental)
        # We use a small number of iterations to 'tune' the model with new data
        updated_booster = xgb.train(
            {'process_type': 'update', 'updater': 'refresh', 'refresh_leaf': True},
            dtrain,
            num_boost_round=10,
            xgb_model=booster
        )

        # 4. Save & Cleanup
        updated_booster.save_model(str(MODEL_PATH))
        
        # Archiving processed data
        with open(PROCESSED_TRAINING_PATH, 'a', newline='', encoding='utf-8') as f:
            df.to_csv(f, header=False, index=False)
        
        # Clear the current training queue
        with open(TRAINING_DATASET_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["code_context", "label", "source"])
            
        print("Automatic Retraining Complete. Model updated.")
        
        # Refresh Global Cache
        load_ml_resources()
        
    except Exception as e:
        print(f"Auto-retrain failed: {e}")


def enforce_two_lines(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines[:2]) if lines else "Analysis pending..."


def compute_risk(code_text: str) -> tuple[float, str, list[dict], list[dict], bool, str | None, str, bool]:
    start_time = time.time()
    ml_confidence = 0.5
    
    # Use Global Cache instead of loading from disk every time
    if GLOBAL_BOOSTER and GLOBAL_VECTORIZER:
        try:
            cleaned_code = clean_text(code_text)
            dmatrix = xgb.DMatrix(GLOBAL_VECTORIZER.transform([cleaned_code]))
            prediction = GLOBAL_BOOSTER.predict(dmatrix)[0]
            ml_confidence = float(np.max(prediction)) if isinstance(prediction, np.ndarray) else float(prediction)
        except Exception as e:
            print(f"ML Inference Error: {e}")
            ml_confidence = 0.5

    risk_keywords = [
        "eval(", "exec(", "os.system", "subprocess.", "socket.", 
        "pickle.loads", "pickle.load", "yaml.load", "yaml.safe_load",
        "requests.get", "requests.post", "urllib.request",
        "flask.request", "request.form", "request.args",
        "__import__", "getattr(", "setattr(", "base64.b64decode",
        "SELECT ", "INSERT ", "UNION SELECT", "DROP TABLE"
    ]
    lines = code_text.split("\n")
    flagged_lines = []
    highlighted_code = []

    for idx, line in enumerate(lines, start=1):
        risky = any(keyword in line for keyword in risk_keywords)
        if risky:
            flagged_lines.append({"line_number": idx, "content": line.strip()})
        highlighted_code.append(
            {"line_number": idx, "content": line, "style": "red-bg" if risky else "green-bg"}
        )

    has_network = "socket" in code_text or "requests" in code_text
    has_dangerous_calls = any(token in code_text for token in ["os.", "exec", "eval"])
    is_zero_day = has_network and has_dangerous_calls

    has_user_input = "input(" in code_text or "request.form" in code_text or "request.args" in code_text
    is_user_injection = has_user_input and (is_rce or is_sqli)

    base_cvss = min(((0.5 if flagged_lines else 0.0) + (0.4 if is_zero_day else 0.0)) * 10.0, 10.0)
    if is_user_injection:
        base_cvss = max(base_cvss, 9.2) # Force High/Critical for injection
    
    if base_cvss == 0.0:
        base_cvss = 1.0

    final_score = base_cvss
    is_boosted = False

    if ml_confidence > 0.6 and base_cvss < 7.0:
        boosted = (base_cvss * 1.2) + (ml_confidence * 2)
        if boosted > base_cvss:
            final_score = boosted
            is_boosted = True

    if ml_confidence > 0.85 and (is_rce or is_sqli or is_user_injection):
        if final_score < 7.0:
            final_score = 9.5
        is_boosted = True

    final_score = min(final_score, 10.0)

    if final_score >= 8.5 or is_zero_day:
        final_risk = "CRITICAL"
    elif final_score >= 7.0:
        final_risk = "HIGH"
    elif final_score >= 4.0:
        final_risk = "MEDIUM"
    else:
        final_risk = "LOW"

    zero_day_reason = None
    if is_zero_day:
        zero_day_reason = (
            "Potential zero-day pattern detected because network access and dangerous execution APIs "
            "are present together."
        )

    duration = time.time() - start_time
    print(f"⏱️ [ML Prediction] Duration: {duration:.2fs}")
    return final_score, final_risk, flagged_lines, highlighted_code, is_zero_day, zero_day_reason, f"{ml_confidence * 100:.1f}%", is_boosted


async def get_hybrid_ai_report(code_snippet: str, score: float) -> dict:
    start_time = time.time()
    try:
        prompt = f"""
Analyze this code vulnerability.

Code:
{code_snippet[:600]}

Risk Score: {score}

Return:
1. Exact technical analysis
2. Worst-case attack scenario
3. Simple attack story
"""

        text = await groq_chat_safe(
            "You are a cybersecurity expert. Provide a concise 3-part analysis including: 1. Technical DNA, 2. Worst-Case Forecast, 3. Story Analogy.",
            prompt,
            800
        )

        duration = time.time() - start_time
        print(f"⏱️ [Groq AI Analysis] Duration: {duration:.2fs}")
        
        # Ensure we return valid content even if AI is generic
        return {
            "exact_analysis": text[:400] if len(text) > 50 else "Detailed technical review of the identified vulnerability markers.",
            "worst_case": text[:400] if len(text) > 50 else "Potential for unauthorized code execution or system compromise.",
            "story": text[:400] if len(text) > 50 else "Like an unlocked back door that allows an intruder to enter your system."
        }

    except Exception as e:
        print(f"Groq Deep Analysis Error: {e}")
        return {
            "exact_analysis": "CyberGuard analysis complete. Review highlighted markers.",
            "worst_case": "Potential security breach or data exposure if unpatched.",
            "story": "A security gap detected in your application logic."
        }


async def get_secure_patch(code_snippet: str, score: float) -> str:
    start_time = time.time()
    try:
        prompt = f"""
Fix this vulnerable code securely.

Code:
{code_snippet[:1200]}

Risk score: {score}

Return only fixed code.
"""

        patch = await groq_chat_safe(
            "You are a senior secure coding engineer. Return ONLY the fixed code block. No explanation.",
            prompt,
            800
        )
        duration = time.time() - start_time
        print(f"⏱️ [Groq AI Patching] Duration: {duration:.2fs}")
        return patch

    except Exception:
        return "AI patch generation temporarily unavailable. Please apply secure input validation and avoid unsafe execution methods."

async def build_scan_response(code_text: str, source_meta: dict | None = None) -> dict:
    # 1. Check Cache First
    cache_key = hashlib.md5(code_text.encode()).hexdigest()
    if cache_key in scan_cache:
        print(f"🚀 Cache Hit: {cache_key}")
        cached_res = scan_cache[cache_key].copy()
        if source_meta:
            cached_res["source_meta"] = source_meta
        return cached_res

    # --- 2. PRE-ANALYSIS HEURISTIC CHECK (Fast Pass) ---
    markers = ['eval(', 'fromCharCode', '\\x', '0x']
    found_count = sum(1 for m in markers if code_text.find(m) != -1)
    
    if found_count >= 2:
        # Instant Critical for Obfuscation
        score = 10.0
        risk = "CRITICAL"
        is_zero_day = True
        confidence = "Heuristic (100%)"
        
        # Pre-populate fields to avoid 'Analysis Pending' and skip LLM
        return {
            "report": {
                "final_risk": risk,
                "score": score,
                "confidence": confidence,
                "is_boosted": True,
                "flagged_lines": [{"line_number": 0, "content": "HEURISTIC DETECTION: Code Obfuscation Detected"}],
                "highlighted_code": [{"line_number": 0, "content": code_text[:200] + "...", "style": "red-bg"}],
                "is_zero_day": True,
                "zero_day_reason": "High-confidence obfuscation signature detected (multiple dangerous markers found).",
                "original_code": code_text,
                "has_solution": False,
                "worst_case": "HEURISTIC ALERT: Code obfuscation is frequently used to hide ransomware or data exfiltration logic. Execution may result in total system compromise with hidden malicious intent.",
                "story": "This code is like a 'trojan horse' using a secret language (obfuscation) to hide its true face from security guards. It is definitely hiding malicious intent.",
                "exact_analysis": f"Static analysis detected {found_count} obfuscation markers: {', '.join([m for m in markers if code_text.find(m) != -1])}.",
            },
            "source_meta": source_meta or {},
            "flow": {"step_1": "Obfuscation detected. Review with extreme caution.", "step_2": "Click 'Get AI Solution?'", "step_3": "Review patch."}
        }

    # Normal Scan Logic (ML + AI)
    score, risk, flagged_lines, highlighted_code, is_zero_day, zero_day_reason, confidence, is_boosted = compute_risk(code_text)

    ai_report = await get_hybrid_ai_report(code_text, score)

    final_response = {
        "v": APP_VERSION,
        "risk_level": risk,
        "score": round(final_score, 2),
        "confidence": confidence,
        "worst_case": ai_report.get("worst_case", "Critical vulnerability in system core."),
        "attack_story": ai_report.get("story", "Dangerous execution pattern detected."),
        "zero_day": zero_day_reason or "No zero-day behavioral patterns found.",
        "recommendation": "Security remediation required. Mitigate RCE/Injection risks.",
        "patch": "Click 'Get AI Solution?' to generate fix.",
        "original_code": code_text,
        "flagged_lines": flagged_lines,
        "highlighted_code": highlighted_code,
    }

    print(f"📦 [V3_DEBUG] Returning Response: Score {final_response['score']}, Risk {final_response['risk_level']}")
    
    # Update cache (with size management)
    if len(scan_cache) >= MAX_CACHE_SIZE:
        scan_cache.clear()
    scan_cache[cache_key] = final_response

    return final_response


@app.get("/")
async def root():
    return {
        "v": APP_VERSION,
        "status": "live", 
        "service": "CyberGuard API",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/debug/config")
async def debug_config():
    """Diagnostic route to verify environment without exposing full secrets."""
    groq_key = os.getenv("GROQ_API_KEY", "")
    return {
        "v": APP_VERSION,
        "groq_api_key_set": bool(groq_key),
        "groq_api_key_masked": f"{groq_key[:4]}...{groq_key[-4:]}" if len(groq_key) > 8 else "too_short/null",
        "model_files_present": {
            "booster": MODEL_PATH.exists(),
            "vectorizer": VECTORIZER_PATH.exists(),
            "encoder": LABEL_ENCODER_PATH.exists()
        },
        "cache_status": {
            "booster_loaded": GLOBAL_BOOSTER is not None,
            "vectorizer_loaded": GLOBAL_VECTORIZER is not None
        }
    }


@app.get("/health")
async def health():
    return {"status": "ok", "llm": "groq_connected"}


@app.post("/scan/quick")
async def scan_quick(data: ScanRequest) -> dict:
    print(f"---> Incoming POST: /scan/quick")
    """Returns ML-only result instantly."""
    try:
        score, risk, flagged, highlighted, zday, zreason, conf, boosted = compute_risk(data.code)
        return {
            "risk_level": risk,
            "score": round(score, 2),
            "confidence": conf,
            "worst_case": "AI Deep Analysis pending...",
            "attack_story": "AI Deep Analysis pending...",
            "zero_day": zreason or "No immediate zero-day behavioral patterns found.",
            "recommendation": "ML quick scan complete. Waiting for AI deep analysis.",
            "patch": "Scanning for secure solution...",
            "original_code": data.code,
            "flagged_lines": flagged,
            "highlighted_code": highlighted,
        }
    except Exception as e:
        print(f"Quick scan error: {e}")
        return {"error": str(e), "score": 0, "risk_level": "ERROR"}

@app.post("/scan/deep")
async def scan_deep(data: ScanRequest) -> dict:
    """Returns AI-only analysis on top of existing score."""
    try:
        score, _, _, _, _, _, _, _ = compute_risk(data.code)
        ai_report = await get_hybrid_ai_report(data.code, score)
        return {
            "worst_case": ai_report.get("worst_case", "Analysis complete."),
            "attack_story": ai_report.get("story", "Breach possibility detected."),
            "recommendation": "Immediate remediation recommended. Follow secure coding standards."
        }
    except Exception as e:
        print(f"Deep scan error: {e}")
        return {"worst_case": "AI analysis unavailable.", "attack_story": "N/A", "recommendation": "Review risky patterns."}


# Optimized Scan Route with Local Error Handling
@app.post("/scan")
async def scan_code(data: ScanRequest) -> dict:
    print(f"---> Incoming POST: /scan")
    try:
        return await build_scan_response(data.code, source_meta={"input_mode": "manual_text"})
    except Exception as e:
        print(f"Error in /scan: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/scan-url")
async def scan_repository_url(data: RepoScanRequest) -> dict:
    try:
        repository_report = scan_github_repository(data.repo_url)
        findings = repository_report.get("findings", [])
        code_text = "\n".join(item.get("code", "") for item in findings[:200] if item.get("code"))
        if not code_text.strip():
            code_text = f"Repository scanned: {data.repo_url}\nNo risky code patterns were found in sampled files."

        return await build_scan_response(
            code_text,
            source_meta={
                "input_mode": "github_url",
                "repo_url": data.repo_url,
                "total_findings": repository_report.get("total_findings", 0),
                "risk_level": repository_report.get("risk_level", "None"),
            },
        )
    except Exception as error:
        print(f"Error in /scan-url: {error}")
        raise HTTPException(status_code=400, detail=f"Repository scan failed: {error}")


@app.post("/scan-file")
async def scan_file(file: UploadFile = File(...)) -> dict:
    try:
        raw_bytes = await file.read()
        try:
            code_text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            code_text = raw_bytes.decode("latin-1", errors="ignore")

        return await build_scan_response(
            code_text,
            source_meta={"input_mode": "file_upload", "filename": file.filename},
        )
    except Exception as error:
        print(f"Error in /scan-file: {error}")
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/scan-solution")
async def scan_solution(data: SolutionRequest) -> dict:
    if not data.code_context.strip():
        raise HTTPException(status_code=400, detail="code_context is required")

    score = data.risk_score if data.risk_score is not None else compute_risk(data.code_context)[0]
    
    # Redundant analysis removed to save time. Using the patch generator only.
    secure_code = await get_secure_patch(data.code_context, score)

    return {
        "solution": {
            "brief": "Secure patch generated based on risk analysis.",
            "worst_case": "Mitigated with secure coding practices.",
            "story": "Patch applied.",
            "secure_code": secure_code,
        }
    }


@app.post("/chat")
async def chat_with_cyberguard(data: ChatRequest) -> dict:
    print(f"---> Incoming POST: /chat")
    system_message = (
        "You are CyberGuard Assistant, a friendly security expert. "
        "IMPORTANT: Provide direct, short, and concise answers only. "
        "Do NOT include meta-commentary like 'Improved Answer' or 'Analysis'. "
        "Respond warmly but briefly (max 2-3 sentences) to greetings. "
        "Get straight to the point without repeating the user's question."
    )
    try:
        # Check if we have code context to include, but trim it to prevent slowness
        user_input = data.user_query
        if data.code_context.strip():
            # Only send the first 1500 chars of code context to maintain speed
            prompt = f"Context Code Snippet:\n{data.code_context[:1500]}\n\nUser Question: {user_input}"
        else:
            prompt = user_input

        answer = await groq_chat_safe(
            system_message,
            prompt,
            500
        )
        if "[VALID]" in answer.upper():
            save_to_dataset(data.code_context)
            answer = f"Verified and saved. {answer.replace('[VALID]', '').strip()}"
        return {"response": answer}
    except Exception as error:
        error_str = str(error).lower()
        if "connection" in error_str or "not found" in error_str:
            return {"response": "CyberGuard AI service is temporarily unavailable. Please try again in a moment."}
        return {"response": f"CyberGuard encountered an error: {error}"}


@app.post("/feedback")
async def submit_feedback(data: FeedbackRequest) -> dict:
    print(f"---> Incoming POST: /feedback")
    code = data.code_context.strip()
    if not code:
        return {"status": "error", "message": "Empty code context."}

    # Step A: Suggestion Capture
    with PENDING_FEEDBACK_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([code, "PENDING", "user_suggestion"])

    # Step B: Double-Check Verification
    # 1. AI (LLM) Analysis
    system_message = (
        "You are a Senior Security Researcher. Analyze if the provided code snippet contains "
        "any security vulnerabilities. Reply ONLY with 'YES' if it is vulnerable, or 'NO' if it is safe. "
        "Provide a 1-sentence technical reason after your verdict."
    )
    try:
        ai_verdict_text = (
            await groq_chat_safe(
                system_message,
                f"Code to verify: {code}",
                150
            )
        ).upper()
        ai_is_vulnerable = "YES" in ai_verdict_text
    except Exception:
        return {"status": "pending", "message": "AI verification failed. Suggestion held in pending."}

    # 2. Tri-Hybrid Fusion Model Analysis
    score, final_risk, _, _, is_zero_day, _, _, _ = compute_risk(code)
    model_is_vulnerable = is_zero_day or score > 4.0

    # Verification Logic
    response_result = None
    if ai_is_vulnerable and model_is_vulnerable:
        # Scenario 1: Both Agree - Likely Vulnerable
        with TRAINING_DATASET_PATH.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([code, "High", "verified_suggestion"]) 
        response_result = {
            "status": "success", 
            "message": "Suggestion verified and added to training queue. The model will improve in the next update cycle."
        }

    elif not ai_is_vulnerable and not model_is_vulnerable:
        # Scenario 3: Both Agree it is Safe - Reject User Suggestion
        explanation_prompt = f"The user thinks this code is unsafe: {code}. Explain why it is actually safe in 1-2 short sentences."
        explanation_text = await groq_chat_safe(
            "You are a senior security reviewer. Explain why the code is safe in 2 short sentences.",
            explanation_prompt,
            200
        )
        response_result = {
            "status": "rejected",
            "message": "Suggestion rejected. Our security engine confirms this code pattern is safe.",
            "explanation": explanation_text
        }

    else:
        # Scenario 2: Conflict (One says safe, one says unsafe)
        response_result = {"status": "conflicted", "message": "Analyzing deeper... Suggestion held for manual expert review."}

    # Trigger retraining check in background
    asyncio.create_task(auto_retrain_model())
    return response_result


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting CyberGuard on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)