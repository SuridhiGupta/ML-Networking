import asyncio
import csv
import hashlib
import re
import pandas as pd
import numpy as np
from pathlib import Path

import joblib
import ollama
import uvicorn
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

ollama_semaphore = asyncio.Semaphore(1)


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


async def ollama_chat_safe(model: str, messages: list[dict], options: dict | None = None):
    async with ollama_semaphore:
        return await asyncio.to_thread(
            ollama.chat,
            model=model,
            messages=messages,
            options=options or {},
        )


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
        
    except Exception as e:
        print(f"Auto-retrain failed: {e}")


def enforce_two_lines(text: str) -> str:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return "\n".join(lines[:2]) if lines else "Analysis pending..."


def compute_risk(code_text: str) -> tuple[float, str, list[dict], list[dict], bool, str | None, str]:
    ml_confidence = 0.5
    if MODEL_PATH.exists() and VECTORIZER_PATH.exists():
        try:
            booster = xgb.Booster()
            booster.load_model(str(MODEL_PATH))
            vectorizer = joblib.load(VECTORIZER_PATH)
            
            # Application of official clean_text for consistency
            cleaned_code = clean_text(code_text)
            dmatrix = xgb.DMatrix(vectorizer.transform([cleaned_code]))
            
            prediction = booster.predict(dmatrix)[0]
            ml_confidence = float(np.max(prediction)) if isinstance(prediction, np.ndarray) else float(prediction)
        except Exception:
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

    base_cvss = min(((0.3 if flagged_lines else 0.0) + (0.4 if is_zero_day else 0.0)) * 10.0, 10.0)
    if base_cvss == 0.0:
        base_cvss = 1.0

    final_score = base_cvss
    is_boosted = False

    if ml_confidence > 0.6 and base_cvss < 7.0:
        boosted = (base_cvss * 1.2) + (ml_confidence * 2)
        if boosted > base_cvss:
            final_score = boosted
            is_boosted = True

    is_rce = "exec(" in code_text or "eval(" in code_text or "os.system" in code_text or "subprocess." in code_text
    is_sqli = "SELECT " in code_text.upper() or "DROP TABLE" in code_text.upper() or "cursor.execute" in code_text

    if ml_confidence > 0.85 and (is_rce or is_sqli):
        if final_score < 7.0:
            final_score = 8.5
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

    return final_score, final_risk, flagged_lines, highlighted_code, is_zero_day, zero_day_reason, f"{ml_confidence * 100:.1f}%", is_boosted


async def get_hybrid_ai_report(code_snippet: str, score: float) -> dict:
    system_message = (
        "You are a cybersecurity expert. Respond with markers [DNA], [FORECAST], [ANALOGY]. "
        "Keep each marker content concise and technical."
    )
    try:
        # Optimized for maximum speed and directness
        options = {
            "num_predict": 250, 
            "temperature": 0.3,
            "top_k": 20,
            "top_p": 0.4,
            "num_ctx": 2048
        }
        response = await ollama_chat_safe(
            model="phi3:latest",
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert. Provide: [DNA]: brief analysis. [FORECAST]: worst case. [ANALOGY]: 1-sentence attack story."},
                {"role": "user", "content": f"Code: {code_snippet[:600]}\nRisk: {score}"},
            ],
            options=options,
        )
        text = response["message"]["content"]
        print(f"DEBUG AI RESP: {text}")
        
        # Robust non-regex 'find' logic to extract the patches
        parts = {
            "exact_analysis": "Expert technical analysis complete.",
            "worst_case": "Potential system-wide impact detected.",
            "story": "Attack scenario identified via analysis."
        }
        
        up_text = text.upper()
        # Find start positions using multiple possible markers
        dna_idx = up_text.find("DNA")
        if dna_idx == -1: dna_idx = up_text.find("ANALYSIS")
        
        fc_idx = up_text.find("FORECAST")
        if fc_idx == -1: fc_idx = up_text.find("WORST")
        
        an_idx = up_text.find("ANALOGY")
        if an_idx == -1: an_idx = up_text.find("STORY")
        if an_idx == -1: an_idx = up_text.find("PATH")
        
        if dna_idx != -1:
            end = fc_idx if fc_idx != -1 else (an_idx if an_idx != -1 else len(text))
            parts["exact_analysis"] = text[dna_idx:end].replace("DNA", "").replace("ANALYSIS", "").strip(": \n\t[]")
            
        if fc_idx != -1:
            end = an_idx if an_idx != -1 else len(text)
            parts["worst_case"] = text[fc_idx:end].replace("FORECAST", "").replace("WORST", "").strip(": \n\t[]")
            
        if an_idx != -1:
            parts["story"] = text[an_idx:].replace("ANALOGY", "").replace("STORY", "").replace("PATH", "").strip(": \n\t[]")
            
        return {
            "exact_analysis": parts["exact_analysis"][:300],
            "worst_case": parts["worst_case"][:300],
            "story": parts["story"][:300]
        }
    except Exception as e:
        print(f"Deep Analysis Exception: {e}")
        return {
            "exact_analysis": "Security review finished.",
            "worst_case": "Impact analysis complete.",
            "story": "Vulnerability path mapped."
        }
    except Exception as e:
        print(f"AI Report Error: {e}")
        return {
            "exact_analysis": "Expert technical review finished.",
            "worst_case": "Critical security impact potential.",
            "story": "Backdoor scenario detected via analysis.",
        }


async def get_secure_patch(code_snippet: str, score: float) -> str:
    try:
        response = await ollama_chat_safe(
            model="phi3:latest",
            messages=[
                {
                    "role": "system",
                    "content": "You are a world-class security engineer. Provide a brief secure patch for the vulnerable code and return code only.",
                },
                {"role": "user", "content": f"Vulnerable Code:\n{code_snippet[:1500]}\n\nPredictive Risk Score: {score}"},
            ],
            options={
                "num_predict": 300, 
                "temperature": 0.3,
                "num_ctx": 2048
            },
        )
        return response["message"]["content"].strip()
    except Exception:
        return "Secure patch generation failed."


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
        "report": {
            "final_risk": risk,
            "score": round(score, 2),
            "confidence": confidence,
            "is_boosted": is_boosted,
            "flagged_lines": flagged_lines,
            "highlighted_code": highlighted_code,
            "is_zero_day": is_zero_day,
            "zero_day_reason": zero_day_reason,
            "original_code": code_text,
            "has_solution": True,
            "worst_case": ai_report.get("worst_case", "Analysis complete."),
            "story": ai_report.get("story", "Breach possibility detected."),
            "exact_analysis": ai_report.get("exact_analysis", "Review risky patterns."),
        },
        "source_meta": source_meta or {},
        "flow": {
            "step_1": "Review highlighted code.",
            "step_2": "Click 'Get AI Solution?' for a fix.",
            "step_3": "Apply secure patch.",
        },
    }

    # Update cache (with size management)
    if len(scan_cache) >= MAX_CACHE_SIZE:
        scan_cache.clear()
    scan_cache[cache_key] = final_response

    return final_response


@app.post("/scan/quick")
async def scan_quick(data: ScanRequest) -> dict:
    """Returns ML-only result instantly."""
    try:
        score, risk, flagged, highlighted, zday, zreason, conf, boosted = compute_risk(data.code)
        return {
            "score": round(score, 2),
            "final_risk": risk,
            "confidence": conf,
            "is_boosted": boosted,
            "flagged_lines": flagged,
            "highlighted_code": highlighted,
            "is_zero_day": zday,
            "zero_day_reason": zreason
        }
    except Exception as e:
        print(f"Quick scan error: {e}")
        return {"error": str(e)}

@app.post("/scan/deep")
async def scan_deep(data: ScanRequest) -> dict:
    """Returns AI-only analysis on top of existing score."""
    try:
        score, _, _, _, _, _, _, _ = compute_risk(data.code)
        ai_report = await get_hybrid_ai_report(data.code, score)
        return ai_report
    except Exception as e:
        print(f"Deep scan error: {e}")
        return {"exact_analysis": "AI analysis unavailable.", "worst_case": "N/A", "story": "N/A"}


# Optimized Scan Route with Local Error Handling
@app.post("/scan")
async def scan_code(data: ScanRequest) -> dict:
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

        response = await ollama_chat_safe(
            model="phi3:latest",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            options={
                "num_predict": 250, 
                "temperature": 0.6,
                "num_ctx": 2048
            },
        )
        answer = response["message"]["content"].strip()
        if "[VALID]" in answer.upper():
            save_to_dataset(data.code_context)
            answer = f"Verified and saved. {answer.replace('[VALID]', '').strip()}"
        return {"response": answer}
    except Exception as error:
        error_str = str(error).lower()
        if "connection" in error_str or "not found" in error_str:
             return {"response": "OLLAMA_NOT_FOUND: CyberGuard cannot find Ollama on your system. Please download and start Ollama (https://ollama.com) to use the chatbot."}
        return {"response": f"CyberGuard encountered an error: {error}"}


@app.post("/feedback")
async def submit_feedback(data: FeedbackRequest) -> dict:
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
        ai_resp = await ollama_chat_safe(
            model="phi3",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Code to verify: {code}"},
            ]
        )
        ai_verdict_text = ai_resp["message"]["content"].upper()
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
        explanation_resp = await ollama_chat_safe(
            model="phi3:latest",
            messages=[{"role": "user", "content": explanation_prompt}],
            options={"num_predict": 100}
        )
        response_result = {
            "status": "rejected",
            "message": "Suggestion rejected. Our security engine confirms this code pattern is safe.",
            "explanation": explanation_resp["message"]["content"]
        }

    else:
        # Scenario 2: Conflict (One says safe, one says unsafe)
        response_result = {"status": "conflicted", "message": "Analyzing deeper... Suggestion held for manual expert review."}

    # Trigger retraining check in background
    asyncio.create_task(auto_retrain_model())
    return response_result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)