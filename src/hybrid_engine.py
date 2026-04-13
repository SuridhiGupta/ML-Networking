import joblib
import os
import ollama
import textwrap
import xgboost as xgb
import numpy as np


# --- LAYER 3: INTERACTIVE INVESTIGATION (The Chatbot Intelligence) ---
def chat_with_guard(user_query, original_code, risk_score):
    """
    This function powers the website chatbot.
    It knows which code we are discussing.
    """
    system_message = (
        "You are the CyberGuard Interactive Assistant. A user is asking about a specific "
        f"code snippet that you flagged with a score of {risk_score}. "
        "Keep your answers in precise manner, technical, and helpful. Use one emoji per response."
    )
    
    try:
        # Switching to phi3 for stability
        response = ollama.chat(model='phi3', messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': f"Code: {original_code}\nQuery: {user_query}"}
        ])
        return response['message']['content']
    except:
        return "I'm recalibrating. Please try in a moment! 🤖"


# --- LAYER 2: LLM ASSISTANCE LAYER (Post-Analysis Intelligence) ---
def get_llm_insights(vuln_type, code_context, score):
    system_message = (
        "You are a Senior Cyber Security Researcher. I need a HEAVY technical report. "
        "Each section MUST have at least 2 lines of preciseexplanation and NO bullet points.\n\n"
        "Follow this EXACT example style:\n"
        "Section 1: 💻 Exact Analysis: Explain the exact technical vulnerability in exactly 2 short  sentences. Mention what's wrong in code.\n"
        "Section 2: ⚠️ Worst Case: Describe a realistic cyber-attack scenario in exactly 2 short  sentences. Explain the impact on data privacy.\n"
        "Section 3: 🛡️ Story: Use a funny and elaborate analogy in exactly 2 short  sentences.\n"
    )   
        
    try:
        response = ollama.chat(
            model='phi3', 
            messages=[{'role': 'system', 'content': system_message},
                      {'role': 'user', 'content': f"Code: {code_context}. Score: {score}"}],
            options={
                "num_predict": 350,
                "temperature": 0.5,
                "num_ctx": 2048 # Reduced context for speed
            }
        )
        return response['message']['content']
    except:
        return "⚠️ High Load: Security engine is busy. Please try again."

# --- LAYER 1: ML PREDICTION ENGINE (Main Core) ---

# Terminal colors
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def get_hybrid_prediction(description, cvss_score, code_snippet):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    model_path = os.path.join(BASE_DIR, 'models', 'xgboost_model.json')
    vec_path = os.path.join(BASE_DIR, 'models', 'tfidf_vectorizer.joblib')
    ml_max_conf = 0.5 

    if os.path.exists(model_path) and os.path.exists(vec_path):
        try:
            # XGBoost loading (Standard way for .json models)
            bst = xgb.Booster()
            bst.load_model(model_path)
            
            vectorizer = joblib.load(vec_path)
            text_features = vectorizer.transform([description])
            
            # Convert to DMatrix (XGBoost's internal format)
            dtrain = xgb.DMatrix(text_features)
            ml_probs = bst.predict(dtrain)[0]
            
            # If binary classification, ml_probs will be a float (0 to 1)
            # If multiclass, it will be an array
            if isinstance(ml_probs, (np.ndarray, list)):
                ml_max_conf = float(np.max(ml_probs))
            else:
                ml_max_conf = float(ml_probs)
                
        except Exception as e: 
            print(f"⚠️ ML Prediction Error: {e}")
            ml_max_conf = 0.5
    else:
        print("❌ Error: Models not found!")

    # --- 1. LINE SCAN & BEHAVIORAL LOGIC ---
    lines = code_snippet.split('\n')
    flagged_lines = []
    highlighted_output = []
    # Risk keywords for behavioral pattern matching (using .find() instead of regex)
    risk_keywords = ['eval(', 'os.system', 'exec(', 'subprocess.', 'cursor.execute']
    
    behavior_score = 0
    # Zero-Day Detection logic using .find() method
    has_network = code_snippet.find('socket') != -1 or code_snippet.find('requests') != -1 or code_snippet.find('input') != -1
    has_dangerous = code_snippet.find('os.') != -1 or code_snippet.find('exec') != -1 or code_snippet.find('eval') != -1
    if has_network and has_dangerous:
        behavior_score += 0.5
    
    for i, line in enumerate(lines):
        is_risky = any(line.find(keyword) != -1 for keyword in risk_keywords)
        if is_risky:
            flagged_lines.append((i + 1, line.strip()))
            highlighted_output.append(f"{RED}{BOLD}Line {i+1}: {line.strip()}{RESET}")
        else:
            highlighted_output.append(f"Line {i+1}: {line.strip()}")

    # --- 2. TRI-HYBRID FUSION (Dynamic CVSS Boost Logic) ---
    base_cvss = cvss_score
    final_score = base_cvss
    
    # 1. Implement Dynamic Boost Algorithm
    if ml_max_conf > 0.6 and base_cvss < 7.0:
        boosted = (base_cvss * 1.2) + (ml_max_conf * 2)
        if boosted > base_cvss:
            final_score = boosted
            
    # 2. Threshold-Based Elevation
    is_rce = code_snippet.find('exec(') != -1 or code_snippet.find('eval(') != -1 or code_snippet.find('os.system') != -1 or code_snippet.find('subprocess') != -1
    is_sqli = code_snippet.upper().find('SELECT ') != -1 or code_snippet.upper().find('DROP TABLE') != -1 or code_snippet.find('cursor.execute') != -1
    
    if ml_max_conf > 0.85 and (is_rce or is_sqli):
        if final_score < 7.0:
            final_score = 8.5
            
    final_score = min(final_score, 10.0)

    # Labeling
    if behavior_score >= 0.5: label = "🛡️ ZERO-DAY SUSPECT"
    elif final_score >= 8.5: label = "🔴 CRITICAL"
    elif final_score >= 7.0: label = "🟠 HIGH"
    elif final_score >= 4.0: label = "🟡 MEDIUM"
    else: label = "🟢 LOW"

    return {
        "final_risk": label,
        "score": round(final_score, 2),
        "confidence": f"{round(ml_max_conf * 100, 1)}%",
        "flagged_lines": flagged_lines,
        "highlighted_code": highlighted_output, 
        "is_zero_day": behavior_score >= 0.5,
        "is_poly": code_snippet.find("base64") != -1 or code_snippet.find("eval") != -1,
        "full_code": code_snippet,
        "insights": get_llm_insights("Security Analysis", code_snippet, final_score)
    }
    
    
    # Step 4: Testing the Hybrid Engine with a Sample Code Snippet
if __name__ == "__main__":
    # Sample Input
    test_code = 'import os\nimport base64\nimport socket\ns = socket.socket()\npayload = base64.b64decode("ZGly")\nos.system(payload)'
    test_desc = "OS Injection"
    test_cvss = 9.8

    result = get_hybrid_prediction(test_desc, test_cvss, test_code)
    WIDTH = 80

    # Smart Tags Logic
    tags = []
    if result.get('is_zero_day'): tags.append(f"{RED}[BEHAVIORAL]{RESET}")
    if result.get('is_poly'): tags.append(f"{RED}[POLYMORPHIC]{RESET}")
    tag_str = " ".join(tags)

    print("\n" + "╔" + "═"*(WIDTH-2) + "╗")
    print("║" + "🛡️  CYBERGUARD: ADAPTIVE INTELLIGENCE REPORT".center(WIDTH-2) + "║")
    print("╠" + "═"*(WIDTH-2) + "╣")
    
    # Updated Status Line with Tags
    status_line = f" STATUS: {result['final_risk']} {tag_str} (Confindence Score: {result['confidence']})"
    # Adjusted for ANSI color length
    print(f"║{status_line.ljust(WIDTH-2 + (len(tag_str) - tag_str.count('[')))}║") 
    
    print("╠" + "═"*(WIDTH-2) + "╣")




    print(f"║ 🔍 THREAT ANALYSIS:".ljust(WIDTH-2) + " ║")
    insight_sections = result['insights'].replace("**", "").split('\n')
    for line in insight_sections:
        if line.strip():
            wrapped = textwrap.wrap(line, width=WIDTH-10)
            for w_line in wrapped:
                print(f"║    {w_line.ljust(WIDTH-8)} ║")
        else:
            print(f"║".ljust(WIDTH-1) + "║")
    print("╚" + "═"*(WIDTH-2) + "╝\n")

    # --- AI-GATED FEEDBACK LOOP ---
    # 1. FIRST FEEDBACK (For Model Improvement)
    feedback = input("🤔 Is this prediction accurate? (y/n): ").lower()
    if feedback == 'n':
        user_reason = input("💬 Why is it wrong? (Provide technical reason): ")
        print("\n🤖 AI Moderator is checking your claim...")
        
        mod_prompt = (
            f"As a Security Expert, resolve this conflict:\n"
            f"Code: {test_code}\nModel Risk: {result['final_risk']}\nUser Argument: {user_reason}\n"
            f"If user is right, reply 'VERIFIED'. If wrong, reply 'REJECTED'. Provide a 1-sentence reason."
        )
        
        verdict = ollama.chat(model='phi3', messages=[{'role': 'user', 'content': mod_prompt}])
        v_text = verdict['message']['content']
        print(f"Verdict: {v_text}")

        if "VERIFIED" in v_text.upper():
            with open("feedback_log.csv", "a") as f:
                f.write(f"'{test_code}',{user_reason},VERIFIED\n")
            print("✅ Verified. Data saved for retraining.")
        else:
            print("🚫 Rejected. Data not saved to prevent poisoning.")
    
    
    # 2. NOW INTERACTIVE CHAT
    print("\n" + "─"*WIDTH)
    print(" 💡 CyberGuard Chatbot is online. Type 'exit' to quit.")
    while True:
        query = input("\n 💬 Ask me anything about this threat: ")
        if query.lower() == 'exit':
            break
        
        # Call the chat_with_guard function (ensure it exists in your code)
        answer = chat_with_guard(query, test_code, result['score'])
        print(f"\n 🛡️  {textwrap.fill(answer, width=70)}")
        print("─"*40)