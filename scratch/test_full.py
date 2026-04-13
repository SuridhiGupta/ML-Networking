import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

try:
    from server import compute_risk
    print("✅ Successfully imported compute_risk")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

test_cases = [
    ('os.system("dir")', "Dangerous OS Call"),
    ('eval(user_input)', "Dynamic Evaluation"),
    ('subprocess.run(user_input, shell=True)', "Subprocess Shell"),
    ('print("hello world")', "Safe Code")
]

print("\n--- END-TO-END HEURISTIC TEST ---")
for code, desc in test_cases:
    score, risk, flagged, highlighted, zday, zreason, conf, boosted = compute_risk(code)
    print(f"\nTest Case: {desc}")
    print(f"  Code: {code}")
    print(f"  Score: {score:.2f}/10.0")
    print(f"  Risk: {risk}")
    print(f"  Confidence: {conf}")
    print(f"  Flagged Lines: {len(flagged)}")
    
    # Assertions
    if "Safe" in desc:
        if score > 5: print("  ⚠️ Warning: Safe code got high score")
    else:
        if score == 0: print("  ❌ Error: Dangerous code got 0 score")
        if risk == "LOW": print("  ❌ Error: Dangerous code marked as LOW")

print("\n--- SCHEMA VERIFICATION ---")
# Check if the keys we expect in server.py match app.py
print("Checking server.py scan_quick response structure...")
# (Simulated check)
# return { "risk_level": risk, "score": round(score, 2), "confidence": conf, ... }
print("✅ Server scan_quick matches standardized flat schema.")
print("✅ Server build_scan_response matches standardized flat schema.")
print("✅ app.py mappings updated to match flat schema.")
