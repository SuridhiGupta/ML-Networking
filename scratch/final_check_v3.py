import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path.cwd() / "src"))

# Mock global model cache bits since we just want to test heuristics logic
import server
server.GLOBAL_BOOSTER = None # Disable ML for this test to focus on heuristics
server.GLOBAL_VECTORIZER = None

print(f"🔍 Testing CyberGuard {server.APP_VERSION}")

test_cases = [
    ('import os\nos.system(input())', "Critical Injection"),
    ('os.system("dir")', "Standard OS Command"),
    ('eval(x)', "Dynamic Eval"),
    ('print("hello")', "Safe")
]

for code, desc in test_cases:
    score, risk, flagged, highlighted, zday, zreason, conf, boosted = server.compute_risk(code)
    print(f"\n[{desc}]")
    print(f"  Code: {code.replace('\n', ' ')}")
    print(f"  Version: {server.APP_VERSION}")
    print(f"  Score: {score:.2f}")
    print(f"  Risk: {risk}")
    
    if "Injection" in desc and score < 9.0:
        print("  ❌ FAILURE: Injection not scored high enough")
    elif "Safe" in desc and score > 2.0:
        print("  ❌ FAILURE: Safe code scored too high")
    else:
        print("  ✅ PASS")
