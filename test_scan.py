import requests
import json

url = "http://localhost:8000/scan"

# Test with a vulnerable code snippet
test_code = """import os
import socket
s = socket.socket()
os.system('rm -rf /')"""

payload = {"code": test_code}

try:
    response = requests.post(url, json=payload)
    print("=== CYBERGUARD SCAN RESULT ===")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"Error: {e}")