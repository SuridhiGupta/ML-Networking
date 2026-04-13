import os
import tempfile
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import requests


CODE_PATTERNS = [
    ("eval(", "eval() usage"),
    ("exec(", "exec() usage"),
    ("subprocess.popen(", "subprocess.Popen usage"),
    ("os.system(", "os.system usage"),
    ("os.popen(", "os.popen usage"),
    ("pickle.loads(", "pickle.loads usage"),
    ("sqlite3.", "possible SQL injection pattern"),
    ("input(", "input() call in scripts"),
    ("execfile(", "execfile usage"),
]


def _contains_risky_pattern(line: str, pattern: str) -> bool:
    line_lower = line.lower()

    if pattern == "shell=true":
        compact = line_lower.replace(" ", "")
        return compact.find("shell=true") != -1

    if pattern == "sql +":
        return line_lower.find("sql+") != -1 or line_lower.find("sql +") != -1

    if pattern == "open_write":
        has_open = line_lower.find("open(") != -1
        has_write_mode = line_lower.find("'w'") != -1 or line_lower.find('"w"') != -1
        return has_open and has_write_mode

    return line_lower.find(pattern) != -1


def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return text.replace("\r\n", "\n")


def scan_code_text(content: str, filename: str = "<text>") -> List[Dict]:
    content = _normalize_text(content)
    lines = content.split("\n")
    findings = []

    for i, line in enumerate(lines, start=1):
        for pattern, message in CODE_PATTERNS:
            if _contains_risky_pattern(line, pattern):
                findings.append({
                    "file": filename,
                    "line": i,
                    "issue": message,
                    "code": line.strip(),
                })

        # Special checks previously handled by regex
        if _contains_risky_pattern(line, "shell=true"):
            findings.append({
                "file": filename,
                "line": i,
                "issue": "subprocess shell=True usage",
                "code": line.strip(),
            })
        if _contains_risky_pattern(line, "sql +"):
            findings.append({
                "file": filename,
                "line": i,
                "issue": "possible SQL injection pattern",
                "code": line.strip(),
            })
        if _contains_risky_pattern(line, "open_write"):
            findings.append({
                "file": filename,
                "line": i,
                "issue": "file write path usage",
                "code": line.strip(),
            })

    return findings


def score_risk_from_findings(findings: List[Dict]) -> str:
    count = len(findings)
    if count >= 5:
        return "High"
    if count >= 2:
        return "Medium"
    if count == 1:
        return "Low"
    return "None"


def is_github_repo_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if "github.com" not in parsed.netloc.lower():
        return False
    return True


def _download_github_repo_zip(repo_url: str, target_dir: str) -> str:
    parsed = urlparse(repo_url)
    parts = parsed.path.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repository URL")

    owner, repo = parts[0], parts[1]
    possible_branches = []

    if len(parts) >= 4 and parts[2] in ("tree", "blob"):
        possible_branches.append(parts[3])

    possible_branches += ["main", "master", "develop"]

    last_error = None
    for branch in possible_branches:
        zip_url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        try:
            # Stream download to disk instead of memory
            with requests.get(zip_url, timeout=30, stream=True) as resp:
                if resp.status_code == 200:
                    target_zip = os.path.join(target_dir, f"{repo}.zip")
                    with open(target_zip, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    return target_zip
        except requests.RequestException as exc:
            last_error = exc

    if last_error:
        raise RuntimeError(f"Unable to download repo archive: {last_error}")

    raise RuntimeError("Unable to download repository archive. Branches tried: " + ",".join(possible_branches))


def scan_github_repository(repo_url: str) -> Dict:
    if not is_github_repo_url(repo_url):
        raise ValueError("Invalid GitHub repository URL")

    tmp = tempfile.mkdtemp(prefix="cve_repo_scan_")
    try:
        archive_path = _download_github_repo_zip(repo_url, tmp)
        scan_results = []

        with zipfile.ZipFile(archive_path, "r") as zipf:
            zipf.extractall(tmp)

        base_dir = None
        for name in os.listdir(tmp):
            candidate = os.path.join(tmp, name)
            if os.path.isdir(candidate):
                base_dir = candidate
                break

        if base_dir is None:
            raise ValueError("Unable to detect repository root after extraction")

        # gather files incrementally with caps to prevent memory issues
        files_scanned = 0
        MAX_FILES = 100  # Cap at 100 files
        MAX_TOTAL_FINDINGS = 500
        
        for root, _, files in os.walk(base_dir):
            if files_scanned >= MAX_FILES:
                break
                
            for file_name in files:
                if len(scan_results) >= MAX_TOTAL_FINDINGS:
                    break
                    
                if file_name.lower().endswith((".py", ".js", ".java", ".c", ".cpp", ".go", ".sh", ".txt")):
                    file_path = os.path.join(root, file_name)
                    try:
                        files_scanned += 1
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            code = f.read()
                    except Exception:
                        continue
                        
                    findings = scan_code_text(code, filename=os.path.relpath(file_path, base_dir))
                    scan_results.extend(findings)
                    
                if len(scan_results) >= MAX_TOTAL_FINDINGS:
                    break

        risk_level = score_risk_from_findings(scan_results)

        return {
            "repo_url": repo_url,
            "total_findings": len(scan_results),
            "risk_level": risk_level,
            "findings": scan_results,
        }

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def scan_uploaded_file(file_buffer, filename: str) -> Dict:
    content_bytes = file_buffer.read()
    if isinstance(content_bytes, bytes):
        try:
            text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = content_bytes.decode("latin-1", errors="ignore")
    else:
        text = str(content_bytes)

    findings = scan_code_text(text, filename=filename)
    risk_level = score_risk_from_findings(findings)

    return {
        "file": filename,
        "total_findings": len(findings),
        "risk_level": risk_level,
        "findings": findings,
    }
