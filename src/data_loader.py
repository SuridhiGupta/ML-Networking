import os
import logging
import time
from pathlib import Path

import requests
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


NVD_ENDPOINT = "https://services.nvd.nist.gov/rest/json/cves/2.0"
MAX_PAGE_SIZE = 2000  # NVD max results per page


def _extract_cvss(cve_item):
    """Extract CVSSv3 score if available, else CVSSv2, else None."""
    metrics = cve_item.get("cve", {}).get("metrics", {})

    # CVSS v3
    cvss3 = metrics.get("cvssMetricV31") or metrics.get("cvssMetricV30")
    if cvss3:
        v = cvss3[0].get("cvssData", {}).get("baseScore")
        if v is not None:
            return float(v)

    # CVSS v2
    cvss2 = metrics.get("cvssMetricV2")
    if cvss2:
        v = cvss2[0].get("cvssData", {}).get("baseScore")
        if v is not None:
            return float(v)

    return None


def _extract_description(cve_item):
    descriptions = cve_item.get("cve", {}).get("descriptions", [])
    for desc in descriptions:
        if desc.get("lang") == "en" and isinstance(desc.get("value"), str):
            return desc.get("value")

    if descriptions and isinstance(descriptions[0].get("value"), str):
        return descriptions[0].get("value")

    return ""


def _risk_label_from_cvss(cvss_score):
    if cvss_score is None:
        return "Unknown"
    if cvss_score >= 9.0:
        return "Critical"
    if cvss_score >= 7.0:
        return "High"
    if cvss_score >= 4.0:
        return "Medium"
    return "Low"


def fetch_cve_data(page=0, results_per_page=1000, api_key=None):
    """Fetch one page of CVE data from NVD.

    Args:
        page (int): Zero-based page index.
        results_per_page (int): Number of results to fetch (max 2000).
        api_key (str): Optional NVD API key.

    Returns:
        pd.DataFrame: (cve_id, description, cvss_score, risk_level)
    """

    results_per_page = min(results_per_page, MAX_PAGE_SIZE)

    headers = {"User-Agent": "ML-Networking-CVE-Loader"}
    if api_key:
        headers["apiKey"] = api_key

    params = {
        "startIndex": page * results_per_page,
        "resultsPerPage": results_per_page,
    }

    logging.info("Fetching CVE page %s using %s resultsPerPage", page, results_per_page)

    try:
        response = requests.get(NVD_ENDPOINT, params=params, headers=headers, timeout=30)
    except requests.RequestException as exc:
        logging.error("Network error while fetching CVE data: %s", exc)
        return pd.DataFrame()

    if response.status_code != 200:
        logging.error("NVD API returned %s - %s", response.status_code, response.text[:200])
        return pd.DataFrame()

    try:
        data = response.json()
    except ValueError as exc:
        logging.error("Invalid JSON from NVD API: %s", exc)
        return pd.DataFrame()

    vulnerabilities = data.get("vulnerabilities", [])

    records = []
    for item in vulnerabilities:
        try:
            cve_id = item.get("cve", {}).get("id", "").strip()
            description = _extract_description(item).strip()
            cvss_score = _extract_cvss(item)
            risk = _risk_label_from_cvss(cvss_score)

            if not cve_id or not description:
                continue

            records.append({
                "cve_id": cve_id,
                "description": description,
                "cvss_score": cvss_score,
                "risk_level": risk,
            })

        except Exception as exc:
            logging.warning("Skipping item due to error: %s", exc)

    if not records:
        logging.warning("No records parsed from NVD result page %s", page)
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def save_cve_csv(df: pd.DataFrame, path: str):
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_csv(dst, index=False)
        logging.info("Saved CVE data to %s (%d rows)", dst, len(df))
    except Exception as exc:
        logging.error("Error writing CSV to %s: %s", dst, exc)
        raise


def collect_cve_data(total=3000, results_per_page=1000, api_key=None, target_path="data/raw/cve_data.csv"):
    pages = (total + results_per_page - 1) // results_per_page
    all_frames = []

    for p in range(pages):
        df_page = fetch_cve_data(page=p, results_per_page=results_per_page, api_key=api_key)
        if df_page.empty:
            logging.warning("Empty frame for page %s, stopping loop", p)
            break

        all_frames.append(df_page)
        time.sleep(1)  # polite pause for rate limiting

    if not all_frames:
        logging.error("No CVE data collected")
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined.drop_duplicates(subset=["cve_id"], inplace=True)

    save_cve_csv(combined, target_path)
    return combined


if __name__ == "__main__":
    api_key = os.environ.get("NVD_API_KEY")
    df = collect_cve_data(total=1000, results_per_page=1000, api_key=api_key)
    print(df.head())
