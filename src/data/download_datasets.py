# =============================================================================
# src/data/download_datasets.py — Download All Raw Datasets
# =============================================================================
# This script downloads the three public datasets used by CloudGuard:
#
#   1. Azure Policy Definitions  — from Microsoft's official GitHub repo
#   2. CloudSploit Scan Results  — from Aqua Security's GitHub repo
#   3. NVD CVE Records           — from NIST's public API (Azure-related CVEs)
#
# Run this script ONCE before anything else:
#   python src/data/download_datasets.py
#
# Downloaded files are saved to data/raw/ and never modified.
# =============================================================================

import os
import sys
import json
import time
import requests
from tqdm import tqdm   # Progress bars

# Add the project root to Python's path so we can import config and helpers
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.utils.helpers import setup_logger, ensure_directories, check_file_exists

# Set up logging for this script
logger = setup_logger(__name__)


# =============================================================================
# DATASET 1: Azure Policy Definitions
# =============================================================================

def download_azure_policy_definitions() -> None:
    """
    Downloads Azure Policy built-in definitions from Microsoft's GitHub repository.

    What this downloads:
        Every built-in Azure Policy definition (JSON files) from:
        https://github.com/Azure/azure-policy/tree/master/built-in-policies

    Why we need it:
        These definitions tell us:
        - What resource types each policy covers
        - Whether the policy uses Deny, Audit, or Disabled effect
        - What security control the policy enforces

    The GitHub API returns file listings as JSON; we then fetch each .json file.
    We save everything merged into one file: data/raw/azure_policy_definitions.json
    """
    logger.info("=" * 60)
    logger.info("Downloading Azure Policy Definitions from GitHub...")
    logger.info("=" * 60)

    # Skip if already downloaded
    if check_file_exists(config.AZURE_POLICY_RAW, "Azure Policy definitions"):
        logger.info("Already downloaded. Delete the file to re-download.")
        return

    # GitHub API endpoint for the built-in policies folder
    # The API returns a list of files and folders in the directory
    api_url = "https://api.github.com/repos/Azure/azure-policy/git/trees/master?recursive=1"

    logger.info(f"Fetching file list from GitHub API...")
    logger.info("(This may take a minute — the repo has thousands of files)")

    try:
        response = requests.get(
            api_url,
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=60
        )
        response.raise_for_status()  # Raises an error if the request failed
        tree = response.json().get("tree", [])

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch GitHub file list: {e}")
        logger.error("Check your internet connection and try again.")
        return

    # Filter to only the built-in policy definition JSON files
    # These follow the pattern: built-in-policies/policyDefinitions/**/*.json
    policy_files = [
        item for item in tree
        if item["path"].startswith("built-in-policies/policyDefinitions/")
        and item["path"].endswith(".json")
        and item["type"] == "blob"
    ]

    logger.info(f"Found {len(policy_files):,} policy definition files")

    # Download each file and collect the JSON content
    all_policies = []
    base_url = "https://raw.githubusercontent.com/Azure/azure-policy/master/"

    # tqdm wraps our loop to show a progress bar in the terminal
    for item in tqdm(policy_files, desc="Downloading policies", unit="file"):
        try:
            file_url = base_url + item["path"]
            file_response = requests.get(file_url, timeout=30)
            file_response.raise_for_status()

            policy_json = file_response.json()
            all_policies.append(policy_json)

            # Be polite to GitHub's servers — don't hammer them with requests
            time.sleep(0.05)  # 50ms pause between requests

        except Exception as e:
            # Log the error but continue — we don't want one bad file to stop everything
            logger.warning(f"  Skipped {item['path']}: {e}")
            continue

    # Save all policies as a single JSON file
    logger.info(f"Saving {len(all_policies):,} policy definitions...")
    with open(config.AZURE_POLICY_RAW, "w", encoding="utf-8") as f:
        json.dump(all_policies, f, indent=2)

    logger.info(f"✓ Saved to {config.AZURE_POLICY_RAW}")


# =============================================================================
# DATASET 2: CloudSploit Scan Results
# =============================================================================

def download_cloudsploit_results() -> None:
    """
    Downloads the CloudSploit cloud security scanner plugin definitions.

    What this downloads:
        The CloudSploit plugins folder contains JSON-structured security checks
        for Azure services. Each plugin represents a security misconfiguration
        check with metadata about severity, resource type, and compliance mapping.

    Why we need it:
        CloudSploit scan results provide our labeled training data:
        - PASS = compliant resource deployment (label: 0)
        - FAIL (Critical/High severity) = non-compliant (label: 1)

    We download the plugin definitions and construct synthetic scan results
    based on the checks (since live scan results require Azure credentials).
    """
    logger.info("=" * 60)
    logger.info("Downloading CloudSploit Plugin Definitions from GitHub...")
    logger.info("=" * 60)

    if check_file_exists(config.CLOUDSPLOIT_RAW, "CloudSploit results"):
        logger.info("Already downloaded. Delete the file to re-download.")
        return

    # CloudSploit Azure plugins directory
    api_url = "https://api.github.com/repos/aquasecurity/cloudsploit/git/trees/master?recursive=1"

    logger.info("Fetching CloudSploit file list from GitHub...")

    try:
        response = requests.get(
            api_url,
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=60
        )
        response.raise_for_status()
        tree = response.json().get("tree", [])

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch CloudSploit file list: {e}")
        return

    # Filter to Azure plugin files only
    azure_plugins = [
        item for item in tree
        if "plugins/azure/" in item["path"]
        and item["path"].endswith(".js")
        and item["type"] == "blob"
    ]

    logger.info(f"Found {len(azure_plugins):,} Azure plugin files")

    base_url = "https://raw.githubusercontent.com/aquasecurity/cloudsploit/master/"
    all_plugins = []

    for item in tqdm(azure_plugins, desc="Downloading CloudSploit plugins", unit="file"):
        try:
            file_url = base_url + item["path"]
            file_response = requests.get(file_url, timeout=30)
            file_response.raise_for_status()

            # Extract metadata from the JavaScript file content
            # CloudSploit plugins follow a consistent structure we can parse
            content = file_response.text
            plugin_meta = _parse_cloudsploit_plugin(content, item["path"])
            all_plugins.append(plugin_meta)

            time.sleep(0.05)

        except Exception as e:
            logger.warning(f"  Skipped {item['path']}: {e}")
            continue

    logger.info(f"Saving {len(all_plugins):,} CloudSploit plugin definitions...")
    with open(config.CLOUDSPLOIT_RAW, "w", encoding="utf-8") as f:
        json.dump(all_plugins, f, indent=2)

    logger.info(f"✓ Saved to {config.CLOUDSPLOIT_RAW}")


def _parse_cloudsploit_plugin(js_content: str, filepath: str) -> dict:
    """
    Extracts metadata from a CloudSploit JavaScript plugin file.

    CloudSploit plugins are JavaScript files but follow a predictable structure.
    We extract the key fields we need for feature engineering.

    Args:
        js_content: The raw JavaScript file content as a string
        filepath:   The file path (used to determine Azure service category)

    Returns:
        A dictionary with the plugin's key metadata fields
    """
    import re

    # Helper to extract a field value from the JS file
    def extract_field(field_name, content):
        # Matches patterns like:  title: 'Some Title',
        pattern = rf"{field_name}:\s*['\"]([^'\"]+)['\"]"
        match = re.search(pattern, content)
        return match.group(1) if match else "Unknown"

    # Extract the Azure service category from the file path
    # e.g., "plugins/azure/storageAccounts/blobPublicAccess.js" → "storageAccounts"
    path_parts = filepath.split("/")
    service_category = path_parts[2] if len(path_parts) > 2 else "unknown"

    return {
        "filepath": filepath,
        "service_category": service_category,
        "title": extract_field("title", js_content),
        "category": extract_field("category", js_content),
        "severity": extract_field("severity", js_content),
        "description": extract_field("description", js_content),
    }


# =============================================================================
# DATASET 3: NVD CVE Records (Azure-related)
# =============================================================================

def download_nvd_cve_data() -> None:
    """
    Downloads Azure-related CVE records from the NIST National Vulnerability Database.

    What this downloads:
        CVE records where the description mentions Azure, filtered to 2019–2024.
        The NVD provides a free public REST API — no authentication required.

    Why we need it:
        CVE timestamps are used to calculate Deployment-to-Vulnerability Lag (DVL):
        DVL = time of first CVE detected - time of resource deployment

    API documentation: https://nvd.nist.gov/developers/vulnerabilities
    """
    logger.info("=" * 60)
    logger.info("Downloading NVD CVE Records (Azure, 2019-2024)...")
    logger.info("=" * 60)

    if check_file_exists(config.NVD_RAW, "NVD CVE data"):
        logger.info("Already downloaded. Delete the file to re-download.")
        return

    all_cves = []

    # The NVD API returns results in pages of 2000 records
    # We query for Azure-related CVEs year by year to stay within rate limits
    years = range(2019, 2025)  # 2019 through 2024

    for year in years:
        logger.info(f"  Fetching CVEs for {year}...")

        # NVD API parameters
        params = {
            "keywordSearch": "Azure",           # Only Azure-related CVEs
            "pubStartDate": f"{year}-01-01T00:00:00.000",
            "pubEndDate":   f"{year}-12-31T23:59:59.999",
            "resultsPerPage": 2000,
        }

        try:
            response = requests.get(
                "https://services.nvd.nist.gov/rest/json/cves/2.0",
                params=params,
                timeout=60
            )
            response.raise_for_status()
            data = response.json()

            cves = data.get("vulnerabilities", [])
            all_cves.extend(cves)
            logger.info(f"    Found {len(cves):,} CVEs for {year}")

            # NVD rate limit: max 5 requests per 30 seconds without API key
            # We wait 6 seconds between requests to be safe
            time.sleep(6)

        except Exception as e:
            logger.warning(f"  Failed to fetch {year} CVEs: {e}")
            time.sleep(6)
            continue

    logger.info(f"Total CVEs collected: {len(all_cves):,}")

    with open(config.NVD_RAW, "w", encoding="utf-8") as f:
        json.dump(all_cves, f, indent=2)

    logger.info(f"✓ Saved to {config.NVD_RAW}")


# =============================================================================
# MAIN — Run all downloads
# =============================================================================

if __name__ == "__main__":
    """
    When you run this file directly (python src/data/download_datasets.py),
    Python executes everything inside this if __name__ == "__main__" block.
    """

    logger.info("CloudGuard Dataset Downloader")
    logger.info("This will download ~3 datasets from public sources.")
    logger.info("Estimated time: 10-20 minutes depending on internet speed.\n")

    # Step 0: Make sure all directories exist before we try to write files
    ensure_directories()

    # Step 1: Azure Policy Definitions (largest download — be patient)
    download_azure_policy_definitions()

    # Step 2: CloudSploit Plugin Definitions
    download_cloudsploit_results()

    # Step 3: NVD CVE Records (slowest due to rate limiting)
    download_nvd_cve_data()

    logger.info("\n" + "=" * 60)
    logger.info("All downloads complete!")
    logger.info("Next step: python src/features/build_features.py")
    logger.info("=" * 60)
