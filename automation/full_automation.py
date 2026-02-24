"""
Consolidated Eviction Case Pipeline
=====================================
Phase 1: Extract case numbers from daily filing PDFs
Phase 2: Download case documents from LINX portal
Phase 3: OCR + LLM classification (residential/commercial/ejectment) + address extraction

Usage:
    python pipeline.py --week 21
    python pipeline.py --week 21 --start-phase 2   # resume from phase 2
    python pipeline.py --week 21 --start-phase 3   # resume from phase 3
"""

import sys
from pathlib import Path

# Project root: C:/dev/right-to-counsel-nudge-2
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import re
import os
import glob
import json
import time
import logging
import zipfile
import argparse

import pandas as pd
import pdfplumber
from tqdm import tqdm
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from config import (
    DATA_DIR,
    AZURE_CV_ENDPOINT,
    AZURE_CV_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_SYSTEM_PROMPT,
    LINX_PORTAL_URL,
    LINX_USERNAME,
    LINX_PASSWORD,
    SCRAPER_DELAY_MIN,
    SCRAPER_HEADLESS,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# PHASE 1 — Extract case numbers from daily filing PDFs
# ===========================================================================

class CaseExtractor:
    """Extracts unlawful detainer cases from daily filing PDFs."""

    EVICTION_KEYWORDS = ["unlawful detainer", "detainer"]

    def __init__(self, week_number: int):
        self.week_number = week_number
        self.pdf_directory = DATA_DIR / "daily_pdfs" / f"week_{week_number}"
        self.output_csv = DATA_DIR / f"week_{week_number}" / "weekly_cases.csv"

        self.pdf_directory.mkdir(parents=True, exist_ok=True)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        self.case_pattern = re.compile(r"\d{2}-\d-\d{5}-\d")

    def extract_cases_from_pdf(self, pdf_path: Path) -> list:
        cases = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)

            lines = text.split("\n")
            for i, line in enumerate(lines):
                case_match = self.case_pattern.search(line)
                if not case_match:
                    continue

                case_number = case_match.group()
                date_match = re.search(r"\d{2}/\d{2}/\d{4}", line)
                filing_date = date_match.group() if date_match else None
                case_type = line.split(filing_date)[-1].strip() if filing_date else ""
                if not case_type and i + 1 < len(lines):
                    case_type = lines[i + 1].strip()

                cases.append(
                    {
                        "case_number": case_number,
                        "filing_date": filing_date,
                        "case_type": case_type,
                        "source_file": pdf_path.name,
                    }
                )

            logger.info(f"[Phase 1] {pdf_path.name}: {len(cases)} cases found")
        except Exception as e:
            logger.error(f"[Phase 1] Error reading {pdf_path.name}: {e}")
        return cases

    def filter_eviction_cases(self, cases: list) -> list:
        eviction_cases = [
            c
            for c in cases
            if any(kw in c["case_type"].lower() for kw in self.EVICTION_KEYWORDS)
        ]
        logger.info(
            f"[Phase 1] {len(eviction_cases)} eviction cases out of {len(cases)} total"
        )
        return eviction_cases

    def run(self) -> list:
        """Run Phase 1. Returns list of eviction case dicts and saves CSV."""
        pdf_files = sorted(self.pdf_directory.glob("**/*.pdf"))
        if not pdf_files:
            logger.warning(f"[Phase 1] No PDFs found in {self.pdf_directory}")
            return []

        logger.info(f"[Phase 1] Processing {len(pdf_files)} PDF(s)")
        all_cases = []
        for pdf_path in pdf_files:
            all_cases.extend(self.extract_cases_from_pdf(pdf_path))

        eviction_cases = self.filter_eviction_cases(all_cases)

        df = pd.DataFrame(eviction_cases)
        df.to_csv(self.output_csv, index=False)
        logger.info(f"[Phase 1] Saved {len(eviction_cases)} cases → {self.output_csv}")
        return eviction_cases


# ===========================================================================
# PHASE 2 — Download case documents from LINX portal
# ===========================================================================

class LinxScraper:
    """Downloads summons and complaint PDFs from Pierce County LINX portal."""

    def __init__(self, week_number: int):
        self.week_number = week_number
        self.week_dir = DATA_DIR / f"week_{week_number}"
        self.case_documents_dir = self.week_dir / "case_documents"
        self.weekly_cases_csv = self.week_dir / "weekly_cases.csv"

        self.case_documents_dir.mkdir(parents=True, exist_ok=True)

        self.driver = None
        self.download_dir = str(self.case_documents_dir.absolute())

    # ------------------------------------------------------------------
    # Browser helpers
    # ------------------------------------------------------------------

    def _start_driver(self) -> bool:
        try:
            opts = Options()
            prefs = {
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "safebrowsing.enabled": False,
            }
            opts.add_experimental_option("prefs", prefs)
            if SCRAPER_HEADLESS:
                opts.add_argument("--headless")

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=opts)
            logger.info("[Phase 2] Chrome WebDriver started")
            return True
        except Exception as e:
            logger.error(f"[Phase 2] WebDriver init failed: {e}")
            return False

    def _login(self) -> bool:
        try:
            self.driver.get(f"{LINX_PORTAL_URL}/linxweb/Account/Logon.cfm")
            self.driver.find_element(By.NAME, "account_num").send_keys(LINX_USERNAME)
            self.driver.find_element(By.NAME, "pin").send_keys(LINX_PASSWORD)
            self.driver.find_element(By.CSS_SELECTOR, 'input[type="Submit"]').click()
            time.sleep(3)
            logger.info("[Phase 2] Logged in to LINX")
            return True
        except Exception as e:
            logger.error(f"[Phase 2] Login failed: {e}")
            return False

    def _latest_zip(self, max_age_secs: int = 60) -> str | None:
        files = sorted(
            glob.glob(os.path.join(self.download_dir, "*.zip")),
            key=os.path.getmtime,
            reverse=True,
        )
        if files and (time.time() - os.path.getmtime(files[0])) < max_age_secs:
            return files[0]
        return None

    # ------------------------------------------------------------------
    # Download + extraction
    # ------------------------------------------------------------------

    def _download_case(self, case_number: str):
        """Navigate to case page and trigger zip download. Returns zip path or None."""
        try:
            self.driver.get(
                f"{LINX_PORTAL_URL}/linxweb/Case/CivilCase.cfm?cause_num={case_number}"
            )
            time.sleep(3)
            self.driver.find_element(By.LINK_TEXT, "download filings").click()
            time.sleep(2)
            self.driver.find_element(By.LINK_TEXT, "select all").click()
            time.sleep(2)
            self.driver.find_element(By.NAME, "btnDownload").click()
            time.sleep(3)
        except Exception as e:
            logger.warning(f"[Phase 2] Could not trigger download for {case_number}: {e}")
            return None

        # Poll for completed download (up to 5 min)
        for _ in range(3000):
            recent = self._latest_zip()
            if recent:
                return recent
            time.sleep(0.1)

        logger.warning(f"[Phase 2] Download timeout: {case_number}")
        return None

    def _extract_documents(self, zip_path: str, case_number: str) -> bool:
        """Unpack summons + complaint PDFs from downloaded zip."""
        case_dir = self.case_documents_dir / case_number
        case_dir.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                file_list = zf.namelist()
                html_files = [f for f in file_list if f.endswith(".htm")]
                if not html_files:
                    logger.warning(f"[Phase 2] No HTML index in zip for {case_number}")
                    return False

                html_content = zf.read(html_files[0]).decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html_content, "html.parser")

                summons_file = complaint_file = None
                for link in soup.find_all("a"):
                    href = link.get("href", "")
                    text = link.get_text().strip().lower()
                    if ".pdf" not in href.lower():
                        continue
                    if "summons" in text:
                        summons_file = href
                    if "complaint" in text:
                        complaint_file = href

                for label, fname in [("summons", summons_file), ("complaint", complaint_file)]:
                    if fname and fname in file_list:
                        out = case_dir / f"{label}.pdf"
                        out.write_bytes(zf.read(fname))
                        logger.info(f"[Phase 2] Extracted {label} → {out}")

                return bool(summons_file or complaint_file)
        except Exception as e:
            logger.error(f"[Phase 2] Extraction error for {case_number}: {e}")
            return False
        finally:
            try:
                os.remove(zip_path)
            except Exception:
                pass

    def _already_downloaded(self) -> set:
        downloaded = set()
        for case_dir in self.case_documents_dir.iterdir():
            if case_dir.is_dir():
                if (case_dir / "summons.pdf").exists() or (case_dir / "complaint.pdf").exists():
                    downloaded.add(case_dir.name)
        return downloaded

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self):
        """Run Phase 2."""
        if not self.weekly_cases_csv.exists():
            logger.error(f"[Phase 2] {self.weekly_cases_csv} not found — run Phase 1 first")
            return

        df = pd.read_csv(self.weekly_cases_csv)
        case_numbers = df["case_number"].astype(str).tolist()

        already = self._already_downloaded()
        to_process = [c for c in case_numbers if c not in already]
        logger.info(
            f"[Phase 2] {len(to_process)} cases to download "
            f"({len(already)} already present)"
        )

        if not to_process:
            logger.info("[Phase 2] Nothing to download — skipping")
            return

        if not self._start_driver():
            return
        if not self._login():
            self.driver.quit()
            return

        failed = []
        try:
            for i, case_number in enumerate(to_process, 1):
                logger.info(f"[Phase 2] {i}/{len(to_process)}: {case_number}")
                zip_file = self._download_case(case_number)
                if zip_file:
                    if not self._extract_documents(zip_file, case_number):
                        failed.append(case_number)
                else:
                    failed.append(case_number)
                time.sleep(SCRAPER_DELAY_MIN)
        finally:
            self.driver.quit()

        logger.info(
            f"[Phase 2] Done. Success: {len(to_process) - len(failed)}, "
            f"Failed: {len(failed)}"
        )
        if failed:
            logger.info(f"[Phase 2] Failed cases: {failed[:10]}")


# ===========================================================================
# PHASE 3 — OCR + LLM: classify + extract address
# ===========================================================================

LLM_USER_PROMPT_TEMPLATE = """Analyze this unlawful detainer document and return a JSON object with:

1. **classification** — one of:
   - "RESIDENTIAL": standard residential tenant eviction
   - "COMMERCIAL": business/commercial property eviction
   - "EJECTMENT": non-tenant occupant (squatter, post-foreclosure, former owner, etc.)

2. **address** — the full property address at issue in the case (street, city, state, zip if present).
   If no address is found, use null.

3. **confidence** — "HIGH", "MEDIUM", or "LOW"

4. **reasoning** — one-sentence explanation

Document text:
{document_text}

Respond with ONLY a JSON object in this exact format:
{{"classification": "RESIDENTIAL"|"COMMERCIAL"|"EJECTMENT", "address": "123 Main St, Tacoma, WA 98401" or null, "confidence": "HIGH"|"MEDIUM"|"LOW", "reasoning": "brief explanation"}}"""


def _normalize_address(address: str | None) -> str | None:
    """Lowercase + collapse whitespace for fuzzy address comparison."""
    if not address:
        return None
    return re.sub(r"\s+", " ", address.strip().lower())


class CaseClassifier:
    """
    OCR + GPT-4o classification and address extraction.

    Each available document (complaint, summons) is processed independently.
    Results are then compared and conflicts flagged in the output CSVs.

    Output columns added to the case CSV:
        classification          — agreed value, or null if conflict
        address                 — agreed value, or null if conflict
        confidence              — from complaint (primary), or summons if complaint absent
        classification_conflict — True if complaint/summons disagree
        address_conflict        — True if complaint/summons disagree
        complaint_classification, complaint_address, complaint_confidence
        summons_classification,  summons_address,  summons_confidence
        conflict_notes          — human-readable description of any conflicts
    """

    MAX_CHARS = 10_000
    VALID_CLASSIFICATIONS = {"RESIDENTIAL", "COMMERCIAL", "EJECTMENT"}
    # Same pattern used in Phase 1 — Pierce County format: ##-#-#####-#
    case_pattern = re.compile(r"\d{2}-\d-\d{5}-\d")

    def __init__(self, week_number: int):
        self.week_number = week_number
        self.week_dir = DATA_DIR / f"week_{week_number}"
        self.case_documents_dir = self.week_dir / "case_documents"
        self.weekly_cases_csv = self.week_dir / "weekly_cases.csv"
        self.outputs_dir = self.week_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        self.cv_client = DocumentAnalysisClient(
            endpoint=AZURE_CV_ENDPOINT,
            credential=AzureKeyCredential(AZURE_CV_KEY),
        )
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
        )
        logger.info("[Phase 3] Azure clients initialized")

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    def _ocr_pdf(self, pdf_path: Path) -> str:
        """Extract text from a PDF via Azure Document Intelligence."""
        try:
            pdf_bytes = pdf_path.read_bytes()
            poller = self.cv_client.begin_analyze_document("prebuilt-read", document=pdf_bytes)
            result = poller.result()
            logger.debug(
                f"[Phase 3] OCR: {pdf_path.name} → {len(result.content)} chars, "
                f"{len(result.pages)} page(s)"
            )
            return result.content
        except Exception as e:
            logger.error(f"[Phase 3] OCR failed for {pdf_path.name}: {e}")
            return ""

    def _available_docs(self, case_number: str) -> dict[str, Path]:
        """
        Return a dict of doc_type → Path for documents that exist on disk.
        Deduplicates by inode so a single combined file isn't OCR'd twice.
        Returns e.g. {"complaint": Path(...)} or {"complaint": ..., "summons": ...}
        """
        case_dir = self.case_documents_dir / case_number
        candidates = {
            "complaint": case_dir / "complaint.pdf",
            "summons": case_dir / "summons.pdf",
        }
        seen_inodes = set()
        docs = {}
        for doc_type, path in candidates.items():
            if not path.exists():
                continue
            inode = path.stat().st_ino
            if inode in seen_inodes:
                logger.debug(
                    f"[Phase 3] {case_number}: {path.name} is same file as a previous doc — skipping duplicate"
                )
                continue
            seen_inodes.add(inode)
            docs[doc_type] = path
        return docs

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _call_llm(self, text: str, case_number: str, doc_type: str) -> dict | None:
        """Call GPT-4o on a single document's text. Returns parsed dict or None."""
        if len(text) > self.MAX_CHARS:
            text = text[: self.MAX_CHARS] + "\n[... text truncated ...]"

        prompt = LLM_USER_PROMPT_TEMPLATE.format(document_text=text)
        try:
            response = self.openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if raw.startswith("```"):
                parts = raw.split("```")
                raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

            result = json.loads(raw)
            classification = result.get("classification", "").upper()
            if classification not in self.VALID_CLASSIFICATIONS:
                raise ValueError(f"Unexpected classification: {classification}")

            result["classification"] = classification
            logger.info(
                f"[Phase 3] {case_number}/{doc_type}: {classification} "
                f"({result.get('confidence', '?')}) | address: {result.get('address', 'n/a')}"
            )
            return result

        except json.JSONDecodeError as e:
            logger.error(f"[Phase 3] JSON parse error ({case_number}/{doc_type}): {e}")
        except Exception as e:
            logger.error(f"[Phase 3] LLM error ({case_number}/{doc_type}): {e}")
        return None

    # ------------------------------------------------------------------
    # Per-case processing
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Case number verification
    # ------------------------------------------------------------------

    def _verify_case_number(self, case_number: str, text: str, doc_type: str) -> tuple[bool, list[str]]:
        """
        Check that the folder case number appears in the OCR'd document text.

        The case number format is ##-#-#####-# (e.g. 24-2-01234-1).
        We also search for common formatting variants found in court documents:
          - Exact match:       24-2-01234-1
          - No leading zero:   24-2-1234-1
          - Spaces/dots:       24 2 01234 1 / 24.2.01234.1

        Returns:
            (verified: bool, found_numbers: list[str])
            verified        — True if the folder case number was found in the text
            found_numbers   — all case numbers detected in the document (for mismatch logging)
        """
        # Build flexible regex variants of the folder case number
        parts = case_number.split("-")  # ['24', '2', '01234', '1']

        # Find ALL case-number-shaped strings in the document
        found_numbers = self.case_pattern.findall(text)

        # Normalise both sides: strip leading zeros from the sequence segment
        # so "01234" == "1234" comparisons work
        def normalise(cn: str) -> str:
            p = cn.split("-")
            if len(p) == 4:
                p[2] = str(int(p[2]))  # remove leading zeros
            return "-".join(p)

        normalised_folder = normalise(case_number)
        normalised_found = [normalise(n) for n in found_numbers]

        verified = normalised_folder in normalised_found

        if not verified:
            if found_numbers:
                logger.warning(
                    f"[Phase 3] Case number mismatch in {case_number}/{doc_type}: "
                    f"folder={case_number}, found in doc={found_numbers}"
                )
            else:
                logger.warning(
                    f"[Phase 3] No case number found in {case_number}/{doc_type} document text"
                )

        return verified, found_numbers

    def _process_case(self, case_number: str) -> dict | None:
        """
        OCR and classify each available document independently.
        Verify case numbers match folder name, then compare results and surface conflicts.
        """
        docs = self._available_docs(case_number)
        if not docs:
            logger.warning(f"[Phase 3] No documents found for {case_number}")
            return None

        # OCR + verify case number + LLM each document
        doc_results: dict[str, dict] = {}
        case_number_verified: dict[str, bool] = {}
        case_numbers_found_in_docs: dict[str, list[str]] = {}

        for doc_type, path in docs.items():
            text = self._ocr_pdf(path)
            if not text or len(text) < 100:
                logger.warning(f"[Phase 3] Insufficient OCR text: {case_number}/{doc_type}")
                continue

            # Verify case number before classifying
            verified, found_numbers = self._verify_case_number(case_number, text, doc_type)
            case_number_verified[doc_type] = verified
            case_numbers_found_in_docs[doc_type] = found_numbers

            result = self._call_llm(text, case_number, doc_type)
            if result:
                doc_results[doc_type] = result
            time.sleep(0.3)  # small gap between LLM calls for the same case

        if not doc_results:
            return None

        # ------------------------------------------------------------------
        # Case number verification summary
        # ------------------------------------------------------------------
        any_unverified = not all(case_number_verified.values())
        # Collect any unexpected case numbers seen across all docs
        all_found = {
            n
            for nums in case_numbers_found_in_docs.values()
            for n in nums
            if n != case_number
        }
        case_number_mismatch = any_unverified  # True if folder number absent from any doc

        # ------------------------------------------------------------------
        # Conflict detection
        # ------------------------------------------------------------------
        classifications = {dt: r["classification"] for dt, r in doc_results.items()}
        addresses = {dt: _normalize_address(r.get("address")) for dt, r in doc_results.items()}

        unique_classifications = set(classifications.values())
        unique_addresses = set(v for v in addresses.values() if v is not None)

        classification_conflict = len(unique_classifications) > 1
        # Address conflict: both docs have an address AND they differ
        address_conflict = (
            len(doc_results) > 1
            and all(a is not None for a in addresses.values())
            and len(unique_addresses) > 1
        )

        # Build conflict notes
        conflict_notes_parts = []
        if case_number_mismatch:
            unverified_docs = [dt for dt, v in case_number_verified.items() if not v]
            found_str = ", ".join(sorted(all_found)) if all_found else "none"
            conflict_notes_parts.append(
                f"case number not found in doc(s): {unverified_docs}; "
                f"case numbers seen in docs: [{found_str}]"
            )
        if classification_conflict:
            detail = ", ".join(f"{dt}={v}" for dt, v in classifications.items())
            conflict_notes_parts.append(f"classification conflict ({detail})")
        if address_conflict:
            detail = ", ".join(
                f"{dt}={doc_results[dt].get('address', 'null')}" for dt in doc_results
            )
            conflict_notes_parts.append(f"address conflict ({detail})")
        conflict_notes = "; ".join(conflict_notes_parts) if conflict_notes_parts else None

        # Agreed values (null if conflict)
        agreed_classification = None if classification_conflict else unique_classifications.pop()
        agreed_address = None if address_conflict else (
            # If only one doc has an address, use it
            next(
                (doc_results[dt].get("address") for dt in ("complaint", "summons") if dt in doc_results and doc_results[dt].get("address")),
                None,
            )
        )

        # Primary confidence comes from complaint; fall back to summons
        primary_doc = "complaint" if "complaint" in doc_results else next(iter(doc_results))
        primary_confidence = doc_results[primary_doc].get("confidence", "UNKNOWN")

        # Flatten per-doc fields for full transparency in output
        row = {
            "case_number": case_number,
            "classification": agreed_classification,
            "address": agreed_address,
            "confidence": primary_confidence,
            "case_number_mismatch": case_number_mismatch,
            "classification_conflict": classification_conflict,
            "address_conflict": address_conflict,
            "conflict_notes": conflict_notes,
        }

        for doc_type in ("complaint", "summons"):
            if doc_type in doc_results:
                r = doc_results[doc_type]
                row[f"{doc_type}_classification"] = r.get("classification")
                row[f"{doc_type}_address"] = r.get("address")
                row[f"{doc_type}_confidence"] = r.get("confidence")
                row[f"{doc_type}_reasoning"] = r.get("reasoning")
                row[f"{doc_type}_case_number_verified"] = case_number_verified.get(doc_type)
            else:
                row[f"{doc_type}_classification"] = None
                row[f"{doc_type}_address"] = None
                row[f"{doc_type}_confidence"] = None
                row[f"{doc_type}_reasoning"] = None
                row[f"{doc_type}_case_number_verified"] = None

        return row

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def run(self) -> pd.DataFrame | None:
        """Run Phase 3. Returns merged DataFrame and saves CSVs."""
        if not self.weekly_cases_csv.exists():
            logger.error(f"[Phase 3] {self.weekly_cases_csv} not found — run Phase 1 first")
            return None

        df = pd.read_csv(self.weekly_cases_csv)
        case_numbers = df["case_number"].astype(str).tolist()
        logger.info(f"[Phase 3] Classifying {len(case_numbers)} cases")

        results = []
        counts = {
            "RESIDENTIAL": 0, "COMMERCIAL": 0, "EJECTMENT": 0, "FAILED": 0,
            "case_number_mismatches": 0, "classification_conflicts": 0, "address_conflicts": 0,
        }

        for case_number in tqdm(case_numbers, desc="Classifying"):
            result = self._process_case(case_number)
            if result:
                results.append(result)
                if result["classification"]:
                    counts[result["classification"]] += 1
                if result["case_number_mismatch"]:
                    counts["case_number_mismatches"] += 1
                if result["classification_conflict"]:
                    counts["classification_conflicts"] += 1
                if result["address_conflict"]:
                    counts["address_conflicts"] += 1
            else:
                counts["FAILED"] += 1
            time.sleep(0.5)

        results_df = pd.DataFrame(results)
        final_df = df.merge(results_df, on="case_number", how="left")

        # Save all classified cases
        all_out = self.outputs_dir / "classified_cases.csv"
        final_df.to_csv(all_out, index=False)
        logger.info(f"[Phase 3] All results → {all_out}")

        # Save residential-only subset
        residential_df = final_df[final_df["classification"] == "RESIDENTIAL"]
        res_out = self.outputs_dir / "residential_cases.csv"
        residential_df.to_csv(res_out, index=False)
        logger.info(f"[Phase 3] Residential cases → {res_out}")

        # Save all flagged cases (case number mismatches + classification/address conflicts)
        conflict_mask = (
            final_df["case_number_mismatch"].fillna(False)
            | final_df["classification_conflict"].fillna(False)
            | final_df["address_conflict"].fillna(False)
        )
        conflicts_df = final_df[conflict_mask]
        if not conflicts_df.empty:
            conflict_out = self.outputs_dir / "conflict_cases.csv"
            conflicts_df.to_csv(conflict_out, index=False)
            logger.info(f"[Phase 3] Flagged cases → {conflict_out}")

        # Summary
        logger.info("=" * 60)
        logger.info("PHASE 3 COMPLETE")
        logger.info(f"  Residential              : {counts['RESIDENTIAL']}")
        logger.info(f"  Commercial               : {counts['COMMERCIAL']}")
        logger.info(f"  Ejectment                : {counts['EJECTMENT']}")
        logger.info(f"  Failed                   : {counts['FAILED']}")
        logger.info(f"  Case number mismatches   : {counts['case_number_mismatches']}")
        logger.info(f"  Classification conflicts  : {counts['classification_conflicts']}")
        logger.info(f"  Address conflicts         : {counts['address_conflicts']}")
        logger.info("=" * 60)

        return final_df


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Consolidated eviction case pipeline (phases 1–3)"
    )
    parser.add_argument("--week", type=int, required=True, help="Week number (e.g. 21)")
    parser.add_argument(
        "--start-phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Phase to start from (default: 1). Use 2 or 3 to resume mid-pipeline.",
    )
    args = parser.parse_args()

    week = args.week
    start = args.start_phase

    logger.info(f"{'=' * 60}")
    logger.info(f"PIPELINE START  |  week={week}  |  start-phase={start}")
    logger.info(f"{'=' * 60}")

    if start <= 1:
        logger.info(">>> PHASE 1: Extract case numbers from PDFs")
        CaseExtractor(week).run()

    if start <= 2:
        logger.info(">>> PHASE 2: Download documents from LINX")
        LinxScraper(week).run()

    if start <= 3:
        logger.info(">>> PHASE 3: OCR + classify + extract addresses")
        CaseClassifier(week).run()

    logger.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()