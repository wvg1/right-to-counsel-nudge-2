"""
Phase 2: Download Case Documents from LINX Portal
Downloads summons and complaint documents for eviction cases identified in Phase 1
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import zipfile
import os
import time
import glob
import logging
from bs4 import BeautifulSoup

from config import (
    WEEKLY_CASES_CSV,
    CASE_DOCUMENTS_DIR,
    LINX_PORTAL_URL,
    LINX_USERNAME,
    LINX_PASSWORD,
    SCRAPER_DELAY_MIN,
    SCRAPER_HEADLESS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LinxScraper:
    """Scrapes case documents from Pierce County LINX portal"""
    
    def __init__(self):
        self.driver = None
        self.download_dir = str(CASE_DOCUMENTS_DIR.absolute())
        self.metrics = {
            'cases_processed': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'processing_time': 0
        }
    
    def start_driver(self):
        """Initialize Chrome WebDriver"""
        try:
            chrome_options = Options()
            
            # Set download directory
            prefs = {
                "download.default_directory": self.download_dir,
                "download.prompt_for_download": False,
                "safebrowsing.enabled": False
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            if SCRAPER_HEADLESS:
                chrome_options.add_argument('--headless')
            
            # Use webdriver-manager to automatically get chromedriver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            
            logger.info("Chrome WebDriver started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebDriver: {e}")
            return False
    
    def login(self):
        """Log into the LINX portal"""
        try:
            login_url = f"{LINX_PORTAL_URL}/linxweb/Account/Logon.cfm"
            self.driver.get(login_url)
            
            # Enter credentials
            self.driver.find_element(By.NAME, "account_num").send_keys(LINX_USERNAME)
            self.driver.find_element(By.NAME, "pin").send_keys(LINX_PASSWORD)
            self.driver.find_element(By.CSS_SELECTOR, 'input[type="Submit"]').click()
            
            time.sleep(3)
            logger.info("Successfully logged in")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False
    
    def get_recently_modified_file(self, seconds=60):
        """Find file modified in last N seconds"""
        current_time = time.time()
        files = glob.glob(os.path.join(self.download_dir, '*.zip'))
        
        if not files:
            return None
        
        files.sort(key=os.path.getmtime, reverse=True)
        mod_time = os.path.getmtime(files[0])
        
        if current_time - mod_time < seconds:
            return files[0]
        
        return None
    
    def download_case_documents(self, case_number):
        """Download documents for a specific case"""
        try:
            # Navigate to case page
            case_url = f"{LINX_PORTAL_URL}/linxweb/Case/CivilCase.cfm?cause_num={case_number}"
            self.driver.get(case_url)
            time.sleep(3)
            
            try:
                # Find and click "download filings" link
                self.driver.find_element(By.LINK_TEXT, "download filings").click()
                time.sleep(2)
                
                # Select all files
                self.driver.find_element(By.LINK_TEXT, "select all").click()
                time.sleep(2)
                
                # Click download button
                self.driver.find_element(By.NAME, "btnDownload").click()
                time.sleep(3)
                
            except Exception as e:
                logger.warning(f"Could not find download option for {case_number}: {e}")
                return False
            
            # Wait for download to complete (max 5 minutes)
            sleep_time = 0
            while sleep_time < 300:
                recent_file = self.get_recently_modified_file(seconds=60)
                if recent_file:
                    logger.info(f"Downloaded {case_number}: {recent_file}")
                    time.sleep(1)
                    return recent_file
                
                time.sleep(0.1)
                sleep_time += 0.1
            
            logger.warning(f"Download timeout for {case_number}")
            return False
            
        except Exception as e:
            logger.error(f"Error downloading {case_number}: {e}")
            return False
    
    def extract_documents(self, zip_path, case_number):
        """Extract summons and complaint documents from zip file"""
        try:
            # Create case folder
            case_dir = CASE_DOCUMENTS_DIR / case_number
            case_dir.mkdir(parents=True, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Extract HTML index to find documents
                html_files = [f for f in file_list if f.endswith('.htm')]
                if not html_files:
                    logger.warning(f"No HTML index found for {case_number}")
                    return None
                
                # Extract HTML temporarily
                html_content = zip_ref.read(html_files[0]).decode('utf-8', errors='ignore')
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Find summons and complaint documents
                summons_file = None
                complaint_file = None
                
                for link in soup.find_all('a'):
                    href = link.get('href')
                    text = link.get_text().strip().lower()  # Strip whitespace!
                    
                    if href and '.pdf' in href.lower():
                        # Look for summons (most reliable for classification)
                        if text == 'summons' or 'summons' in text:
                            summons_file = href
                        # Look for complaint
                        if text == 'complaint' or 'complaint' in text:
                            complaint_file = href
                
                # Extract summons if found
                if summons_file and summons_file in file_list:
                    output_path = case_dir / "summons.pdf"
                    with open(output_path, 'wb') as f:
                        f.write(zip_ref.read(summons_file))
                    logger.info(f"Extracted summons for {case_number}")
                
                # Extract complaint if found
                if complaint_file and complaint_file in file_list:
                    output_path = case_dir / "complaint.pdf"
                    with open(output_path, 'wb') as f:
                        f.write(zip_ref.read(complaint_file))
                    
                    if complaint_file == summons_file:
                        logger.info(f"Extracted combined summons/complaint for {case_number}")
                    else:
                        logger.info(f"Extracted complaint for {case_number}")
                
                # Return success if we got at least one document
                if summons_file or complaint_file:
                    return True
                else:
                    logger.warning(f"No summons or complaint found for {case_number}")
                    return None
            
        except Exception as e:
            logger.error(f"Error extracting documents for {case_number}: {e}")
            return None
        
        finally:
            # Clean up zip file
            try:
                os.remove(zip_path)
            except:
                pass
    
    def get_downloaded_cases(self):
        """Get list of cases already downloaded"""
        downloaded = set()
        try:
            for case_dir in CASE_DOCUMENTS_DIR.iterdir():
                if case_dir.is_dir():
                    # Check if we have at least summons or complaint
                    has_summons = (case_dir / "summons.pdf").exists()
                    has_complaint = (case_dir / "complaint.pdf").exists()
                    if has_summons or has_complaint:
                        downloaded.add(case_dir.name)
        except Exception as e:
            logger.warning(f"Error reading downloaded cases: {e}")
        
        return downloaded
    
    def process_cases(self):
        """Main processing loop"""
        start_time = time.time()
        
        try:
            # Read case numbers from Phase 1 output
            if not WEEKLY_CASES_CSV.exists():
                logger.error(f"Case file not found: {WEEKLY_CASES_CSV}")
                logger.error("Run Phase 1 (pdf_extractor.py) first!")
                return
            
            df = pd.read_csv(WEEKLY_CASES_CSV)
            case_numbers = df['case_number'].astype(str).tolist()
            
            logger.info(f"Found {len(case_numbers)} cases to process")
            
            # Check which cases already downloaded
            downloaded = self.get_downloaded_cases()
            cases_to_process = [c for c in case_numbers if c not in downloaded]
            
            if len(cases_to_process) < len(case_numbers):
                logger.info(f"Resuming: {len(downloaded)} already downloaded, {len(cases_to_process)} remaining")
            
            # Start browser and login
            if not self.start_driver():
                return
            
            if not self.login():
                return
            
            # Process each case
            failed_cases = []
            
            for i, case_number in enumerate(cases_to_process, 1):
                logger.info(f"Processing {i}/{len(cases_to_process)}: {case_number}")
                
                try:
                    # Download case documents
                    zip_file = self.download_case_documents(case_number)
                    
                    if zip_file:
                        # Extract summons and complaint
                        if self.extract_documents(zip_file, case_number):
                            self.metrics['successful_downloads'] += 1
                        else:
                            self.metrics['failed_downloads'] += 1
                            failed_cases.append(case_number)
                    else:
                        self.metrics['failed_downloads'] += 1
                        failed_cases.append(case_number)
                    
                    self.metrics['cases_processed'] += 1
                    
                    # Progress update every 10 cases
                    if i % 10 == 0:
                        logger.info(f"Progress: {i}/{len(cases_to_process)} "
                                  f"(Success: {self.metrics['successful_downloads']}, "
                                  f"Failed: {self.metrics['failed_downloads']})")
                    
                    # Be respectful - delay between requests
                    time.sleep(SCRAPER_DELAY_MIN)
                    
                except Exception as e:
                    logger.error(f"Error processing {case_number}: {e}")
                    failed_cases.append(case_number)
                    self.metrics['failed_downloads'] += 1
            
            # Calculate metrics
            self.metrics['processing_time'] = time.time() - start_time
            
            # Log summary
            self._log_summary(failed_cases)
            
        except Exception as e:
            logger.error(f"Fatal error in process_cases: {e}")
        
        finally:
            if self.driver:
                self.driver.quit()
    
    def _log_summary(self, failed_cases):
        """Log processing summary"""
        logger.info("=" * 60)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Cases processed: {self.metrics['cases_processed']}")
        logger.info(f"Successful: {self.metrics['successful_downloads']}")
        logger.info(f"Failed: {self.metrics['failed_downloads']}")
        logger.info(f"Processing time: {self.metrics['processing_time']:.2f} seconds")
        
        if failed_cases:
            logger.info(f"\nFailed cases ({len(failed_cases)}):")
            for case in failed_cases[:10]:
                logger.info(f"  - {case}")
            if len(failed_cases) > 10:
                logger.info(f"  ... and {len(failed_cases) - 10} more")
        
        logger.info("=" * 60)


def main():
    """Run the scraper"""
    scraper = LinxScraper()
    scraper.process_cases()


if __name__ == '__main__':
    main()