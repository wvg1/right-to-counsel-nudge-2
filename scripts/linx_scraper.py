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
import regex as re
from pathlib import Path
from bs4 import BeautifulSoup

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(r'C:\Users\wvg1\Documents\right-to-counsel-nudge-2\scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_recently_modified_file(directory, seconds=60):
    # find file modified in last 60 seconds
    current_time = time.time()
    files = glob.glob(os.path.join(directory, '*.zip'))  # only look for zip files
    
    if not files:
        return None
    
    # get the most recently modified zip file
    files.sort(key=os.path.getmtime, reverse=True)
    mod_time = os.path.getmtime(files[0])
    
    if current_time - mod_time < seconds:
        return files[0]
    
    return None

def label_file(old_path, new_path, number=1):
    # rename file, handling duplicates by adding numbers
    try:
        os.rename(old_path, new_path)
    except OSError:
        new_path_with_number = re.sub(r'\.pdf$', f'{number}.pdf', new_path)
        label_file(old_path, new_path_with_number, number + 1)


def relabel_files(cause, case_folder):
    # rename files based on HTML index
    try:
        html_path = os.path.join(case_folder, f'{cause}.htm')
        
        if not os.path.exists(html_path):
            logger.warning(f"HTML file not found: {html_path}")
            return
        
        with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_doc = f.read()
        
        # parse HTML to get file mappings
        soup = BeautifulSoup(html_doc, 'html.parser')
        
        old_names = []
        new_names = []
        
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and '.pdf' in href.lower():
                old_names.append(href)
                # get the link text
                for string in link.stripped_strings:
                    new_names.append(repr(string))
                    break  # only take first string
        
        # clean new names for use as filenames
        new_names = [re.sub(r'^\W|\W$', '', x) for x in new_names]
        new_names = [re.sub(r'\W+', '_', x) for x in new_names]
        new_names = [x.lower() + '.pdf' for x in new_names]
        
        # create mapping dictionary
        file_dict = dict(zip(old_names, new_names))
        
        # only keep PDFs
        file_dict = {old: file_dict[old] for old in file_dict 
                    if '.pdf' in old.lower()}
        
        # get files actually present in folder
        files_present = list(set(os.listdir(case_folder)).intersection(file_dict.keys()))
        
        # rename files
        for old_filename in files_present:
            try:
                old_path = os.path.join(case_folder, old_filename)
                new_path = os.path.join(case_folder, file_dict[old_filename])
                label_file(old_path, new_path)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f"Could not label file {old_filename}: {e}")
        
        logger.info(f"Relabeled files for case {cause}")
        
    except Exception as e:
        logger.error(f"Error in relabel_files for {cause}: {e}")


def get_downloaded_cases(output_dir):
    # get list of cases that have already been downloaded
    downloaded = set()
    try:
        # look for year folders
        for year_dir in glob.glob(os.path.join(output_dir, '20*')):
            # look for case folders in each year
            for case_dir in glob.glob(os.path.join(year_dir, '*')):
                case_name = os.path.basename(case_dir)
                if os.path.isdir(case_dir):
                    downloaded.add(case_name)
    except Exception as e:
        logger.warning(f"Error reading downloaded cases: {e}")
    
    return downloaded


def start_driver(chromedriver_path=None):
    # initialize Chrome WebDriver
    try:
        chrome_options = Options()
        # set download directory
        prefs = {
            "download.default_directory": os.path.abspath(r'C:\Users\wvg1\Documents\eviction-data\data\doc_downloads'),
            "download.prompt_for_download": False,
            "safebrowsing.enabled": False
        }
        chrome_options.add_experimental_option("prefs", prefs)
        # use webdriver-manager to automatically get the right chromedriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        logger.info("Chrome WebDriver started")
        return driver
    except Exception as e:
        logger.error(f"Failed to start WebDriver: {e}")
        raise


def login(driver, username, password, base_url):
    # log into the LINX portal
    try:
        driver.get(f"{base_url}/Account/Logon.cfm")
        
        # enter account number and PIN
        driver.find_element(By.NAME, "account_num").send_keys(username)
        driver.find_element(By.NAME, "pin").send_keys(password)
        driver.find_element(By.CSS_SELECTOR, 'input[type="Submit"]').click()
        
        time.sleep(3)  # wait for login to complete
        logger.info("Successfully logged in")
        return True
        
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return False


def read_case_numbers(excel_file):
    # read case numbers from Excel file
    try:
        df = pd.read_excel(excel_file)
        # assumes case numbers are in 'case numbers' column or first column
        if 'case numbers' in df.columns:
            case_numbers = df['case numbers'].astype(str).tolist()
        elif 'Case Number' in df.columns:
            case_numbers = df['Case Number'].astype(str).tolist()
        else:
            case_numbers = df.iloc[:, 0].astype(str).tolist()
        
        logger.info(f"Read {len(case_numbers)} case numbers from Excel")
        return case_numbers
    except Exception as e:
        logger.error(f"Error reading Excel file: {e}")
        return []


def download_case_documents(driver, case_number, base_url, output_dir):
    # download documents for a specific case
    try:
        # navigate to case page
        case_url = f"{base_url}/Case/CivilCase.cfm?cause_num={case_number}"
        driver.get(case_url)
        time.sleep(3)
        
        try:
            # find and click "download filings" link
            driver.find_element(By.LINK_TEXT, "download filings").click()
            time.sleep(2)
            
            # select all files
            driver.find_element(By.LINK_TEXT, "select all").click()
            time.sleep(2)
            
            # click download button
            driver.find_element(By.NAME, "btnDownload").click()
            time.sleep(3)
        except:
            # case might not have documents available
            logger.warning(f"Could not find download option for {case_number}")
            return False
        
        # wait for download to complete
        sleep_time = 0
        
        while sleep_time < 300:  # max 300 second wait (5 minutes)
            recent_file = get_recently_modified_file(output_dir, seconds=60)
            if recent_file:
                logger.info(f"Downloaded {case_number}: {recent_file}")
                time.sleep(1)  # give file time to finish writing
                return True
            
            time.sleep(0.1)
            sleep_time += 0.1
        
        logger.warning(f"Download timeout for {case_number}")
        return False
        
    except Exception as e:
        logger.error(f"Error downloading {case_number}: {e}")
        return False


def process_zip_files(output_dir):
    # unzip files and organize by year/case number
    try:
        zip_files = glob.glob(os.path.join(output_dir, '*.zip'))
        
        if not zip_files:
            logger.info("No zip files to process")
            return 0
        
        logger.info(f"Processing {len(zip_files)} zip file(s)...")
        processed = 0
        
        for zip_path in zip_files:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    # get case number from first HTML file found
                    html_files = [f for f in file_list if f.endswith('.htm')]
                    
                    if not html_files:
                        logger.warning(f"No HTML files in {zip_path}")
                        continue
                    
                    case_number = os.path.basename(html_files[0]).replace('.htm', '')
                    year_prefix = case_number.split('-')[0]  # extract year prefix (e.g., "23", "05")
                    
                    # convert 2-digit year to 4-digit
                    year_int = int(year_prefix)
                    if year_int <= 30:  # assume 00-30 is 2000-2030
                        year = f"20{year_prefix}"
                    else:  # assume 31-99 is 1931-1999
                        year = f"19{year_prefix}"
                    
                    # create year/case folder structure
                    year_dir = os.path.join(output_dir, year)
                    case_dir = os.path.join(year_dir, case_number)
                    os.makedirs(case_dir, exist_ok=True)
                    
                    # extract to case folder
                    zip_ref.extractall(case_dir)
                    
                    # relabel files based on HTML index
                    relabel_files(case_number, case_dir)
                
                    # remove zip file after processing
                    os.remove(zip_path)
                    processed += 1
                
            except Exception as e:
                logger.error(f"Error processing zip file {zip_path}: {e}")
        
        logger.info(f"Processed {processed} zip files")
        return processed
        
    except Exception as e:
        logger.error(f"Error in process_zip_files: {e}")
        return 0


def main():
    # config
    username = 'P2148238'
    password = 'Research2025'
    excel_file = r'C:\Users\wvg1\Documents\right-to-counsel-nudge-2\data\case_numbers.xlsx'
    output_dir = r'C:\Users\wvg1\Documents\right-to-counsel-nudge-2\data\doc_downloads'
    base_url = 'https://linxonline.co.pierce.wa.us/linxweb'
    
    # maintenance intervals
    PROCESS_ZIPS_EVERY = 100  # process zips every N cases
    RESTART_CHROME_EVERY = 200  # restart Chrome every N cases
    
    # create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    driver = None
    try:
        # start driver and login
        driver = start_driver()
        
        if not login(driver, username, password, base_url):
            logger.error("Failed to login")
            return
        
        # read case numbers
        case_numbers = read_case_numbers(excel_file)
        if not case_numbers:
            logger.error("No case numbers found")
            return
        
        # get already downloaded cases
        downloaded_cases = get_downloaded_cases(output_dir)
        cases_to_process = [c for c in case_numbers if c not in downloaded_cases]
        
        if len(cases_to_process) < len(case_numbers):
            logger.info(f"Resuming: {len(downloaded_cases)} cases already downloaded, {len(cases_to_process)} remaining")
        
        # download documents for each case
        successful = 0
        failed = 0
        failed_cases = []
        
        for i, case_number in enumerate(cases_to_process, 1):
            logger.info(f"Processing {i}/{len(cases_to_process)}: {case_number}")
            
            try:
                if download_case_documents(driver, case_number, base_url, output_dir):
                    successful += 1
                else:
                    failed += 1
                    failed_cases.append(case_number)
            except Exception as e:
                logger.error(f"Error with case {case_number}: {e}")
                failed += 1
                failed_cases.append(case_number)
                # try to reconnect if session dies
                try:
                    driver.quit()
                except:
                    pass
                driver = start_driver()
                if not login(driver, username, password, base_url):
                    logger.error("Failed to reconnect and login")
                    break
            
            # process zips every 100 cases
            if i % PROCESS_ZIPS_EVERY == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"PROCESSING ACCUMULATED ZIP FILES (every {PROCESS_ZIPS_EVERY} cases)")
                logger.info(f"{'='*60}")
                processed_count = process_zip_files(output_dir)
                logger.info(f"Processed {processed_count} zip files, continuing downloads...")
                logger.info(f"{'='*60}\n")
            
            # restart Chrome every 200 cases to prevent memory issues
            if i % RESTART_CHROME_EVERY == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"RESTARTING CHROME (every {RESTART_CHROME_EVERY} cases)")
                logger.info(f"{'='*60}")
                try:
                    driver.quit()
                    logger.info("Chrome closed")
                except:
                    pass
                
                time.sleep(3)  # brief pause before restart
                driver = start_driver()
                
                if not login(driver, username, password, base_url):
                    logger.error("Failed to login after Chrome restart")
                    break
                
                logger.info("Chrome restarted successfully")
                logger.info(f"{'='*60}\n")
            
            # print progress every 50 cases
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(cases_to_process)} (Success: {successful}, Failed: {failed})")
            
            # add delay between requests
            time.sleep(5)
        
        # process any remaining zip files
        logger.info("\n{'='*60}")
        logger.info("Processing remaining zip files...")
        logger.info(f"{'='*60}")
        process_zip_files(output_dir)
        
        # final summary
        logger.info(f"\n{'='*60}")
        logger.info(f"SCRAPING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Cases processed this run: {len(cases_to_process)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total downloaded so far: {len(downloaded_cases) + successful}")
        if failed_cases:
            logger.info(f"\nFailed cases:")
            for case in failed_cases[:50]:  # show first 50 failed cases
                logger.info(f"  - {case}")
            if len(failed_cases) > 50:
                logger.info(f"  ... and {len(failed_cases) - 50} more")
        logger.info(f"{'='*60}\n")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    
    finally:
        if driver:
            driver.quit()


if __name__ == '__main__':
    main()