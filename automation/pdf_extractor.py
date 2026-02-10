"""
Phase 1: PDF Case Number Extractor
Extracts case numbers and types from Pierce County daily filing PDFs
"""

import sys
from pathlib import Path
# add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))

import re
import logging
from typing import List, Dict
import time
import argparse
import pdfplumber
from config import DATA_DIR

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CaseExtractor:
    """Extracts unlawful detainer cases from daily filing PDFs"""
    
    def __init__(self, week_number: int):
        """
        Initialize the case extractor
        
        Args:
            week_number: Week number for organizing data
        """
        self.week_number = week_number
        self.pdf_directory = DATA_DIR / "daily_pdfs" / f"week_{week_number}"
        self.output_csv = DATA_DIR / f"week_{week_number}" / "weekly_cases.csv"
        
        # create directories if they don't exist
        self.pdf_directory.mkdir(parents=True, exist_ok=True)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        self.case_pattern = re.compile(r'\d{2}-\d-\d{5}-\d')
        self.metrics = {
            'pdfs_processed': 0,
            'total_cases_found': 0,
            'eviction_cases_found': 0,
            'processing_time': 0
        }
    
    def extract_cases_from_pdf(self, pdf_path: Path) -> List[Dict[str, str]]:
        """
        Extract all cases from a single PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with case_number, case_type, and filing_date
        """
        cases = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                
                # split by case number pattern
                lines = text.split('\n')
                current_case = None
                
                for i, line in enumerate(lines):
                    # check if line contains a case number
                    case_match = self.case_pattern.search(line)
                    
                    if case_match:
                        case_number = case_match.group()
                        
                        # extract filing date (should be on same line)
                        date_match = re.search(r'\d{2}/\d{2}/\d{4}', line)
                        filing_date = date_match.group() if date_match else None
                        
                        # extract case type (typically on same line after date)
                        case_type = line.split(filing_date)[-1].strip() if filing_date else ""
                        
                        # if case type is empty, check next few lines
                        if not case_type and i + 1 < len(lines):
                            case_type = lines[i + 1].strip()
                        
                        current_case = {
                            'case_number': case_number,
                            'filing_date': filing_date,
                            'case_type': case_type,
                            'source_file': pdf_path.name
                        }
                        cases.append(current_case)
                
                logger.info(f"Extracted {len(cases)} cases from {pdf_path.name}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path.name}: {str(e)}")
        
        return cases
    
    def filter_eviction_cases(self, cases: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Filter for unlawful detainer (eviction) cases
        
        Args:
            cases: List of all cases
            
        Returns:
            List of eviction cases only
        """
        eviction_keywords = ['unlawful detainer', 'detainer']
        
        eviction_cases = [
            case for case in cases
            if any(keyword in case['case_type'].lower() for keyword in eviction_keywords)
        ]
        
        logger.info(f"Found {len(eviction_cases)} eviction cases out of {len(cases)} total")
        return eviction_cases
    
    def process_weekly_pdfs(self) -> List[Dict[str, str]]:
        """
        Process all PDFs in the directory and extract eviction cases
        
        Returns:
            List of all eviction cases from the week
        """
        start_time = time.time()
        all_cases = []
        
        # get all PDF files (recursively to handle subdirectories)
        pdf_files = sorted(self.pdf_directory.glob('**/*.pdf'))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return []
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        for pdf_path in pdf_files:
            cases = self.extract_cases_from_pdf(pdf_path)
            all_cases.extend(cases)
            self.metrics['pdfs_processed'] += 1
        
        self.metrics['total_cases_found'] = len(all_cases)
        
        # filter for evictions
        eviction_cases = self.filter_eviction_cases(all_cases)
        self.metrics['eviction_cases_found'] = len(eviction_cases)
        
        # calculate processing time
        self.metrics['processing_time'] = time.time() - start_time
        
        self._log_metrics()
        
        return eviction_cases
    
    def _log_metrics(self):
        """Log processing metrics"""
        logger.info("=" * 50)
        logger.info("PROCESSING METRICS")
        logger.info("=" * 50)
        logger.info(f"Week Number: {self.week_number}")
        logger.info(f"PDFs Processed: {self.metrics['pdfs_processed']}")
        logger.info(f"Total Cases Found: {self.metrics['total_cases_found']}")
        logger.info(f"Eviction Cases Found: {self.metrics['eviction_cases_found']}")
        logger.info(f"Processing Time: {self.metrics['processing_time']:.2f} seconds")
        logger.info("=" * 50)
    
    def get_metrics(self) -> Dict:
        """Return metrics for resume/reporting purposes"""
        return self.metrics
    
    def save_to_csv(self, eviction_cases: List[Dict[str, str]]):
        """Save eviction cases to CSV"""
        import pandas as pd
        df = pd.DataFrame(eviction_cases)
        df.to_csv(self.output_csv, index=False)
        logger.info(f"Saved {len(eviction_cases)} cases to {self.output_csv}")


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(description='Extract eviction cases from daily filing PDFs')
    parser.add_argument('--week', type=int, required=True, help='Week number')
    args = parser.parse_args()
    
    # initialize extractor with week number
    extractor = CaseExtractor(week_number=args.week)
    
    # process all PDFs from the week
    eviction_cases = extractor.process_weekly_pdfs()
    
    # display results
    print(f"\nFound {len(eviction_cases)} eviction cases:")
    for case in eviction_cases[:5]:  # show first 5
        print(f"  {case['case_number']} - {case['case_type']} ({case['filing_date']})")
    
    if len(eviction_cases) > 5:
        print(f"  ... and {len(eviction_cases) - 5} more")
    
    # save to CSV for next phase
    extractor.save_to_csv(eviction_cases)
    
    return eviction_cases


if __name__ == "__main__":
    main()