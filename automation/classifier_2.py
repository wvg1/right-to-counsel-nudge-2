"""
Phase 3: LLM-Based Residential Classification with Address & Name Extraction
Uses Azure Computer Vision OCR + GPT-4o to classify eviction cases and extract tenant info
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import time
import json
import argparse
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

# Azure imports
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
    LLM_SYSTEM_PROMPT
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResidentialClassifier:
    """Classifies eviction cases as residential or commercial using AI"""
    
    def __init__(self, week_number: int):
        self.week_number = week_number
        self.week_dir = DATA_DIR / f"week_{week_number}"
        self.case_documents_dir = self.week_dir / "case_documents"
        self.weekly_cases_csv = self.week_dir / "weekly_cases.csv"
        self.outputs_dir = self.week_dir / "outputs"
        
        # Create outputs directory
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        
        self.cv_client = None
        self.openai_client = None
        self.metrics = {
            'cases_processed': 0,
            'residential_count': 0,
            'flagged_count': 0,
            'failed_count': 0,
            'processing_time': 0
        }
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure clients"""
        try:
            # Document Intelligence client for multi-page OCR
            self.cv_client = DocumentAnalysisClient(
                endpoint=AZURE_CV_ENDPOINT,
                credential=AzureKeyCredential(AZURE_CV_KEY)
            )
            
            # OpenAI client for LLM classification
            self.openai_client = AzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            
            logger.info("Azure clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {e}")
            raise
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from multi-page PDF using Azure Document Intelligence"""
        try:
            # Read PDF as bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            # Use Azure Document Intelligence Read model for multi-page OCR
            poller = self.cv_client.begin_analyze_document(
                "prebuilt-read",
                document=pdf_bytes
            )
            
            result = poller.result()
            
            # Extract all text from all pages
            full_text = result.content
            
            logger.debug(f"Extracted {len(full_text)} characters from {pdf_path.name} ({len(result.pages)} pages)")
            return full_text
                
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path.name}: {e}")
            return ""
    
    def analyze_document_with_llm(self, document_text: str, case_number: str, doc_type: str) -> Optional[Dict]:
        """Use GPT-4o to classify and extract information from a document"""
        try:
            # Truncate text if too long
            max_chars = 10000
            if len(document_text) > max_chars:
                document_text = document_text[:max_chars] + "\n[... text truncated ...]"
            
            # Create prompt
            user_prompt = f"""Analyze this unlawful detainer {doc_type} and extract the following information:

1. Classification: RESIDENTIAL, COMMERCIAL, or EJECTMENT
2. Property address (where eviction is happening)
3. Defendant/tenant names (up to 5)

Document text:
{document_text}

IMPORTANT NAME PARSING RULES:
- Parse names as: first, middle (or middle initial), last
- Watch for multi-part last names like: "de la Rosa", "von Geldern", "Al Shaim", "Heiden-Mostert"
- If middle name is absent, leave it blank
- Extract ALL defendants listed (up to 5)

Respond with ONLY a JSON object in this exact format:
{{
  "classification": "RESIDENTIAL" or "COMMERCIAL" or "EJECTMENT",
  "confidence": "HIGH" or "MEDIUM" or "LOW",
  "reasoning": "brief explanation",
  "address": "full street address",
  "address_unit": "apt/unit number or empty string",
  "address_city": "city",
  "address_state": "state",
  "address_zip": "zip code",
  "tenants": [
    {{"first": "John", "middle": "Q", "last": "Smith"}},
    {{"first": "Jane", "middle": "", "last": "Doe"}}
  ]
}}"""
            
            # Call GPT-4o
            response = self.openai_client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON
            try:
                # Remove markdown code blocks if present
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                
                result = json.loads(response_text)
                
                # Validate response
                if 'classification' not in result:
                    raise ValueError("Missing classification in response")
                
                classification = result['classification'].upper()
                if classification not in ['RESIDENTIAL', 'COMMERCIAL', 'EJECTMENT']:
                    raise ValueError(f"Invalid classification: {classification}")
                
                logger.info(f"{case_number} ({doc_type}): {classification} ({result.get('confidence', 'UNKNOWN')} confidence)")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON for {case_number}: {e}")
                logger.error(f"Response was: {response_text}")
                return None
            
        except Exception as e:
            logger.error(f"Error in LLM analysis for {case_number}: {e}")
            return None
    
    def normalize_address(self, addr: str) -> str:
        """Normalize address for comparison (fuzzy matching)"""
        if not addr:
            return ""
        addr = addr.lower().strip()
        # Normalize common variations
        addr = addr.replace('street', 'st').replace('avenue', 'ave').replace('road', 'rd')
        addr = addr.replace('drive', 'dr').replace('lane', 'ln').replace('court', 'ct')
        addr = addr.replace('apartment', 'apt').replace('suite', 'ste').replace('#', 'apt')
        addr = addr.replace('.', '').replace(',', '')
        return ' '.join(addr.split())  # normalize whitespace
    
    def addresses_match(self, addr1: Dict, addr2: Dict) -> bool:
        """Check if two address dictionaries match (with fuzzy logic)"""
        # Compare normalized street addresses
        street1 = self.normalize_address(addr1.get('address', ''))
        street2 = self.normalize_address(addr2.get('address', ''))
        
        if street1 != street2:
            return False
        
        # Compare other fields (exact match)
        for field in ['address_city', 'address_state', 'address_zip']:
            val1 = (addr1.get(field, '') or '').strip().lower()
            val2 = (addr2.get(field, '') or '').strip().lower()
            if val1 != val2:
                return False
        
        return True
    
    def process_case(self, case_number: str) -> Dict:
        """Process a single case: analyze both summons and complaint"""
        try:
            case_dir = self.case_documents_dir / case_number
            summons_path = case_dir / "summons.pdf"
            complaint_path = case_dir / "complaint.pdf"
            
            summons_data = None
            complaint_data = None
            flag_reasons = []
            
            # Extract and analyze summons if exists
            if summons_path.exists():
                summons_text = self.extract_text_from_pdf(summons_path)
                if summons_text and len(summons_text) > 100:
                    summons_data = self.analyze_document_with_llm(summons_text, case_number, "summons")
            
            # Extract and analyze complaint if exists
            if complaint_path.exists():
                complaint_text = self.extract_text_from_pdf(complaint_path)
                if complaint_text and len(complaint_text) > 100:
                    complaint_data = self.analyze_document_with_llm(complaint_text, case_number, "complaint")
            
            # If neither document processed successfully, fail
            if not summons_data and not complaint_data:
                logger.warning(f"No valid documents for {case_number}")
                return None
            
            # Determine final classification and data
            if summons_data and complaint_data:
                # Both documents available - check for consistency
                
                # Check classification consistency
                if summons_data['classification'] != complaint_data['classification']:
                    flag_reasons.append(f"classification_mismatch (summons: {summons_data['classification']}, complaint: {complaint_data['classification']})")
                
                # Check address consistency
                if not self.addresses_match(summons_data, complaint_data):
                    flag_reasons.append("inconsistent_addresses")
                
                # Use complaint data as primary (more detailed)
                final_data = complaint_data.copy()
                final_classification = complaint_data['classification']
                
            elif complaint_data:
                # Only complaint available
                final_data = complaint_data.copy()
                final_classification = complaint_data['classification']
            else:
                # Only summons available
                final_data = summons_data.copy()
                final_classification = summons_data['classification']
            
            # Check tenant count
            tenants = final_data.get('tenants', [])
            if len(tenants) > 5:
                flag_reasons.append(f"too_many_tenants ({len(tenants)} tenants)")
            
            # Flatten tenant data into columns (up to 5 tenants)
            result = {
                'case_number': case_number,
                'classification': final_classification,
                'confidence': final_data.get('confidence', 'UNKNOWN'),
                'reasoning': final_data.get('reasoning', ''),
                'address': final_data.get('address', ''),
                'address_unit': final_data.get('address_unit', ''),
                'address_city': final_data.get('address_city', ''),
                'address_state': final_data.get('address_state', ''),
                'address_zip': final_data.get('address_zip', ''),
            }
            
            # Add tenant columns (up to 5)
            for i in range(5):
                if i < len(tenants):
                    tenant = tenants[i]
                    result[f'tenant_{i+1}_first'] = tenant.get('first', '')
                    result[f'tenant_{i+1}_middle'] = tenant.get('middle', '')
                    result[f'tenant_{i+1}_last'] = tenant.get('last', '')
                else:
                    result[f'tenant_{i+1}_first'] = ''
                    result[f'tenant_{i+1}_middle'] = ''
                    result[f'tenant_{i+1}_last'] = ''
            
            # Add flag info
            result['flagged'] = len(flag_reasons) > 0
            result['flag_reasons'] = '; '.join(flag_reasons) if flag_reasons else ''
            
            return result
                
        except Exception as e:
            logger.error(f"Error processing case {case_number}: {e}")
            return None
    
    def process_all_cases(self):
        """Process all cases from Phase 1 output"""
        start_time = time.time()
        
        try:
            # Read case numbers from Phase 1
            if not self.weekly_cases_csv.exists():
                logger.error(f"Case file not found: {self.weekly_cases_csv}")
                logger.error("Run Phase 1 (pdf_extractor.py) first!")
                return
            
            df = pd.read_csv(self.weekly_cases_csv)
            case_numbers = df['case_number'].astype(str).tolist()
            
            logger.info(f"Processing {len(case_numbers)} cases for classification")
            
            # Process each case
            results = []
            
            for case_number in tqdm(case_numbers, desc="Classifying cases"):
                result = self.process_case(case_number)
                
                if result:
                    results.append(result)
                    
                    if result['classification'] == 'RESIDENTIAL' and not result['flagged']:
                        self.metrics['residential_count'] += 1
                    if result['flagged']:
                        self.metrics['flagged_count'] += 1
                else:
                    self.metrics['failed_count'] += 1
                
                self.metrics['cases_processed'] += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Merge with original case data
            final_df = df.merge(results_df, on='case_number', how='left')
            
            # Split into residential and flagged
            # Residential: RESIDENTIAL classification AND not flagged
            residential_df = final_df[
                (final_df['classification'] == 'RESIDENTIAL') & 
                (final_df['flagged'] == False)
            ]
            
            # Flagged: either flagged OR not residential
            flagged_df = final_df[
                (final_df['flagged'] == True) | 
                (final_df['classification'] != 'RESIDENTIAL')
            ]
            
            # Save residential cases
            residential_path = self.outputs_dir / "residential_cases.csv"
            residential_df.to_csv(residential_path, index=False)
            logger.info(f"Saved {len(residential_df)} residential cases to {residential_path}")
            
            # Save flagged cases
            flagged_path = self.outputs_dir / "flagged_cases.csv"
            flagged_df.to_csv(flagged_path, index=False)
            logger.info(f"Saved {len(flagged_df)} flagged cases to {flagged_path}")
            
            # Calculate metrics
            self.metrics['processing_time'] = time.time() - start_time
            
            # Log summary
            self._log_summary()
            
            return residential_df
            
        except Exception as e:
            logger.error(f"Fatal error in process_all_cases: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _log_summary(self):
        """Log processing summary"""
        logger.info("=" * 60)
        logger.info("CLASSIFICATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Week Number: {self.week_number}")
        logger.info(f"Cases processed: {self.metrics['cases_processed']}")
        logger.info(f"Clean residential: {self.metrics['residential_count']}")
        logger.info(f"Flagged for review: {self.metrics['flagged_count']}")
        logger.info(f"Failed: {self.metrics['failed_count']}")
        logger.info(f"Processing time: {self.metrics['processing_time']:.2f} seconds")
        logger.info("=" * 60)


def main():
    """Run the classifier"""
    parser = argparse.ArgumentParser(description='Classify eviction cases and extract tenant info')
    parser.add_argument('week', type=int, help='Week number')
    args = parser.parse_args()
    
    classifier = ResidentialClassifier(week_number=args.week)
    classifier.process_all_cases()


if __name__ == '__main__':
    main()