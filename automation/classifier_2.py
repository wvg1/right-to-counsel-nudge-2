"""
Phase 3: LLM-Based Residential Classification
Uses Azure Computer Vision OCR + GPT-4o to classify eviction cases as residential vs commercial
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import time
import json
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

# Azure imports
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

from config import (
    WEEKLY_CASES_CSV,
    CASE_DOCUMENTS_DIR,
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
    
    def __init__(self):
        self.cv_client = None
        self.openai_client = None
        self.metrics = {
            'cases_processed': 0,
            'residential_count': 0,
            'commercial_count': 0,
            'ejectment_count': 0,
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
    
    def classify_with_llm(self, complaint_text: str, case_number: str) -> Dict:
        """Use GPT-4o to classify as residential or commercial"""
        try:
            # Truncate text if too long (GPT-4o has token limits)
            max_chars = 10000
            if len(complaint_text) > max_chars:
                complaint_text = complaint_text[:max_chars] + "\n[... text truncated ...]"
            
            # Create prompt
            user_prompt = f"""Analyze this unlawful detainer complaint and classify it into one of three categories:
- RESIDENTIAL: Standard residential tenant eviction
- COMMERCIAL: Business/commercial property eviction  
- EJECTMENT: Non-tenant occupant (squatter, post-foreclosure, former owner, etc.)

Complaint text:
{complaint_text}

Classify this case and respond with ONLY a JSON object in this exact format:
{{"classification": "RESIDENTIAL" or "COMMERCIAL" or "EJECTMENT", "confidence": "HIGH" or "MEDIUM" or "LOW", "reasoning": "brief explanation"}}"""
            
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
                
                logger.info(f"{case_number}: {classification} ({result.get('confidence', 'UNKNOWN')} confidence)")
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON for {case_number}: {e}")
                logger.error(f"Response was: {response_text}")
                return None
            
        except Exception as e:
            logger.error(f"Error in LLM classification for {case_number}: {e}")
            return None
    
    def process_case(self, case_number: str) -> Dict:
        """Process a single case: check summons for keywords, then analyze complaint if needed"""
        try:
            case_dir = CASE_DOCUMENTS_DIR / case_number
            summons_path = case_dir / "summons.pdf"
            complaint_path = case_dir / "complaint.pdf"
            
            # Strategy 1: Check summons for classification keywords
            if summons_path.exists():
                summons_text = self.extract_text_from_pdf(summons_path)
                summons_lower = summons_text.lower()
                
                # Look for explicit classification in summons
                if 'residential landlord' in summons_lower or 'residential tenant' in summons_lower:
                    logger.info(f"{case_number}: RESIDENTIAL (from summons - residential landlord/tenant)")
                    return {
                        'case_number': case_number,
                        'classification': 'RESIDENTIAL',
                        'confidence': 'HIGH',
                        'reasoning': 'Summons document references "Residential Landlord" or "Residential Tenant" act'
                    }
                elif 'commercial lease' in summons_lower or 'commercial landlord' in summons_lower:
                    logger.info(f"{case_number}: COMMERCIAL (from summons)")
                    return {
                        'case_number': case_number,
                        'classification': 'COMMERCIAL',
                        'confidence': 'HIGH',
                        'reasoning': 'Summons document references commercial lease or landlord'
                    }
                elif 'ejectment' in summons_lower and 'unlawful detainer' not in summons_lower:
                    logger.info(f"{case_number}: EJECTMENT (from summons)")
                    return {
                        'case_number': case_number,
                        'classification': 'EJECTMENT',
                        'confidence': 'HIGH',
                        'reasoning': 'Summons document is for ejectment action'
                    }
            
            # Strategy 2: Analyze complaint content with LLM if summons inconclusive
            if not complaint_path.exists():
                logger.warning(f"No complaint found for {case_number}")
                return None
            
            # Extract text via OCR
            complaint_text = self.extract_text_from_pdf(complaint_path)
            
            if not complaint_text or len(complaint_text) < 100:
                logger.warning(f"Insufficient text extracted from {case_number}")
                return None
            
            # Classify with LLM
            classification = self.classify_with_llm(complaint_text, case_number)
            
            if classification:
                return {
                    'case_number': case_number,
                    'classification': classification['classification'],
                    'confidence': classification.get('confidence', 'UNKNOWN'),
                    'reasoning': classification.get('reasoning', '')
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error processing case {case_number}: {e}")
            return None
    
    def process_all_cases(self):
        """Process all cases from Phase 1 output"""
        start_time = time.time()
        
        try:
            # Read case numbers from Phase 1
            if not WEEKLY_CASES_CSV.exists():
                logger.error(f"Case file not found: {WEEKLY_CASES_CSV}")
                logger.error("Run Phase 1 (pdf_extractor.py) first!")
                return
            
            df = pd.read_csv(WEEKLY_CASES_CSV)
            case_numbers = df['case_number'].astype(str).tolist()
            
            logger.info(f"Processing {len(case_numbers)} cases for classification")
            
            # Process each case
            results = []
            
            for case_number in tqdm(case_numbers, desc="Classifying cases"):
                result = self.process_case(case_number)
                
                if result:
                    results.append(result)
                    
                    if result['classification'] == 'RESIDENTIAL':
                        self.metrics['residential_count'] += 1
                    elif result['classification'] == 'COMMERCIAL':
                        self.metrics['commercial_count'] += 1
                    elif result['classification'] == 'EJECTMENT':
                        self.metrics['ejectment_count'] += 1
                else:
                    self.metrics['failed_count'] += 1
                
                self.metrics['cases_processed'] += 1
                
                # Small delay to avoid rate limits
                time.sleep(0.5)
            
            # Create results dataframe
            results_df = pd.DataFrame(results)
            
            # Merge with original case data
            final_df = df.merge(results_df, on='case_number', how='left')
            
            # Save results
            output_path = WEEKLY_CASES_CSV.parent / "classified_cases.csv"
            final_df.to_csv(output_path, index=False)
            logger.info(f"Saved classification results to {output_path}")
            
            # Filter for residential only
            residential_df = final_df[final_df['classification'] == 'RESIDENTIAL']
            residential_path = WEEKLY_CASES_CSV.parent / "residential_cases.csv"
            residential_df.to_csv(residential_path, index=False)
            logger.info(f"Saved {len(residential_df)} residential cases to {residential_path}")
            
            # Calculate metrics
            self.metrics['processing_time'] = time.time() - start_time
            
            # Log summary
            self._log_summary()
            
            return residential_df
            
        except Exception as e:
            logger.error(f"Fatal error in process_all_cases: {e}")
            return None
    
    def _log_summary(self):
        """Log processing summary"""
        logger.info("=" * 60)
        logger.info("CLASSIFICATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Cases processed: {self.metrics['cases_processed']}")
        logger.info(f"Residential: {self.metrics['residential_count']}")
        logger.info(f"Commercial: {self.metrics['commercial_count']}")
        logger.info(f"Ejectment: {self.metrics['ejectment_count']}")
        logger.info(f"Failed: {self.metrics['failed_count']}")
        logger.info(f"Processing time: {self.metrics['processing_time']:.2f} seconds")
        logger.info("=" * 60)


def main():
    """Run the classifier"""
    classifier = ResidentialClassifier()
    classifier.process_all_cases()


if __name__ == '__main__':
    main()