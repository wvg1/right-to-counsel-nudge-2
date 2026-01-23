"""
configuration file for eviction case automation pipeline
handles Azure credentials, file paths, and pipeline settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# base directory (project root)
BASE_DIR = Path(__file__).parent

# data directories
DATA_DIR = BASE_DIR / "data"
DAILY_PDFS_DIR = DATA_DIR / "daily_pdfs"
CASE_DOCUMENTS_DIR = DATA_DIR / "case_documents"
ASSIGNMENTS_DIR = DATA_DIR / "assignments"
LABELS_DIR = DATA_DIR / "labels"

# database and existing sample
EXISTING_SAMPLE_PATH = DATA_DIR / "existing_sample.xlsx"
DATABASE_PATH = DATA_DIR / "sample_database.db"

# output files
WEEKLY_CASES_CSV = DATA_DIR / "weekly_eviction_cases.csv"

# create directories if they don't exist
for directory in [DATA_DIR, DAILY_PDFS_DIR, CASE_DOCUMENTS_DIR, 
                  ASSIGNMENTS_DIR, LABELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# AZURE CREDENTIALS
# ============================================================================

# Azure Computer Vision (OCR)
AZURE_CV_ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
AZURE_CV_KEY = os.getenv("AZURE_CV_KEY")

# Azure OpenAI (LLM for classification)
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# ============================================================================
# PIERCE COUNTY LINX PORTAL
# ============================================================================

LINX_PORTAL_URL = os.getenv("LINX_PORTAL_URL", "https://linxonline.co.pierce.wa.us/")
LINX_USERNAME = os.getenv("LINX_USERNAME")
LINX_PASSWORD = os.getenv("LINX_PASSWORD")

# portal scraping settings
SCRAPER_DELAY_MIN = 2  # minimum seconds between requests
SCRAPER_DELAY_MAX = 5  # maximum seconds between requests
SCRAPER_HEADLESS = True  # run browser in headless mode

# ============================================================================
# PIPELINE SETTINGS
# ============================================================================

# case extraction
CASE_NUMBER_PATTERN = r'\d{2}-\d-\d{5}-\d'
EVICTION_KEYWORDS = ['unlawful detainer', 'detainer']

# LLM Classification
# LLM Classification
LLM_TEMPERATURE = 0.1  # Low temperature for consistent classification
LLM_MAX_TOKENS = 500
LLM_SYSTEM_PROMPT = LLM_SYSTEM_PROMPT = LLM_SYSTEM_PROMPT = """You are an expert legal document classifier. 
Your task is to analyze unlawful detainer complaints and classify them into one of three categories:

1. RESIDENTIAL - Standard residential tenant eviction (apartments, houses, mobile homes)
2. COMMERCIAL - Business/commercial property eviction
3. EJECTMENT - Cases involving non-tenant occupants (squatters, post-foreclosure, holdover owners, trespassers)

Look for indicators such as:
- Property type mentions (apartment, house, residence vs. commercial lease, office, retail)
- Tenant names: LLC, Inc., Corporation = likely commercial; individual names = likely residential
- Legal basis: breach of lease/nonpayment of rent = eviction; unlawful occupancy without lease = ejectment
- Notice periods: 3-day, 10-day, 14-day, or 30-day notice = tenant eviction; 20-day notice = likely ejectment

CRITICAL: If the complaint mentions ANY of the following, it is a tenant eviction (RESIDENTIAL or COMMERCIAL), NOT ejectment:
- "Tenancy" or "tenant"
- Rent, rental price, or rental payments
- Lease agreement or rental agreement  
- "Eviction" or "evict"
- Payment of utilities or other charges under a rental arrangement

Ejectment cases involve NO landlord-tenant relationship at all. They use terms like:
- "Former owner" (post-foreclosure)
- "No rental agreement" AND no mention of rent or tenancy
- "Unauthorized occupant" ONLY if there is no mention of rent, tenancy, or lease
- Divorce decree property disputes
- Adverse possession or boundary disputes

If you see both "unauthorized occupant" AND references to tenancy/rent, classify as RESIDENTIAL eviction, not ejectment.

Only classify as RESIDENTIAL if there is a clear landlord-tenant relationship for residential property.
"""

# deduplication settings
DUPLICATE_NAME_THRESHOLD = 0.85  # fuzzy matching threshold for names
DUPLICATE_ADDRESS_EXACT = True   # require exact address match

# randomization
TREATMENT_PROBABILITY = 0.5  # 50% treatment, 50% control
RANDOM_SEED = 651  # for reproducibility

# label generation
LABEL_TEMPLATE = "Avery 5164"  # 6 labels per sheet, 4" x 3.33"
LABEL_FONT_SIZE = 11
LABEL_FONT = "Helvetica"

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = BASE_DIR / "automation.log"

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate that required configuration is present"""
    errors = []
    
    # check Azure credentials
    if not AZURE_CV_ENDPOINT or not AZURE_CV_KEY:
        errors.append("Azure Computer Vision credentials missing (AZURE_CV_ENDPOINT, AZURE_CV_KEY)")
    
    if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_KEY:
        errors.append("Azure OpenAI credentials missing (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_KEY)")
    
    # check LINX portal credentials
    if not LINX_USERNAME or not LINX_PASSWORD:
        errors.append("LINX portal credentials missing (LINX_USERNAME, LINX_PASSWORD)")
    
    if errors:
        error_msg = "Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return True

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("Configuration loaded successfully!")
    print(f"Data directory: {DATA_DIR}")
    print(f"Azure OpenAI deployment: {AZURE_OPENAI_DEPLOYMENT}")
    print(f"LINX portal URL: {LINX_PORTAL_URL}")
    
    try:
        validate_config()
        print("\n✓ All required credentials present")
    except ValueError as e:
        print(f"\n✗ {e}")
        print("\nCreate a .env file with your credentials:")
        print("  AZURE_CV_ENDPOINT=your_endpoint")
        print("  AZURE_CV_KEY=your_key")
        print("  AZURE_OPENAI_ENDPOINT=your_endpoint")
        print("  AZURE_OPENAI_KEY=your_key")
        print("  LINX_USERNAME=your_username")
        print("  LINX_PASSWORD=your_password")