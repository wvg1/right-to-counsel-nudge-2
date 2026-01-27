"""
email agent for weekly PDF collection
sends email requests, monitors for responses, downloads PDFs, runs pipeline
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import base64
import pickle
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import subprocess

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from config import DATA_DIR

# Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/gmail.send']

# configuration
ASHLEY_EMAIL = "will.vongeldern@gmail.com" # need to update this
MY_EMAIL = "wvg1@uw.edu"  
REQUEST_SUBJECT = "Weekly case docs"
DEADLINE_HOUR = 17
CHECK_INTERVAL_MINUTES = 30

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmailAgent:
    """autonomous email agent for weekly PDF automation"""
    
    def __init__(self, week_number: int):
        self.week_number = week_number
        self.pdf_dir = DATA_DIR / "daily_pdfs" / f"week_{week_number}"
        self.week_dir = DATA_DIR / f"week_{week_number}"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        
        self.service = None
        self.credentials = None
        self._authenticate()
    
    def _authenticate(self):
        """authenticate with Gmail API"""
        creds = None
        token_path = Path('token.pickle')
        
        # load existing credentials
        if token_path.exists():
            with open(token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # if no valid credentials, log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            # save credentials for next time
            with open(token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        self.credentials = creds
        self.service = build('gmail', 'v1', credentials=creds)
        logger.info("Gmail API authenticated successfully")
    
    def send_request_email(self):
        """send weekly PDF request to Ashley"""
        try:
            message = MIMEMultipart()
            message['to'] = ASHLEY_EMAIL
            message['subject'] = REQUEST_SUBJECT
            
            body = """Hi Ashley,

Hope you had a good weekend. Could you please send over last week's cases when you have a moment?

Thanks!"""
            
            message.attach(MIMEText(body, 'plain'))
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            send_message = {'raw': raw}
            
            self.service.users().messages().send(
                userId='me', body=send_message).execute()
            
            logger.info(f"request email sent to {ASHLEY_EMAIL}")
            return True
            
        except Exception as e:
            logger.error(f"failed to send email: {e}")
            return False
    
    def check_for_response(self) -> Optional[str]:
        """check if Ashley has responded with PDFs"""
        try:
            # search for emails from Ashley with the subject
            query = f'from:{ASHLEY_EMAIL} subject:"{REQUEST_SUBJECT}" has:attachment newer_than:1d'
            
            results = self.service.users().messages().list(
                userId='me', q=query).execute()
            
            messages = results.get('messages', [])
            
            if messages:
                logger.info(f"found response email with attachments")
                return messages[0]['id']
            
            return None
            
        except Exception as e:
            logger.error(f"error checking for response: {e}")
            return None
    
    def download_attachments(self, message_id: str) -> int:
        """download PDF attachments from email"""
        try:
            message = self.service.users().messages().get(
                userId='me', id=message_id).execute()
            
            pdf_count = 0
            
            for part in message['payload'].get('parts', []):
                if part['filename'] and part['filename'].endswith('.pdf'):
                    attachment_id = part['body'].get('attachmentId')
                    
                    if attachment_id:
                        attachment = self.service.users().messages().attachments().get(
                            userId='me', messageId=message_id, id=attachment_id).execute()
                        
                        data = attachment['data']
                        file_data = base64.urlsafe_b64decode(data)
                        
                        filepath = self.pdf_dir / part['filename']
                        with open(filepath, 'wb') as f:
                            f.write(file_data)
                        
                        logger.info(f"downloaded: {part['filename']}")
                        pdf_count += 1
            
            return pdf_count
            
        except Exception as e:
            logger.error(f"error downloading attachments: {e}")
            return 0
    
    def run_pipeline(self):
        """run the 3-phase automation pipeline"""
        try:
            logger.info("starting phase 1: PDF extraction...")
            result1 = subprocess.run(
                ['python', 'automation/pdf_extractor.py', '--week', str(self.week_number)],
                capture_output=True, text=True
            )
            
            if result1.returncode != 0:
                logger.error(f"phase 1 failed: {result1.stderr}")
                return False
            
            logger.info("starting phase 2: document download...")
            result2 = subprocess.run(
                ['python', 'automation/download_case_docs.py', '--week', str(self.week_number)],
                capture_output=True, text=True
            )
            
            if result2.returncode != 0:
                logger.error(f"Phase 2 failed: {result2.stderr}")
                return False
            
            logger.info("starting phase 3: LLM classification...")
            result3 = subprocess.run(
                ['python', 'automation/llm_classifier.py', '--week', str(self.week_number)],
                capture_output=True, text=True
            )
            
            if result3.returncode != 0:
                logger.error(f"phase 3 failed: {result3.stderr}")
                return False
            
            logger.info("all phases completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"pipeline error: {e}")
            return False
    
    def send_alert_email(self, subject: str, body: str):
        """send alert email to myself"""
        try:
            message = MIMEText(body)
            message['to'] = MY_EMAIL
            message['subject'] = subject
            
            raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
            send_message = {'raw': raw}
            
            self.service.users().messages().send(
                userId='me', body=send_message).execute()
            
            logger.info(f"alert email sent: {subject}")
            
        except Exception as e:
            logger.error(f"failed to send alert: {e}")
    
    def send_success_summary(self, pdf_count: int):
        """send success summary email"""
        residential_csv = self.week_dir / "outputs" / "residential_cases.csv"
        
        import pandas as pd
        if residential_csv.exists():
            df = pd.read_csv(residential_csv)
            case_count = len(df)
        else:
            case_count = "Unknown"
        
        body = f"""weekly processing complete - week {self.week_number}

PDFs received: {pdf_count}
residential cases identified: {case_count}
pipeline completed successfully

Results saved to: data/week_{self.week_number}/outputs/
"""
        
        self.send_alert_email(
            subject=f"Week {self.week_number} processing complete",
            body=body
        )
    
    def run(self):
        """Main agent loop"""
        logger.info(f"=== email agent started for week {self.week_number} ===")
        
        # step 1: send request email
        if not self.send_request_email():
            self.send_alert_email(
                subject="ERROR: failed to send request email",
                body="the agent could not send the weekly PDF request email, check logs"
            )
            return
        
        # step 2: monitor for response until 5pm
        start_time = datetime.now()
        deadline = start_time.replace(hour=DEADLINE_HOUR, minute=0, second=0)
        
        logger.info(f"monitoring for response until {deadline.strftime('%I:%M %p')}...")
        
        while datetime.now() < deadline:
            # check for response
            message_id = self.check_for_response()
            
            if message_id:
                # step 3: download PDFs
                pdf_count = self.download_attachments(message_id)
                
                if pdf_count == 0:
                    self.send_alert_email(
                        subject="WARNING: no PDFs found in response",
                        body="ashley replied but no PDF attachments were found, check manually."
                    )
                    return
                
                logger.info(f"downloaded {pdf_count} PDFs")
                
                # step 4: run the pipeline
                success = self.run_pipeline()
                
                if success:
                    # step 5: send success summary
                    self.send_success_summary(pdf_count)
                    logger.info("=== Agent completed successfully ===")
                else:
                    self.send_alert_email(
                        subject="ERROR: Pipeline failed",
                        body="PDFs were downloaded but the processing pipeline failed. Check logs for details."
                    )
                
                return
            
            # wait before checking again
            logger.info(f"no response yet, checking again in {CHECK_INTERVAL_MINUTES} minutes...")
            time.sleep(CHECK_INTERVAL_MINUTES * 60)
        
        # if deadline passed with no response
        self.send_alert_email(
            subject="ALERT: no PDFs received by deadline",
            body=f"It's now {datetime.now().strftime('%I:%M %p')} and Ashley has not responded with PDFs yet. Please follow up manually."
        )
        logger.info("=== agent timeout: No response by deadline ===")


def main():
    """run the email agent"""
    import argparse
    
    parser = argparse.ArgumentParser(description='email agent for weekly PDF automation')
    parser.add_argument('--week', type=int, required=True, help='week number')
    args = parser.parse_args()
    
    agent = EmailAgent(week_number=args.week)
    agent.run()


if __name__ == '__main__':
    main()