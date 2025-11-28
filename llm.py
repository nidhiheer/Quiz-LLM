import os
import json
import time
import base64
import requests
import re
import logging
from urllib.parse import urljoin, urlparse
from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager
from groq import Groq
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
SECRET = os.getenv('SECRET')
EMAIL = os.getenv('EMAIL')

client = Groq(api_key=GROQ_API_KEY)

def setup_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--remote-debugging-port=9222')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_quiz_page(url):
    driver = setup_driver()
    try:
        logging.info(f"Scraping URL: {url}")
        
        driver.set_page_load_timeout(30)
        driver.get(url)
        
        WebDriverWait(driver, 20).until(
            lambda driver: driver.execute_script('return document.readyState') == 'complete'
        )
        
        time.sleep(2)
        
        body_element = driver.find_element(By.TAG_NAME, 'body')
        content = body_element.text
        
        logging.info(f"Page title: '{driver.title}'")
        logging.info(f"Current URL: {driver.current_url}")
        logging.info(f"Content preview: {content[:200]}...")
        
        return content, driver.current_url
        
    except Exception as e:
        logging.error(f"Scraping failed: {str(e)}")
        raise e
    finally:
        driver.quit()

def scrape_data_page(base_url, relative_path):
    """Scrape data from a relative URL"""
    full_url = urljoin(base_url, relative_path)
    driver = setup_driver()
    try:
        logging.info(f"Scraping data from: {full_url}")
        
        driver.set_page_load_timeout(30)
        driver.get(full_url)
        
        WebDriverWait(driver, 20).until(
            lambda driver: driver.execute_script('return document.readyState') == 'complete'
        )
        
        time.sleep(2)
        
        body_element = driver.find_element(By.TAG_NAME, 'body')
        content = body_element.text
        
        logging.info(f"Scraped data: {content}")
        return content
        
    except Exception as e:
        logging.error(f"Data scraping failed: {str(e)}")
        raise e
    finally:
        driver.quit()

def extract_submit_url(quiz_content, current_url):
    """Extract submit URL from quiz content"""
    
    patterns = [
        r'POST (?:this JSON to|to) (https?://[^\s]+)',
        r'Post your answer to (https?://[^\s]+)',
        r'POST the secret code back to (/submit)',
        r'post to (https?://[^\s]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, quiz_content, re.IGNORECASE)
        if match:
            url = match.group(1)
            if url.startswith('/'):
                base_domain = re.match(r'(https?://[^/]+)', current_url)
                if base_domain:
                    url = base_domain.group(1) + url
            logging.info(f"Found submit URL: {url}")
            return url
    
    return "https://tds-llm-analysis.s-anand.net/submit"

def extract_data_url(quiz_content, current_url):
    """Extract data URL to scrape from quiz content"""
    
    patterns = [
        r'Scrape ([^\s]+)',
        r'scrape ([^\s]+)',
        r'visit ([^\s]+)',
        r'data from ([^\s]+)',
        r'(/demo-scrape-data[^\s]*)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, quiz_content, re.IGNORECASE)
        if match:
            relative_path = match.group(1)
            relative_path = re.sub(r'[.,)\]]', '', relative_path)
            # Fix: Properly handle the email parameter
            if 'email=' in quiz_content:
                email_match = re.search(r'email=([^\s&]+)', quiz_content)
                if email_match:
                    email = email_match.group(1)
                    if '?' in relative_path:
                        relative_path = relative_path + '&email=' + email
                    else:
                        relative_path = relative_path + '?email=' + email
            full_url = urljoin(current_url, relative_path)
            logging.info(f"Found data URL: {full_url}")
            return full_url
    
    return None

def extract_secret_code(scraped_data):
    """Extract secret code from scraped data - FIXED VERSION"""
    # The data shows: "Secret code is 27098 and not 27305."
    # So we need to extract the correct code
    
    # Pattern to find "Secret code is XXXX and not YYYY"
    pattern = r'Secret code is (\d+) and not \d+'
    match = re.search(pattern, scraped_data)
    if match:
        code = match.group(1)
        logging.info(f"Extracted secret code: {code}")
        return code
    
    # Fallback patterns
    patterns = [
        r'secret[:\s]*is[:\s]*(\d+)',
        r'code[:\s]*is[:\s]*(\d+)',
        r'(\d{4,6})',  # 4-6 digit numbers
    ]
    
    for pattern in patterns:
        match = re.search(pattern, scraped_data, re.IGNORECASE)
        if match:
            code = match.group(1)
            logging.info(f"Extracted code with fallback: {code}")
            return code
    
    return scraped_data.strip()

def solve_quiz_with_llm(quiz_text, scraped_data=None):
    if scraped_data:
        # For scraping tasks, extract the code directly - NO LLM NEEDED
        secret_code = extract_secret_code(scraped_data)
        return secret_code
    else:
        # For other tasks, use LLM
        quiz_text = quiz_text.replace('"', "'")[:2000]
        
        prompt = f"""
        Analyze this quiz and provide ONLY the answer in JSON format: {{"answer": "value"}}
        
        Quiz: {quiz_text}
        
        If this is a demo, answer with "demo_test".
        Return ONLY JSON, no other text.
        """
        
        try:
            logging.info("Calling Groq LLM")
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.1
            )
            raw_content = response.choices[0].message.content.strip()
            logging.info(f"LLM response: {raw_content}")
            
            json_match = re.search(r'\{[^}]*\}', raw_content)
            if json_match:
                result = json.loads(json_match.group(0))
                return result.get('answer', 'demo_test')
            return 'demo_test'
            
        except Exception as e:
            logging.error(f"LLM failed: {str(e)}")
            return 'demo_test'

def submit_answer(submit_url, payload):
    try:
        logging.info(f"Submitting to: {submit_url}")
        logging.info(f"Full payload: {json.dumps(payload, indent=2)}")
        
        headers = {'Content-Type': 'application/json'}
        response = requests.post(submit_url, json=payload, headers=headers, timeout=10)
        result = response.json()
        logging.info(f"Submission result: {result}")
        return result
    except requests.RequestException as e:
        logging.error(f"Submission failed: {str(e)}")
        raise e

@app.route('/solve', methods=['POST'])
def solve_quiz():
    start_time = time.time()
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'secret' not in data or 'url' not in data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        if data['secret'] != SECRET:
            return jsonify({"error": "Invalid secret"}), 403
        
        current_url = data['url']
        max_time = 180
        
        while current_url and (time.time() - start_time) < max_time:
            try:
                logging.info(f"Processing URL: {current_url}")
                
                # Scrape the quiz page
                quiz_content, final_url = scrape_quiz_page(current_url)
                current_url = final_url
                
                # Check if this is a scraping task
                data_url = extract_data_url(quiz_content, current_url)
                scraped_data = None
                
                if data_url:
                    logging.info(f"Scraping data from: {data_url}")
                    scraped_data = scrape_data_page(current_url, data_url)
                
                # Extract submit URL
                submit_url = extract_submit_url(quiz_content, current_url)
                
                # Solve the quiz
                answer = solve_quiz_with_llm(quiz_content, scraped_data)
                
                # IMPORTANT: Use the exact URL from the quiz page, not the current_url
                # The quiz expects the original URL, not the final redirected URL
                payload_url = data['url']  # Use the original URL from the request
                
                payload = {
                    "email": EMAIL,
                    "secret": SECRET,
                    "url": payload_url,  # Use original URL, not current_url
                    "answer": answer
                }
                
                # Submit answer
                result = submit_answer(submit_url, payload)
                
                if result.get('correct'):
                    current_url = result.get('url')
                    logging.info(f"Correct! Next URL: {current_url}")
                    if not current_url:
                        return jsonify({"status": "Quiz completed successfully!"}), 200
                else:
                    logging.warning(f"Incorrect: {result.get('reason', 'No reason')}")
                    # Check if we got a new URL to proceed to
                    if result.get('url'):
                        current_url = result.get('url')
                        logging.info(f"Proceeding to next URL despite incorrect: {current_url}")
                    else:
                        break
                    
            except Exception as e:
                logging.error(f"Error in quiz loop: {str(e)}")
                return jsonify({"error": f"Processing failed: {str(e)}"}), 500
        
        return jsonify({"status": "Timeout or completed"}), 200
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)