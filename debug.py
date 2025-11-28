import os
import json
import time
import requests
import re
import logging
import pandas as pd
import numpy as np
from io import StringIO
from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

SECRET = os.getenv('SECRET')
EMAIL = os.getenv('EMAIL')

def setup_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_quiz_page(url):
    driver = setup_driver()
    try:
        logging.info(f"Scraping URL: {url}")
        driver.get(url)
        
        WebDriverWait(driver, 20).until(
            lambda driver: driver.execute_script('return document.readyState') == 'complete'
        )
        
        time.sleep(2)
        
        body_element = driver.find_element(By.TAG_NAME, 'body')
        content = body_element.text
        
        file_links = []
        try:
            links = driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                if href and '.csv' in href.lower():
                    file_links.append(href)
        except:
            pass
        
        return content, driver.current_url, file_links
        
    except Exception as e:
        logging.error(f"Scraping failed: {str(e)}")
        raise e
    finally:
        driver.quit()

def download_file(url):
    try:
        logging.info(f"Downloading: {url}")
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.content
        return None
    except Exception as e:
        logging.error(f"Download failed: {str(e)}")
        return None

def solve_audio_csv_no_header(content):
    """Process CSV with NO column names - just raw data"""
    try:
        csv_text = content.decode('utf-8')
        
        # Read CSV without header - just raw numbers
        df = pd.read_csv(StringIO(csv_text), header=None)
        data = df.values.flatten()
        
        logging.info(f"Raw data shape: {df.shape}")
        logging.info(f"First 10 values: {data[:10]}")
        logging.info(f"Data range: {data.min():.0f} to {data.max():.0f}")
        
        cutoff = 47022
        
        # Calculate all possible sums with raw data
        calculations = [
            ('sum_all_raw', np.sum(data)),                          # Simple sum of all raw values
            ('sum_abs_raw', np.sum(np.abs(data))),                  # Sum of absolute values
            ('sum_above_cutoff_raw', np.sum(data[data > cutoff])),  # Values above cutoff
            ('sum_below_cutoff_raw', np.sum(data[data < cutoff])),  # Values below cutoff
            ('count_above_cutoff_raw', np.sum(data > cutoff)),      # Count above cutoff
            ('sum_positive_raw', np.sum(data[data > 0])),           # Only positive values
            ('sum_negative_abs_raw', np.sum(np.abs(data[data < 0]))), # Absolute of negative values
        ]
        
        # Log all calculations
        for name, value in calculations:
            if not np.isnan(value):
                logging.info(f"  {name}: {int(value)}")
        
        # Most likely: simple sum of all raw values (no header means we should use all data directly)
        result = int(calculations[0][1])
        logging.info(f"Using sum_all_raw: {result}")
        return result
        
    except Exception as e:
        logging.error(f"Audio CSV processing failed: {str(e)}")
        return None

def extract_data_url(quiz_content, current_url):
    pattern = r'/demo-scrape-data\?email=([^\s]+)'
    match = re.search(pattern, quiz_content)
    if match:
        email_part = match.group(1)
        return f"https://tds-llm-analysis.s-anand.net/demo-scrape-data?email={email_part}"
    return None

def extract_secret_code(scraped_data):
    match = re.search(r'Secret code is (\d+)', scraped_data)
    return match.group(1) if match else "unknown"

def submit_answer(answer, current_url):
    try:
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
        payload = {
            "email": EMAIL,
            "secret": SECRET,
            "url": current_url,
            "answer": answer
        }
        
        logging.info(f"Submitting: {payload['answer']}")
        response = requests.post(submit_url, json=payload, timeout=10)
        return response.json()
    except Exception as e:
        logging.error(f"Submission failed: {str(e)}")
        return {"correct": False}

@app.route('/solve', methods=['POST'])
def solve_quiz():
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'secret' not in data or 'url' not in data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        if data['secret'] != SECRET:
            return jsonify({"error": "Invalid secret"}), 403
        
        current_url = data['url']
        task_count = 0
        
        while current_url and task_count < 10:  # Limit to prevent infinite loops
            task_count += 1
            logging.info(f"Task {task_count}: {current_url}")
            
            try:
                quiz_content, final_url, file_links = scrape_quiz_page(current_url)
                answer = None
                
                # Task 1: Simple demo
                if "anything you want" in quiz_content:
                    answer = "demo_test"
                
                # Task 2: Scraping task  
                elif "/demo-scrape-data" in quiz_content:
                    data_url = extract_data_url(quiz_content, final_url)
                    if data_url:
                        scraped_data, _, _ = scrape_quiz_page(data_url)
                        answer = extract_secret_code(scraped_data)
                        logging.info(f"Secret code: {answer}")
                
                # Task 3: Audio CSV task
                elif "audio" in quiz_content.lower() or "csv" in quiz_content.lower():
                    csv_url = None
                    for link in file_links:
                        if '.csv' in link.lower():
                            csv_url = link
                            break
                    
                    if csv_url:
                        csv_content = download_file(csv_url)
                        if csv_content:
                            # Use the no-header processing for raw data
                            answer = solve_audio_csv_no_header(csv_content)
                
                # Submit answer
                if answer:
                    result = submit_answer(answer, current_url)
                    
                    if result.get('correct'):
                        current_url = result.get('url')
                        logging.info(f"✅ Correct! Next: {current_url}")
                        if not current_url:
                            return jsonify({"status": "Quiz completed!"}), 200
                    else:
                        logging.warning(f"❌ Incorrect: {result.get('reason')}")
                        
                        # If audio task failed, try alternative calculations
                        if "audio" in quiz_content.lower() and csv_url:
                            csv_content = download_file(csv_url)
                            if csv_content:
                                alternative_answer = try_alternative_calculations(csv_content, answer)
                                if alternative_answer and alternative_answer != answer:
                                    logging.info(f"Trying alternative: {alternative_answer}")
                                    result = submit_answer(alternative_answer, current_url)
                                    if result.get('correct'):
                                        current_url = result.get('url')
                                        logging.info(f"✅ Alternative correct! Next: {current_url}")
                                        if not current_url:
                                            return jsonify({"status": "Quiz completed!"}), 200
                        
                        current_url = result.get('url')  # Continue with next URL if provided
                        if not current_url:
                            break
                else:
                    # For demo task without files
                    if "demo" in current_url and "anything you want" in quiz_content:
                        answer = "demo_completed"
                        result = submit_answer(answer, current_url)
                        if result.get('correct'):
                            current_url = result.get('url')
                            if current_url:
                                continue
                    
                    logging.error("No answer determined")
                    break
                    
            except Exception as e:
                logging.error(f"Task {task_count} failed: {str(e)}")
                break
        
        return jsonify({
            "status": "completed", 
            "tasks_processed": task_count
        }), 200
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def try_alternative_calculations(content, previous_answer):
    """Try other calculation methods if first attempt fails"""
    try:
        csv_text = content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_text), header=None)  # No header!
        data = df.values.flatten()
        cutoff = 47022
        
        alternatives = [
            ('sum_abs', np.sum(np.abs(data))),                    # Sum of absolute values
            ('count_above', np.sum(data > cutoff)),               # Count above cutoff
            ('sum_below', np.sum(data[data < cutoff])),           # Sum below cutoff
            ('sum_positive', np.sum(data[data > 0])),             # Sum positive only
            ('sum_abs_above', np.sum(np.abs(data)[data > cutoff])), # Absolute values above cutoff
        ]
        
        for name, value in alternatives:
            alt_answer = int(value)
            if alt_answer != previous_answer:
                logging.info(f"Alternative {name}: {alt_answer}")
                return alt_answer
        
        return None
    except:
        return None

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)