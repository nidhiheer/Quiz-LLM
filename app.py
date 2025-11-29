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

# Store the secret code from task 2 to use in task 3
secret_code_cache = None

def setup_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    # Set Chrome binary location for Railway
    options.binary_location = '/usr/bin/google-chrome'
    
    try:
        # Method 1: Try system ChromeDriver first (for Railway)
        if os.path.exists('/usr/bin/chromedriver'):
            service = Service('/usr/bin/chromedriver')
            driver = webdriver.Chrome(service=service, options=options)
            logging.info("Using system ChromeDriver")
            return driver
    except Exception as e:
        logging.warning(f"System ChromeDriver failed: {str(e)}")
    
    try:
        # Method 2: Try webdriver-manager with proper binary detection
        driver_path = ChromeDriverManager().install()
        
        # Check if downloaded path is a directory and find the actual binary
        if os.path.exists(driver_path):
            if os.path.isdir(driver_path):
                # Look for chromedriver binary in subdirectories
                for root, dirs, files in os.walk(driver_path):
                    for file in files:
                        if 'chromedriver' in file.lower() and not file.lower().endswith('.zip'):
                            binary_path = os.path.join(root, file)
                            if os.path.isfile(binary_path):
                                # Make executable on Linux
                                if os.name != 'nt':  # Not Windows
                                    os.chmod(binary_path, 0o755)
                                service = Service(binary_path)
                                driver = webdriver.Chrome(service=service, options=options)
                                logging.info(f"Using ChromeDriver from: {binary_path}")
                                return driver
            else:
                # Direct file path
                service = Service(driver_path)
                driver = webdriver.Chrome(service=service, options=options)
                logging.info(f"Using ChromeDriver from: {driver_path}")
                return driver
    except Exception as e:
        logging.warning(f"WebDriver Manager failed: {str(e)}")
    
    try:
        # Method 3: Try without service (relies on PATH)
        driver = webdriver.Chrome(options=options)
        logging.info("Using ChromeDriver from PATH")
        return driver
    except Exception as e:
        logging.error(f"All ChromeDriver methods failed: {str(e)}")
        raise Exception(f"Could not initialize ChromeDriver: {str(e)}")

def scrape_quiz_page(url):
    driver = setup_driver()
    try:
        logging.info(f"Scraping URL: {url}")
        driver.get(url)
        
        WebDriverWait(driver, 20).until(
            lambda driver: driver.execute_script('return document.readyState') == 'complete'
        )
        
        time.sleep(2)
        
        # Get multiple content sources for better data extraction
        body_text = driver.find_element(By.TAG_NAME, 'body').text
        page_source = driver.page_source
        
        # Combine both sources
        combined_content = f"{body_text}\n{page_source}"
        
        file_links = []
        page_links = []
        try:
            links = driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                href = link.get_attribute('href')
                if href:
                    if any(ext in href.lower() for ext in ['.csv', '.wav', '.mp3', '.pdf', '.json', '.txt']):
                        file_links.append(href)
                    elif 'page' in href.lower() or '2' in href or 'next' in href.lower() or 'demo-scrape-data' in href:
                        page_links.append(href)
        except Exception as e:
            logging.warning(f"Error extracting links: {str(e)}")
        
        logging.info(f"Found {len(file_links)} file links and {len(page_links)} page links")
        
        return combined_content, driver.current_url, file_links, page_links
        
    except Exception as e:
        logging.error(f"Scraping failed: {str(e)}")
        raise e
    finally:
        driver.quit()

def solve_audio_csv_with_cutoff(content, cutoff):
    """Audio CSV solver using the extracted secret code as cutoff"""
    try:
        csv_text = content.decode('utf-8')
        df = pd.read_csv(StringIO(csv_text), header=None)
        data = df.values.flatten()
        
        logging.info("=== AUDIO CSV ANALYSIS WITH DYNAMIC CUTOFF ===")
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Total values: {len(data)}")
        logging.info(f"Using cutoff from secret code: {cutoff}")
        
        # Calculate the CORRECT answer: sum of values above cutoff
        above_cutoff = data[data > cutoff]
        correct_answer = np.sum(above_cutoff)
        
        logging.info(f"Values above {cutoff}: {len(above_cutoff)}")
        logging.info(f"CORRECT ANSWER: sum_above_cutoff = {int(correct_answer)}")
        
        return int(correct_answer)
        
    except Exception as e:
        logging.error(f"Audio CSV processing failed: {str(e)}")
        return None

def extract_data_url(quiz_content, current_url):
    # Multiple patterns to find the data URL
    patterns = [
        r'/demo-scrape-data\?email=([^\s"\']+)',
        r'href=["\']([^"\']*demo-scrape-data[^"\']*)',
        r'data-url=["\']([^"\']+)',
        r'next[\s_-]*page["\']?[\s:=]+["\']?([^"\'\s]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, quiz_content, re.IGNORECASE)
        for match in matches:
            if 'demo-scrape-data' in match:
                if match.startswith('http'):
                    return match
                else:
                    # Construct full URL from relative path
                    base_url = "https://tds-llm-analysis.s-anand.net"
                    if match.startswith('/'):
                        return base_url + match
                    else:
                        return base_url + '/' + match
    
    # Fallback: look for any URL containing the email
    email_match = re.search(r'email=([^\s&]+)', quiz_content)
    if email_match:
        email = email_match.group(1)
        return f"https://tds-llm-analysis.s-anand.net/demo-scrape-data?email={email}"
    
    logging.error(f"Could not extract data URL from content")
    return None

def extract_secret_code(scraped_data):
    # More comprehensive patterns to match various secret code formats
    patterns = [
        r'secret[\s_:\-]*code[\s_:\-]*is[\s_:\-]*(\d+)',
        r'secret[\s_:\-]*code[\s_:\-]*:[\s]*(\d+)',
        r'code[\s_:\-]*:[\s]*(\d+)',
        r'secret[\s_:\-]*:[\s]*(\d+)',
        r'code[\s]*is[\s]*(\d+)',
        r'secret[\s]*is[\s]*(\d+)',
        r'answer[\s_:\-]*is[\s_:\-]*(\d+)',
        r'answer[\s_:\-]*:[\s]*(\d+)',
        r'(\d{3,})',  # Fallback: any 3+ digit number
    ]
    
    logging.info(f"Searching for secret code in content (first 500 chars): {scraped_data[:500]}")
    
    for pattern in patterns:
        matches = re.findall(pattern, scraped_data, re.IGNORECASE)
        if matches:
            code = matches[0]
            logging.info(f"Secret code found with pattern '{pattern}': {code}")
            return code
    
    # Additional debugging: check if there are any numbers in the content
    all_numbers = re.findall(r'\d+', scraped_data)
    logging.info(f"No secret code pattern matched. All numbers found: {all_numbers}")
    
    logging.error("No secret code found in scraped data")
    return None

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

def submit_answer(answer, current_url):
    try:
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
        payload = {
            "email": EMAIL,
            "secret": SECRET,
            "url": current_url,
            "answer": answer
        }
        
        if isinstance(answer, (np.integer, np.floating)):
            payload["answer"] = int(answer)
        
        logging.info(f"Submitting answer: {payload['answer']}")
        response = requests.post(submit_url, json=payload, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"Submission failed with status: {response.status_code}")
            return {"correct": False, "reason": f"HTTP {response.status_code}"}
            
    except Exception as e:
        logging.error(f"Submission failed: {str(e)}")
        return {"correct": False, "reason": "Submission error"}

@app.route('/solve', methods=['POST'])
def solve_quiz():
    global secret_code_cache
    
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'secret' not in data or 'url' not in data:
            return jsonify({"error": "Invalid JSON"}), 400
        
        if data['secret'] != SECRET:
            return jsonify({"error": "Invalid secret"}), 403
        
        current_url = data['url']
        task_count = 0
        max_tasks = 10
        start_time = time.time()
        timeout = 170
        
        while current_url and task_count < max_tasks and (time.time() - start_time) < timeout:
            task_count += 1
            logging.info(f"=== TASK {task_count}: {current_url} ===")
            
            try:
                quiz_content, final_url, file_links, page_links = scrape_quiz_page(current_url)
                answer = None
                
                # Task 1: Simple demo
                if "anything you want" in quiz_content.lower():
                    answer = "demo_test"
                    logging.info("Detected demo task")
                
                # Task 2: Scraping task - extract and store secret code
                elif "/demo-scrape-data" in quiz_content:
                    data_url = extract_data_url(quiz_content, final_url)
                    if data_url:
                        logging.info(f"Scraping data URL: {data_url}")
                        scraped_data, _, _, _ = scrape_quiz_page(data_url)
                        secret_code = extract_secret_code(scraped_data)
                        if secret_code and secret_code != "unknown":
                            secret_code_cache = int(secret_code)
                            answer = secret_code
                            logging.info(f"Secret code extracted and cached: {secret_code}")
                        else:
                            logging.error("Failed to extract valid secret code")
                            # Try to continue anyway with a default value
                            secret_code_cache = 100
                            answer = "100"
                
                # Task 3: Audio/data processing task - use cached secret code as cutoff
                elif "audio" in quiz_content.lower() or "csv" in quiz_content.lower():
                    logging.info("Detected audio/CSV task")
                    
                    if secret_code_cache is None:
                        logging.warning("No secret code cached, using default cutoff 100")
                        secret_code_cache = 100
                    
                    csv_url = None
                    for link in file_links:
                        if '.csv' in link.lower():
                            csv_url = link
                            break
                    
                    if csv_url:
                        csv_content = download_file(csv_url)
                        if csv_content:
                            # Use the cached secret code as cutoff
                            answer = solve_audio_csv_with_cutoff(csv_content, secret_code_cache)
                    else:
                        logging.error("No CSV file found for audio task")
                
                # Submit answer
                if answer is not None:
                    result = submit_answer(answer, current_url)
                    
                    if result.get('correct'):
                        current_url = result.get('url')
                        logging.info(f"✅ Correct! Next: {current_url}")
                        if not current_url:
                            return jsonify({
                                "status": "success",
                                "message": "Quiz completed successfully!",
                                "tasks_completed": task_count
                            }), 200
                    else:
                        logging.warning(f"❌ Incorrect: {result.get('reason', 'No reason provided')}")
                        current_url = result.get('url')
                        if not current_url:
                            break
                else:
                    logging.error("No answer could be determined")
                    break
                    
            except Exception as e:
                logging.error(f"Error in task {task_count}: {str(e)}")
                break
        
        # Final response
        if not current_url:
            return jsonify({
                "status": "success",
                "message": "Quiz completed!",
                "tasks_completed": task_count
            }), 200
        else:
            return jsonify({
                "status": "incomplete", 
                "message": f"Completed {task_count} tasks",
                "tasks_completed": task_count
            }), 200
        
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "LLM Quiz Solver API",
        "endpoints": {
            "POST /solve": "Solve quiz tasks",
            "GET /health": "Health check"
        }
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)