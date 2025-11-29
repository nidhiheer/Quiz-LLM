import os
import json
import time
import requests
import re
import logging
import pandas as pd
import numpy as np
import platform
from io import StringIO
from flask import Flask, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

SECRET = os.getenv('SECRET')
EMAIL = os.getenv('EMAIL')

secret_code_cache = None

# ==============================
# CHROME DRIVER SETUP
# ==============================

def setup_driver():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation', 'enable-logging'])
    options.add_experimental_option('useAutomationExtension', False)
    
    system = platform.system().lower()
    is_windows = system == 'windows'
    is_linux = system == 'linux'
    
    logging.info(f"Detected platform: {system}")
    
    if is_windows:
        pass
    elif is_linux:
        options.binary_location = '/usr/bin/google-chrome'
        if os.path.exists('/usr/bin/chromedriver'):
            service = Service('/usr/bin/chromedriver')
    
    driver_attempts = []
    
    if is_windows:
        driver_attempts = [
            lambda: webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options),
            lambda: webdriver.Chrome(options=options),
            lambda: webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options),
        ]
    else:
        driver_attempts = [
            lambda: webdriver.Chrome(service=Service('/usr/bin/chromedriver'), options=options),
            lambda: webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options),
            lambda: webdriver.Chrome(options=options),
        ]
    
    errors = []
    for i, attempt in enumerate(driver_attempts):
        try:
            driver = attempt()
            logging.info(f"ChromeDriver initialized successfully with method {i+1} on {system}")
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except Exception as e:
            error_msg = f"Method {i+1} failed: {str(e)}"
            errors.append(error_msg)
            logging.warning(f"{error_msg}")
            continue
    
    logging.info("Trying fallback driver configuration...")
    try:
        fallback_options = Options()
        fallback_options.add_argument('--headless=new')
        fallback_options.add_argument('--no-sandbox')
        fallback_options.add_argument('--disable-dev-shm-usage')
        
        if is_windows:
            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=fallback_options)
        else:
            driver = webdriver.Chrome(options=fallback_options)
            
        logging.info("Fallback ChromeDriver initialization successful")
        return driver
    except Exception as e:
        logging.error(f"Fallback also failed: {str(e)}")
    
    raise Exception(f"All ChromeDriver initialization methods failed on {system}. Errors: {'; '.join(errors)}")

# ==============================
# WEB SCRAPING FUNCTIONS
# ==============================

def scrape_quiz_page(url):
    max_retries = 2
    for attempt in range(max_retries):
        driver = None
        try:
            driver = setup_driver()
            logging.info(f"Scraping URL (attempt {attempt + 1}/{max_retries}): {url}")
            
            driver.set_page_load_timeout(45)
            driver.get(url)
            
            WebDriverWait(driver, 30).until(
                lambda driver: driver.execute_script('return document.readyState') == 'complete'
            )
            
            time.sleep(1)
            
            body_text = driver.find_element(By.TAG_NAME, 'body').text
            page_source = driver.page_source
            current_url = driver.current_url
            
            file_links = []
            page_links = []
            try:
                links = driver.find_elements(By.TAG_NAME, 'a')
                for link in links:
                    href = link.get_attribute('href')
                    text = link.text.strip()
                    if href:
                        if any(ext in href.lower() for ext in ['.csv', '.wav', '.mp3', '.pdf', '.json', '.txt']):
                            file_links.append({'url': href, 'text': text})
                        elif any(keyword in text.lower() or keyword in href.lower() 
                               for keyword in ['page', 'next', 'continue', 'demo-scrape-data', 'task', 'question']):
                            page_links.append({'url': href, 'text': text})
            except Exception as e:
                logging.warning(f"Error extracting links: {str(e)}")
            
            logging.info(f"Found {len(file_links)} file links and {len(page_links)} page links")
            
            return {
                'content': body_text,
                'source': page_source,
                'current_url': current_url,
                'file_links': file_links,
                'page_links': page_links,
                'full_content': body_text + "\n" + page_source
            }
            
        except Exception as e:
            logging.error(f"Scraping attempt {attempt + 1} failed for {url}: {str(e)}")
            if attempt == max_retries - 1:
                raise e
            else:
                logging.info(f"Retrying in 2 seconds...")
                time.sleep(2)
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass

# ==============================
# QUIZ SOLVING FUNCTIONS
# ==============================

def solve_audio_csv_with_cutoff(content, cutoff):
    try:
        if isinstance(content, bytes):
            csv_text = content.decode('utf-8')
        else:
            csv_text = content
            
        df = pd.read_csv(StringIO(csv_text), header=None)
        data = df.values.flatten()
        
        logging.info("=== AUDIO CSV ANALYSIS ===")
        logging.info(f"Data shape: {df.shape}")
        logging.info(f"Data range: {np.min(data)} to {np.max(data)}")
        logging.info(f"Using cutoff: {cutoff}")
        
        above_cutoff = data[data > cutoff]
        correct_answer = np.sum(above_cutoff)
        
        logging.info(f"Values above {cutoff}: {len(above_cutoff)}")
        logging.info(f"Sum of values above cutoff: {int(correct_answer)}")
        
        return int(correct_answer)
        
    except Exception as e:
        logging.error(f"Audio CSV processing failed: {str(e)}")
        return None

def extract_next_url(scraped_data, current_task_data):
    content = scraped_data['full_content']
    file_links = scraped_data['file_links']
    page_links = scraped_data['page_links']
    
    next_keywords = ['next', 'continue', 'page', 'task', 'question', 'proceed']
    for link_info in page_links:
        link_text = link_info['text'].lower()
        link_url = link_info['url']
        if any(keyword in link_text for keyword in next_keywords):
            logging.info(f"Found next URL via keyword: {link_url}")
            return link_url
    
    for link_info in page_links:
        if 'demo-scrape-data' in link_info['url']:
            logging.info(f"Found demo-scrape-data URL: {link_info['url']}")
            return link_info['url']
    
    for link_info in file_links:
        if '.csv' in link_info['url'].lower():
            logging.info(f"Found CSV URL: {link_info['url']}")
            return link_info['url']
    
    patterns = [
        r'href=["\']([^"\']*demo-scrape-data[^"\']*)',
        r'next[\s_-]*page["\']?[\s:=]+["\']?([^"\'\s]+)',
        r'continue[\s_-]*to["\']?[\s:=]+["\']?([^"\'\s]+)',
        r'url[\s:=]+["\']?([^"\'\s]+)',
        r'/(task|question|page)-\d+',
    ]
    
    base_url = "https://tds-llm-analysis.s-anand.net"
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            if match and not match.startswith('#'):
                url = match
                if not url.startswith('http'):
                    if url.startswith('/'):
                        url = base_url + url
                    else:
                        url = base_url + '/' + url
                logging.info(f"Found URL with pattern '{pattern}': {url}")
                return url
    
    email_match = re.search(r'email=([^\s&]+)', content)
    if email_match:
        email = email_match.group(1)
        url = f"{base_url}/demo-scrape-data?email={email}"
        logging.info(f"Constructed URL from email: {url}")
        return url
    
    logging.warning("No next URL found in content")
    return None

def extract_secret_code(scraped_data):
    content = scraped_data['content']
    
    clean_text = re.sub(r'<[^>]+>', ' ', content)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    
    logging.info(f"Searching for secret code in: {clean_text[:200]}...")
    
    patterns = [
        r'secret[\s_:\-]*code[\s_:\-]*is[\s_:\-]*[#]?[\s]*(\d{2,})',
        r'secret[\s_:\-]*code[\s_:\-]*:[\s]*[#]?[\s]*(\d{2,})',
        r'code[\s_:\-]*:[\s]*[#]?[\s]*(\d{2,})',
        r'secret[\s_:\-]*:[\s]*[#]?[\s]*(\d{2,})',
        r'code[\s]*is[\s]*[#]?[\s]*(\d{2,})',
        r'secret[\s]*is[\s]*[#]?[\s]*(\d{2,})',
        r'answer[\s_:\-]*is[\s_:\-]*[#]?[\s]*(\d{2,})',
        r'answer[\s_:\-]*:[\s]*[#]?[\s]*(\d{2,})',
        r'final[\s_:\-]*code[\s_:\-]*:[\s]*[#]?[\s]*(\d{2,})',
        r'use[\s_:\-]*code[\s_:\-]*(\d{2,})',
        r'cutoff[\s_:\-]*:[\s]*(\d{2,})',
        r'cutoff[\s]*is[\s]*(\d{2,})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, clean_text, re.IGNORECASE)
        if matches:
            code = matches[0]
            logging.info(f"Secret code found with pattern '{pattern}': {code}")
            return int(code)
    
    all_numbers = re.findall(r'\b\d{2,4}\b', clean_text)
    logging.info(f"All numbers found: {all_numbers}")
    
    if all_numbers:
        for num in all_numbers:
            num_int = int(num)
            if 50 <= num_int <= 500:
                logging.info(f"Selected likely secret code: {num_int}")
                return num_int
        
        largest = max(map(int, all_numbers))
        logging.info(f"Using largest number as secret code: {largest}")
        return largest
    
    logging.error("No secret code found")
    return None

def download_file(url):
    for attempt in range(2):
        try:
            logging.info(f"Downloading ({attempt+1}/2): {url}")
            response = requests.get(url, timeout=25)
            if response.status_code == 200:
                return response.content
            logging.warning(f"Download failed with status {response.status_code}")
        except Exception as e:
            logging.warning(f"Download attempt {attempt+1} failed: {str(e)}")
            time.sleep(1)
    
    logging.error(f"All download attempts failed for: {url}")
    return None

def submit_answer(answer, current_url):
    try:
        submit_url = "https://tds-llm-analysis.s-anand.net/submit"
        
        if isinstance(answer, (np.integer, np.int64)):
            answer = int(answer)
        elif isinstance(answer, (np.floating, np.float64)):
            answer = float(answer)
        
        payload = {
            "email": EMAIL,
            "secret": SECRET,
            "url": current_url,
            "answer": answer
        }
        
        logging.info(f"Submitting answer: {payload['answer']}")
        response = requests.post(submit_url, json=payload, timeout=25)
        
        if response.status_code == 200:
            result = response.json()
            logging.info(f"Submission response: {result}")
            return result
        else:
            logging.error(f"Submission failed with status: {response.status_code}")
            return {"correct": False, "reason": f"HTTP {response.status_code}"}
            
    except Exception as e:
        logging.error(f"Submission failed: {str(e)}")
        return {"correct": False, "reason": "Submission error"}

def solve_task(task_data):
    global secret_code_cache
    
    content = task_data['full_content']
    current_url = task_data['current_url']
    file_links = task_data['file_links']
    
    logging.info(f"Solving task from: {current_url}")
    
    if "anything you want" in content.lower() or "demo" in content.lower():
        logging.info("Detected demo task")
        return "demo_test"
    
    elif "demo-scrape-data" in content.lower() or "secret code" in content.lower():
        logging.info("Detected secret code extraction task")
        
        next_url = extract_next_url(task_data, None)
        if next_url:
            logging.info(f"Scraping data from: {next_url}")
            scraped_data = scrape_quiz_page(next_url)
            secret_code = extract_secret_code(scraped_data)
            
            if secret_code:
                secret_code_cache = secret_code
                logging.info(f"Secret code found and cached: {secret_code}")
                return secret_code
            else:
                logging.warning("No secret code found, using default 100")
                secret_code_cache = 100
                return 100
        else:
            logging.error("No data URL found for scraping task")
            return None
    
    elif "audio" in content.lower() or "csv" in content.lower() or "cutoff" in content.lower():
        logging.info("Detected audio CSV processing task")
        
        if secret_code_cache is None:
            logging.warning("No secret code cached, using default cutoff 100")
            secret_code_cache = 100
        
        csv_url = None
        for link_info in file_links:
            if '.csv' in link_info['url'].lower():
                csv_url = link_info['url']
                break
        
        if csv_url:
            csv_content = download_file(csv_url)
            if csv_content:
                answer = solve_audio_csv_with_cutoff(csv_content, secret_code_cache)
                return answer
            else:
                logging.error("Failed to download CSV file")
                return None
        else:
            logging.error("No CSV file found")
            return None
    
    else:
        logging.warning("Unknown task type, attempting to find answer in content")
        numbers = re.findall(r'\b\d{3,}\b', content)
        if numbers:
            logging.info(f"Found potential answer in content: {numbers[0]}")
            return int(numbers[0])
        
        return None

# ==============================
# FLASK ROUTES
# ==============================

@app.route('/solve', methods=['POST'])
def solve_quiz():
    global secret_code_cache
    
    try:
        data = request.get_json()
        if not data or 'email' not in data or 'secret' not in data or 'url' not in data:
            return jsonify({"error": "Invalid JSON: missing email, secret, or url"}), 400
        
        if data['secret'] != SECRET:
            return jsonify({"error": "Invalid secret"}), 403
        
        current_url = data['url']
        task_count = 0
        max_tasks = 10
        start_time = time.time()
        overall_timeout = 300
        
        task_chain = []
        
        while current_url and task_count < max_tasks and (time.time() - start_time) < overall_timeout:
            task_count += 1
            task_start_time = time.time()
            task_timeout = 180
            
            logging.info(f"TASK {task_count}: {current_url}")
            
            try:
                if (time.time() - task_start_time) > task_timeout:
                    logging.warning(f"Task {task_count} timeout after 3 minutes")
                    break
                
                task_data = scrape_quiz_page(current_url)
                task_chain.append({
                    'task_number': task_count,
                    'url': current_url,
                    'content_preview': task_data['content'][:200] + '...',
                    'time_taken': round(time.time() - task_start_time, 2)
                })
                
                answer = solve_task(task_data)
                
                if answer is not None:
                    result = submit_answer(answer, current_url)
                    
                    if result.get('correct'):
                        current_url = result.get('url')
                        task_time = round(time.time() - task_start_time, 2)
                        logging.info(f"Task {task_count} CORRECT (took {task_time}s)")
                        
                        if not current_url:
                            total_time = round(time.time() - start_time, 2)
                            logging.info(f"QUIZ CHAIN COMPLETED SUCCESSFULLY! Total time: {total_time}s")
                            return jsonify({
                                "status": "success",
                                "message": "Quiz chain completed successfully!",
                                "tasks_completed": task_count,
                                "total_time_seconds": total_time,
                                "task_chain": task_chain,
                                "final_secret_code": secret_code_cache
                            }), 200
                        else:
                            logging.info(f"Next task: {current_url}")
                    else:
                        reason = result.get('reason', 'No reason provided')
                        task_time = round(time.time() - task_start_time, 2)
                        logging.warning(f"Task {task_count} INCORRECT: {reason} (took {task_time}s)")
                        current_url = result.get('url')
                        if not current_url:
                            break
                else:
                    task_time = round(time.time() - task_start_time, 2)
                    logging.error(f"Could not solve task {task_count} (took {task_time}s)")
                    break
                    
            except Exception as e:
                task_time = round(time.time() - task_start_time, 2)
                logging.error(f"Error in task {task_count}: {str(e)} (took {task_time}s)")
                if 'current_url' in locals() and current_url:
                    continue
                else:
                    break
        
        total_time = round(time.time() - start_time, 2)
        if not current_url:
            return jsonify({
                "status": "success",
                "message": "Quiz chain completed!",
                "tasks_completed": task_count,
                "total_time_seconds": total_time,
                "task_chain": task_chain,
                "final_secret_code": secret_code_cache
            }), 200
        else:
            return jsonify({
                "status": "incomplete", 
                "message": f"Completed {task_count} tasks in the chain",
                "tasks_completed": task_count,
                "total_time_seconds": total_time,
                "task_chain": task_chain,
                "next_url": current_url,
                "final_secret_code": secret_code_cache
            }), 200
        
    except Exception as e:
        logging.error(f"Unexpected error in solve_quiz: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "LLM Quiz Chain Solver API",
        "description": "Solves sequential quiz tasks in a chain with 3-minute timeout per task",
        "endpoints": {
            "POST /solve": "Solve quiz task chain starting from given URL",
            "GET /health": "Health check"
        },
        "timeouts": {
            "per_task": "180 seconds",
            "overall": "1800 seconds"
        }
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)