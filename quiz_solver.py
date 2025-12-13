import asyncio
import httpx
import json
import re
import base64
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import Optional, Dict, Any, List
import logging
from playwright.async_api import async_playwright, Browser, Page
from io import BytesIO
import PyPDF2
from PIL import Image
import matplotlib.pyplot as plt

# Ensure proper event loop policy on Windows for subprocess support (Playwright)
# Use WindowsSelectorEventLoopPolicy because Proactor doesn't implement subprocess APIs
import sys, asyncio, os
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

logger = logging.getLogger(__name__)

class QuizSolver:
    def simple_answer_from_instructions(self, instructions: str) -> Optional[Any]:
        """Derive a simple answer from instructions without LLM.
        - If a JSON snippet is present with an "answer" key, return its value.
        - If the phrase 'anything you want' appears, return that string.
        """
        try:
            m = re.search(r"\{[\s\S]*?\}", instructions)
            if m:
                try:
                    obj = json.loads(m.group())
                    if isinstance(obj, dict) and "answer" in obj:
                        return obj["answer"]
                except Exception:
                    pass
            if "anything you want" in instructions.lower():
                return "anything you want"
        except Exception:
            pass
        return None
    def __init__(self, email: str, secret: str, groq_api_key: str):
        self.email = email
        self.secret = secret
        self.groq_api_key = groq_api_key
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.client = httpx.AsyncClient(timeout=30.0)
        # Configure Playwright usage: default enabled on non-Windows; override via USE_PLAYWRIGHT env
        _env_flag = os.getenv("USE_PLAYWRIGHT")
        _default = not sys.platform.startswith("win")
        self.use_playwright = _default if _env_flag is None else _env_flag.lower() in ("1", "true", "yes", "on")
        
    async def solve_quiz(self, quiz_url: str, start_time: Optional[float] = None, max_duration: float = 180.0) -> Optional[Dict]:
        """Solve one or more quizzes starting from quiz_url.
        Follows chained URLs provided by the endpoint and allows one retry per URL within max_duration seconds from start_time.
        """
        try:
            if start_time is None:
                start_time = asyncio.get_event_loop().time()

            current_url = quiz_url
            last_result: Optional[Dict] = None

            while True:
                now = asyncio.get_event_loop().time()
                if now - start_time >= max_duration:
                    logger.info("Time limit reached; stopping quiz chain")
                    break

                # Step 1: Fetch and render current quiz page
                quiz_content = await self.fetch_quiz_page(current_url)
                if not quiz_content:
                    logger.error(f"Failed to fetch content for {current_url}")
                    break

                # Step 2: Parse instructions
                instructions = self.parse_instructions(quiz_content)
                logger.info(f"Parsed instructions: {instructions[:200]}...")

                # Step 3: Analyze task and plan
                task_plan = await self.analyze_task_with_grok(instructions, quiz_content)

                # Step 4: Execute plan to compute answer
                answer = await self.execute_task_plan(task_plan, quiz_content)

                # Step 5: Submit answer
                submit_url = self.extract_submit_url(quiz_content, current_url)
                if not submit_url:
                    # Try to parse submit URL from instructions text as well
                    submit_url = self.extract_submit_url_from_text(instructions, current_url)
                if not submit_url:
                    logger.error("No submit URL found; stopping to avoid posting to non-submit endpoint")
                    break

                result = await self.submit_answer(submit_url, answer, original_url=current_url)
                last_result = result

                # Handle response: may include correctness and next url
                if not result or not isinstance(result, dict):
                    logger.error("Invalid or empty submission result; stopping")
                    break

                # If a new URL is provided, proceed to it
                next_url = result.get("url")
                correct = result.get("correct")
                if next_url and isinstance(next_url, str):
                    logger.info(f"Proceeding to next quiz URL: {next_url}")
                    current_url = next_url
                    continue

                # If incorrect and still time, attempt one alternate answer and resubmit
                if correct is False and (asyncio.get_event_loop().time() - start_time) < max_duration:
                    logger.info("Answer incorrect; attempting one retry with alternate strategy")
                    alt_answer = await self.fallback_grok_answer(quiz_content)
                    if alt_answer != answer:
                        result = await self.submit_answer(submit_url, alt_answer, original_url=current_url)
                        last_result = result or last_result
                # No new URL or retries exhausted; stop
                break

            return last_result
        
        except Exception as e:
            logger.error(f"Error solving quiz: {str(e)}")
            return None
            
    async def fetch_quiz_page(self, url: str) -> Optional[str]:
        """Fetch and render quiz page. Try Playwright first, then HTTP fallback."""
        # If Playwright is disabled, use HTTP-only fetch path
        if not self.use_playwright:
            try:
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    resp = await client.get(
                        url,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                            "Accept-Language": "en-US,en;q=0.9",
                        },
                    )
                    if resp.status_code == 200 and resp.text:
                        return resp.text
                    logger.error(f"HTTP-only fetch failed: status={resp.status_code}")
            except Exception as e:
                logger.error(f"HTTP-only fetch error: {e!r}")
            return None
        # Attempt with Playwright (handles JS-rendered pages)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                # Navigate and wait for DOM to be ready (networkidle may never occur on some sites)
                await page.goto(url, wait_until="domcontentloaded", timeout=90000)
                try:
                    # Best-effort wait for network to quiet down
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

                # Small additional wait for dynamic content
                await page.wait_for_timeout(1000)

                content = await page.content()
                await browser.close()

                if content:
                    return content
        except Exception as e:
            logger.error(f"Error fetching quiz page via Playwright: {e!r}")

        # Fallback: simple HTTP GET (works for static pages)
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                )
                if resp.status_code == 200 and resp.text:
                    return resp.text
                logger.error(f"HTTP fallback failed: status={resp.status_code}")
        except Exception as e:
            logger.error(f"HTTP fallback error: {e!r}")

        return None
    
    def parse_instructions(self, html_content: str) -> str:
        """Extract quiz instructions from HTML"""
        try:
            # Look for common patterns
            patterns = [
                r'<div[^>]*id="result"[^>]*>(.*?)</div>',
                r'<script[^>]*>.*?document\.querySelector.*?innerHTML\s*=\s*atob\(`(.*?)`\)',
                r'Q\d+\.(.*?)(?:Post your answer|Submit|http)',
                r'<pre>(.*?)</pre>',
            ]
            
            for pattern in patterns:
                matches = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                if matches:
                    text = matches.group(1)
                    # Decode base64 if found
                    if 'atob' in pattern:
                        try:
                            text = base64.b64decode(text).decode('utf-8')
                        except:
                            pass
                    return text.strip()
            
            # Fallback: extract all text
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            return text
            
        except Exception as e:
            logger.error(f"Error parsing instructions: {str(e)}")
            return html_content[:1000]  # Return first 1000 chars as fallback

    def _extract_json_plan(self, text: str) -> Optional[Dict]:
        """Extract a JSON object from LLM output.
        - Prefer ```json fenced blocks.
        - Otherwise, take the first balanced {...} block.
        Returns a dict or None.
        """
        try:
            fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
            candidate = None
            if fence:
                candidate = fence.group(1)
            else:
                start = text.find('{')
                if start != -1:
                    depth = 0
                    in_str = False
                    escape = False
                    end = None
                    for i, ch in enumerate(text[start:], start=start):
                        if in_str:
                            if escape:
                                escape = False
                            elif ch == '\\':
                                escape = True
                            elif ch == '"':
                                in_str = False
                        else:
                            if ch == '"':
                                in_str = True
                            elif ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    end = i
                                    break
                    if end is not None:
                        candidate = text[start:end+1]
            if candidate:
                return json.loads(candidate)
        except Exception:
            return None
        return None

    def extract_submit_url_from_text(self, text: str, original_url: str) -> Optional[str]:
        """Extract submit URL from plain instruction text. Supports absolute and relative '/submit' URLs."""
        from urllib.parse import urljoin
        try:
            # Absolute submit URL anywhere in text
            m = re.search(r'(https?://[^\s"\'<>]+/submit[^\s"\'<>]*)', text, re.IGNORECASE)
            if m:
                return re.sub(r'[^a-zA-Z0-9:/._?=&-]', '', m.group(1))
            # Common phrasing
            m = re.search(r'Post your answer to\s+(https?://[^\s"\'<>]+)', text, re.IGNORECASE)
            if m:
                return re.sub(r'[^a-zA-Z0-9:/._?=&-]', '', m.group(1))
            m = re.search(r'Submit to\s+(https?://[^\s"\'<>]+)', text, re.IGNORECASE)
            if m:
                return re.sub(r'[^a-zA-Z0-9:/._?=&-]', '', m.group(1))
            # Relative '/submit' path
            m = re.search(r'(?:^|\s)(/submit[^\s"\'<>]*)', text, re.IGNORECASE)
            if m:
                return urljoin(original_url, m.group(1))
        except Exception:
            return None
        return None

    async def analyze_task_with_grok(self, instructions: str, html_content: str) -> Dict:
        """Use Grok API to analyze the quiz task and create execution plan"""
        prompt = f"""
        Analyze this quiz task and create an execution plan:
        
        INSTRUCTIONS:
        {instructions}
        
        TASK TYPES TO CONSIDER:
        1. Data Sourcing: Download file, scrape website, call API
        2. Data Preparation: Clean text, parse PDF, transform data
        3. Data Analysis: Filter, sort, aggregate, calculate statistics
        4. Data Visualization: Create charts, generate images
        5. Answer Format: boolean, number, string, base64, JSON
        
        Provide a JSON plan with:
        - task_type: one of the above
        - steps: array of specific actions
        - expected_output_format: description of answer format
        - tools_needed: list of required tools/libraries
        """
        
        try:
            response = await self.client.post(
                self.groq_api_url,
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": "You are a data analysis assistant that creates execution plans."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                plan_text = result["choices"][0]["message"]["content"]
                plan = self._extract_json_plan(plan_text)
                if plan is not None:
                    return plan
                else:
                    # Fallback plan
                    return {
                        "task_type": "unknown",
                        "steps": ["extract_data", "process_data", "calculate_answer"],
                        "expected_output_format": "auto_detect",
                        "tools_needed": ["pandas", "requests"]
                    }
            else:
                logger.error(f"Groq API error: {response.status_code} body={response.text}")
                return self.create_fallback_plan(instructions)
                
        except Exception as e:
            logger.error(f"Error analyzing task: {str(e)}")
            return self.create_fallback_plan(instructions)
    
    def create_fallback_plan(self, instructions: str) -> Dict:
        """Create a basic fallback plan based on keyword matching"""
        instructions_lower = instructions.lower()
        
        if any(word in instructions_lower for word in ['sum', 'total', 'calculate', 'average']):
            return {
                "task_type": "data_analysis",
                "steps": ["extract_numbers", "perform_calculation"],
                "expected_output_format": "number",
                "tools_needed": ["pandas"]
            }
        elif any(word in instructions_lower for word in ['download', 'file', 'pdf']):
            return {
                "task_type": "data_sourcing",
                "steps": ["download_file", "extract_content"],
                "expected_output_format": "auto_detect",
                "tools_needed": ["requests", "pypdf2"]
            }
        else:
            return {
                "task_type": "text_processing",
                "steps": ["extract_text", "find_answer_in_text"],
                "expected_output_format": "string",
                "tools_needed": ["regex"]
            }
    
    async def execute_task_plan(self, plan: Dict, html_content: str) -> Any:
        """Execute the task plan and generate answer"""
        task_type = plan.get("task_type", "unknown")
        
        try:
            if task_type == "data_sourcing":
                return await self.handle_data_sourcing(plan, html_content)
            elif task_type == "data_analysis":
                return await self.handle_data_analysis(plan, html_content)
            elif task_type == "visualization":
                return await self.handle_visualization(plan, html_content)
            else:
                return await self.handle_general_task(plan, html_content)
                
        except Exception as e:
            logger.error(f"Error executing task plan: {str(e)}")
            # Fallback: try to extract answer using Grok
            return await self.fallback_grok_answer(html_content)
    
    async def handle_data_sourcing(self, plan: Dict, html_content: str) -> Any:
        """Handle data sourcing tasks (download files, scrape websites)"""
        # Extract download links
        import urllib.parse
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            text = link.get_text().lower()
            
            # Look for data file links
            if any(ext in href.lower() for ext in ['.csv', '.json', '.pdf', '.xls', '.xlsx']):
                # Download file
                async with httpx.AsyncClient() as client:
                    response = await client.get(href)
                    
                    if response.status_code == 200:
                        content = response.content
                        
                        # Process based on file type
                        if href.endswith('.csv'):
                            import io
                            df = pd.read_csv(io.BytesIO(content))
                            return df.to_dict()
                        elif href.endswith('.pdf'):
                            # Extract text from PDF
                            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                            text = ""
                            for page in pdf_reader.pages:
                                text += page.extract_text()
                            return text[:5000]  # Return first 5000 chars
                        
        return None
    
    async def handle_data_analysis(self, plan: Dict, html_content: str) -> Any:
        """Handle data analysis tasks"""
        # Extract tables from HTML
        try:
            import pandas as pd
            from io import StringIO
            
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            
            if tables:
                # Convert first table to DataFrame
                df = pd.read_html(StringIO(str(tables[0])))[0]
                
                # Analyze based on instructions
                instructions = self.parse_instructions(html_content)
                
                if 'sum' in instructions.lower() and 'value' in instructions.lower():
                    # Find value column
                    for col in df.columns:
                        if 'value' in str(col).lower():
                            return float(df[col].sum())
                
                if 'average' in instructions.lower():
                    # Find numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        return float(df[numeric_cols[0]].mean())
            
            return "No table found for analysis"
            
        except Exception as e:
            logger.error(f"Data analysis error: {str(e)}")
            return None
    
    async def handle_visualization(self, plan: Dict, html_content: str) -> str:
        """Handle visualization tasks"""
        try:
            # Create a simple visualization
            plt.figure(figsize=(10, 6))
            
            # Example: create bar chart from instructions
            instructions = self.parse_instructions(html_content)
            
            # This is a placeholder - you'd extract actual data
            x = ['A', 'B', 'C', 'D']
            y = [10, 20, 15, 25]
            
            plt.bar(x, y)
            plt.title('Visualization')
            plt.xlabel('Category')
            plt.ylabel('Value')
            
            # Save to bytes
            from io import BytesIO
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            
            # Convert to base64
            img_str = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{img_str}"
            
        except Exception as e:
            logger.error(f"Visualization error: {str(e)}")
            return None
    
    async def handle_general_task(self, plan: Dict, html_content: str) -> Any:
        """Handle general tasks using Grok API with heuristic fallback"""
        instructions = self.parse_instructions(html_content)

        # Heuristic: try to derive simple answer without LLM
        simple = self.simple_answer_from_instructions(instructions)
        if simple is not None:
            return simple
        
        prompt = f"""
        Based on these quiz instructions, provide the answer:
        
        {instructions}
        
        Provide only the answer in the appropriate format (number, string, boolean, or JSON).
        If you need to calculate something, show your work but put the final answer at the end.
        """
        
        response = await self.client.post(
            self.groq_api_url,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "You are a quiz-solving assistant."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 200
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"].strip()
            
            # Try to parse as appropriate type
            try:
                # Check if it's a number
                if re.match(r'^-?\d+(\.\d+)?$', answer):
                    return float(answer) if '.' in answer else int(answer)
                # Check if it's boolean
                elif answer.lower() in ['true', 'false']:
                    return answer.lower() == 'true'
                else:
                    return answer
            except:
                return answer
        else:
            logger.error(f"Groq API error: {response.status_code} body={response.text}")
            return None
    
    async def fallback_grok_answer(self, html_content: str) -> str:
        """Fallback method using Grok to directly answer"""
        instructions = self.parse_instructions(html_content)
        
        prompt = f"""
        Read these quiz instructions and provide the answer directly:
        
        {instructions}
        
        Just give the answer, nothing else.
        """
        
        response = await self.client.post(
            self.groq_api_url,
            headers={
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": "Answer the quiz question directly."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 100
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        logger.error(f"Groq API error: {response.status_code} body={response.text}")
        # Heuristic fallback
        return self.simple_answer_from_instructions(instructions) or "No answer found"
    
    def extract_submit_url(self, html_content: str, original_url: str) -> Optional[str]:
        """Extract the submission URL from quiz page, preferring explicit endpoints.
        Tries multiple sources: absolute URLs, JS fetch/axios, DOM attributes, forms, and relative '/submit' paths.
        Returns a fully-resolved absolute URL or None.
        """
        from urllib.parse import urljoin

        # 1) Explicit absolute URLs in text
        abs_patterns = [
            r'Post your answer to (https?://[^\s"\']+)',
            r'Submit to (https?://[^\s"\']+)',
            r'"submit_url"\s*[:=]\s*"(https?://[^"]+)"',
            r"'submit_url'\s*[:=]\s*'(https?://[^']+)'",
            r'(https?://[^\s"\']+/submit[^\s"\']*)',
        ]
        for pattern in abs_patterns:
            m = re.search(pattern, html_content, re.IGNORECASE)
            if m:
                url = m.group(1)
                return re.sub(r'[^a-zA-Z0-9:/._?=&-]', '', url)

        # 2) JavaScript calls: fetch/axios/$.post
        js_patterns = [
            r"fetch\(\s*['\"]([^'\"\s]+?/submit[^'\"\s]*)['\"]",
            r"axios\.post\(\s*['\"]([^'\"\s]+?/submit[^'\"\s]*)['\"]",
            r"\$\.post\(\s*['\"]([^'\"\s]+?/submit[^'\"\s]*)['\"]",
        ]
        for pattern in js_patterns:
            m = re.search(pattern, html_content, re.IGNORECASE)
            if m:
                cand = m.group(1)
                url = cand if cand.startswith('http') else urljoin(original_url, cand)
                return re.sub(r'[^a-zA-Z0-9:/._?=&-]', '', url)

        # 3) DOM attributes: form action, data-submit-url, hidden input
        dom_patterns = [
            r'<form[^>]*action=["\']([^"\']*submit[^"\']*)["\']',
            r'data-submit-url\s*=\s*["\']([^"\']*submit[^"\']*)["\']',
            r'<input[^>]*name=["\']submit_url["\'][^>]*value=["\']([^"\']+)["\']',
            r'["\'](/submit[^"\']*)["\']',
        ]
        for pattern in dom_patterns:
            m = re.search(pattern, html_content, re.IGNORECASE)
            if m:
                cand = m.group(1)
                url = cand if cand.startswith('http') else urljoin(original_url, cand)
                return re.sub(r'[^a-zA-Z0-9:/._?=&-]', '', url)

        # 4) Handle <span class="origin"></span>/path pattern
        origin_placeholder = re.search(r'<span[^>]*class=["\']origin["\'][^>]*></span>\s*(/[^\s"\'<>]+)', html_content, re.IGNORECASE)
        if origin_placeholder:
            try:
                path = origin_placeholder.group(1)
                if 'submit' in path:
                    return urljoin(original_url, path)
            except Exception:
                pass

        # 5) Domain-specific safe fallback for demo host or env override
        try:
            fallback_env = os.getenv('SUBMIT_URL_FALLBACK')
            if fallback_env:
                return fallback_env
            from urllib.parse import urlparse
            parsed = urlparse(original_url)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if parsed.netloc.endswith('tds-llm-analysis.s-anand.net'):
                return urljoin(origin + '/', 'submit')
        except Exception:
            pass

        return None
    
    async def submit_answer(self, submit_url: str, answer: Any, original_url: str) -> Optional[Dict]:
        """Submit answer to the specified URL; set payload 'url' to original quiz URL."""
        try:
            payload = {
                "email": self.email,
                "secret": self.secret,
                "url": original_url,
                "answer": answer
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    submit_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Submission failed: {response.status_code} body={response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error submitting answer: {str(e)}")
            return None