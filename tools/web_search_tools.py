# tools/web_search_tools.py
import requests
import json
import re
import logging
import time
import random
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Union, Optional
from urllib.parse import quote_plus, urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_search_tools")

class SearchRateLimiter:
    """Simple rate limiter to prevent overloading search engines."""
    
    def __init__(self, min_delay: float = 1.0, max_delay: float = 3.0):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0
    
    def wait(self):
        """Wait an appropriate amount of time between requests."""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_delay:
            # Add a random delay to be polite to search engines
            delay = self.min_delay - elapsed + random.uniform(0, self.max_delay - self.min_delay)
            logger.info(f"Rate limiting: Waiting {delay:.2f} seconds before next request")
            time.sleep(delay)
        
        self.last_request_time = time.time()

# Create a global rate limiter instance
rate_limiter = SearchRateLimiter()

def search_duckduckgo(query: str, num_results: int = 5, safe_search: bool = True) -> List[Dict[str, str]]:
    """
    Perform a web search using DuckDuckGo and return the results.
    
    Args:
        query: The search query
        num_results: The number of results to return (max 25)
        safe_search: Whether to enable safe search filtering
    
    Returns:
        A list of dictionaries containing search results with title, url, and snippet
    """
    # Validate input
    if not query or not isinstance(query, str):
        logger.error("Invalid query provided")
        return [{"error": "Invalid search query"}]
    
    # Convert num_results to int if it's a string
    if isinstance(num_results, str):
        try:
            num_results = int(num_results)
        except ValueError:
            logger.warning(f"Invalid num_results: {num_results}, using default of 5")
            num_results = 5
    
    # Limit the number of results to prevent abuse
    num_results = min(max(1, num_results), 25)
    
    # Convert safe_search to boolean if it's a string
    if isinstance(safe_search, str):
        safe_search = safe_search.lower() == 'true'
    
    # Respect rate limits
    rate_limiter.wait()
    
    # Use DuckDuckGo's HTML API
    safe_search_param = "1" if safe_search else "-1"
    encoded_query = quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}&kp={safe_search_param}"
    
    logger.info(f"Searching DuckDuckGo for: {query}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the HTML response
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract search results
        results = []
        result_elements = soup.select('.result')
        
        for i, result in enumerate(result_elements):
            if i >= num_results:
                break
                
            # Extract result details
            title_element = result.select_one('.result__title')
            title = title_element.get_text(strip=True) if title_element else "No title"
            
            link_element = result.select_one('.result__url')
            url = link_element.get_text(strip=True) if link_element else ""
            
            # Fix URL if needed (DuckDuckGo sometimes shows partial URLs)
            if url and not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            
            # Get the actual URL from the a tag if available
            a_tag = title_element.find('a') if title_element else None
            href = a_tag.get('href') if a_tag else None
            if href and href.startswith('/'):
                # This is a DuckDuckGo redirect URL, try to extract the actual URL
                href_match = re.search(r'uddg=([^&]+)', href)
                if href_match:
                    from urllib.parse import unquote
                    url = unquote(href_match.group(1))
            
            snippet_element = result.select_one('.result__snippet')
            snippet = snippet_element.get_text(strip=True) if snippet_element else "No description available"
            
            results.append({
                "title": title,
                "url": url,
                "snippet": snippet
            })
        
        logger.info(f"Found {len(results)} search results for query: {query}")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error performing search: {str(e)}")
        return [{"error": f"Search request failed: {str(e)}"}]
    except Exception as e:
        logger.error(f"Unexpected error during search: {str(e)}")
        return [{"error": f"Unexpected error: {str(e)}"}]

def fetch_webpage(url: str) -> Dict[str, Any]:
    """
    Fetch a webpage and return its content.
    
    Args:
        url: The URL of the webpage to fetch
    
    Returns:
        A dictionary containing the webpage content, status, and metadata
    """
    # Validate URL
    if not url or not isinstance(url, str):
        logger.error("Invalid URL provided")
        return {"error": "Invalid URL", "content": "", "status_code": 0, "headers": {}}
    
    # Add scheme if missing
    if not url.startswith(('http://', 'https://')):
        url = f"https://{url}"
    
    # Validate URL structure
    try:
        parsed_url = urlparse(url)
        if not parsed_url.netloc:
            logger.error(f"Invalid URL structure: {url}")
            return {"error": "Invalid URL structure", "content": "", "status_code": 0, "headers": {}}
    except Exception as e:
        logger.error(f"URL parsing error: {str(e)}")
        return {"error": f"URL parsing error: {str(e)}", "content": "", "status_code": 0, "headers": {}}
    
    # Respect rate limits
    rate_limiter.wait()
    
    logger.info(f"Fetching webpage: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        status_code = response.status_code
        
        # Get response headers
        headers_dict = dict(response.headers)
        
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type or 'application/xhtml+xml' in content_type:
            # Parse HTML content
            try:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else "No title"
                
                # Extract metadata
                meta_tags = {}
                for meta in soup.find_all('meta'):
                    name = meta.get('name', meta.get('property', ''))
                    content = meta.get('content', '')
                    if name and content:
                        meta_tags[name] = content
                
                return {
                    "title": title,
                    "content": response.text,
                    "status_code": status_code,
                    "headers": headers_dict,
                    "url": response.url,  # Final URL after any redirects
                    "meta": meta_tags,
                    "html": soup,  # Return the BeautifulSoup object for further processing
                }
            except Exception as e:
                logger.error(f"Error parsing HTML: {str(e)}")
                return {
                    "error": f"Error parsing HTML: {str(e)}",
                    "content": response.text,
                    "status_code": status_code,
                    "headers": headers_dict,
                    "url": response.url
                }
        else:
            # Handle non-HTML content
            logger.info(f"Non-HTML content detected: {content_type}")
            return {
                "content_type": content_type,
                "content": response.text if len(response.content) < 1024 * 1024 else "Content too large to display",
                "status_code": status_code,
                "headers": headers_dict,
                "url": response.url
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching webpage: {str(e)}")
        return {"error": f"Request failed: {str(e)}", "content": "", "status_code": 0, "headers": {}}
    except Exception as e:
        logger.error(f"Unexpected error fetching webpage: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}", "content": "", "status_code": 0, "headers": {}}

def extract_content(html_result: Dict[str, Any], max_length: int = 5000) -> str:
    """
    Extract the main content from a webpage, removing boilerplate.
    
    Args:
        html_result: The result from fetch_webpage
        max_length: Maximum length of the extracted content
    
    Returns:
        A string containing the main content of the webpage
    """
    if "error" in html_result and html_result["error"]:
        logger.error(f"Cannot extract content from error result: {html_result['error']}")
        return f"Error: {html_result['error']}"
    
    if "html" not in html_result or not html_result["html"]:
        logger.error("No HTML content to extract from")
        return "No HTML content to extract from"
    
    try:
        soup = html_result["html"]
        
        # Convert max_length to int if it's a string
        if isinstance(max_length, str):
            try:
                max_length = int(max_length)
            except ValueError:
                logger.warning(f"Invalid max_length: {max_length}, using default of 5000")
                max_length = 5000
        
        # Extract title
        title = soup.title.string if soup.title else "No title"
        
        # Remove unwanted elements
        for element in soup.select('script, style, header, footer, nav, aside, iframe, .ad, .advertisement, .banner'):
            element.decompose()
        
        # Try to find the main content
        main_content = soup.select_one('main, #main, #content, .main, .content, article, .article')
        
        if main_content:
            # Use the identified main content
            content = main_content.get_text(separator=' ', strip=True)
        else:
            # Fallback to extracting paragraphs
            paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]
            content = ' '.join(paragraphs)
        
        # Clean up the content
        content = re.sub(r'\s+', ' ', content).strip()
        
        # Limit the length
        if len(content) > max_length:
            content = content[:max_length] + "... [content truncated]"
        
        result = f"Title: {title}\n\n{content}"
        logger.info(f"Successfully extracted content (length: {len(result)})")
        return result
        
    except Exception as e:
        logger.error(f"Error extracting content: {str(e)}")
        return f"Error extracting content: {str(e)}"

def summarize_search_results(search_results: List[Dict[str, str]], query: str) -> str:
    """
    Create a formatted summary of search results.
    
    Args:
        search_results: The results from search_duckduckgo
        query: The original search query
    
    Returns:
        A formatted string with search results summary
    """
    if not search_results:
        return "No search results found."
    
    if "error" in search_results[0]:
        return f"Search error: {search_results[0]['error']}"
    
    summary = f"Search results for: {query}\n\n"
    
    for i, result in enumerate(search_results):
        title = result.get("title", "No title")
        url = result.get("url", "No URL")
        snippet = result.get("snippet", "No description available")
        
        summary += f"{i+1}. {title}\n"
        summary += f"   URL: {url}\n"
        summary += f"   {snippet}\n\n"
    
    return summary

def search_and_summarize(query: str, num_results: int = 5, fetch_top_result: bool = False) -> str:
    """
    Perform a search and create a summary of the results.
    Optionally fetch and summarize the top result.
    
    Args:
        query: The search query
        num_results: Number of results to include in summary
        fetch_top_result: Whether to fetch and summarize the top result
    
    Returns:
        A string with search results and optionally top result content
    """
    # Convert arguments if they're strings
    if isinstance(num_results, str):
        try:
            num_results = int(num_results)
        except ValueError:
            num_results = 5
    
    if isinstance(fetch_top_result, str):
        fetch_top_result = fetch_top_result.lower() == 'true'
    
    # Perform the search
    search_results = search_duckduckgo(query, num_results)
    
    if not search_results or "error" in search_results[0]:
        if "error" in search_results[0]:
            return f"Search error: {search_results[0]['error']}"
        return "No search results found."
    
    # Create the summary
    summary = summarize_search_results(search_results, query)
    
    # Optionally fetch and summarize the top result
    if fetch_top_result and search_results:
        top_result_url = search_results[0].get("url")
        if top_result_url:
            summary += "\n--- TOP RESULT CONTENT ---\n\n"
            try:
                webpage = fetch_webpage(top_result_url)
                content = extract_content(webpage)
                summary += content
            except Exception as e:
                summary += f"Error fetching top result: {str(e)}"
    
    return summary
