# tools/search_tools.py
import re
import json
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Union

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("search_tools")

def ensure_soup(potential_soup):
    """Ensures the input is a BeautifulSoup object."""
    if isinstance(potential_soup, BeautifulSoup):
        return potential_soup
    
    # If it's a string, try to parse it
    if isinstance(potential_soup, str):
        logger.warning("Converting string to BeautifulSoup object")
        try:
            return BeautifulSoup(potential_soup, 'html.parser')
        except Exception as e:
            logger.error(f"Failed to convert string to BeautifulSoup: {str(e)}")
            raise ValueError(f"Input cannot be converted to BeautifulSoup: {str(e)}")
    
    # Try to cast to string and then parse
    try:
        logger.warning(f"Converting {type(potential_soup)} to BeautifulSoup object")
        return BeautifulSoup(str(potential_soup), 'html.parser')
    except Exception as e:
        logger.error(f"Failed to convert input to BeautifulSoup: {str(e)}")
        raise ValueError(f"Input must be a BeautifulSoup object or valid HTML string, not {type(potential_soup)}")

def search_text(content: str, search_term: str, case_sensitive: bool = False) -> List[str]:
    """
    Searches for a term in the given text content and returns matching lines.
    
    Args:
        content: The text content to search in
        search_term: The term to search for
        case_sensitive: Whether the search should be case sensitive
    
    Returns:
        A list of matching lines with the search term
    """
    # Handle boolean parameters that might come as strings
    if isinstance(case_sensitive, str):
        case_sensitive = case_sensitive.lower() == 'true'
    
    # Convert content to string if it's not already
    if not isinstance(content, str):
        content = str(content)
    
    if not case_sensitive:
        search_term = search_term.lower()
    
    results = []
    lines = content.splitlines()
    
    for line in lines:
        if not case_sensitive and search_term in line.lower():
            results.append(line)
        elif case_sensitive and search_term in line:
            results.append(line)
    
    return results

def search_html(soup, search_term: str, case_sensitive: bool = False) -> Dict[str, List[str]]:
    """
    Searches for a term in different parts of an HTML document.
    
    Args:
        soup: A BeautifulSoup object representing the HTML
        search_term: The term to search for
        case_sensitive: Whether the search should be case sensitive
    
    Returns:
        A dictionary with categories of matches (text content, attributes, scripts, etc.)
    """
    # Handle boolean parameters that might come as strings
    if isinstance(case_sensitive, str):
        case_sensitive = case_sensitive.lower() == 'true'
    
    # Ensure we have a BeautifulSoup object
    try:
        soup = ensure_soup(soup)
    except Exception as e:
        logger.error(f"Failed to process soup: {str(e)}")
        return {
            "error": [f"Failed to process HTML: {str(e)}"],
            "text_content": [],
            "attributes": [],
            "comments": [],
            "scripts": []
        }
    
    if not case_sensitive:
        search_term = search_term.lower()
    
    results = {
        "text_content": [],
        "attributes": [],
        "comments": [],
        "scripts": [],
    }
    
    # Search in text content
    try:
        for text in soup.stripped_strings:
            if not case_sensitive and search_term in text.lower():
                results["text_content"].append(text)
            elif case_sensitive and search_term in text:
                results["text_content"].append(text)
    except Exception as e:
        logger.error(f"Error searching text content: {str(e)}")
        results["text_content"].append(f"Error: {str(e)}")
    
    # Search in attributes
    try:
        for tag in soup.find_all(True):  # Find all tags
            for attr, value in tag.attrs.items():
                if isinstance(value, str):  # Handle string attributes
                    if not case_sensitive and search_term in value.lower():
                        results["attributes"].append(f"{tag.name}[{attr}] = {value}")
                    elif case_sensitive and search_term in value:
                        results["attributes"].append(f"{tag.name}[{attr}] = {value}")
                elif isinstance(value, list):  # Handle list attributes (like class)
                    for item in value:
                        if isinstance(item, str):
                            if not case_sensitive and search_term in item.lower():
                                results["attributes"].append(f"{tag.name}[{attr}] = {item}")
                            elif case_sensitive and search_term in item:
                                results["attributes"].append(f"{tag.name}[{attr}] = {item}")
    except Exception as e:
        logger.error(f"Error searching attributes: {str(e)}")
        results["attributes"].append(f"Error: {str(e)}")
    
    # Search in comments
    try:
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            if not case_sensitive and search_term in comment.lower():
                results["comments"].append(comment)
            elif case_sensitive and search_term in comment:
                results["comments"].append(comment)
    except Exception as e:
        logger.error(f"Error searching comments: {str(e)}")
        results["comments"].append(f"Error: {str(e)}")
    
    # Search in script tags
    try:
        for script in soup.find_all('script'):
            if script.string:
                if not case_sensitive and search_term in script.string.lower():
                    script_lines = script.string.splitlines()
                    for line in script_lines:
                        if not case_sensitive and search_term in line.lower():
                            results["scripts"].append(line)
                        elif case_sensitive and search_term in line:
                            results["scripts"].append(line)
    except Exception as e:
        logger.error(f"Error searching scripts: {str(e)}")
        results["scripts"].append(f"Error: {str(e)}")
    
    return results

def find_by_selector(soup, css_selector: str) -> List[Any]:
    """
    Finds elements in the HTML using CSS selectors.
    
    Args:
        soup: A BeautifulSoup object representing the HTML
        css_selector: A CSS selector string
    
    Returns:
        A list of matching elements
    """
    # Ensure we have a BeautifulSoup object
    try:
        soup = ensure_soup(soup)
    except Exception as e:
        logger.error(f"Failed to process soup: {str(e)}")
        return [f"Error: Failed to process HTML: {str(e)}"]
    
    try:
        return soup.select(css_selector)
    except Exception as e:
        logger.error(f"Error executing selector: {str(e)}")
        return [f"Error executing selector: {str(e)}"]

def extract_links(soup, filter_pattern: str = None) -> List[Dict[str, str]]:
    """
    Extracts all links from the HTML document with optional filtering.
    
    Args:
        soup: A BeautifulSoup object representing the HTML
        filter_pattern: Optional regex pattern to filter links
    
    Returns:
        A list of dictionaries with link details (text, href, title)
    """
    # Ensure we have a BeautifulSoup object
    try:
        soup = ensure_soup(soup)
    except Exception as e:
        logger.error(f"Failed to process soup: {str(e)}")
        return [{"error": f"Failed to process HTML: {str(e)}"}]
    
    links = []
    
    try:
        for a_tag in soup.find_all('a', href=True):
            href = a_tag.get('href', '')
            
            # Skip if there's a filter and it doesn't match
            if filter_pattern and not re.search(filter_pattern, href):
                continue
                
            link_info = {
                'text': a_tag.get_text(strip=True),
                'href': href,
                'title': a_tag.get('title', '')
            }
            links.append(link_info)
    except Exception as e:
        logger.error(f"Error extracting links: {str(e)}")
        links.append({"error": f"Error: {str(e)}"})
    
    return links

def find_forms(soup) -> List[Dict[str, Any]]:
    """
    Extracts details about forms in the HTML document.
    
    Args:
        soup: A BeautifulSoup object representing the HTML
    
    Returns:
        A list of dictionaries with form details
    """
    # Ensure we have a BeautifulSoup object
    try:
        soup = ensure_soup(soup)
    except Exception as e:
        logger.error(f"Failed to process soup: {str(e)}")
        return [{"error": f"Failed to process HTML: {str(e)}"}]
    
    forms_data = []
    
    try:
        for form in soup.find_all('form'):
            form_info = {
                'action': form.get('action', ''),
                'method': form.get('method', 'get').upper(),
                'id': form.get('id', ''),
                'name': form.get('name', ''),
                'inputs': []
            }
            
            # Get all inputs
            for input_tag in form.find_all(['input', 'textarea', 'select', 'button']):
                input_info = {
                    'type': input_tag.get('type', 'text') if input_tag.name == 'input' else input_tag.name,
                    'name': input_tag.get('name', ''),
                    'id': input_tag.get('id', ''),
                    'value': input_tag.get('value', ''),
                    'required': input_tag.has_attr('required'),
                    'placeholder': input_tag.get('placeholder', '')
                }
                form_info['inputs'].append(input_info)
                
            forms_data.append(form_info)
    except Exception as e:
        logger.error(f"Error finding forms: {str(e)}")
        forms_data.append({"error": f"Error: {str(e)}"})
    
    return forms_data
