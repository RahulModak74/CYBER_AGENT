# tools/bs4_tools.py
from bs4 import BeautifulSoup
import sys
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bs4_tools")

def parse_html(html_content, parser="html.parser"):
    """Parses HTML content and creates a BeautifulSoup object.
    
    This function has been updated to handle both string and raw HTML content,
    and to provide better error handling for parser selection.
    """
    # For debugging
    logger.info(f"Parsing HTML content (type: {type(html_content)}, length: {len(str(html_content)[:100])}...)")
    
    # Default to a more commonly available parser if the requested one isn't available
    available_parsers = ["html.parser", "lxml", "html5lib"]
    
    # If parser is wrapped in quotes, remove them
    if isinstance(parser, str) and (parser.startswith('"') or parser.startswith("'")):
        parser = parser.strip('"\'')
    
    # Handle the 'html_content' placeholder in example calls
    if html_content == 'html_content':
        # This is a placeholder - we need actual HTML content
        raise ValueError("Missing actual HTML content. Replace 'html_content' with the HTML you want to parse.")
    
    # Try the requested parser first
    try:
        soup = BeautifulSoup(html_content, parser)
        logger.info(f"Successfully parsed HTML with {parser}")
        return soup
    except Exception as e:
        # If the requested parser fails, try the others
        for backup_parser in available_parsers:
            if backup_parser != parser:
                try:
                    logger.warning(f"Warning: '{parser}' parser failed, using '{backup_parser}' instead")
                    return BeautifulSoup(html_content, backup_parser)
                except:
                    continue
                
        # If all parsers fail, raise the original error
        logger.error(f"Failed to parse HTML: {str(e)}")
        raise Exception(f"Failed to parse HTML with any available parser. Original error: {str(e)}")

def find_element(soup, tag, attributes=None):
    """Finds the first HTML element matching the given tag or attributes."""
    # Ensure soup is a BeautifulSoup object
    if not isinstance(soup, BeautifulSoup):
        logger.warning(f"Input is not a BeautifulSoup object but {type(soup)}. Attempting to convert.")
        try:
            soup = BeautifulSoup(str(soup), 'html.parser')
        except Exception as e:
            logger.error(f"Failed to convert input to BeautifulSoup: {str(e)}")
            raise ValueError(f"Input must be a BeautifulSoup object or valid HTML string, not {type(soup)}")
    
    # Handle attributes being a string representation of a dictionary
    if isinstance(attributes, str):
        if attributes.strip() in ['{}', 'None', '']:
            attributes = {}
        else:
            try:
                # Try to evaluate the string as a dict (be careful with this!)
                import ast
                attributes = ast.literal_eval(attributes)
            except:
                logger.error(f"Failed to parse attributes string: {attributes}")
                attributes = {}
    
    logger.info(f"Finding element: tag={tag}, attributes={attributes}")
    return soup.find(tag, attributes)

def find_all_elements(soup, tag, attributes=None):
    """Finds all HTML elements matching the given tag or attributes."""
    # Ensure soup is a BeautifulSoup object
    if not isinstance(soup, BeautifulSoup):
        logger.warning(f"Input is not a BeautifulSoup object but {type(soup)}. Attempting to convert.")
        try:
            soup = BeautifulSoup(str(soup), 'html.parser')
        except Exception as e:
            logger.error(f"Failed to convert input to BeautifulSoup: {str(e)}")
            raise ValueError(f"Input must be a BeautifulSoup object or valid HTML string, not {type(soup)}")
    
    # Handle attributes being a string representation of a dictionary
    if isinstance(attributes, str):
        if attributes.strip() in ['{}', 'None', '']:
            attributes = {}
        else:
            try:
                # Try to evaluate the string as a dict (be careful with this!)
                import ast
                attributes = ast.literal_eval(attributes)
            except:
                logger.error(f"Failed to parse attributes string: {attributes}")
                attributes = {}
    
    logger.info(f"Finding all elements: tag={tag}, attributes={attributes}")
    return soup.find_all(tag, attributes)

def get_text(element):
    """Extracts text content from an HTML element."""
    if element is None:
        return None
    
    # For BeautifulSoup objects, use get_text()
    if hasattr(element, 'get_text'):
        return element.get_text()
    
    # For strings, return as is
    if isinstance(element, str):
        return element
    
    # For Tag objects, convert to string
    return str(element)
