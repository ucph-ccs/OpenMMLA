import socket
import time
from urllib.parse import urlparse, urlunparse

import requests

from openmmla.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_url(url: str) -> str:
    """Resolve hostname in URL to IP address.
    
    Args:
        url: URL with hostname (e.g., 'http://uber-server.local:8080/infer')
        
    Returns:
        URL with resolved IP (e.g., 'http://192.168.1.100:8080/infer')
    """
    try:
        parsed = urlparse(url)
        if not parsed.hostname:
            return url

        ip = socket.gethostbyname(parsed.hostname)
        resolved = parsed._replace(netloc=f"{ip}:{parsed.port}" if parsed.port else ip)
        return urlunparse(resolved)
    except socket.gastrror as e:
        logger.warning(f"Could not resolve hostname in {url}: {e}")
        return url


def send_request_with_retry(url, files, data, max_retries=4, timeout=10, process_response=None):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, files=files, data=data, timeout=timeout)
            if response.status_code == 200 and process_response:
                return process_response(response)
            else:
                logger.warning(
                    f"Error in sending/requesting {url} with status code {response.status_code}: {response.text}")
                return None
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout occurred: {e}, retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}, retrying... Attempt {attempt + 1} of {max_retries}")
            time.sleep(2)
    logger.warning("Max retries exceeded.")
    return None
