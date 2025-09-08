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

        result = urlunparse(resolved)
        if isinstance(result, bytes):
            result = result.decode("utf-8")
        return result
    except socket.gaierror as e:
        logger.warning(f"Could not resolve hostname in {url}: {e}")
        return url


def send_request_with_retry(url, files, data, max_retries=4, timeout=10, process_response=None):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, files=files, data=data, timeout=timeout)
            if response.status_code == 200:
                return process_response(response) if process_response else response
            elif 500 <= response.status_code < 600:  # retry on all server errors (5xx)
                logger.warning(f"Server error {response.status_code}, {response.text} retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:  # client errors (4xx) or other status codes - don't retry
                logger.warning(f"Client error {response.status_code} - {response.text} - not retrying")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2)
    logger.warning("Max retries exceeded")
    return None
