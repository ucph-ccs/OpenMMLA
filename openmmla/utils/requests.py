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


def build_service_url(config: dict, value: str, resolve: bool = True) -> str:
    """Resolve a Server-section endpoint value into a full, IP-resolved URL.

    Two forms are accepted and may be mixed per endpoint:
      * A full URL ('http://host:8080/enhance', 'https://...', 'rtmp://...') is
        used as-is, bypassing the shared gateway (direct connection), then
        DNS-resolved via ``resolve_url``.
      * A bare gateway-relative endpoint name ('enhance', 'vad') is composed as
        ``{Gateway.scheme}://{Gateway.host}:{Gateway.http_port}/{endpoint}`` from
        the shared ``Gateway`` section, then DNS-resolved.

    Args:
        config: full pipeline config dict (needs a ``Gateway`` section for
            bare-name composition).
        value: the Server-section value for a single service endpoint.
        resolve: False keeps the host name (a URL to show, or to ask once).

    Returns:
        A full URL with the hostname resolved to an IP where possible.
    """
    if not value:
        return value
    keep = resolve_url if resolve else str
    if str(value).startswith(("http://", "https://", "rtmp://")):
        return keep(value)
    gateway = (config or {}).get("Gateway") or {}
    host = gateway.get("host", "localhost")
    scheme = gateway.get("scheme", "http")
    port = gateway.get("http_port", 8080)
    endpoint = str(value).lstrip("/")
    return keep(f"{scheme}://{host}:{port}/{endpoint}")


# what the services behind a URL have, by what asking it gave (service_problems)
_CAUSES = {
    "no-route": ("the Gateway at {origin} has no route to {paths}: it was started while none of their servers "
                 "could be reached, so start the Gateway (Nginx) card again"),
    "no-server": "the Gateway at {origin} routes {paths}, but none of the servers behind it answers: is the server running?",
    "down": "nothing answers at {origin} ({paths})",
    "slow": "{origin} does not answer in time ({paths})",
}


def _gateway_cause(response) -> str | None:
    """what an error page of Nginx's own (the Gateway's) says of the service
    behind it: "no-route" (it has no location for the path), "no-server" (it
    has one, and no server there answers); None for any other answer, which
    came from a service."""
    if "<center>nginx" not in (response.text or ""):
        return None
    if response.status_code == 404:
        return "no-route"
    if response.status_code in (502, 503, 504):
        return "no-server"
    return None


def _said(response) -> str:
    """a failed answer in a few words: its status, what it means when it is
    an error page of the Gateway's own, and else its text unless that is an
    HTML page."""
    said = f"{response.status_code} {response.reason or ''}".rstrip()
    cause = _gateway_cause(response)
    if cause == "no-route":
        return f"{said} (the Gateway has no route to it)"
    if cause == "no-server":
        return f"{said} (the Gateway has no server for it that answers)"
    text = (response.text or "").strip()
    if text and not text.startswith("<"):
        said += f": {text[:300]}"
    return said


def service_problems(urls: list[str], timeout: float = 5.0) -> list[str]:
    """why the services at `urls` cannot serve, asked with a bare GET (a
    service that runs answers it, if only with 405): one sentence per cause,
    naming the paths it hits. Empty when every one answers."""
    causes: dict[tuple[str, str], list[str]] = {}
    for url in urls:
        parsed = urlparse(url)
        try:
            cause = _gateway_cause(requests.get(url, timeout=timeout))
        except requests.exceptions.Timeout:
            cause = "slow"
        except requests.exceptions.RequestException:
            cause = "down"
        if cause:
            causes.setdefault((cause, f"{parsed.scheme}://{parsed.netloc}"), []).append(parsed.path or "/")
    return [_CAUSES[cause].format(origin=origin, paths=_listed(paths)) for (cause, origin), paths in causes.items()]


def _listed(items: list[str]) -> str:
    """a, b and c."""
    return f"{', '.join(items[:-1])} and {items[-1]}" if len(items) > 1 else items[0]


def send_request_with_retry(url, files, data, max_retries=4, timeout=10, process_response=None):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, files=files, data=data, timeout=timeout)
            if response.status_code == 200:
                return process_response(response) if process_response else response
            elif 500 <= response.status_code < 600:  # retry on all server errors (5xx)
                logger.warning(f"Server error {_said(response)} from {url}, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:  # client errors (4xx) or other status codes - don't retry
                logger.warning(f"Client error {_said(response)} from {url} - not retrying")
                return None
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error: {e}, retrying... ({attempt + 1}/{max_retries})")
            time.sleep(2)
    logger.warning("Max retries exceeded")
    return None
