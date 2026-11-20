
from urllib.parse import urlparse

MAX_URL_LENGTH = 2048
TIMEOUT = 10


def validate_url(raw_url: str) -> str:
    url = (raw_url or "").strip()

    if not url or len(url) > MAX_URL_LENGTH:
        raise ValueError("Please provide a valid website URL.")

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    if not parsed.hostname or not parsed.netloc:
        raise ValueError("Invalid website URL.")

    return url