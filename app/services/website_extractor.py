import re
from bs4 import BeautifulSoup, Comment
from urllib.parse import urlparse, urljoin, urldefrag, unquote

from app.config import settings
import requests


NOISE_ATTR_WORDS = ("cookie", "consent", "popup", "modal", "newsletter", "subscribe", "breadcrumb")
NOISE_MAX_CHARS = 2000
HEADERS = {"User-Agent": "Mozilla/5.0"}

NOISE_TAGS = ["script", "style", "noscript", "svg", "iframe",
              "nav", "footer", "aside", "button"]

LINK_KEYWORDS = {"service": 3, "solution": 3, "product": 3,
                 "about": 2, "company": 1, "pricing": 1}
LINK_SKIP_WORDS = ("blog", "career", "privacy", "terms", "login", "cart")
SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".zip")


def clean_body_text(soup: BeautifulSoup, heading_texts: set[str])-> str:
    for c in soup.find_all(string= lambda s: isinstance(s, Comment)):
        c.extract()

    for tag in soup.find_all(NOISE_TAGS):
        tag.decompose()

    for tag in soup.select('[hidden], [aria-hidden="true"]'):
        tag.decompose()

    for tag in soup.find_all(["div", "section", "ul", "p", "span"]):
        if tag.decomposed:  # already deleted along with its parent
            continue
        label = " ".join([tag.get("id") or "", *(tag.get("class") or [])]).lower()
        if any(word in label for word in NOISE_ATTR_WORDS):
            if len(tag.get_text(strip=True)) < NOISE_MAX_CHARS:  # big wrapper = real content, keep
                tag.decompose()

    area = soup.body or soup
    for name in ("main", "article"):
        found = soup.find(name)
        if found and len(found.get_text(strip=True)) >= 500:
            area = found
            break

    lines, seen = [], set()
    for raw in area.get_text("\n").split("\n"):
        line = re.sub(r"\s+", " ", raw).strip()          # 6. tidy spaces
        if not line:
            continue
        if len(line.split()) < 2 and len(line) < 20 and line.lower() not in heading_texts:
            continue                                      # 7. drop "Menu", "Login"
        if line.lower() in seen:                          # 8. drop duplicates
            continue
        seen.add(line.lower())
        lines.append(line)
    return "\n".join(lines)


def fetch_html(url: str) -> tuple[str, bytes]:
   try:
       resp = requests.get(url, headers= HEADERS, timeout = settings.FETCH_TIMEOUT)
   except requests.Timeout:
       raise ValueError(f"Timeout after { settings.FETCH_TIMEOUT } seconds")
   except requests.RequestException:
        raise ValueError("Could not connect to the website.")

   if resp.status_code in (401, 403, 429):
        raise ValueError("The website blocked access.")
   if resp.status_code >= 400:
        raise ValueError(f"The website returned HTTP {resp.status_code}.")
   if "html" not in resp.headers.get("content-type", "").lower():
        raise ValueError("The URL did not return an HTML page.")

   return resp.url, resp.content

def parse_page(html: bytes, page_url: str)-> dict:
    soup = BeautifulSoup(html, "lxml")

    title = soup.title.get_text(strip = True)[:200] if soup.title else ""

    meta_tag = soup.find("meta", attrs = {"name": "description"})
    meta_description = (meta_tag.get("content") or "").strip()[:400] if meta_tag else ""

    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = re.sub(r"\s+", " ", tag.get_text(" ", strip=True))
        if 2 < len(text) <= 150 and text not in headings:
            headings.append(text)

    candidate_links = find_candidate_links(soup, page_url)

    text = clean_body_text(soup, {h.lower() for h in headings})

    return {
        "title": title,
        "meta_description": meta_description,
        "headings": headings[:20],
        "candidate_links": candidate_links,
        "text": text,
    }


def find_candidate_links(soup, page_url):
    home_host = urlparse(page_url).netloc.lower().removeprefix("www.")
    seen, scored = set(), []

    for a in soup.find_all("a", href = True):
        link, _ = urldefrag(urljoin(page_url, a["href"].strip()))
        parsed = urlparse(link)

        if parsed.scheme not in ("http", "https"):
            continue
        if parsed.netloc.lower().removeprefix("www.") != home_host:
            continue
        path = unquote(parsed.path).lower()
        if path in ("", "/") or path.endswith(SKIP_EXTENSIONS):
            continue

        key = link.rstrip("/")
        if key in seen:
            continue
        seen.add(key)

        # 5. text to check = URL path + visible link text
        text = f"{path} {a.get_text(' ', strip=True)}".lower()
        text = text.replace("-", " ").replace("_", " ")

        # 6. skip unwanted pages (login, blog...)
        if any(word in text for word in LINK_SKIP_WORDS):
            continue

        # 7. score by keywords
        score = sum(w for word, w in LINK_KEYWORDS.items() if word in text)
        if score > 0:
            scored.append((score, link))

    scored.sort(key=lambda item: -item[0])   # best score first
    return [link for _, link in scored[:6]]

def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0]

def extract_website(url: str) -> dict:
    # 1. homepage
    final_url, html = fetch_html(url)
    home = parse_page(html, final_url)

    # 2. subpages
    subpages = []
    for link in home["candidate_links"][:4]:
        if len(subpages) >= settings.MAX_EXTRA_PAGES:
            break
        try:
            sub_url, sub_html = fetch_html(link)
            sub_text = parse_page(sub_html, sub_url)["text"]
        except ValueError:
            continue
        if sub_text:
            subpages.append({"url": sub_url, "text": sub_text})

    # 3. budget (runs once, after the loop)
    for page in subpages:
        page["text"] = _truncate(page["text"], settings.MAX_CONTENT_CHARS // 4)
    home_budget = settings.MAX_CONTENT_CHARS - sum(len(p["text"]) for p in subpages)
    pages = [{"url": final_url, "text": _truncate(home["text"], home_budget)}] + subpages

    # 4. empty check
    if sum(len(p["text"]) for p in pages) < settings.MIN_CONTENT_CHARS:
        raise ValueError("Not enough readable text was found. The site may need JavaScript.")

    # 5. result
    return {
        "url": final_url,
        "title": home["title"],
        "meta_description": home["meta_description"],
        "headings": home["headings"],
        "pages": pages,
    }


def build_prompt(data: dict) -> str:
    """Turn the extractor dict into one text block for the LLM."""
    parts = [
        f"Website: {data['url']}",
        f"Title: {data['title'] or 'not found'}",
        f"Meta description: {data['meta_description'] or 'not found'}",
        "Headings: " + (" | ".join(data["headings"]) or "not found"),
    ]

    for page in data["pages"]:
        parts.append(f"\n--- Page: {page['url']} ---\n{page['text']}")

    parts.append("\nAnalyze this website and return the JSON.")
    return "\n".join(parts)




