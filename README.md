# AI Link Analysis API

A FastAPI service that analyzes a website with an LLM. Send a URL, and the API fetches the site, cleans the content, and returns a structured JSON analysis with five sections:

1. Competitor vs your brand
2. Top 5 target search queries (Google, Facebook, Instagram, LinkedIn, X)
3. SEO improvements
4. Top 5 services
5. Target audience

## Tech stack

- Python 3.12, FastAPI, Uvicorn, Pydantic
- `requests` + BeautifulSoup (`lxml`) for fetching and cleaning websites
- Groq (OpenAI-compatible API) for the LLM call
- `cachetools` for an in-memory result cache
- `pytest` for tests

## How it works

```
POST /link-analysis {"url": "..."}
 1. validate_url()        check format, add https:// if missing
 2. cache lookup          same URL within 1 hour -> return the stored result
 3. extract_website()     fetch homepage + up to 2 useful subpages, clean the text
 4. llm_service           build the prompt, one LLM call, validate the JSON
 5. analysis_service      label competitors, build the response, store in cache
 6. return JSON
```

### Website extraction

- Downloads the homepage and reads title, meta description and headings.
- Removes noise: scripts, styles, nav, footer, hidden elements, cookie/popup boxes.
- Prefers the `<main>` / `<article>` area when it holds real content.
- Drops one-word fragments and duplicate lines.
- Scores same-site links by keywords (services, solutions, products, about...) and reads the best 2 as extra pages. Blog, login and legal pages are skipped. A subpage that fails is skipped silently.
- Total text is capped (22,000 characters) to control LLM token cost.
- If almost no text is found (for example a JavaScript-only site), the API returns an error instead of letting the LLM guess.

### LLM step

- **One LLM call** produces all five sections, which is cheaper and faster than several calls.
- The output is validated with Pydantic (`schemas.py`). The schema enforces exactly 5 search queries (one per platform), exactly 5 services, and at least 3 competitors.
- The prompt tells the model to use only the website content, not to invent facts, and not to report any search volumes.
- Each service is labeled `source: "website"` (stated on the site) or `"inferred"`.

### Competitors

Competitors come from the LLM's own knowledge, not from live web research. The response always carries the note **"Competitors are AI-generated suggestions"**.

### Caching

Successful results are cached in memory by normalized URL (1 hour, max 200 entries). Errors are never cached.

## Project structure

```
ai-link-analysis/
├── app/
│   ├── main.py                    # FastAPI app, route, error handlers
│   ├── config.py                  # settings from .env
│   ├── schemas.py                 # request/response and LLM output models
│   ├── llm_client.py              # Groq client: retries, rate limit, JSON validation
│   ├── services/
│   │   ├── website_extractor.py   # fetch + clean website content
│   │   ├── llm_service.py         # prompt building + LLM call
│   │   └── analysis_service.py    # orchestrator + cache
│   └── utils/
│       └── validator.py           # URL validation
├── tests/
│   └── test_link_analysis.py
├── Dockerfile
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd ai-link-analysis

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and add your Groq API key:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

```env
GROQ_API_KEY=your_key_here
```

Get a key from the Groq console. **Never commit `.env`**; only `.env.example` belongs in the repo.

### 3. Run the server

Run from the project root (not from inside `app/`):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

- API: `http://localhost:4000/link-analysis`
- Interactive docs: `http://localhost:4000/docs`

Use a single uvicorn worker, because the cache lives in process memory.

## API

### `POST /link-analysis`

**Request**

```json
{
  "url": "https://example.com"
}
```

**curl**

```bash
curl -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://basecamp.com"}'
```

On Windows `cmd`, use `-d "{\"url\": \"https://basecamp.com\"}"`. In PowerShell, call `curl.exe`.

**Success response (abridged, shape only)**

```json
{
  "success": true,
  "website": "https://basecamp.com/",
  "brand": {
    "name": "Basecamp",
    "category": "Project management software",
    "summary": "..."
  },
  "analysis": {
    "competitor_analysis": {
      "brand_overview": "...",
      "strengths": ["..."],
      "opportunities": ["..."],
      "competitors": [
        { "name": "...", "strength": "...", "weakness": "..." }
      ],
      "note": "Competitors are AI-generated suggestions"
    },
    "target_search_queries": [
      { "query": "...", "platform": "Google", "intent": "...", "audience": "..." }
    ],
    "seo_improvements": {
      "primary_keyword": "...",
      "secondary_keywords": ["..."],
      "recommended_title": "...",
      "meta_description": "...",
      "supporting_content": ["..."]
    },
    "top_services": [
      {
        "name": "...",
        "description": "...",
        "target_industry": "...",
        "target_customer": "...",
        "main_benefit": "...",
        "source": "website"
      }
    ],
    "target_audience": {
      "primary_audience": ["..."],
      "secondary_audience": ["..."],
      "industries": ["..."],
      "customer_needs": ["..."]
    }
  }
}
```

`target_search_queries` contains exactly 5 items (Google, Facebook, Instagram, LinkedIn, X) and `top_services` exactly 5. `competitors` has at least 3.

### Error responses

Every error uses the same shape:

```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Could not connect to the website."
  }
}
```

| Situation | Status | Code |
|---|---|---|
| Bad URL, unreachable site, site blocked us, non-HTML page, too little text | 400 | `INVALID_REQUEST` (the message explains which) |
| Missing or wrong request body | 422 | `INVALID_BODY` |
| LLM API failed, or returned invalid JSON after retries | 502 | `HTTP_502` |
| Unknown route / wrong method | 404 / 405 | `HTTP_404` / `HTTP_405` |
| Unexpected server error | 500 | `INTERNAL_ERROR` |

## Tests

```bash
pip install pytest httpx
python -m pytest -v
```

The tests do not call the network or the real LLM; both are mocked. They cover:

- URL validation
- Text cleaning, link filtering and page parsing
- Schema rules (5 queries, one per platform, 5 services, 3+ competitors)
- The API: response shape, forced competitor note, caching, and each error path

## Docker

```bash
docker build -t link-analysis .
docker run -p 4000:4000 --env-file .env link-analysis
```

The image reads the `PORT` environment variable when the host provides one, and defaults to 4000. `.env` is excluded from the image through `.dockerignore`.

## Known limitations

- **JavaScript-rendered sites** return almost no text to a plain HTTP fetch, so the API reports "not enough readable text" for them.
- **Some sites block automated requests** (HTTP 401/403/429) and cannot be analyzed.
- **Competitors are AI-generated suggestions**, not verified live research.
- **The cache is in memory.** It is cleared on restart and is not shared between workers.
- **Subpages are fetched one after another**, so a slow site can make a request take longer.
- **No search volumes or rankings** are returned, because no data source for them is used.
