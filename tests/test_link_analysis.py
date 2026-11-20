import pytest
from bs4 import BeautifulSoup
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas import LLMOutput
from app.services import analysis_service
from app.services.website_extractor import clean_body_text, find_candidate_links, parse_page
from app.utils.validator import validate_url

# raise_server_exceptions=False lets the catch-all 500 handler answer instead of the test crashing
client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------- sample data
def sample_llm_dict() -> dict:
    platforms = ["Google", "Facebook", "Instagram", "LinkedIn", "X"]
    return {
        "brand": {"name": "Acme", "category": "Web agency", "summary": "Builds websites."},
        "analysis": {
            "competitor_analysis": {
                "brand_overview": "A small web agency.",
                "strengths": ["Affordable"],
                "opportunities": ["Improve visibility"],
                "competitors": [
                    {"name": f"Rival {i}", "strength": "Big brand", "weakness": "Expensive"}
                    for i in range(3)
                ],
                "note": "whatever the LLM wrote",
            },
            "target_search_queries": [
                {"query": f"web agency {p}", "platform": p, "intent": "Service search", "audience": "Startups"}
                for p in platforms
            ],
            "seo_improvements": {
                "primary_keyword": "web agency",
                "secondary_keywords": ["web design"],
                "recommended_title": "Web Agency for Small Business | Acme",
                "meta_description": "Acme builds fast, affordable websites.",
                "supporting_content": ["FAQ page"],
            },
            "top_services": [
                {
                    "name": f"Service {i}", "description": "d", "target_industry": "Retail",
                    "target_customer": "Small business", "main_benefit": "More customers",
                    "source": "website",
                }
                for i in range(5)
            ],
            "target_audience": {
                "primary_audience": ["Small businesses"],
                "secondary_audience": ["Startups"],
                "industries": ["Retail"],
                "customer_needs": ["Online presence"],
            },
        },
    }


def sample_extractor_data() -> dict:
    return {
        "url": "https://example.com",
        "title": "Acme",
        "meta_description": "We build websites",
        "headings": ["Services"],
        "pages": [{"url": "https://example.com", "text": "We build websites for small businesses."}],
    }


# ---------------------------------------------------------------- validator
def test_validator_adds_https_and_strips_spaces():
    assert validate_url("  example.com  ") == "https://example.com"


def test_validator_keeps_http_scheme():
    assert validate_url("http://example.com") == "http://example.com"


@pytest.mark.parametrize("bad", ["", "   ", "https://", "a" * 3000])
def test_validator_rejects_bad_urls(bad):
    with pytest.raises(ValueError):
        validate_url(bad)


# ---------------------------------------------------------------- extractor: cleaning
def test_clean_removes_noise_duplicates_and_fragments():
    html = """
    <html><body>
      <nav>Home About</nav>
      <h1>Services</h1>
      <p>We build custom websites for small businesses.</p>
      <p>We build custom websites for small businesses.</p>
      <span>Menu</span>
      <footer>Copyright 2026</footer>
      <script>var x = 1;</script>
    </body></html>
    """
    text = clean_body_text(BeautifulSoup(html, "lxml"), {"services"})
    assert "Services" in text                      # short heading kept
    assert text.count("We build custom websites") == 1   # duplicate removed
    assert "Menu" not in text                      # one-word fragment dropped
    assert "Copyright" not in text                 # footer removed
    assert "var x" not in text                     # script removed


def test_clean_removes_cookie_banner():
    html = '<body><div class="cookie-banner">Accept all cookies now</div><p>Real business content here.</p></body>'
    text = clean_body_text(BeautifulSoup(html, "lxml"), set())
    assert "cookies" not in text
    assert "Real business content" in text


# ---------------------------------------------------------------- extractor: links
def test_candidate_links_filtering_and_order():
    html = """
    <a href="/about-us">About us</a>
    <a href="/services">Our Services</a>
    <a href="/blog/post-1">Blog</a>
    <a href="/login">Login</a>
    <a href="/brochure.pdf">Brochure services</a>
    <a href="https://other.com/services">Partner</a>
    <a href="/">Home</a>
    <a href="#top">Top</a>
    """
    links = find_candidate_links(BeautifulSoup(html, "lxml"), "https://site.com")
    assert links == ["https://site.com/services", "https://site.com/about-us"]


# ---------------------------------------------------------------- extractor: parse_page
def test_parse_page_reads_metadata_and_text():
    html = b"""
    <html><head><title>Acme Agency</title>
    <meta name="description" content="We build websites"></head>
    <body><h1>Web design</h1><p>We design modern websites for local shops.</p></body></html>
    """
    page = parse_page(html, "https://acme.com")
    assert page["title"] == "Acme Agency"
    assert page["meta_description"] == "We build websites"
    assert "Web design" in page["headings"]
    assert "modern websites" in page["text"]


# ---------------------------------------------------------------- schemas
def test_valid_llm_output_passes():
    assert LLMOutput.model_validate(sample_llm_dict())


def test_fewer_than_5_queries_rejected():
    data = sample_llm_dict()
    data["analysis"]["target_search_queries"].pop()
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(data)


def test_duplicate_platform_rejected():
    data = sample_llm_dict()
    data["analysis"]["target_search_queries"][4]["platform"] = "Google"
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(data)


def test_unknown_platform_rejected():
    data = sample_llm_dict()
    data["analysis"]["target_search_queries"][0]["platform"] = "TikTok"
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(data)


def test_fewer_than_3_competitors_rejected():
    data = sample_llm_dict()
    data["analysis"]["competitor_analysis"]["competitors"].pop()
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(data)


def test_service_source_must_be_website_or_inferred():
    data = sample_llm_dict()
    data["analysis"]["top_services"][0]["source"] = "guess"
    with pytest.raises(ValidationError):
        LLMOutput.model_validate(data)


# ---------------------------------------------------------------- API (extractor + LLM mocked)
@pytest.fixture(autouse=True)
def clear_cache():
    analysis_service._cache.clear()


@pytest.fixture
def mock_pipeline(monkeypatch):
    calls = {"extract": 0, "llm": 0}

    def fake_extract(url):
        calls["extract"] += 1
        return sample_extractor_data()

    def fake_llm(data):
        calls["llm"] += 1
        return LLMOutput.model_validate(sample_llm_dict())

    # patch the names inside analysis_service, because that is where they were imported
    monkeypatch.setattr(analysis_service, "extract_website", fake_extract)
    monkeypatch.setattr(analysis_service, "analyze_content", fake_llm)
    return calls


def post(url="https://example.com"):
    return client.post("/link-analysis", json={"url": url})


def test_api_success_shape(mock_pipeline):
    res = post()
    body = res.json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["website"] == "https://example.com"
    assert body["brand"]["name"] == "Acme"
    assert len(body["analysis"]["target_search_queries"]) == 5
    assert len(body["analysis"]["top_services"]) == 5


def test_api_forces_competitor_note(mock_pipeline):
    note = post().json()["analysis"]["competitor_analysis"]["note"]
    assert note == "Competitors are AI-generated suggestions"


def test_api_second_call_uses_cache(mock_pipeline):
    post()
    post()
    assert mock_pipeline["extract"] == 1
    assert mock_pipeline["llm"] == 1


def test_api_empty_url_returns_400(mock_pipeline):
    res = post("")
    assert res.status_code == 400
    assert res.json()["success"] is False


def test_api_missing_body_returns_422(mock_pipeline):
    res = client.post("/link-analysis", json={})
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "INVALID_BODY"


def test_api_website_error_returns_400(monkeypatch):
    def boom(url):
        raise ValueError("Could not connect to the website.")
    monkeypatch.setattr(analysis_service, "extract_website", boom)
    res = post()
    assert res.status_code == 400
    assert res.json()["error"]["message"] == "Could not connect to the website."


def test_api_llm_failure_returns_502(monkeypatch):
    monkeypatch.setattr(analysis_service, "extract_website", lambda url: sample_extractor_data())

    def boom(data):
        raise RuntimeError("Groq down")
    monkeypatch.setattr(analysis_service, "analyze_content", boom)
    assert post().status_code == 502


def test_api_llm_invalid_output_returns_502(monkeypatch):
    monkeypatch.setattr(analysis_service, "extract_website", lambda url: sample_extractor_data())

    def boom(data):
        raise ValueError("bad json")
    monkeypatch.setattr(analysis_service, "analyze_content", boom)
    assert post().status_code == 502


def test_api_failures_are_not_cached(monkeypatch):
    calls = {"n": 0}

    def flaky(url):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("temporary problem")
        return sample_extractor_data()

    monkeypatch.setattr(analysis_service, "extract_website", flaky)
    monkeypatch.setattr(analysis_service, "analyze_content", lambda d: LLMOutput.model_validate(sample_llm_dict()))
    assert post().status_code == 400
    assert post().status_code == 200   # retried, not served from cache


def test_api_unexpected_error_returns_500(monkeypatch):
    def boom(url):
        raise KeyError("bug")
    monkeypatch.setattr(analysis_service, "extract_website", boom)
    res = post()
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "INTERNAL_ERROR"