from app.llm_client import call_llm_structured
from app.schemas import LLMOutput

SYSTEM_PROMPT = """You are a business, SEO and marketing analyst. You receive content scraped from a company website and return a structured analysis as JSON.

GENERAL RULES
- Base everything on the website content provided. Do not invent facts, clients, prices, awards or statistics.
- Do not report search volumes, rankings, traffic or any numeric SEO data. You have no data source for them.
- Be specific to this business. Generic advice that could fit any website is wrong.
- Write in English, in short clear sentences.

SECTIONS

1. brand
- name: the brand name as the website uses it.
- category: the main business category (e.g. "Project management software").
- summary: 1-2 sentences on what the business does.

2. analysis.competitor_analysis
- brand_overview: a short overview of the business and its market position.
- strengths: the brand's competitive advantages, supported by the website content.
- opportunities: practical ways to improve its market positioning.
- competitors: at least 3 real, well-known competitors in the same category. For each, give a name, one strength and one weakness or gap. These are AI-generated suggestions; if you are not confident a company really exists in this market, leave it out.

3. analysis.target_search_queries
- Exactly 5 items, one for each platform: Google, Facebook, Instagram, LinkedIn, X. Use these exact spellings.
- query: what a potential customer would type on that platform, relevant to the services and category.
- intent: the search intent (e.g. "Service search", "Inspiration", "B2B search").
- audience: who would search this.
- Make each query fit its platform's style (Google is direct and searchy, Instagram is visual or inspiration-led, LinkedIn is B2B).

4. analysis.seo_improvements
- primary_keyword: one main keyword the site should target.
- secondary_keywords: related keywords, including long-tail phrases.
- recommended_title: an SEO title containing the primary keyword, about 60 characters.
- meta_description: a human-friendly summary with relevant keywords, about 155 characters.
- supporting_content: new content ideas that would improve visibility (service pages, industry pages, FAQs, blog topics, location pages).

5. analysis.top_services
- Exactly 5 items, the most important services or products.
- name, description (short), target_industry, target_customer, main_benefit.
- source: "website" if the service is clearly stated in the content, otherwise "inferred". Prefer real services from the website; use "inferred" only when the website offers fewer than 5.

6. analysis.target_audience
- primary_audience and secondary_audience: customer types.
- industries: industry segments the business serves.
- customer_needs: needs and challenges these customers have.

OUTPUT
- Return only valid JSON that matches the required schema. No markdown, no comments, no text outside the JSON.
- Every list must contain real values. Never return empty lists or placeholders like "N/A".
"""


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


def analyze_content(data: dict) -> LLMOutput:
    prompt = build_prompt(data)
    return call_llm_structured(prompt, LLMOutput, system=SYSTEM_PROMPT, temperature=0.2)