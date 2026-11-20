from typing import Literal

from pydantic import BaseModel, Field, field_validator

Platform = Literal["Google", 'Facebook', 'Instagram', "LinkedIn", "X"]



class LinkAnalysisRequest(BaseModel):
    url: str = Field(..., description="the url to analyze", json_schema_extra={"example": "https://example.com"})

class Brand(BaseModel):
    name: str
    category: str
    summary: str

class Competitor(BaseModel):
    name: str
    strength: str
    weakness: str


class CompetitorAnalysis(BaseModel):
    brand_overview: str
    strengths: list[str]  # the brand's competitive advantages
    opportunities: list[str]  # ways to improve market positioning
    competitors: list[Competitor] = Field(min_length=3)
    note: str = "These competitors are AI-generated suggestions"
    


class SearchQuery(BaseModel):
    query: str
    platform: Platform
    intent: str
    audience: str


class SeoImprovements(BaseModel):
    primary_keyword: str
    secondary_keywords: list[str]
    recommended_title: str  # aim for ~60 characters (stated in the prompt, not enforced here)
    meta_description: str  # aim for ~155 characters
    supporting_content: list[str]

class Service(BaseModel):
    name: str
    description: str
    target_industry: str
    target_customer: str
    main_benefit: str
    source: Literal["website", "inferred"]


class TargetAudience(BaseModel):
    primary_audience: list[str]
    secondary_audience: list[str]
    industries: list[str]
    customer_needs: list[str]


class Analysis(BaseModel):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery] = Field(min_length=5, max_length=5)
    seo_improvements: SeoImprovements
    top_services: list[Service] = Field(min_length=5, max_length=5)
    target_audience: TargetAudience

    @field_validator("target_search_queries")
    @classmethod
    def one_query_per_platform(cls, queries: list[SearchQuery]) -> list[SearchQuery]:
        if len({q.platform for q in queries}) != 5:
            raise ValueError("need exactly one query for each platform: Google, Facebook, Instagram, LinkedIn, X")
        return queries


class LLMOutput(BaseModel):
    """What we ask the LLM to return."""

    brand: Brand
    analysis: Analysis


# ---------------------------------------------------------------- API responses
class LinkAnalysisResponse(BaseModel):
    success: bool = True
    website: str
    brand: Brand
    analysis: Analysis


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail