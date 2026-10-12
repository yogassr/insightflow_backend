
from datetime import date as Date, datetime

from pydantic import BaseModel, ConfigDict, Field

from typing import Literal


class FeedbackResponse(BaseModel):
    id: int
    text: str
    source: str
    category: str
    sentiment: str
    date: Date
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class FeedbackCreate(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    source: str = Field(default="Manual", max_length=100)
    category: str = Field(default="Uncategorized", max_length=100)
    sentiment: str = Field(default="Neutral", max_length=20)
    date: Date | None = None


class FeedbackImportResponse(BaseModel):
    message: str
    imported: int
    skipped: int
    summary: dict
    records: list[FeedbackResponse]


class AdaptionStatusResponse(BaseModel):
    api_connected: bool
    platform: str
    dataset_id: str
    run_id: str
    trained_model: str
    run_status: str
    categories: list[str]
    best_win_rate: float | None = None
    row_count: int | None = None
    dataset_name: str | None = None

from pydantic import BaseModel, ConfigDict, Field


class IssueResponse(BaseModel):
    id: int
    name: str
    description: str
    count: int = Field(ge=0)
    negative: float = Field(ge=0, le=100)
    positive_count: int = Field(default=0, ge=0)
    neutral_count: int = Field(default=0, ge=0)
    negative_count: int = Field(default=0, ge=0)
    unclassified_count: int = Field(default=0, ge=0)
    priority: str
    status: str
    category: str
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class IssueStatusUpdate(BaseModel):
    status: Literal["Open", "In Progress", "Resolved"]
    completed_count: int | None = Field(default=None, ge=0)


class ReportCategoryMetrics(BaseModel):
    category: str = Field(max_length=100)
    total: int = Field(ge=0)
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)
    unclassified: int = Field(ge=0)


class ReportIssueMetrics(BaseModel):
    issue: str = Field(max_length=200)
    category: str = Field(max_length=100)
    frequency: int = Field(ge=0)
    positive_feedback_count: int = Field(ge=0)
    neutral_feedback_count: int = Field(ge=0)
    negative_feedback_count: int = Field(ge=0)
    unclassified_feedback_count: int = Field(ge=0)
    negative_feedback_percent: float = Field(ge=0, le=100)
    priority: Literal["Critical", "High", "Medium", "Low"]
    status: Literal["Open", "In Progress", "Resolved"]


class ReportPeriodMetrics(BaseModel):
    month: str = Field(max_length=32)
    total: int = Field(ge=0)
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)
    unclassified: int = Field(ge=0)
    is_partial: bool = False


class ReportInsightsRequest(BaseModel):
    feedback_total: int = Field(ge=0)
    sentiment_valid_total: int = Field(ge=0)
    unclassified_sentiment_total: int = Field(ge=0)
    reporting_period: str = Field(max_length=64)
    feedback_text_included: bool = False
    positive_total: int = Field(ge=0)
    neutral_total: int = Field(ge=0)
    negative_total: int = Field(ge=0)
    categories: list[ReportCategoryMetrics] = Field(max_length=20)
    issues: list[ReportIssueMetrics] = Field(max_length=15)
    monthly_sentiment: list[ReportPeriodMetrics] = Field(default_factory=list, max_length=60)


class ReportInsightsResponse(BaseModel):
    report: str
    model: str
