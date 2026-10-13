# main.py

import csv
import io
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Annotated
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Ensure local modules resolve whether started from repo root or backend/
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import and_, case, func, insert, select
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine, settings
from models import Feedback
from schemas import FeedbackCreate, FeedbackResponse, FeedbackImportResponse, AdaptionStatusResponse
from adaption_service import adaption_service

from models import Issue, issue_completed_feedback, issue_feedback
from schemas import IssueResponse, IssueStatusUpdate, ReportInsightsRequest, ReportInsightsResponse
from google import genai
from google.genai.errors import APIError

#loggers
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically create database tables on startup
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        logger.error("Database table initialization warning: %s", exc)
    yield


app = FastAPI(
    title="InsightFlow API",
    description="Customer feedback storage and import API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https://.*\.(vercel\.app|onrender\.com)",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DbSession = Annotated[Session, Depends(get_db)]


@app.get("/")
def root():
    return {
        "message": "InsightFlow API is running",
        "docs": "/docs",
    }


@app.get("/health")
def health(db: DbSession):
    # This also verifies that the database is reachable.
    from sqlalchemy import text

    db.execute(text("SELECT 1"))

    return {
        "status": "ok",
        "database": "connected",
    }


@app.get("/adaption/status", response_model=AdaptionStatusResponse)
def get_adaption_status():
    return adaption_service.get_status()


@app.get(
    "/feedback",
    response_model=list[FeedbackResponse],
)
def get_feedback(db: DbSession):
    statement = select(Feedback).order_by(
        Feedback.created_at.desc(),
        Feedback.id.desc(),
    )

    return db.scalars(statement).all()


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
def create_feedback(
    payload: FeedbackCreate,
    db: DbSession,
):
    feedback_text = payload.text.strip()

    if not feedback_text:
        raise HTTPException(
            status_code=422,
            detail="Feedback text cannot be empty.",
        )

    # Use Adaption AI to categorize and determine sentiment if not provided
    processed = adaption_service.process_review(
        text=feedback_text,
        existing_category=payload.category,
        existing_sentiment=payload.sentiment,
    )

    record = Feedback(
        text=feedback_text,
        source=payload.source.strip() or "Manual",
        category=processed["category"],
        sentiment=processed["sentiment"],
        date=payload.date or date.today(),
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    _track_feedback(db, record)
    db.commit()

    return record


@app.post(
    "/feedback/import",
    response_model=FeedbackImportResponse,
)
def import_feedback(
    file: Annotated[UploadFile, File()],
    db: DbSession,
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a .csv file.",
        )

    try:
        raw = file.file.read()

        # Prevent accidentally importing an excessively large file.
        if len(raw) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="CSV file must be 10 MB or smaller.",
            )

        content = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))

        if not reader.fieldnames:
            raise HTTPException(
                status_code=400,
                detail="The CSV file is empty or has no header.",
            )

        # Normalize header names.
        headers = {
            name.strip().lower(): name
            for name in reader.fieldnames
            if name
        }

        def find_column(*names):
            for name in names:
                if name in headers:
                    return headers[name]
            return None

        text_column = find_column(
            "feedbacktext",
            "feedback_text",
            "feedback",
            "text",
            "review",
            "comment",
        )

        date_column = find_column(
            "createdat",
            "created_at",
            "date",
            "timestamp",
        )

        source_column = find_column(
            "source",
            "channel",
            "platform",
        )

        category_column = find_column(
            "category",
            "topic",
            "tag",
            "type",
            "department",
        )

        sentiment_column = find_column(
            "sentiment",
            "polarity",
            "feeling",
            "rating",
            "label",
        )

        if not text_column:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Missing feedback text column. "
                    "Use feedbackText, feedback, text, review or comment."
                ),
            )

        records = []
        skipped = 0

        for row in reader:
            text_value = (row.get(text_column) or "").strip()

            if not text_value:
                skipped += 1
                continue

            if len(text_value) > 20000:
                skipped += 1
                continue

            source = (
                (row.get(source_column) or "").strip()
                if source_column
                else "CSV"
            )

            feedback_date = date.today()

            if date_column:
                raw_date = (row.get(date_column) or "").strip()

                if raw_date:
                    try:
                        # Supports common ISO dates and timestamps.
                        feedback_date = date.fromisoformat(
                            raw_date[:10]
                        )
                    except ValueError:
                        skipped += 1
                        continue

            raw_cat = (row.get(category_column) or "").strip() if category_column else None
            raw_sent = (row.get(sentiment_column) or "").strip() if sentiment_column else None

            # Use Adaption AI to categorize and determine Positive / Negative sentiment
            processed = adaption_service.process_review(
                text=text_value,
                existing_category=raw_cat,
                existing_sentiment=raw_sent,
            )

            records.append(
                Feedback(
                    text=text_value,
                    source=source or "CSV",
                    category=processed["category"],
                    sentiment=processed["sentiment"],
                    date=feedback_date,
                )
            )

        if not records:
            raise HTTPException(
                status_code=400,
                detail="No valid feedback rows were found.",
            )

        # Save the batch in a single transaction.
        db.add_all(records)
        db.commit()

        _backfill_unlinked_feedback(db)
        db.commit()

        # Build classification summary
        pos_count = sum(1 for r in records if r.sentiment == "Positive")
        neg_count = sum(1 for r in records if r.sentiment == "Negative")
        neu_count = sum(1 for r in records if r.sentiment == "Neutral")

        cat_counts: dict[str, int] = {}
        for r in records:
            cat_counts[r.category] = cat_counts.get(r.category, 0) + 1

        return {
            "message": f"Successfully uploaded {len(records)} feedback entries with Adaption AI categorization.",
            "imported": len(records),
            "skipped": skipped,
            "summary": {
                "positive": pos_count,
                "negative": neg_count,
                "neutral": neu_count,
                "categories": cat_counts,
            },
            "records": records,
        }

    except HTTPException:
        db.rollback()
        raise

    except UnicodeDecodeError:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="CSV must be UTF-8 encoded.",
        )

    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Import failed. Check the backend logs.",
        )



@app.get("/issues", response_model=list[IssueResponse])
def get_issues(db: Session = Depends(get_db)):
    _backfill_unlinked_feedback(db)
    db.commit()

    counts = db.execute(
        select(
            issue_feedback.c.issue_id,
            func.count(Feedback.id).label("frequency"),
            func.sum(case((func.lower(Feedback.sentiment) == "positive", 1), else_=0)).label("positive_count"),
            func.sum(case((func.lower(Feedback.sentiment) == "neutral", 1), else_=0)).label("neutral_count"),
            func.sum(case((func.lower(Feedback.sentiment) == "negative", 1), else_=0)).label("negative_count"),
        )
        .join(Feedback, Feedback.id == issue_feedback.c.feedback_id)
        .outerjoin(
            issue_completed_feedback,
            and_(
                issue_completed_feedback.c.issue_id == issue_feedback.c.issue_id,
                issue_completed_feedback.c.feedback_id == issue_feedback.c.feedback_id,
            ),
        )
        .where(issue_completed_feedback.c.feedback_id.is_(None))
        .group_by(issue_feedback.c.issue_id)
    ).all()
    count_by_issue = {
        row.issue_id: (
            row.frequency,
            row.positive_count or 0,
            row.neutral_count or 0,
            row.negative_count or 0,
            max(0, row.frequency - (row.positive_count or 0) - (row.neutral_count or 0) - (row.negative_count or 0)),
        )
        for row in counts
    }

    results = []
    for issue in db.scalars(select(Issue).order_by(Issue.created_at.desc())).all():
        frequency, positive_count, neutral_count, negative_count, unclassified_count = count_by_issue.get(issue.id, (0, 0, 0, 0, 0))

        # If frequency is not zero, the issue must remain Open or In Progress, NEVER Resolved.
        if frequency > 0 and issue.status == "Resolved":
            issue.status = "Open"
            db.commit()

        # If frequency is zero or the issue is resolved, remove it from tracked issues
        if frequency == 0 or issue.status == "Resolved":
            continue

        negative_percentage = negative_count * 100 / frequency
        priority = "High" if negative_percentage >= 70 else "Medium" if negative_percentage >= 40 else "Low"
        results.append(IssueResponse(
            id=issue.id,
            name=issue.name,
            description=issue.description,
            count=frequency,
            negative=round(negative_percentage, 1),
            positive_count=positive_count,
            neutral_count=neutral_count,
            negative_count=negative_count,
            unclassified_count=unclassified_count,
            priority=priority,
            status=issue.status,
            category=issue.category,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
        ))
    return results


@app.patch("/issues/{issue_id}", response_model=IssueResponse)
def update_issue_status(
    issue_id: int,
    payload: IssueStatusUpdate,
    db: Session = Depends(get_db),
):
    issue = db.query(Issue).filter(Issue.id == issue_id).first()

    if issue is None:
        raise HTTPException(status_code=404, detail="Issue not found")

    active_feedback = db.execute(
        select(Feedback.id, Feedback.sentiment)
        .join(issue_feedback, issue_feedback.c.feedback_id == Feedback.id)
        .outerjoin(
            issue_completed_feedback,
            and_(
                issue_completed_feedback.c.issue_id == issue_feedback.c.issue_id,
                issue_completed_feedback.c.feedback_id == issue_feedback.c.feedback_id,
            ),
        )
        .where(issue_feedback.c.issue_id == issue_id, issue_completed_feedback.c.feedback_id.is_(None))
        .order_by(Feedback.id)
    ).all()

    current_frequency = len(active_feedback)

    if payload.completed_count is not None:
        if payload.completed_count > current_frequency:
            raise HTTPException(
                status_code=422,
                detail="Completed enquiries cannot exceed the current frequency.",
            )
        completed_rows = [
            {"issue_id": issue_id, "feedback_id": row.id}
            for row in active_feedback[:payload.completed_count]
        ]
        if completed_rows:
            db.execute(insert(issue_completed_feedback), completed_rows)

        remaining_count = current_frequency - payload.completed_count
        if remaining_count == 0:
            issue.status = "Resolved"
        else:
            # Frequency is not zero -> cannot be Resolved, must remain Open or In Progress
            if payload.status == "In Progress" or issue.status == "In Progress":
                issue.status = "In Progress"
            else:
                issue.status = "Open"
    else:
        if payload.status == "Resolved":
            if current_frequency > 0:
                raise HTTPException(
                    status_code=422,
                    detail="Cannot mark as Resolved while frequency is greater than zero. Complete all enquiries to resolve.",
                )
            issue.status = "Resolved"
        elif payload.status in ["Open", "In Progress"]:
            issue.status = payload.status

    db.commit()
    db.refresh(issue)

    completed_num = payload.completed_count or 0
    active_feedback = active_feedback[completed_num:]
    total_count = len(active_feedback)

    positive_count = sum(1 for row in active_feedback if row.sentiment.lower() == "positive")
    neutral_count = sum(1 for row in active_feedback if row.sentiment.lower() == "neutral")
    negative_count = sum(1 for row in active_feedback if row.sentiment.lower() == "negative")
    unclassified_count = total_count - positive_count - neutral_count - negative_count

    negative_percentage = (
        (negative_count / total_count) * 100
        if total_count > 0
        else 0
    )

    if total_count == 0:
        return IssueResponse(
            id=issue.id,
            name=issue.name,
            description=issue.description,
            count=0,
            negative=0,
            positive_count=0,
            neutral_count=0,
            negative_count=0,
            unclassified_count=0,
            priority=issue.priority,
            status="Resolved",
            category=issue.category,
            created_at=issue.created_at,
            updated_at=issue.updated_at,
        )

    priority = "High" if negative_percentage >= 70 else "Medium" if negative_percentage >= 40 else "Low"

    return IssueResponse(
        id=issue.id,
        name=issue.name,
        description=issue.description,
        count=total_count,
        negative=round(negative_percentage, 1),
        positive_count=positive_count,
        neutral_count=neutral_count,
        negative_count=negative_count,
        unclassified_count=unclassified_count,
        priority=priority,
        status=issue.status,
        category=issue.category,
        created_at=issue.created_at,
        updated_at=issue.updated_at,
    )


def _track_feedback(db: Session, feedback: Feedback) -> None:
    category = (feedback.category or "Uncategorized").strip() or "Uncategorized"
    issue = db.query(Issue).filter(Issue.category == category).first()
    if issue is None:
        issue = Issue(name=category, description=f"Customer feedback about {category.lower()}.", category=category)
        db.add(issue)
        db.flush()
    db.execute(insert(issue_feedback).values(issue_id=issue.id, feedback_id=feedback.id))
    if issue.status == "Resolved":
        issue.status = "Open"


def _backfill_unlinked_feedback(db: Session) -> None:
    """Attach old unlinked feedback in batches, without per-row relationship loads."""
    unlinked = db.execute(
        select(Feedback.id, Feedback.category)
        .outerjoin(issue_feedback, issue_feedback.c.feedback_id == Feedback.id)
        .where(issue_feedback.c.feedback_id.is_(None))
    ).all()
    if not unlinked:
        return

    categories = {(category or "Uncategorized").strip() or "Uncategorized" for _, category in unlinked}
    issues = db.scalars(select(Issue).where(Issue.category.in_(categories))).all()
    issue_by_category = {issue.category: issue for issue in issues}

    for category in categories - issue_by_category.keys():
        issue = Issue(
            name=category,
            description=f"Customer feedback about {category.lower()}.",
            category=category,
        )
        db.add(issue)
        issue_by_category[category] = issue
    db.flush()

    rows = []
    for feedback_id, raw_category in unlinked:
        category = (raw_category or "Uncategorized").strip() or "Uncategorized"
        issue = issue_by_category[category]
        rows.append({"issue_id": issue.id, "feedback_id": feedback_id})
        if issue.status == "Resolved":
            issue.status = "Open"
    if rows:
        db.execute(insert(issue_feedback), rows)


def _get_feedback_context(db: Session, neg_limit: int = 15, pos_limit: int = 5) -> dict[str, list[dict]]:
    """Fetch representative real customer comments from database for enhanced context."""
    context = {"sample_negative_feedback": [], "sample_positive_feedback": []}
    try:
        neg_rows = db.execute(
            select(Feedback.category, Feedback.text, Feedback.source)
            .where(func.lower(Feedback.sentiment) == "negative")
            .order_by(Feedback.id.desc())
            .limit(neg_limit)
        ).all()
        context["sample_negative_feedback"] = [
            {"category": row[0], "text": row[1], "source": row[2]}
            for row in neg_rows
        ]

        pos_rows = db.execute(
            select(Feedback.category, Feedback.text, Feedback.source)
            .where(func.lower(Feedback.sentiment) == "positive")
            .order_by(Feedback.id.desc())
            .limit(pos_limit)
        ).all()
        context["sample_positive_feedback"] = [
            {"category": row[0], "text": row[1], "source": row[2]}
            for row in pos_rows
        ]
    except Exception as exc:
        logger.warning("Could not fetch feedback context from DB: %s", exc)
    return context


def _generate_gemini_text(
    prompt: str,
    *,
    max_output_tokens: int = 4096,
    task_label: str = "content",
) -> tuple[str, str]:
    """
    Generate text using Google GenAI SDK with automatic multi-model fallback.
    Tries the configured model first, falling back gracefully to available high-throughput models.
    """
    api_key = settings.gemini_api_key
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is not configured. Add GEMINI_API_KEY "
                "to backend/.env and restart FastAPI."
            ),
        )

    # Candidate models in priority order
    candidate_models = [
        settings.gemini_model,
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3.8-flash",
    ]
    models_to_try: list[str] = []
    for model_name in candidate_models:
        if model_name and model_name not in models_to_try:
            models_to_try.append(model_name)

    client = genai.Client(api_key=api_key)
    last_error: Exception | None = None

    try:
        for model in models_to_try:
            for attempt in range(2):
                try:
                    logger.info("Calling Gemini model %s for %s (attempt %d)", model, task_label, attempt + 1)
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    text = response.text if response and response.text else None
                    if text and text.strip():
                        logger.info("Gemini %s successfully generated with model %s", task_label, model)
                        return text.strip(), model
                except Exception as error:
                    last_error = error
                    err_str = str(error)
                    logger.warning(
                        "Gemini model %s failed for %s on attempt %d: %s",
                        model,
                        task_label,
                        attempt + 1,
                        err_str,
                    )
                    # If 429 quota exhausted or 404 model not found, immediately break attempt loop to try next candidate model
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str or "NOT_FOUND" in err_str:
                        break
                    if attempt == 0:
                        time.sleep(1)
    finally:
        client.close()

    logger.exception("All Gemini candidate models failed for %s. Last error: %s", task_label, last_error)
    raise HTTPException(
        status_code=502,
        detail=f"Gemini could not generate {task_label}. All models failed. Last error: {last_error}",
    )


@app.post("/reports/ai", response_model=ReportInsightsResponse)
def generate_ai_report(payload: ReportInsightsRequest, db: Session = Depends(get_db)):
    report_data = payload.model_dump()
    db_context = _get_feedback_context(db, neg_limit=18, pos_limit=6)

    # Generate using Adaption AI API & AutoScientist intelligence
    report_text, used_model = adaption_service.generate_report(
        data=report_data,
        feedback_context=db_context,
        llm_generate_fn=_generate_gemini_text,
    )
    return {
        "report": report_text,
        "model": used_model,
    }


@app.post("/insights/ai", response_model=ReportInsightsResponse)
def generate_ai_insights(payload: ReportInsightsRequest, db: Session = Depends(get_db)):
    insight_data = payload.model_dump()
    db_context = _get_feedback_context(db, neg_limit=12, pos_limit=4)

    # Generate using Adaption AI API & AutoScientist intelligence
    report_text, used_model = adaption_service.generate_insights(
        data=insight_data,
        feedback_context=db_context,
        llm_generate_fn=_generate_gemini_text,
    )
    return {
        "report": report_text,
        "model": used_model,
    }


if __name__ == "__main__":
    import os
    import uvicorn

    port = int(os.environ.get("PORT", settings.port))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

