
# models.py

from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    source: Mapped[str] = mapped_column(
        String(100),
        default="CSV",
        nullable=False,
    )

    category: Mapped[str] = mapped_column(
        String(100),
        default="Uncategorized",
        nullable=False,
    )

    sentiment: Mapped[str] = mapped_column(
        String(20),
        default="Neutral",
        nullable=False,
    )

    date: Mapped[date] = mapped_column(
        Date,
        default=date.today,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

# Add these imports if they are not already present
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Table,
    Text,
)
from sqlalchemy.orm import relationship

from database import Base


# Links existing feedback records to tracked issues.
# This creates a new table; it does not modify or delete feedback.
issue_feedback = Table(
    "issue_feedback",
    Base.metadata,
    Column(
        "issue_id",
        Integer,
        ForeignKey("issues.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "feedback_id",
        Integer,
        ForeignKey("feedback.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)
Index("ix_issue_feedback_feedback_id", issue_feedback.c.feedback_id)

issue_completed_feedback = Table(
    "issue_completed_feedback",
    Base.metadata,
    Column("issue_id", Integer, ForeignKey("issues.id", ondelete="CASCADE"), primary_key=True),
    Column("feedback_id", Integer, ForeignKey("feedback.id", ondelete="CASCADE"), primary_key=True),
)


class Issue(Base):
    __tablename__ = "issues"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=False, default="")
    category = Column(String(100), nullable=False, default="Uncategorized")

    # Open, In Progress, or Resolved
    status = Column(String(30), nullable=False, default="Open")

    # High, Medium, or Low
    priority = Column(String(20), nullable=False, default="Low")

    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    feedback_items = relationship(
        "Feedback",
        secondary=issue_feedback,
        backref="tracked_issues",
        passive_deletes=True,
    )
