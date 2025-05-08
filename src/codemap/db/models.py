"""Database models for CodeMap using SQLModel."""

from datetime import UTC, datetime

from sqlmodel import Field, SQLModel


class ChatHistory(SQLModel, table=True):
	"""Represents a single entry in the chat history table."""

	__tablename__: str = "chat_history"  # type: ignore[assignment]

	id: int | None = Field(default=None, primary_key=True)
	session_id: str = Field(index=True)
	timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC), index=True)
	user_query: str
	ai_response: str | None = Field(default=None)
	context: str | None = Field(default=None)  # JSON string or similar
	tool_calls: str | None = Field(default=None)  # JSON string
