"""
Message models for session conversations.
"""
from __future__ import annotations


from datetime import datetime
from typing import Any, Literal
from pydantic import BaseModel, Field
from oats.date import utc
from oats.core.id import generate_id


class TextPart(BaseModel):
    """A text content part."""

    type: Literal["text"] = "text"
    id: str = Field(default_factory=generate_id)
    content: str


class ImagePart(BaseModel):
    """An image content part (base64-encoded or URL)."""

    type: Literal["image"] = "image"
    id: str = Field(default_factory=generate_id)
    media_type: str  # e.g. "image/png", "image/jpeg"
    data: str | None = None  # base64-encoded image bytes
    url: str | None = None   # or a URL pointing to the image
    detail: str = "auto"     # "auto", "low", "high" — provider hint for resolution


class ToolCallPart(BaseModel):
    """A tool call from the assistant."""

    type: Literal["tool_call"] = "tool_call"
    id: str = Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolResultPart(BaseModel):
    """Result from a tool execution."""

    type: Literal["tool_result"] = "tool_result"
    id: str = Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    title: str
    output: str
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# Union type for all message parts
MessagePart = TextPart | ImagePart | ToolCallPart | ToolResultPart


class Message(BaseModel):
    """A message in a session conversation."""

    id: str = Field(default_factory=generate_id)
    session_id: str
    role: Literal["user", "assistant", "system"]
    parts: list[MessagePart] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc)
    model: str | None = None
    provider: str | None = None
    usage: dict[str, int] | None = None

    def add_text(self, content: str) -> TextPart:
        """Add a text part to the message."""
        part = TextPart(content=content)
        self.parts.append(part)
        return part

    def add_image(
        self,
        media_type: str,
        *,
        data: str | None = None,
        url: str | None = None,
        detail: str = "auto",
    ) -> ImagePart:
        """Add an image part (base64 data or URL) to the message."""
        part = ImagePart(media_type=media_type, data=data, url=url, detail=detail)
        self.parts.append(part)
        return part

    def get_images(self) -> list[ImagePart]:
        """Get all image parts."""
        return [p for p in self.parts if isinstance(p, ImagePart)]

    def has_images(self) -> bool:
        """Check whether the message contains any image parts."""
        return any(isinstance(p, ImagePart) for p in self.parts)

    def add_tool_call(
        self, tool_call_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallPart:
        """Add a tool call part to the message."""
        part = ToolCallPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            arguments=arguments,
        )
        self.parts.append(part)
        return part

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        title: str,
        output: str,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResultPart:
        """Add a tool result part to the message."""
        part = ToolResultPart(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            title=title,
            output=output,
            error=error,
            metadata=metadata or {},
        )
        self.parts.append(part)
        return part

    def get_text_content(self) -> str:
        """Get concatenated text content from all text parts."""
        texts = [p.content for p in self.parts if isinstance(p, TextPart)]
        return "\n".join(texts)

    def get_tool_calls(self) -> list[ToolCallPart]:
        """Get all tool call parts."""
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    def get_tool_results(self) -> list[ToolResultPart]:
        """Get all tool result parts."""
        return [p for p in self.parts if isinstance(p, ToolResultPart)]

    def _build_multimodal_content(self) -> list[dict[str, Any]]:
        """Build a list of content blocks (text + images) for multimodal LLM calls."""
        blocks: list[dict[str, Any]] = []
        text = self.get_text_content()
        if text:
            blocks.append({"type": "text", "text": text})
        for img in self.get_images():
            if img.data:
                blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.media_type};base64,{img.data}",
                        "detail": img.detail,
                    },
                })
            elif img.url:
                blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img.url,
                        "detail": img.detail,
                    },
                })
        return blocks

    def to_llm_format(self) -> dict[str, Any]:
        """Convert to format for LLM API."""
        if self.role == "user":
            if self.has_images():
                return {
                    "role": "user",
                    "content": self._build_multimodal_content(),
                }
            return {
                "role": "user",
                "content": self.get_text_content(),
            }
        elif self.role == "system":
            return {
                "role": "system",
                "content": self.get_text_content(),
            }
        elif self.role == "assistant":
            tool_calls = self.get_tool_calls()
            if tool_calls:
                return {
                    "role": "assistant",
                    "content": self.get_text_content() or None,
                    "tool_calls": [
                        {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": str(tc.arguments),
                            },
                        }
                        for tc in tool_calls
                    ],
                }
            else:
                return {
                    "role": "assistant",
                    "content": self.get_text_content(),
                }
        else:
            return {
                "role": self.role,
                "content": self.get_text_content(),
            }
