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
    """A text content part within a message.

    Attributes:
        type: Always "text".
        id: Unique identifier for this part.
        content: The text content.
    """

    type: Literal["text"] = "text"
    id: str = Field(default_factory=generate_id)
    content: str


class ImagePart(BaseModel):
    """An image content part (base64-encoded or URL).

    Attributes:
        type: Always "image".
        id: Unique identifier for this part.
        media_type: MIME type of the image (e.g. "image/png").
        data: Base64-encoded image bytes (mutually exclusive with url).
        url: URL pointing to the image (mutually exclusive with data).
        detail: Resolution hint for the provider ("auto", "low", "high").
    """

    type: Literal["image"] = "image"
    id: str = Field(default_factory=generate_id)
    media_type: str  # e.g. "image/png", "image/jpeg"
    data: str | None = None  # base64-encoded image bytes
    url: str | None = None   # or a URL pointing to the image
    detail: str = "auto"     # "auto", "low", "high" — provider hint for resolution


class ToolCallPart(BaseModel):
    """A tool call from the assistant.

    Attributes:
        type: Always "tool_call".
        id: Unique identifier for this part.
        tool_call_id: The ID used to correlate with the tool result.
        tool_name: Name of the tool being called.
        arguments: Dict of arguments passed to the tool.
    """

    type: Literal["tool_call"] = "tool_call"
    id: str = Field(default_factory=generate_id)
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolResultPart(BaseModel):
    """Result from a tool execution.

    Attributes:
        type: Always "tool_result".
        id: Unique identifier for this part.
        tool_call_id: The ID of the tool call this result corresponds to.
        tool_name: Name of the tool that was executed.
        title: Short title describing the tool result.
        output: The tool's output text.
        error: Error message if the tool failed.
        metadata: Additional metadata about the tool execution.
    """

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
    """A message in a session conversation.

    Messages can contain text, images, tool calls, and tool results.
    Supports conversion to the format expected by LLM provider APIs.

    Attributes:
        id: Unique message identifier.
        session_id: The session this message belongs to.
        role: The sender role ("user", "assistant", "system").
        parts: Ordered list of content parts (text, images, tool calls, results).
        created_at: When the message was created.
        model: The model that generated this message (for assistant messages).
        provider: The provider that generated this message.
        usage: Token usage stats for this message.
    """

    id: str = Field(default_factory=generate_id)
    session_id: str
    role: Literal["user", "assistant", "system"]
    parts: list[MessagePart] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc)
    model: str | None = None
    provider: str | None = None
    usage: dict[str, int] | None = None

    def add_text(self, content: str) -> TextPart:
        """Add a text part to the message.

        Args:
            content: The text content to add.

        Returns:
            The created TextPart.
        """
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
        """Add an image part (base64 data or URL) to the message.

        Args:
            media_type: MIME type of the image.
            data: Base64-encoded image bytes.
            url: URL pointing to the image.
            detail: Resolution hint for the provider.

        Returns:
            The created ImagePart.
        """
        part = ImagePart(media_type=media_type, data=data, url=url, detail=detail)
        self.parts.append(part)
        return part

    def get_images(self) -> list[ImagePart]:
        """Get all image parts.

        Returns:
            List of ImagePart objects in this message.
        """
        return [p for p in self.parts if isinstance(p, ImagePart)]

    def has_images(self) -> bool:
        """Check whether the message contains any image parts.

        Returns:
            True if the message has at least one image part.
        """
        return any(isinstance(p, ImagePart) for p in self.parts)

    def add_tool_call(
        self, tool_call_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> ToolCallPart:
        """Add a tool call part to the message.

        Args:
            tool_call_id: Unique ID for correlating with the result.
            tool_name: Name of the tool being called.
            arguments: Arguments to pass to the tool.

        Returns:
            The created ToolCallPart.
        """
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
        """Add a tool result part to the message.

        Args:
            tool_call_id: ID of the tool call this result corresponds to.
            tool_name: Name of the tool that was executed.
            title: Short title for the result.
            output: The tool's output text.
            error: Error message if the tool failed.
            metadata: Additional metadata about the execution.

        Returns:
            The created ToolResultPart.
        """
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
        """Get concatenated text content from all text parts.

        Returns:
            All text parts joined with newlines.
        """
        texts = [p.content for p in self.parts if isinstance(p, TextPart)]
        return "\n".join(texts)

    def get_tool_calls(self) -> list[ToolCallPart]:
        """Get all tool call parts.

        Returns:
            List of ToolCallPart objects in this message.
        """
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    def get_tool_results(self) -> list[ToolResultPart]:
        """Get all tool result parts.

        Returns:
            List of ToolResultPart objects in this message.
        """
        return [p for p in self.parts if isinstance(p, ToolResultPart)]

    def _build_multimodal_content(self) -> list[dict[str, Any]]:
        """Build a list of content blocks (text + images) for multimodal LLM calls.

        Returns:
            List of dicts in the format expected by OpenAI-compatible APIs.
        """
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
        """Convert to format for LLM API.

        Handles user (with optional multimodal), system, and assistant
        roles. For assistant messages with tool calls, formats them
        in the OpenAI-compatible tool_calls structure.

        Returns:
            A dict ready to be passed to a provider API.
        """
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
