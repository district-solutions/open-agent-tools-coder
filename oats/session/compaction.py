"""
Conversation compaction — auto-summarize older messages when approaching context limits.

Prevents context overflow by detecting when token count approaches the model's limit
and replacing older messages with a compact summary.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Optional

from oats.session.message import Message
from oats.core.features import rich_compaction_enabled, context_collapse_candidate_enabled
from oats.provider.provider import (
    CompletionRequest,
    Message as ProviderMessage,
    Provider,
    get_provider,
)
from oats.log import cl

log = cl("session.compact")


# Compact when estimated tokens reach this fraction of context window
COMPACTION_THRESHOLD_RATIO = 0.75

# Always preserve this many recent messages
PRESERVE_RECENT_MESSAGES = 8

# Summary request max tokens
SUMMARY_MAX_TOKENS = 2000


COMPACTION_SYSTEM_PROMPT = """You are summarizing an engineering work session for continuation.

Respond with plain text only. Do not call tools.
Preserve concrete technical details needed to continue work accurately.
Prefer exact file paths, tool names, decisions, errors, and unfinished tasks over vague narration.
Use the following sections:
1. Primary request
2. State capsule
3. Important files and code areas
4. Tools used and why
5. Decisions and changes made
6. Errors, risks, and blockers
7. Outstanding work / next likely step
"""


class ConversationCompactor:
    """Detects context limit approaching and compacts older messages.

    Uses the LLM itself to generate a summary of older messages,
    then replaces them with a single summary message. Supports both
    simple summarization and rich context-capsule mode with deterministic
    state extraction.

    Attributes:
        _max_tokens: The model's context window size in tokens.
        _provider_id: Provider ID for the summarization call.
        _model_id: Model ID for the summarization call.
    """

    def __init__(
        self,
        model_context_length: int = 32768,
        provider_id: str | None = None,
        model_id: str | None = None,
    ) -> None:
        """Initialize the compactor.

        Args:
            model_context_length: The model's context window in tokens.
            provider_id: Provider ID for summarization calls.
            model_id: Model ID for summarization calls.
        """
        self._max_tokens = model_context_length
        self._provider_id = provider_id
        self._model_id = model_id

    def should_compact(self, messages: list[Message]) -> bool:
        """Check if compaction is needed based on estimated token count.

        Args:
            messages: The current conversation messages.

        Returns:
            True if the estimated token count exceeds the compaction threshold.
        """
        if len(messages) <= PRESERVE_RECENT_MESSAGES:
            return False
        estimated = self._estimate_tokens(messages)
        threshold = int(self._max_tokens * COMPACTION_THRESHOLD_RATIO)
        should = estimated > threshold
        if should:
            log.info(
                f"compaction_needed: estimated={estimated} "
                f"threshold={threshold} messages={len(messages)}"
            )
        return should

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Token estimate across text, tool calls, and tool results.

        Delegates to :mod:`oats.core.tokens` so the compactor, the session
        token budget, and the context ledger all agree on what a token costs
        — previously each used chars/4 independently.
        """
        from oats.core.tokens import count_message_tokens
        return count_message_tokens(messages)

    async def compact(
        self,
        messages: list[Message],
        session_id: str,
    ) -> list[Message]:
        """Compact older messages into a summary.

        Splits messages into older (to summarize) and recent (to preserve).
        Generates an LLM summary of the older messages and replaces them
        with a single summary message.

        Args:
            messages: The full conversation message list.
            session_id: The session ID for the summary message.

        Returns:
            A new message list with older messages replaced by a summary.
        """
        if len(messages) <= PRESERVE_RECENT_MESSAGES:
            return messages

        # Pin the original task brief verbatim. The first user message is the
        # task spec; truncating or summarizing it loses concrete steps the
        # model needs to finish (e.g. "step 5: write SUMMARY.txt").
        pinned_brief: Optional[Message] = None
        for m in messages:
            if m.role == "user" and (m.get_text_content() or "").strip():
                pinned_brief = m
                break

        # Split: older messages to summarize, recent to preserve
        cutoff = len(messages) - PRESERVE_RECENT_MESSAGES
        older = messages[:cutoff]
        recent = messages[cutoff:]

        log.info(
            f"compacting: summarizing {len(older)} messages, "
            f"preserving {len(recent)} recent"
            + (" (task brief pinned)" if pinned_brief is not None else "")
        )

        # Generate summary
        summary = await self._summarize(older)

        # Create summary message
        summary_role = "system" if context_collapse_candidate_enabled() else "user"
        header = (
            f"[Context Capsule — {len(older)} messages compacted]\n\n"
            if context_collapse_candidate_enabled()
            else f"[Conversation Summary — {len(older)} messages compacted]\n\n"
        )
        summary_msg = Message(
            session_id=session_id,
            role=summary_role,
        )
        summary_msg.add_text(f"{header}{summary}")

        # Don't double-pin if the brief is already inside `recent`.
        result: list[Message] = []
        if pinned_brief is not None and pinned_brief not in recent:
            result.append(pinned_brief)
        result.append(summary_msg)
        result.extend(recent)
        return result

    async def _summarize(self, messages: list[Message]) -> str:
        """Use the LLM to summarize a batch of messages.

        Extracts a deterministic state capsule from tool activity, then
        asks the LLM to produce a continuation-grade summary.

        Args:
            messages: The messages to summarize.

        Returns:
            The summary text.
        """
        state_capsule = self._extract_state_capsule(messages)
        rendered_capsule = self._render_state_capsule(state_capsule)

        # Build summarization prompt
        conversation_text = []
        for msg in messages:
            role = msg.role.upper()
            text = msg.get_text_content() or ""

            # Include tool activity summaries
            tool_calls = msg.get_tool_calls()
            tool_results = msg.get_tool_results()

            parts = []
            if text:
                parts.append(text[:500])
            for tc in tool_calls:
                parts.append(f"[Called tool: {tc.tool_name}({str(tc.arguments)[:200]})]")
            for tr in tool_results:
                status = "error" if tr.error else "success"
                output_preview = (tr.output or "")[:200]
                parts.append(f"[Tool result ({status}): {output_preview}]")

            if parts:
                conversation_text.append(f"{role}: {' | '.join(parts)}")

        conv_str = "\n".join(conversation_text)

        # Truncate if too long for summarization
        if len(conv_str) > 30000:
            conv_str = conv_str[:30000] + "\n...(truncated)"

        if context_collapse_candidate_enabled():
            summarization_prompt = (
                "Create a continuation-grade context capsule for the following engineering session.\n"
                "You are given a deterministic state capsule extracted from the session.\n"
                "Preserve and refine that state rather than replacing it with generic prose.\n"
                "Keep the response dense, factual, and optimized for continuing work after compaction.\n\n"
                f"DETERMINISTIC STATE CAPSULE:\n{rendered_capsule}\n\n"
                f"CONVERSATION:\n{conv_str}"
            )
            system_prompt = COMPACTION_SYSTEM_PROMPT
        elif rich_compaction_enabled():
            summarization_prompt = (
                "Create a continuation-grade summary of the following engineering session.\n"
                "Preserve file paths, code areas, tool usage, decisions, errors, unfinished work, and active state.\n"
                "Be dense and factual rather than conversational.\n\n"
                f"CONVERSATION:\n{conv_str}"
            )
            system_prompt = COMPACTION_SYSTEM_PROMPT
        else:
            summarization_prompt = (
                "Summarize the following conversation between a user and an AI assistant. "
                "Focus on:\n"
                "1. What the user asked for\n"
                "2. Key decisions made\n"
                "3. Files that were read or modified\n"
                "4. Important findings or results\n"
                "5. Any unresolved issues or next steps\n\n"
                "Be concise but preserve critical details like file paths and decisions.\n\n"
                f"CONVERSATION:\n{conv_str}"
            )
            system_prompt = (
                "You are a conversation summarizer. Produce a concise, factual summary."
            )

        try:
            provider = get_provider(self._provider_id)
            request = CompletionRequest(
                messages=[
                    ProviderMessage(
                        role="system",
                        content=system_prompt,
                    ),
                    ProviderMessage(role="user", content=summarization_prompt),
                ],
                model=self._model_id,
                provider_id=self._provider_id,
                max_tokens=SUMMARY_MAX_TOKENS,
            )
            response = await provider.complete(request)
            summary = response.content or "[Summary generation failed]"
            if context_collapse_candidate_enabled():
                return self._merge_context_collapse_summary(rendered_capsule, summary)
            return summary
        except Exception as e:
            log.error(f"summarization_failed: {e}")
            # Fallback: create a basic summary without LLM
            fallback = self._fallback_summary(messages)
            if context_collapse_candidate_enabled():
                return self._merge_context_collapse_summary(rendered_capsule, fallback)
            return fallback

    def _fallback_summary(self, messages: list[Message]) -> str:
        """Create a basic summary without using the LLM."""
        user_msgs = []
        tools_used = set()
        files_mentioned = set()

        for msg in messages:
            if msg.role == "user":
                text = msg.get_text_content() or ""
                if text:
                    user_msgs.append(text[:100])
            for tc in msg.get_tool_calls():
                tools_used.add(tc.tool_name)
                # Extract file paths from arguments
                for key in ("file_path", "path", "file"):
                    if key in tc.arguments:
                        files_mentioned.add(str(tc.arguments[key]))

        lines = [
            f"Primary request: compacted {len(messages)} earlier messages for continuation."
        ]
        if context_collapse_candidate_enabled():
            lines.append("State capsule: continue from the preserved recent messages, tool results, and unresolved work without redoing completed steps.")
        if user_msgs:
            lines.append(f"Recent user requests: {'; '.join(user_msgs[:5])}")
        if tools_used:
            lines.append(f"Tools used: {', '.join(sorted(tools_used))}")
        if files_mentioned:
            lines.append(f"Files involved: {', '.join(sorted(files_mentioned)[:10])}")
        lines.append("Outstanding work: continue from the latest preserved messages and pending user request.")

        return "\n".join(lines)

    def _extract_state_capsule(self, messages: list[Message]) -> dict[str, Any]:
        """Extract deterministic working state from a message batch."""
        first_user_request = ""
        recent_user_requests: list[str] = []
        tool_counts: Counter[str] = Counter()
        files_read: list[str] = []
        files_modified: list[str] = []
        files_searched: list[str] = []
        unresolved_errors: list[str] = []
        key_results: list[str] = []
        continuation_hints: list[str] = []

        # Ordered (path, action) events so we can reconstruct the final state
        # per file — "read before edit" becomes "edited (prior read stale)"
        # rather than two independent facts in the capsule.
        file_events: list[tuple[str, str]] = []

        seen_read: set[str] = set()
        seen_modified: set[str] = set()
        seen_search: set[str] = set()
        seen_errors: set[str] = set()
        seen_results: set[str] = set()

        for msg in messages:
            text = (msg.get_text_content() or "").strip()
            if msg.role == "user" and text:
                if not first_user_request:
                    first_user_request = text[:400]
                recent_user_requests.append(text[:240])

            for tc in msg.get_tool_calls():
                tool_counts[tc.tool_name] += 1
                path = self._extract_path(tc.arguments)
                if tc.tool_name == "read" and path:
                    file_events.append((path, "read"))
                    if path not in seen_read:
                        seen_read.add(path)
                        files_read.append(path)
                elif tc.tool_name in {"write", "edit", "multiedit", "patch"} and path:
                    file_events.append((path, "modified"))
                    if path not in seen_modified:
                        seen_modified.add(path)
                        files_modified.append(path)
                elif tc.tool_name in {"glob", "grep", "lsp"} and path and path not in seen_search:
                    seen_search.add(path)
                    files_searched.append(path)

            for tr in msg.get_tool_results():
                meta_path = None
                if tr.metadata:
                    meta_path = tr.metadata.get("file_path")
                    if meta_path and tr.tool_name in {"write", "edit", "multiedit", "patch"} and meta_path not in seen_modified:
                        seen_modified.add(str(meta_path))
                        files_modified.append(str(meta_path))
                    if tr.metadata.get("lsp_diagnostics"):
                        diag_preview = f"{tr.tool_name}: {str(tr.metadata['lsp_diagnostics'])[:220]}"
                        if diag_preview not in seen_results:
                            seen_results.add(diag_preview)
                            key_results.append(diag_preview)
                if tr.error:
                    err_line = f"{tr.tool_name}: {tr.error[:240]}"
                    if err_line not in seen_errors:
                        seen_errors.add(err_line)
                        unresolved_errors.append(err_line)
                elif tr.output:
                    output = tr.output.strip()
                    if output:
                        compact = f"{tr.tool_name}: {output[:220]}"
                        if compact not in seen_results:
                            seen_results.add(compact)
                            key_results.append(compact)

        if recent_user_requests:
            continuation_hints.append(f"Latest user request: {recent_user_requests[-1]}")
        if unresolved_errors:
            continuation_hints.append("Address unresolved tool errors before broad new exploration.")
        if files_modified:
            continuation_hints.append(
                "Preserve and verify the modified files before making wider changes."
            )

        return {
            "primary_request": first_user_request or "Continue the active engineering task accurately.",
            "recent_user_requests": recent_user_requests[-5:],
            "tool_counts": tool_counts,
            "files_read": files_read[:12],
            "files_modified": files_modified[:12],
            "files_searched": files_searched[:12],
            "file_state": self._build_file_state(file_events),
            "unresolved_errors": unresolved_errors[:8],
            "key_results": key_results[:8],
            "continuation_hints": continuation_hints[:5],
        }

    def _build_file_state(self, events: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Collapse per-file event streams into a final-state table.

        Returns [(path, status)] where status is one of:
        - 'modified'           : edited; prior reads (if any) are stale
        - 'modified (re-read)' : edited, then re-read — in-memory content fresh
        - 'read'               : read-only; no modifications seen

        The goal is: at continuation time, the model should know which files
        have been touched and whether its recollection of their contents is
        still valid. Without this, a read earlier in the session gets
        summarized identically whether or not an edit since invalidated it.
        """
        if not events:
            return []
        per_file: dict[str, list[str]] = {}
        for path, action in events:
            per_file.setdefault(path, []).append(action)

        out: list[tuple[str, str]] = []
        for path, actions in per_file.items():
            if "modified" not in actions:
                out.append((path, "read"))
                continue
            last_mod = len(actions) - 1 - actions[::-1].index("modified")
            # Any read after the last modification means the in-memory read is fresh.
            if "read" in actions[last_mod + 1:]:
                out.append((path, "modified (re-read)"))
            else:
                out.append((path, "modified"))
        # Bound the output — 20 entries is plenty for any realistic session.
        return out[:20]

    def _render_state_capsule(self, capsule: dict[str, Any]) -> str:
        """Render deterministic state into a continuation-friendly text block."""
        tool_counts: Counter[str] = capsule["tool_counts"]
        tool_summary = ", ".join(
            f"{name} x{count}" for name, count in tool_counts.most_common(8)
        ) or "none"

        file_state = capsule.get("file_state") or []
        file_state_lines = (
            [f"- {path}: {status}" for path, status in file_state]
            or ["- none"]
        )
        sections = [
            "Primary request:",
            capsule["primary_request"],
            "",
            "File state (final status per touched file):",
            *file_state_lines,
            "",
            "Search/navigation targets:",
            *([f"- {path}" for path in capsule["files_searched"]] or ["- none"]),
            "",
            "Tool usage:",
            f"- {tool_summary}",
            "",
            "Key tool results:",
            *([f"- {item}" for item in capsule["key_results"]] or ["- none"]),
            "",
            "Unresolved errors or risks:",
            *([f"- {item}" for item in capsule["unresolved_errors"]] or ["- none"]),
            "",
            "Recent user requests:",
            *([f"- {item}" for item in capsule["recent_user_requests"]] or ["- none"]),
            "",
            "Continuation hints:",
            *([f"- {item}" for item in capsule["continuation_hints"]] or ["- continue from the preserved recent messages and avoid redoing completed work."]),
        ]
        return "\n".join(sections)

    def _merge_context_collapse_summary(
        self,
        rendered_capsule: str,
        llm_summary: str,
    ) -> str:
        """Combine deterministic state with the model-written continuation summary."""
        return (
            "## Deterministic State Capsule\n\n"
            f"{rendered_capsule}\n\n"
            "## Narrative Continuation Summary\n\n"
            f"{llm_summary.strip()}"
        )

    def _extract_path(self, arguments: dict[str, Any]) -> str | None:
        """Best-effort extraction of a relevant path-like argument."""
        for key in ("file_path", "path", "file"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None
