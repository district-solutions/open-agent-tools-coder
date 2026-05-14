"""Stream a user message through the session processor and render the response in the TUI.

Handles all event types emitted by the processor (text deltas, tool calls,
tool results, usage stats, compaction, errors, warnings) and displays
them with appropriate icons, formatting, and a live status indicator.
"""

import uuid
import time
from rich.console import Console
from oats.session.processor import SessionProcessor
from oats.date import utc
from oats.session.session import Session
from oats.cli.tui.tui_utils import (
    _StatusTracker,
    SYM_OK,
    SYM_ITER,
    SYM_TOOL,
    SYM_ERR,
    SYM_WARN,
    SYM_COMPACT,
    TOOL_ICONS,
    _format_tokens,
    _strip_think_blocks,
    _format_tool_args,
    _get_provider_display,
)

async def process_message(
    console: Console,
    processor: SessionProcessor,
    message: str,
    session: Session,
    auto_approve: bool = True,
    images: list[dict[str, str]] | None = None,
    max_tokens: int = 8192,
) -> str:
    """Process a single message through the session processor, streaming output."""
    provider_id, model_id = _get_provider_display()
    uid = str(uuid.uuid4()).replace('-', '')
    now = utc()
    tid = f'oats_{now.strftime("%Y%m%d%H%M%S")}_{uid[:8]}'

    final_text = ""
    tool_call_count = 0
    turn_start = time.monotonic()
    turn_input_tokens = 0
    turn_output_tokens = 0

    status = _StatusTracker(console=console)
    status.start()

    try:
        async for event in processor.process_message(
            message, auto_approve_tools=auto_approve,
            max_tokens=max_tokens, images=images,
        ):
            etype = event.get("type")

            if etype == "llm_request":
                iteration = event.get("iteration", 1)
                if iteration > 1:
                    status.stop()
                    console.print(f"  [dim]{SYM_ITER} iter {iteration}[/dim]")
                    status.set_phase("thinking")
                    status.start()

            elif etype == "assistant_text_delta":
                delta = event.get("content", "")
                if delta:
                    status.set_phase("generating")
                    status.add_output_chars(len(delta))

            elif etype == "assistant_text":
                status.stop()
                content = event.get("content", "")
                final_text = content
                clean = _strip_think_blocks(content)
                if clean.strip():
                    from rich.markdown import Markdown
                    console.print()
                    console.print(Markdown(clean))

            elif etype == "tool_call":
                status.stop()
                tool_name = event.get("tool_name", "")
                args = event.get("arguments", {})
                tool_call_count += 1
                icon = TOOL_ICONS.get(tool_name, SYM_TOOL)
                args_preview = _format_tool_args(tool_name, args)
                console.print(f"  {icon} [yellow]{tool_name}[/yellow] [dim]{args_preview}[/dim]")
                status.set_phase(f"running {tool_name}")
                status.start()

            elif etype == "tool_result":
                status.stop()
                tool_name = event.get("tool_name", "")
                error = event.get("error")
                output = str(event.get("output", ""))
                if error:
                    console.print(f"    [red]{SYM_ERR} {error[:120]}[/red]")
                else:
                    preview = output[:100].replace('\n', ' ').strip()
                    if len(output) > 100:
                        preview += "…"
                    if preview:
                        console.print(f"    [dim]{SYM_OK} {preview}[/dim]")
                    else:
                        console.print(f"    [dim]{SYM_OK} done[/dim]")
                status.set_phase("thinking")
                status.start()

            elif etype == "usage":
                usage = event.get("usage", {})
                in_tok = usage.get("prompt_tokens", 0)
                out_tok = usage.get("completion_tokens", 0)
                turn_input_tokens += in_tok
                turn_output_tokens += out_tok
                status.add_tokens(in_tok, out_tok)

            elif etype == "compaction":
                status.stop()
                console.print(f"  [dim]{SYM_COMPACT} context compacted[/dim]")
                status.start()

            elif etype == "error":
                status.stop()
                err = event.get("error", "unknown")
                console.print(f"  [red]{SYM_ERR} {err}[/red]")

            elif etype == "warning":
                status.stop()
                msg = event.get("message", "")
                console.print(f"  [yellow]{SYM_WARN} {msg}[/yellow]")
                status.start()

            elif etype == "complete":
                pass

    except KeyboardInterrupt:
        console.print(f"\n  [yellow]interrupted[/yellow]")
    except Exception as e:
        console.print(f"  [red]{SYM_ERR} {e}[/red]")
    finally:
        status.stop()

    # ── Turn stats ────────────────────────────────────────────────
    elapsed = time.monotonic() - turn_start
    stats_parts = []
    if turn_input_tokens or turn_output_tokens:
        stats_parts.append(f"in:{_format_tokens(turn_input_tokens)}")
        stats_parts.append(f"out:{_format_tokens(turn_output_tokens)}")
    if tool_call_count:
        stats_parts.append(f"tools:{tool_call_count}")
    stats_parts.append(f"{elapsed:.1f}s")
    console.print(f"\n  [dim]{' · '.join(stats_parts)}[/dim]")

    return final_text
