import asyncio
import sqlite3
import sys
import argparse
import uuid
import os
import select
import termios
import time
import tty
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompt_toolkit import HTML
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from rich.console import Console

from .agent import create_eval_agent

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding.key_processor import KeyPressEvent


def _load_env(project_dir: Path) -> None:
    """Load .env files from project dir (tries common names), then home dir."""
    for name in (".env.local", ".env"):
        if load_dotenv(project_dir / name, override=False, verbose=False):
            return
    load_dotenv(Path.home() / ".env", override=False, verbose=False)


console = Console()

WELCOME = """\
[bold green]promptloop[/bold green] — prompt engineering & evaluation agent
Type your message and press [bold]Enter[/bold] to send.
Commands: /help /clear /quit /threads /thread <id>
"""

HELP = """\
[bold]Commands[/bold]
  /help      — show this message
  /clear     — start a new conversation thread
  /threads   — list saved chat threads (resume with: promptloop --thread <id>)
  /thread <id> — switch to thread (in-session)
  /quit      — exit

[bold]Resuming a chat[/bold]
  Run [bold]promptloop --thread <thread_id>[/bold] to continue a previous conversation.
  Use [bold]/threads[/bold] to see available thread IDs.

[bold]Input[/bold]
  [bold]Enter[/bold] to send.
  Press [bold]Esc[/bold] while streaming to interrupt the current response.
  Multi-line paste keeps formatting for prompts, JSON examples, and markdown.

[bold]Getting started[/bold]
  Tell me a prompt file path: "evaluate the prompt at src/prompts/summarize.txt"
  Or just describe what you want to evaluate and I'll help you find it.
"""

# Increase above deepagents default of 25 to handle multi-step eval workflows
RECURSION_LIMIT = 100

CHAT_DB_NAME = "chat.db"
ESCAPE_KEY = "\x1b"


def _list_thread_ids(project_dir: Path) -> list[str]:
    """List thread IDs from the chat SQLite DB."""
    db_path = project_dir / ".evals" / CHAT_DB_NAME
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        with conn:
            cur = conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            )
            return [row[0] for row in cur.fetchall()]
    except sqlite3.OperationalError:
        return []


def _resolve_thread_id(project_dir: Path, requested_id: str) -> tuple[Optional[str], Optional[str]]:
    thread_ids = _list_thread_ids(project_dir)
    if requested_id in thread_ids:
        return requested_id, None

    matches = [tid for tid in thread_ids if tid.startswith(requested_id)]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        return None, f"Ambiguous thread prefix '{requested_id}' matches {len(matches)} threads."
    return None, f"No saved thread found for '{requested_id}'. Use /threads to list saved threads."


def _create_prompt_session() -> "PromptSession[str]":
    """Create a PromptSession with Enter=send and formatting-preserving paste."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event: "KeyPressEvent") -> None:
        event.current_buffer.validate_and_handle()

    @kb.add(Keys.BracketedPaste)
    def _bracketed_paste(event: "KeyPressEvent") -> None:
        """Preserve pasted multi-paragraph input for prompts and examples."""
        pasted_text = event.data if isinstance(event.data, str) else ""
        normalized = _normalize_user_input(pasted_text)
        if not normalized:
            return

        buffer = event.current_buffer
        if buffer.document.text_before_cursor and not buffer.document.text_before_cursor.endswith(("\n", " ")):
            normalized = f"\n{normalized}"
        buffer.insert_text(normalized)

    @kb.add("c-c")
    def _ctrl_c(event: "KeyPressEvent") -> None:
        # Ask prompt_toolkit to end the current prompt with KeyboardInterrupt
        # so callers can handle it without an asyncio traceback.
        event.app.exit(exception=KeyboardInterrupt())

    return PromptSession(key_bindings=kb, multiline=True)


async def _read_user_input(session: "PromptSession[str]", prompt: Any) -> str:
    """Read user input via prompt_toolkit. Enter sends the message."""
    return await session.prompt_async(prompt)


def _normalize_user_input(raw_input: str) -> str:
    """Normalize terminal line endings without changing user formatting."""
    return raw_input.replace("\r\n", "\n").replace("\r", "\n")


def _hide_cursor() -> None:
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def _show_cursor() -> None:
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def _format_tool_args(args: dict) -> str:
    if not args:
        return ""
    parts = []
    for k, v in list(args.items())[:2]:
        v_str = str(v)
        if len(v_str) > 35:
            v_str = v_str[:32] + "…"
        parts.append(f"{k}={v_str!r}" if isinstance(v, str) else f"{k}={v_str}")
    preview = ", ".join(parts)
    if len(args) > 2:
        preview += ", …"
    return f"({preview})"


def _format_tool_result(event: dict) -> str:
    name = event.get("name", "")
    output = event.get("data", {}).get("output")
    if not output:
        return ""
    out_str = str(output)
    lines = [l for l in out_str.splitlines() if l.strip()]
    if any(x in name for x in ("read_file", "read")):
        return f"{len(lines)} lines" if lines else ""
    if any(x in name for x in ("glob", "grep", "search", "ls")):
        return f"{len(lines)} results" if lines else ""
    if out_str.strip() and len(out_str.strip()) < 60:
        return out_str.strip()
    return ""


async def _stream_response(agent, message: str, thread_id: str) -> None:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    console.print("\n[bold green]agent[/bold green]", end=" ")
    _hide_cursor()

    # ── In-place status ticker ────────────────────────────────────────────────
    # Shows "verb… Ns" in place, updating every 300 ms.
    # Call _status_start(verb) to begin, await _status_stop() before any output.
    _st_task: Optional[asyncio.Task[None]] = None
    _st_active = False
    _st_verb = ""
    _st_start = 0.0
    _st_rendered = 0  # visible chars currently on stdout from the ticker

    def _st_erase() -> None:
        nonlocal _st_rendered
        if _st_rendered:
            sys.stdout.write("\b" * _st_rendered + " " * _st_rendered + "\b" * _st_rendered)
            sys.stdout.flush()
            _st_rendered = 0

    async def _st_run() -> None:
        nonlocal _st_rendered
        while _st_active:
            elapsed = int(time.monotonic() - _st_start)
            if _st_verb.startswith("Drafting "):
                text = f"{_st_verb}… {elapsed}s (Esc cancels)"
            else:
                text = f"{_st_verb}… {elapsed}s" if _st_verb else f"… {elapsed}s"
            _st_erase()
            sys.stdout.write(text)
            sys.stdout.flush()
            _st_rendered = len(text)
            await asyncio.sleep(0.3)

    def _status_start(verb: str) -> None:
        nonlocal _st_task, _st_active, _st_verb, _st_start, _st_rendered
        _st_active = False
        if _st_task and not _st_task.done():
            _st_task.cancel()
        _st_erase()
        _st_verb = verb
        _st_start = time.monotonic()
        _st_active = True
        _st_rendered = 0
        _st_task = asyncio.create_task(_st_run())

    async def _status_stop() -> None:
        nonlocal _st_active, _st_task
        _st_active = False
        _st_erase()
        if _st_task and not _st_task.done():
            _st_task.cancel()
            with suppress(asyncio.CancelledError):
                await _st_task
        _st_task = None

    # ─────────────────────────────────────────────────────────────────────────

    interrupted = False
    stop_escape_listener = asyncio.Event()
    escape_task: Optional[asyncio.Task[bool]] = None
    active_tool_run_ids: set[str] = set()
    streaming_text = False
    pending_tool_name: Optional[str] = None

    async def _wait_for_escape(stop_event: asyncio.Event) -> bool:
        if not sys.stdin.isatty():
            await stop_event.wait()
            return False

        fd = sys.stdin.fileno()
        original_mode = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                readable, _, _ = select.select([fd], [], [], 0.1)
                if not readable:
                    await asyncio.sleep(0)
                    continue
                key = os.read(fd, 1).decode("utf-8", errors="ignore")
                if key == ESCAPE_KEY:
                    return True
            return False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, original_mode)

    async def _consume_stream() -> None:
        nonlocal pending_tool_name, streaming_text

        def _is_inside_tool(event: dict[str, Any]) -> bool:
            parent_ids = event.get("parent_ids") or []
            return any(pid in active_tool_run_ids for pid in parent_ids)

        _status_start("Thinking")

        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            version="v2",
        ):
            kind = event.get("event")

            if kind == "on_chat_model_stream":
                if _is_inside_tool(event):
                    continue
                chunk = event["data"].get("chunk")
                if not chunk:
                    continue

                # When model streams a tool call, show which tool is being called
                # before on_tool_start fires (which only fires after model finishes).
                for tc in getattr(chunk, "tool_call_chunks", None) or []:
                    if isinstance(tc, dict) and tc.get("name"):
                        tool_name = tc["name"]
                        if pending_tool_name != tool_name:
                            pending_tool_name = tool_name
                            if streaming_text:
                                sys.stdout.write("\n")
                                sys.stdout.flush()
                                streaming_text = False
                            _status_start(f"Drafting {tool_name} input")
                        break

                content = chunk.content
                text = ""
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                if text:
                    if not streaming_text:
                        await _status_stop()
                        streaming_text = True
                    sys.stdout.write(text)
                    sys.stdout.flush()

            elif kind == "on_tool_start":
                run_id = event.get("run_id")
                if isinstance(run_id, str):
                    active_tool_run_ids.add(run_id)
                await _status_stop()
                streaming_text = False
                name = event.get("name", "tool")
                args = event.get("data", {}).get("input") or {}
                args_preview = _format_tool_args(args)
                prefix = "" if pending_tool_name else "\n"
                pending_tool_name = None
                sys.stdout.write(f"{prefix}  → {name}{args_preview}")
                sys.stdout.flush()
                _status_start("")  # show elapsed on the tool line

            elif kind == "on_tool_end":
                run_id = event.get("run_id")
                if isinstance(run_id, str):
                    active_tool_run_ids.discard(run_id)
                await _status_stop()
                result_summary = _format_tool_result(event)
                suffix = f" · {result_summary}" if result_summary else ""
                sys.stdout.write(f" ✓{suffix}\n")
                sys.stdout.flush()
                _status_start("Thinking")  # restart for next model step

            elif kind == "on_chat_model_end":
                # Model finished its turn but LangGraph may still be routing.
                # Restart status so the user sees activity instead of silence.
                if not _is_inside_tool(event) and streaming_text:
                    streaming_text = False
                    _status_start("Processing")

    try:
        stream_task = asyncio.create_task(_consume_stream())
        escape_task = asyncio.create_task(_wait_for_escape(stop_escape_listener))
        done, _ = await asyncio.wait(
            {stream_task, escape_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if escape_task in done and escape_task.result():
            interrupted = True
            stream_task.cancel()
            with suppress(asyncio.CancelledError):
                await stream_task
        elif stream_task in done:
            await stream_task
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_escape_listener.set()
        await _status_stop()
        if escape_task is not None and not escape_task.done():
            escape_task.cancel()
            with suppress(asyncio.CancelledError):
                await escape_task
        _show_cursor()

    if interrupted:
        console.print("\n[yellow]interrupted[/yellow]", end="")
    console.print()


async def amain() -> None:
    parser = argparse.ArgumentParser(
        prog="promptloop",
        description="Interactive prompt evaluation agent",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root to evaluate. Must contain the prompts you want to eval (default: cwd)",
    )
    parser.add_argument(
        "--model",
        default="anthropic:claude-sonnet-4-6",
        help="Orchestrator model (default: anthropic:claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--thread",
        metavar="ID",
        help="Resume a previous chat by thread ID (use /threads to list)",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    _load_env(project_dir)

    # Ensure .evals exists for chat DB
    (project_dir / ".evals").mkdir(parents=True, exist_ok=True)
    db_path = project_dir / ".evals" / CHAT_DB_NAME

    async with AsyncSqliteSaver.from_conn_string(str(db_path)) as checkpointer:
        agent = create_eval_agent(project_dir, args.model, checkpointer=checkpointer)
        thread_id = args.thread if args.thread else str(uuid.uuid4())

        console.print(WELCOME)
        console.print(f"[dim]project: {project_dir}[/dim]")
        if args.thread:
            console.print(f"[dim]thread: {thread_id}[/dim] (resuming)")
        console.print()

        session = _create_prompt_session()
        prompt_template = (
            '<style fg="ansicyan"><b>you</b></style> '
            '<style fg="#6b7280">[{thread}…]: </style>'
        )

        while True:
            try:
                prompt_str = HTML(prompt_template).format(thread=thread_id[:8])
                user_input = _normalize_user_input(await _read_user_input(session, prompt_str))
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]bye[/dim]")
                break

            stripped_input = user_input.strip()
            if not stripped_input:
                continue

            if stripped_input.startswith("/") and "\n" not in stripped_input:
                parts = stripped_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1].strip() if len(parts) > 1 else ""

                if cmd in ("/quit", "/exit", "/q"):
                    console.print("[dim]bye[/dim]")
                    break
                elif cmd == "/help":
                    console.print(HELP)
                elif cmd == "/clear":
                    thread_id = str(uuid.uuid4())
                    console.print(
                        "[dim]conversation cleared — new thread started[/dim]"
                    )
                elif cmd == "/threads":
                    thread_ids = _list_thread_ids(project_dir)
                    if not thread_ids:
                        console.print("[dim]No saved threads yet.[/dim]")
                    else:
                        console.print("[bold]Saved threads:[/bold]")
                        for tid in thread_ids:
                            marker = " ←" if tid == thread_id else ""
                            console.print(f"  {tid}{marker}")
                        console.print(
                            "[dim]Resume with: promptloop --thread <id>[/dim]"
                        )
                elif cmd == "/thread" and arg:
                    resolved_thread_id, error = _resolve_thread_id(project_dir, arg)
                    if error:
                        console.print(f"[red]{error}[/red]")
                    else:
                        thread_id = resolved_thread_id or arg
                        console.print(f"[dim]switched to thread {thread_id}[/dim]")
                elif cmd == "/thread":
                    console.print("[red]Usage: /thread <thread_id>[/red]")
                else:
                    console.print(f"[red]unknown command: {cmd}[/red]")
                continue

            await _stream_response(agent, user_input, thread_id)


def main() -> None:
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        # Final safety net for interrupts outside prompt input flow.
        console.print("\n[dim]bye[/dim]")
