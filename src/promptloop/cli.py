import asyncio
import sqlite3
import sys
import argparse
import uuid
import os
import select
import termios
import tty
from contextlib import suppress
from pathlib import Path
from typing import Any, TYPE_CHECKING

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
  Multi-line paste is flattened to a single line before sending.

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


def _create_prompt_session() -> "PromptSession[str]":
    """Create a PromptSession with Enter=send and single-line paste behavior."""
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys

    kb = KeyBindings()

    @kb.add("enter")
    def _enter(event: "KeyPressEvent") -> None:
        event.current_buffer.validate_and_handle()

    @kb.add(Keys.BracketedPaste)
    def _bracketed_paste(event: "KeyPressEvent") -> None:
        """Flatten pasted multi-paragraph input into one line before insert."""
        pasted_text = event.data if isinstance(event.data, str) else ""
        flattened = _normalize_user_input(pasted_text)
        if not flattened:
            return

        buffer = event.current_buffer
        if buffer.document.text_before_cursor and not buffer.document.text_before_cursor.endswith(" "):
            flattened = f" {flattened}"
        buffer.insert_text(flattened)

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
    """Flatten pasted/typed multi-line input into a single line."""
    return " ".join(raw_input.split())


def _hide_cursor() -> None:
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def _show_cursor() -> None:
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


async def _stream_response(agent, message: str, thread_id: str) -> None:
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    console.print("\n[bold green]agent[/bold green] ", end="")
    _hide_cursor()

    async def _wait_for_escape(stop_event: asyncio.Event) -> bool:
        if not sys.stdin.isatty():
            await stop_event.wait()
            return False

        fd = sys.stdin.fileno()
        original_mode = termios.tcgetattr(fd)
        try:
            # cbreak mode lets us read Esc immediately without waiting for Enter.
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

    interrupted = False
    stop_escape_listener = asyncio.Event()
    escape_task: asyncio.Task[bool] | None = None
    waiting_animation_task: asyncio.Task[None] | None = None
    waiting_animation_done = asyncio.Event()
    waiting_animation_stopped = False
    active_tool_run_ids: set[str] = set()
    needs_newline_before_next_streamed_text = False

    async def _animate_waiting() -> None:
        # Symbol-only bar: fill, pause at full, drain, pause at empty.
        phases: list[tuple[str, float]] = [
            ("[.    ]", 0.20),
            ("[..   ]", 0.20),
            ("[...  ]", 0.20),
            ("[.... ]", 0.20),
            ("[.....]", 0.20),
            ("[.....]", 0.45),
            ("[.... ]", 0.20),
            ("[...  ]", 0.20),
            ("[..   ]", 0.20),
            ("[.    ]", 0.20),
            ("[     ]", 0.35),
        ]
        frame_len = max(len(phase[0]) for phase in phases)
        phase_idx = 0
        first_frame = True
        while not waiting_animation_done.is_set():
            frame, delay = phases[phase_idx]
            padded_frame = frame.ljust(frame_len)
            if first_frame:
                first_frame = False
                sys.stdout.write(padded_frame)
            else:
                sys.stdout.write("\b" * frame_len + padded_frame)
            sys.stdout.flush()
            phase_idx = (phase_idx + 1) % len(phases)
            await asyncio.sleep(delay)

        # Erase the waiting indicator so streamed content starts cleanly.
        if first_frame:
            # Nothing was rendered; avoid writing backspaces.
            return
        sys.stdout.write("\b" * frame_len + (" " * frame_len) + ("\b" * frame_len))
        sys.stdout.flush()

    async def _stop_waiting_animation() -> None:
        nonlocal waiting_animation_stopped
        if waiting_animation_stopped:
            return
        waiting_animation_stopped = True
        waiting_animation_done.set()
        if waiting_animation_task is not None and not waiting_animation_task.done():
            await waiting_animation_task

    async def _consume_stream() -> None:
        nonlocal needs_newline_before_next_streamed_text

        def _is_inside_tool(event: dict[str, Any]) -> bool:
            parent_ids = event.get("parent_ids") or []
            return any(parent_id in active_tool_run_ids for parent_id in parent_ids)

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
                if chunk:
                    content = chunk.content
                    if isinstance(content, str) and content:
                        await _stop_waiting_animation()
                        if needs_newline_before_next_streamed_text:
                            sys.stdout.write("\n")
                            needs_newline_before_next_streamed_text = False
                        sys.stdout.write(content)
                        sys.stdout.flush()
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    await _stop_waiting_animation()
                                    if needs_newline_before_next_streamed_text:
                                        sys.stdout.write("\n")
                                        needs_newline_before_next_streamed_text = False
                                    sys.stdout.write(text)
                                    sys.stdout.flush()

            elif kind == "on_tool_start":
                run_id = event.get("run_id")
                if isinstance(run_id, str):
                    active_tool_run_ids.add(run_id)
                await _stop_waiting_animation()
                name = event.get("name", "tool")
                console.print(f"\n[dim]  → {name}[/dim]", end="")

            elif kind == "on_tool_end":
                run_id = event.get("run_id")
                if isinstance(run_id, str):
                    active_tool_run_ids.discard(run_id)
                console.print(" [dim]✓[/dim]", end="")
                needs_newline_before_next_streamed_text = True

    try:
        waiting_animation_task = asyncio.create_task(_animate_waiting())
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
            # Propagate errors from streaming (if any).
            await stream_task
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop_escape_listener.set()
        await _stop_waiting_animation()
        if escape_task is not None and not escape_task.done():
            escape_task.cancel()
            with suppress(asyncio.CancelledError):
                await escape_task
        _show_cursor()

    if interrupted:
        console.print("\n[yellow]interrupted[/yellow]", end="")
    console.print()  # final newline


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
                user_input = _normalize_user_input(
                    await _read_user_input(session, prompt_str)
                )
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]bye[/dim]")
                break

            if not user_input.strip():
                continue

            if user_input.startswith("/"):
                parts = user_input.strip().split(maxsplit=1)
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
                    thread_id = arg
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
