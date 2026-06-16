import json
import difflib
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool

SAFE_PROMPT_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class ApprovalGate:
    """Shared approval state between the CLI and tools.

    The CLI sets a decision at on_tool_start (after showing the diff and
    reading a keypress). The tool reads it synchronously when it runs,
    which always happens after on_tool_start completes.
    """

    def __init__(self) -> None:
        self._decisions: dict[str, bool] = {}

    def resolve(self, key: str, approved: bool) -> None:
        self._decisions[key] = approved

    def consume(self, key: str) -> bool:
        return self._decisions.pop(key, True)


def make_prompt_tools(project_dir: Path, gate: ApprovalGate):
    prompts_dir = project_dir / ".evals" / "prompts"

    def _next_version(history_dir: Path) -> int:
        existing = list(history_dir.glob("v*.txt"))
        if not existing:
            return 1
        return max(int(p.stem[1:]) for p in existing) + 1

    def _save_prompt(
        content: str,
        prompt_id: str,
        source_path: Optional[str],
        source_type: str,
    ) -> tuple[int, Path]:
        prompt_dir = _prompt_dir(prompt_id)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        history_dir = prompt_dir / "history"
        history_dir.mkdir(exist_ok=True)

        version = _next_version(history_dir)
        (prompt_dir / "current.txt").write_text(content)
        (history_dir / f"v{version}.txt").write_text(content)

        meta = {
            "source_path": source_path,
            "source_type": source_type,
            "prompt_id": prompt_id,
            "registered_at": datetime.now().isoformat(),
            "version": version,
        }
        (prompt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return version, prompt_dir

    def _inline_prompt_id() -> str:
        return f"inline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _prompt_dir(prompt_id: str) -> Path:
        if not SAFE_PROMPT_ID.fullmatch(prompt_id):
            raise ValueError(
                "prompt_id may only contain letters, numbers, dots, underscores, and hyphens."
            )
        return prompts_dir / prompt_id

    def _source_path_if_file(source: str) -> Optional[Path]:
        if "\n" in source or len(source) > 240:
            return None

        p = Path(source)
        if not p.is_absolute():
            p = project_dir / p

        try:
            return p if p.exists() and p.is_file() else None
        except OSError:
            return None

    def _diff(current: str, proposed: str, prompt_id: str) -> str:
        diff_lines = list(difflib.unified_diff(
            current.splitlines(keepends=True),
            proposed.splitlines(keepends=True),
            fromfile=f"{prompt_id} (current)",
            tofile=f"{prompt_id} (proposed)",
        ))
        return "".join(diff_lines)

    def _build_prompt_from_edits(current: str, edits: list[dict]) -> tuple[str, Optional[str]]:
        if not edits:
            return current, "Error: no edits provided."

        validated_edits = []
        for i, edit in enumerate(edits, start=1):
            old = edit.get("old")
            new = edit.get("new")
            if not isinstance(old, str) or not isinstance(new, str):
                return current, f"Error: edit {i} must include string 'old' and 'new' values."
            if not old:
                return current, f"Error: edit {i} has an empty 'old' value."

            count = current.count(old)
            if count == 0:
                return current, f"Error: edit {i} did not match the current prompt exactly."
            if count > 1:
                return current, f"Error: edit {i} matched {count} places. Make the 'old' text more specific."
            validated_edits.append((old, new))

        proposed = current
        for old, new in validated_edits:
            proposed = proposed.replace(old, new, 1)
        return proposed, None

    def _apply_content(prompt_id: str, new_content: str) -> str:
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not found."

        current = (prompt_dir / "current.txt").read_text()
        if current == new_content:
            return "No changes detected between current and proposed content."

        meta = json.loads((prompt_dir / "meta.json").read_text())
        new_version = meta["version"] + 1
        source_path_value = meta.get("source_path")
        if not source_path_value:
            source_note = "No source file to update for inline prompt."
        else:
            source_path = Path(source_path_value)
            if not source_path.is_absolute():
                source_path = project_dir / source_path

            if not source_path.exists():
                return f"Error: source file {source_path} not found; changes were not applied."
            source_path.write_text(new_content)
            source_note = f"Source file updated at {source_path}."

        history_dir = prompt_dir / "history"
        (history_dir / f"v{new_version}.txt").write_text(new_content)
        (prompt_dir / "current.txt").write_text(new_content)

        meta["version"] = new_version
        meta["updated_at"] = datetime.now().isoformat()
        (prompt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        return f"Applied. Prompt '{prompt_id}' is now v{new_version}. {source_note}"

    @tool(parse_docstring=True)
    def register_prompt(source: str, prompt_id: Optional[str] = None) -> str:
        """Register a prompt file or inline prompt text into the eval system.

        If source is an existing path, reads the file and records the path for
        future syncing. Otherwise, treats source as the literal prompt content.
        Call this first before any eval work.

        Args:
            source: Path to a prompt file, or literal inline prompt text.
            prompt_id: Optional short identifier. Defaults to the file stem for
                files, or an inline timestamp ID for pasted prompt text.

        Returns:
            Confirmation with the registered prompt ID and a content preview.
        """
        p = _source_path_if_file(source)

        if p is not None:
            content = p.read_text()
            pid = prompt_id or p.stem.replace(" ", "_").lower()
            source_path = str(p)
            source_type = "file"
            source_label = f"Source: {p}"
        else:
            content = source
            if not content.strip():
                return "Error: inline prompt content is empty."
            pid = prompt_id or _inline_prompt_id()
            source_path = None
            source_type = "inline"
            source_label = "Source: inline prompt"

        try:
            version, _ = _save_prompt(content, pid, source_path, source_type)
        except ValueError as e:
            return f"Error: {e}"
        preview = content[:300] + "\n..." if len(content) > 300 else content
        return f"Registered prompt '{pid}' (v{version})\n{source_label}\n\n{preview}"

    @tool(parse_docstring=True)
    def read_current_prompt(prompt_id: str) -> str:
        """Read the current version of a registered prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            The full prompt content with version and source metadata.
        """
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not registered. Use register_prompt first."
        meta = json.loads((prompt_dir / "meta.json").read_text())
        content = (prompt_dir / "current.txt").read_text()
        source = meta.get("source_path") or "inline prompt"
        return f"Prompt '{prompt_id}' — v{meta['version']} — source: {source}\n\n{content}"

    @tool(parse_docstring=True)
    def list_prompts() -> str:
        """List all prompts registered in the eval system.

        Returns:
            A formatted list of prompt IDs with version and source path.
        """
        if not prompts_dir.exists():
            return "No prompts registered yet."
        entries = [
            d for d in prompts_dir.iterdir()
            if d.is_dir() and (d / "meta.json").exists()
        ]
        if not entries:
            return "No prompts registered yet."
        lines = ["Registered prompts:"]
        for d in entries:
            meta = json.loads((d / "meta.json").read_text())
            source = meta.get("source_path") or "inline prompt"
            lines.append(f"  {d.name}  v{meta['version']}  →  {source}")
        return "\n".join(lines)

    @tool(parse_docstring=True)
    def edit_prompt(prompt_id: str, edits: list[dict]) -> str:
        """Edit a prompt with targeted find/replace changes.

        The TUI will show a diff and ask for approval before writing. If the
        user denies, the edit is cancelled and nothing is written.

        Args:
            prompt_id: The prompt identifier.
            edits: List of edit dicts with "old" and "new" string keys. Each
                "old" string must appear exactly once in the current prompt.

        Returns:
            Confirmation with the new version number, or cancellation message.
        """
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not found."
        current = (prompt_dir / "current.txt").read_text()
        new_content, error = _build_prompt_from_edits(current, edits)
        if error:
            return error
        if not gate.consume(prompt_id):
            return f"Edit to '{prompt_id}' cancelled by user."
        return _apply_content(prompt_id, new_content)

    @tool(parse_docstring=True)
    def show_prompt_history(prompt_id: str) -> str:
        """Show version history for a prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            List of versions with size info and current marker.
        """
        try:
            prompt_dir = _prompt_dir(prompt_id)
        except ValueError as e:
            return f"Error: {e}"
        history_dir = prompt_dir / "history"
        if not history_dir.exists():
            return f"No history found for '{prompt_id}'."

        meta = json.loads((prompt_dir / "meta.json").read_text())
        versions = sorted(history_dir.glob("v*.txt"), key=lambda p: int(p.stem[1:]))
        lines = [f"Version history for '{prompt_id}':"]
        for v in versions:
            num = int(v.stem[1:])
            marker = " ← current" if num == meta["version"] else ""
            lines.append(f"  v{num}: {len(v.read_text())} chars{marker}")
        return "\n".join(lines)

    return [
        register_prompt,
        read_current_prompt,
        list_prompts,
        edit_prompt,
        show_prompt_history,
    ]
