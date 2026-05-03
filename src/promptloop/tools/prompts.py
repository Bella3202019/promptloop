import json
import difflib
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool


def make_prompt_tools(project_dir: Path):
    prompts_dir = project_dir / ".evals" / "prompts"

    def _next_version(history_dir: Path) -> int:
        existing = list(history_dir.glob("v*.txt"))
        if not existing:
            return 1
        return max(int(p.stem[1:]) for p in existing) + 1

    def _save_prompt(
        content: str,
        prompt_id: str,
        source_path: str | None,
        source_type: str,
    ) -> tuple[int, Path]:
        prompt_dir = prompts_dir / prompt_id
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

    @tool(parse_docstring=True)
    def register_prompt(source: str, prompt_id: str | None = None) -> str:
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
        p = Path(source)
        if not p.is_absolute():
            p = project_dir / p

        if p.exists() and p.is_file():
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

        version, _ = _save_prompt(content, pid, source_path, source_type)
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
        prompt_dir = prompts_dir / prompt_id
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
    def propose_prompt_changes(prompt_id: str, new_content: str) -> str:
        """Propose changes to a prompt and show a unified diff.

        Saves the proposed content temporarily. After showing the diff,
        ask the user to confirm with 'yes' before calling apply_prompt_changes.

        Args:
            prompt_id: The prompt identifier.
            new_content: The full proposed new prompt content.

        Returns:
            A unified diff of current vs proposed content.
        """
        prompt_dir = prompts_dir / prompt_id
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not found."

        current = (prompt_dir / "current.txt").read_text()
        diff_lines = list(difflib.unified_diff(
            current.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"{prompt_id} (current)",
            tofile=f"{prompt_id} (proposed)",
        ))

        if not diff_lines:
            return "No changes detected between current and proposed content."

        (prompt_dir / "proposed.txt").write_text(new_content)
        diff_str = "".join(diff_lines)
        return f"Proposed changes to '{prompt_id}':\n\n```diff\n{diff_str}\n```\n\nShall I apply this? (yes / no)"

    @tool(parse_docstring=True)
    def apply_prompt_changes(prompt_id: str) -> str:
        """Apply previously proposed prompt changes after explicit user approval.

        Only call this after the user has said 'yes' to the proposed diff.
        Writes the new content to the source file and saves a history entry.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            Confirmation with the new version number.
        """
        prompt_dir = prompts_dir / prompt_id
        proposed_file = prompt_dir / "proposed.txt"
        if not proposed_file.exists():
            return "No pending changes found. Use propose_prompt_changes first."

        new_content = proposed_file.read_text()
        history_dir = prompt_dir / "history"
        meta = json.loads((prompt_dir / "meta.json").read_text())

        new_version = meta["version"] + 1
        (history_dir / f"v{new_version}.txt").write_text(new_content)
        (prompt_dir / "current.txt").write_text(new_content)
        proposed_file.unlink()

        meta["version"] = new_version
        meta["updated_at"] = datetime.now().isoformat()
        (prompt_dir / "meta.json").write_text(json.dumps(meta, indent=2))

        source_path_value = meta.get("source_path")
        if not source_path_value:
            source_note = "No source file to update for inline prompt."
        else:
            source_path = Path(source_path_value)
            if not source_path.is_absolute():
                source_path = project_dir / source_path

            if source_path.exists():
                source_path.write_text(new_content)
                source_note = f"Source file updated at {source_path}."
            else:
                source_note = f"Warning: source file {source_path} not found; only .evals copy updated."

        return f"Applied. Prompt '{prompt_id}' is now v{new_version}. {source_note}"

    @tool(parse_docstring=True)
    def show_prompt_history(prompt_id: str) -> str:
        """Show version history for a prompt.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            List of versions with size info and current marker.
        """
        prompt_dir = prompts_dir / prompt_id
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
        propose_prompt_changes,
        apply_prompt_changes,
        show_prompt_history,
    ]
