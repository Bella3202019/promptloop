from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from .system_prompt import SYSTEM_PROMPT
from .tools import (
    make_prompt_tools,
    make_test_case_tools,
    make_runner_tools,
    make_report_tools,
)


def create_eval_agent(
    project_dir: Path,
    model_name: str = "anthropic:claude-sonnet-4-6",
    checkpointer: Any | None = None,
):
    """Create the prompt eval agent scoped to a project directory.

    Initialises the .evals/ directory structure and registers all custom tools.
    The agent has full filesystem read access to the project dir via deepagents
    built-in tools (read_file, glob, grep, ls, etc.).

    Args:
        project_dir: Root directory of the project being evaluated.
        model_name: The orchestrating agent's model (not the model under test).

    Returns:
        A compiled LangGraph agent ready to invoke.
    """
    # Ensure .evals/ dirs exist
    for subdir in ["prompts", "test_cases", "eval_configs", "results"]:
        (project_dir / ".evals" / subdir).mkdir(parents=True, exist_ok=True)

    tools = [
        *make_prompt_tools(project_dir),
        *make_test_case_tools(project_dir),
        *make_runner_tools(project_dir),
        *make_report_tools(project_dir),
    ]

    model = init_chat_model(model_name, temperature=0, streaming=True)

    backend = FilesystemBackend(
        root_dir=str(project_dir),
        virtual_mode=False,
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT.format(project_dir=project_dir),
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
        backend=backend,
    )
