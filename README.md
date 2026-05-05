# Promptloop

An interactive CLI agent that runs the **full prompt-eval loop** — *create test cases → run evals → generate reports → improve the prompt* — without leaving your terminal.

Point it at a project, tell it which prompt to evaluate, and it will help you design test cases, pick metrics, run evals across models, and propose prompt edits as diffs you approve.

Built on [deepagents](https://github.com/langchain-ai/deepagents) + LangGraph.

## Why

Prompt iteration is usually a loop of *edit → eyeball outputs → tweak*. promptloop turns that loop into a structured workflow:

- **Locate prompts in real code**, not isolated playgrounds.
- **Save eval methodology once** per prompt — no re-deciding metrics each run.
- **Compare across models and versions** with persisted reports.
- **Apply prompt edits via diffs** with version history.

## Install

```bash
git clone <this repo>
cd promptloop
uv sync
```

Set API keys (in `.env.local`, `.env`, or your shell). Environment files are loaded from the target project directory first:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

## Usage

Run against any project that contains the prompts you want to evaluate:

```bash
uv run promptloop --project-dir /path/to/your/project
```

You'll get an interactive chat. Try things like:

- *"Evaluate the prompt at `src/prompts/summarize.txt`"*
- *"Add three more test cases for edge cases"*
- *"Re-run with `openai:gpt-4o-mini` and compare to the last run"*
- *"Propose a fix for the failing JSON schema cases"*

### Commands

| Command | Description |
| --- | --- |
| `/help` | Show help |
| `/clear` | Start a new conversation thread |
| `/threads` | List saved threads |
| `/thread <id>` | Switch to a thread in-session |
| `/quit` | Exit |

Resume past sessions with `promptloop --thread <id>`. Press **Esc** to interrupt a streaming response.

## How it works

The agent has a small set of typed tools on top of deepagents' filesystem access:

- `register_prompt`, `propose_prompt_changes`, `apply_prompt_changes`, `show_prompt_history`
- `add_test_case`, `infer_json_schema`, `save_eval_config`
- `run_eval`, `list_eval_runs`
- `generate_report`, `read_report`, `compare_runs`

Everything is persisted under `.evals/` in the target project:

```
.evals/
  prompts/        # registered prompts + version history
  test_cases/     # per-prompt test suites
  eval_configs/   # methodology (metrics, models, judges)
  results/        # eval runs and reports
  chat.db         # SQLite checkpoint of conversation threads
```

## Metrics

promptloop can help create an eval methodology for each prompt, including metrics such as:

| Metric | What it measures |
| --- | --- |
| `latency` | Response time (always recorded) |
| `json_schema` | Output validates against an inferred schema |
| `fuzzy_match` | Text similarity vs expected output (configurable threshold) |
| `llm_judge` | LLM scores the output 0–10 with a judge prompt you approve |

## Status

Early / experimental. Feedback and issues welcome.
