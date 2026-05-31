# Promptloop

An interactive CLI agent for the full prompt-eval loop: create test cases, run evals, generate reports, and approve prompt diffs without leaving your terminal.

## The Prompt Eval Loop

Agent harnesses are getting better, but prompts still shape what they do. promptloop turns a prompt and eval intent into a repeatable loop:

<img src="https://github.com/Bella3202019/promptloop/releases/download/v0.1-media/prompt_flow.png" alt="Prompt eval loop" style="max-width: 640px; width: 100%; height: auto;">

It saves the methodology, test cases, reports, prompt history, and chat checkpoints under `.evals/` in the target project.

```text
.evals/
  prompts/        # registered prompts + version history
  test_cases/     # per-prompt test suites
  eval_configs/   # methodology (metrics, models, judges)
  results/        # eval runs and reports
  chat.db         # SQLite checkpoint of conversation threads
```

Example metrics:

- `latency`: response time
- `json_schema`: validates structured output
- `fuzzy_match`: compares text similarity
- `llm_judge`: scores output with a judge prompt

## Install and Run

```bash
git clone <this repo>
cd promptloop
uv sync
uv run promptloop --project-dir /path/to/your/project
```

You'll get an interactive chat. Try things like:

- *"Evaluate the prompt at `src/prompts/summarize.txt`"*
- *"Add three more test cases for edge cases"*
- *"Re-run with `openai:gpt-4o-mini` and compare to the last run"*
- *"Propose a fix for the failing JSON schema cases"*

## Commands

| Command | Description |
| --- | --- |
| `/help` | Show help |
| `/clear` | Start a new conversation thread |
| `/threads` | List saved threads |
| `/thread <id>` | Switch to a thread in-session |
| `/quit` | Exit |

Resume past sessions with `promptloop --thread <id>`. Press **Esc** to interrupt a streaming response.

## Quick Demo

**Stage 1: Register a prompt** — point promptloop at a prompt file and it registers it with version tracking:

![register prompt](https://github.com/Bella3202019/promptloop/releases/download/v0.1-media/01-register-prompt.gif)

**Stage 2: Add test cases** — the agent proposes test cases based on your prompt and intent; you pick what to keep:

![add test cases](https://github.com/Bella3202019/promptloop/releases/download/v0.1-media/02-add-test-cases.gif)

**Stage 3: Run the eval** — see pass/fail results per test case with metrics and latency:

![run eval](https://github.com/Bella3202019/promptloop/releases/download/v0.1-media/03-run-eval.gif)

**Stage 4: Propose and update the prompt** — when cases fail, ask for a fix. promptloop reads the report, proposes a diff, and saves the updated prompt as a new version on approval:

![propose and update prompt](https://github.com/Bella3202019/promptloop/releases/download/v0.1-media/04-propose-and-update-prompt.gif)

**Stage 5: Next iteration** — re-run the eval on the new prompt version and keep iterating:

![next iteration](https://github.com/Bella3202019/promptloop/releases/download/v0.1-media/05-next-iteration.gif)

Every run persists the full loop — versioned prompts, test cases, eval configs, and reports — so nothing is lost between sessions:

![promptloop show result](https://github.com/Bella3202019/promptloop/releases/download/v0.1-media/promptloop.-.show.result.gif)

## How It Works

The agent has a small set of typed tools on top of deepagents' filesystem access:

- `register_prompt`, `propose_prompt_changes`, `apply_prompt_changes`, `show_prompt_history`
- `add_test_case`, `infer_json_schema`, `save_eval_config`
- `run_eval`, `list_eval_runs`
- `generate_report`, `read_report`, `compare_runs`

For more detail on the agent runtime behind this project, see [The Harness Behind Deep Agent](docs/The_Harness_Behind_Deep_Agent.md).



Early / experimental. Feedback and issues welcome.

## What's Next

This is a starting point. A few directions I'm thinking about:

**1. Interface** — the CLI works, but the prompt loop deserves a less friction interface. Extending toward a richer chat UI or editor integration so the loop feels more natural to run.

**2. Less human in the loop** — the current flow still relies on you to drive each stage. 

> *"It would be extremely cool to be able to write one or two lines of prompt in my harness, and have a light model iterate with me a few times writing/proposing requirements, guidelines and explanations, refining the prompt until it's ready to be sent to the actual LLM."* ---- [HN commenters](https://news.ycombinator.com/item?id=48325073):

Ideally it would be a lightweight thing(could be tool, skill, cli, chat interface or a snippet) that co-authors the prompt with you, proposes requirements, flags gaps, and tightens the spec before it ever hits your production model.

Built on LangChain [deepagents](https://github.com/langchain-ai/deepagents).
