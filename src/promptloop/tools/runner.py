import json
import asyncio
import uuid
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
import jsonschema

DEFAULT_MAX_CONCURRENCY = 4


def make_runner_tools(project_dir: Path):
    prompts_dir = project_dir / ".evals" / "prompts"
    test_cases_dir = project_dir / ".evals" / "test_cases"
    eval_configs_dir = project_dir / ".evals" / "eval_configs"
    results_dir = project_dir / ".evals" / "results"

    def _get_model(model_cache: dict[str, Any], model_name: str) -> Any:
        if model_name not in model_cache:
            model_cache[model_name] = init_chat_model(model_name, temperature=0)
        return model_cache[model_name]

    async def _call_model(
        model_cache: dict[str, Any],
        model_name: str,
        system_prompt: str,
        user_input: str,
    ) -> tuple[str, float]:
        model = _get_model(model_cache, model_name)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
        start = time.monotonic()
        response = await model.ainvoke(messages)
        latency_ms = (time.monotonic() - start) * 1000

        content = response.content
        if isinstance(content, str):
            return content, latency_ms
        if isinstance(content, list):
            return "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ), latency_ms
        return str(content), latency_ms

    async def _run_llm_judge(
        model_cache: dict[str, Any],
        judge_prompt_template: str,
        actual_output: str,
        user_input: str,
        expected_output: Optional[str],
        judge_model: str,
    ) -> dict:
        prompt = judge_prompt_template.format(
            input=user_input,
            output=actual_output,
            expected=expected_output or "N/A",
        )
        model = _get_model(model_cache, judge_model)
        response = await model.ainvoke([HumanMessage(content=prompt)])
        raw = response.content if isinstance(response.content, str) else str(response.content)

        # Parse a numeric score out of the response (0-10 or 0.0-1.0).
        score = None
        m = re.search(r'\b(10|[0-9](?:\.\d+)?)(?:/10)?\b', raw)
        if m:
            raw_score = float(m.group(1))
            score = raw_score if raw_score <= 1 else raw_score / 10.0
            score = min(max(score, 0.0), 1.0)

        return {
            "type": "llm_judge",
            "passed": (score >= 0.7) if score is not None else None,
            "score": round(score, 3) if score is not None else None,
            "detail": raw[:600],
        }

    def _run_sync_metric(metric: dict, actual_output: str, expected_output: Optional[str]) -> dict:
        t = metric.get("type")
        result: dict = {"type": t, "passed": None, "score": None, "detail": ""}

        if t == "latency":
            result["detail"] = "measured separately"
            result["passed"] = True

        elif t == "json_schema":
            try:
                parsed = json.loads(actual_output)
                schema = metric.get("schema")
                if schema:
                    jsonschema.validate(parsed, schema)
                result["passed"] = True
                result["detail"] = "valid JSON" + (" matching schema" if schema else "")
            except json.JSONDecodeError as e:
                result["passed"] = False
                result["detail"] = f"invalid JSON: {e}"
            except jsonschema.ValidationError as e:
                result["passed"] = False
                result["detail"] = f"schema mismatch: {e.message}"

        elif t == "fuzzy_match":
            if expected_output is None:
                result["detail"] = "no expected_output provided"
            else:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, actual_output.strip(), expected_output.strip()).ratio()
                threshold = metric.get("threshold", 0.8)
                result["score"] = round(ratio, 3)
                result["passed"] = ratio >= threshold
                result["detail"] = f"similarity {ratio:.1%} (threshold {threshold:.0%})"

        elif t == "llm_judge":
            result["detail"] = "__pending__"

        return result

    async def _run_test_case(
        tc: dict,
        prompt_content: str,
        model_name: str,
        model_cache: dict[str, Any],
    ) -> dict:
        result: dict = {
            "test_id": tc["id"],
            "model": model_name,
            "input": tc["input"],
            "expected_output": tc.get("expected_output"),
            "actual_output": None,
            "latency_ms": None,
            "metrics": [],
            "error": None,
        }
        try:
            actual_output, latency_ms = await _call_model(
                model_cache,
                model_name,
                prompt_content,
                tc["input"],
            )
            result["actual_output"] = actual_output
            result["latency_ms"] = round(latency_ms, 1)

            default_judge_model = "anthropic:claude-sonnet-4-6"
            metrics = tc.get("metrics") or [{"type": "latency"}]

            async_tasks = []
            sync_results = []

            for metric in metrics:
                if metric.get("type") == "llm_judge":
                    async_tasks.append(_run_llm_judge(
                        model_cache,
                        metric.get("judge_prompt", "Rate the output quality 0-10.\nInput: {input}\nOutput: {output}"),
                        actual_output,
                        tc["input"],
                        tc.get("expected_output"),
                        metric.get("judge_model", default_judge_model),
                    ))
                else:
                    sync_results.append(_run_sync_metric(metric, actual_output, tc.get("expected_output")))

            judge_results = await asyncio.gather(*async_tasks) if async_tasks else []

            # Merge latency into the latency metric result
            for r in sync_results:
                if r["type"] == "latency":
                    r["latency_ms"] = result["latency_ms"]
                    r["detail"] = f"{result['latency_ms']:.0f}ms"

            result["metrics"] = sync_results + list(judge_results)

        except Exception as e:
            result["error"] = str(e)

        return result

    @tool(parse_docstring=True)
    async def run_eval(
        prompt_id: str,
        models: Optional[list[str]] = None,
        test_case_ids: Optional[list[str]] = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> str:
        """Run evaluation for a prompt against all (or selected) test cases.

        Runs each test case × model combination with bounded concurrency.
        Measures latency, validates JSON schema, computes fuzzy match, and
        calls LLM-as-judge as configured per test case.

        Args:
            prompt_id: The prompt identifier.
            models: Models to test, e.g. ["anthropic:claude-sonnet-4-6", "openai:gpt-4o"].
                    Falls back to eval config defaults, then claude-sonnet-4-6.
            test_case_ids: Run only these test case IDs. Runs all if omitted.
            max_concurrency: Maximum model/judge calls to run at once.

        Returns:
            Run ID and a pass/fail summary. Use generate_report() for the full analysis.
        """
        prompt_dir = prompts_dir / prompt_id
        if not prompt_dir.exists():
            return f"Error: prompt '{prompt_id}' not registered."
        prompt_content = (prompt_dir / "current.txt").read_text()

        tc_dir = test_cases_dir / prompt_id
        all_cases = list(tc_dir.glob("*.json")) if tc_dir.exists() else []
        if not all_cases:
            return f"Error: no test cases found for '{prompt_id}'. Add some with add_test_case."

        test_cases = []
        for f in all_cases:
            tc = json.loads(f.read_text())
            if test_case_ids is None or tc["id"] in test_case_ids:
                test_cases.append(tc)

        if not test_cases:
            return "No test cases matched the given IDs."

        # Resolve models and default metrics.
        default_metrics = [{"type": "latency"}]
        if models is None:
            config_file = eval_configs_dir / f"{prompt_id}.json"
            if config_file.exists():
                config_data = json.loads(config_file.read_text())
                resolved_models: list[str] = config_data.get("default_models", ["anthropic:claude-sonnet-4-6"])
                default_metrics = config_data.get("default_metrics") or default_metrics
            else:
                resolved_models = ["anthropic:claude-sonnet-4-6"]
        else:
            resolved_models = models
            config_file = eval_configs_dir / f"{prompt_id}.json"
            if config_file.exists():
                config_data = json.loads(config_file.read_text())
                default_metrics = config_data.get("default_metrics") or default_metrics

        for tc in test_cases:
            if not tc.get("metrics"):
                tc["metrics"] = default_metrics

        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
        run_dir = results_dir / prompt_id / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        prompt_version = json.loads((prompt_dir / "meta.json").read_text()).get("version")
        started_at = datetime.now().isoformat()
        try:
            requested_concurrency = int(max_concurrency or DEFAULT_MAX_CONCURRENCY)
        except (TypeError, ValueError):
            requested_concurrency = DEFAULT_MAX_CONCURRENCY

        model_cache: dict[str, Any] = {}
        concurrency = max(1, min(requested_concurrency, len(test_cases) * len(resolved_models)))
        semaphore = asyncio.Semaphore(concurrency)
        results: list[dict] = []

        async def _run_limited(tc: dict, model: str) -> dict:
            async with semaphore:
                return await _run_test_case(tc, prompt_content, model, model_cache)

        tasks = [
            asyncio.create_task(_run_limited(tc, model))
            for tc in test_cases
            for model in resolved_models
        ]

        for completed in asyncio.as_completed(tasks):
            result = await completed
            results.append(result)
            partial_data = {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "models": resolved_models,
                "prompt_version": prompt_version,
                "started_at": started_at,
                "status": "running",
                "max_concurrency": concurrency,
                "completed": len(results),
                "total": len(tasks),
                "results": results,
            }
            (run_dir / "raw.json").write_text(json.dumps(partial_data, indent=2))

        run_data = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "models": resolved_models,
            "prompt_version": prompt_version,
            "started_at": started_at,
            "status": "complete",
            "max_concurrency": concurrency,
            "results": [r for r in results],
        }
        (run_dir / "raw.json").write_text(json.dumps(run_data, indent=2))

        # Summary
        passed = sum(
            1 for r in results
            if not r.get("error") and all(m.get("passed") is not False for m in r.get("metrics", []))
        )
        failed = len(results) - passed
        latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        lines = [
            f"Run complete — ID: {run_id}",
            f"Results: {passed} passed / {failed} failed / {len(results)} total",
            f"Avg latency: {avg_lat:.0f}ms",
            f"Max concurrency: {concurrency}",
            "",
        ]
        for r in results:
            status = "✓" if not r.get("error") and all(m.get("passed") is not False for m in r.get("metrics", [])) else "✗"
            metric_summary = " | ".join(
                f"{m['type']}: {m.get('detail', '') or m.get('score', '')}"
                for m in r.get("metrics", [])
            )
            lines.append(f"  {status} [{r['test_id']}] {r['model']}  {metric_summary}")
            if r.get("error"):
                lines.append(f"      error: {r['error']}")

        lines.append(f"\nUse generate_report('{prompt_id}', '{run_id}') for full analysis.")
        return "\n".join(lines)

    @tool(parse_docstring=True)
    def list_eval_runs(prompt_id: str) -> str:
        """List all eval runs for a prompt, most recent first.

        Args:
            prompt_id: The prompt identifier.

        Returns:
            A list of run IDs with pass/fail counts and model info.
        """
        runs_dir = results_dir / prompt_id
        if not runs_dir.exists():
            return f"No eval runs found for '{prompt_id}'."

        runs = sorted(runs_dir.iterdir(), reverse=True)
        if not runs:
            return f"No eval runs found for '{prompt_id}'."

        lines = [f"Eval runs for '{prompt_id}':"]
        for run_dir in runs[:15]:
            raw_file = run_dir / "raw.json"
            if not raw_file.exists():
                continue
            data = json.loads(raw_file.read_text())
            passed = sum(
                1 for r in data["results"]
                if not r.get("error") and all(m.get("passed") is not False for m in r.get("metrics", []))
            )
            total = len(data["results"])
            has_report = (run_dir / "report.md").exists()
            models = ", ".join(data.get("models", []))
            v = data.get("prompt_version", "?")
            lines.append(
                f"  {run_dir.name}  v{v}  {passed}/{total} passed  [{models}]"
                + (" [report]" if has_report else "")
            )
        return "\n".join(lines)

    return [run_eval, list_eval_runs]
