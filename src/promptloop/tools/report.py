import json
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model


REPORT_SYSTEM_PROMPT = (
    "You are an expert prompt engineer. Generate clear, actionable eval reports in markdown. "
    "Focus on root causes of failures and give specific, concrete improvement suggestions "
    "that reference exact parts of the prompt. Be concise."
)


def make_report_tools(project_dir: Path):
    results_dir = project_dir / ".evals" / "results"
    prompts_dir = project_dir / ".evals" / "prompts"

    @tool(parse_docstring=True)
    async def generate_report(
        prompt_id: str,
        run_id: str,
        model: str = "anthropic:claude-sonnet-4-6",
    ) -> str:
        """Generate a markdown evaluation report for a completed run.

        Analyzes raw results with an LLM to produce a structured report with
        per-test breakdown, failure analysis, and specific prompt improvement recommendations.
        Saves the report to .evals/results/{prompt_id}/{run_id}/report.md.

        Args:
            prompt_id: The prompt identifier.
            run_id: The run ID returned by run_eval.
            model: Model to use for report generation.

        Returns:
            The full report content (also saved to disk).
        """
        run_dir = results_dir / prompt_id / run_id
        raw_file = run_dir / "raw.json"
        if not raw_file.exists():
            return f"Error: no results found for run '{run_id}'. Run run_eval first."

        run_data = json.loads(raw_file.read_text())
        results = run_data["results"]

        passed = sum(
            1 for r in results
            if not r.get("error") and all(m.get("passed") is not False for m in r.get("metrics", []))
        )
        total = len(results)
        latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0
        pass_rate = passed / total if total else 0

        prompt_content = ""
        prompt_file = prompts_dir / prompt_id / "current.txt"
        if prompt_file.exists():
            prompt_content = prompt_file.read_text()

        analysis_input = f"""Prompt being evaluated (v{run_data.get('prompt_version', '?')}):
---
{prompt_content[:2000]}
---

Eval run: {run_id}
Models tested: {", ".join(run_data.get("models", []))}
Pass rate: {pass_rate:.0%} ({passed}/{total})
Avg latency: {avg_lat:.0f}ms

Raw results:
{json.dumps(results, indent=2)[:5000]}

Generate a markdown report with these sections:
1. **Summary** — one-paragraph overview with key stats table
2. **Per-Test Results** — for each test case: input snippet, actual output snippet, pass/fail per metric, scores
3. **Failure Analysis** — group failures by root cause; quote specific parts of the prompt that likely caused each failure
4. **Latency Analysis** — if latency metric is present, note p50/p95 and any outliers
5. **Recommendations** — numbered list of specific prompt edits (quote the problematic text, show what to change it to)

Be specific. If a JSON schema check failed, show the actual vs expected structure.
If an LLM judge gave a low score, explain what aspect of quality was lacking.
"""

        lm = init_chat_model(model, temperature=0)
        response = await lm.ainvoke([
            SystemMessage(content=REPORT_SYSTEM_PROMPT),
            HumanMessage(content=analysis_input),
        ])

        report = response.content if isinstance(response.content, str) else str(response.content)

        # Prepend metadata header
        header = f"# Prompt Eval Report: {prompt_id}\n\n**Run:** {run_id}  \n**Date:** {run_data.get('started_at', '')[:19]}  \n**Models:** {', '.join(run_data.get('models', []))}  \n**Pass rate:** {pass_rate:.0%} ({passed}/{total})  \n**Avg latency:** {avg_lat:.0f}ms\n\n---\n\n"
        full_report = header + report

        (run_dir / "report.md").write_text(full_report)
        return full_report

    @tool(parse_docstring=True)
    def read_report(prompt_id: str, run_id: str) -> str:
        """Read a previously generated evaluation report.

        Args:
            prompt_id: The prompt identifier.
            run_id: The run ID.

        Returns:
            The report content, or a message if no report exists yet.
        """
        report_file = results_dir / prompt_id / run_id / "report.md"
        if not report_file.exists():
            return f"No report found for run '{run_id}'. Use generate_report first."
        return report_file.read_text()

    @tool(parse_docstring=True)
    def compare_runs(prompt_id: str, run_id_a: str, run_id_b: str) -> str:
        """Compare two eval runs for the same prompt.

        Useful for seeing whether a prompt change improved results.

        Args:
            prompt_id: The prompt identifier.
            run_id_a: First run ID (e.g. the baseline).
            run_id_b: Second run ID (e.g. after a prompt change).

        Returns:
            A side-by-side comparison of pass rates, latency, and per-test changes.
        """
        def _load(run_id: str) -> dict:
            f = results_dir / prompt_id / run_id / "raw.json"
            if not f.exists():
                return {}
            return json.loads(f.read_text())

        a = _load(run_id_a)
        b = _load(run_id_b)
        if not a:
            return f"Run '{run_id_a}' not found."
        if not b:
            return f"Run '{run_id_b}' not found."

        def _stats(data: dict) -> dict:
            results = data["results"]
            passed = sum(
                1 for r in results
                if not r.get("error") and all(m.get("passed") is not False for m in r.get("metrics", []))
            )
            latencies = [r["latency_ms"] for r in results if r.get("latency_ms")]
            return {
                "passed": passed,
                "total": len(results),
                "avg_lat": sum(latencies) / len(latencies) if latencies else 0,
                "version": data.get("prompt_version", "?"),
                "by_test": {r["test_id"]: r for r in results},
            }

        sa = _stats(a)
        sb = _stats(b)

        lines = [
            f"Comparison: {prompt_id}",
            f"",
            f"{'Metric':<20} {'Run A':>12} {'Run B':>12} {'Delta':>10}",
            f"{'-'*55}",
            f"{'Pass rate':<20} {sa['passed']}/{sa['total']:>10} {sb['passed']}/{sb['total']:>10}",
            f"{'Avg latency (ms)':<20} {sa['avg_lat']:>12.0f} {sb['avg_lat']:>12.0f} {sb['avg_lat'] - sa['avg_lat']:>+10.0f}",
            f"{'Prompt version':<20} {sa['version']:>12} {sb['version']:>12}",
            f"",
            f"Per-test changes:",
        ]

        all_ids = sorted(set(list(sa["by_test"].keys()) + list(sb["by_test"].keys())))
        for tid in all_ids:
            ra = sa["by_test"].get(tid)
            rb = sb["by_test"].get(tid)

            def _pass(r):
                if r is None:
                    return "—"
                if r.get("error"):
                    return "ERR"
                return "✓" if all(m.get("passed") is not False for m in r.get("metrics", [])) else "✗"

            lines.append(f"  [{tid}]  A: {_pass(ra)}  →  B: {_pass(rb)}")

        return "\n".join(lines)

    return [generate_report, read_report, compare_runs]
