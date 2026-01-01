#!/usr/bin/env python3
"""
Async evaluation harness with progress tracking, grading, and resumable state.

Uses OpenRouter with sensible defaults. No config files needed.
Concurrently collects responses AND grades them using an LLM judge.

Usage:
    python harness.py --dataset data/misguided_attention_v4.scr \
                      --models google/gemini-2.5-pro openai/gpt-4o \
                      --samples 3 --concurrency 20

Model syntax supports reasoning control suffix:
    model_id:effort   - effort level (low/medium/high/max) for OpenAI models
    model_id:NUMBER   - max thinking tokens for Gemini/Anthropic models

    Examples:
        openai/o3-mini:high
        google/gemini-3-pro-preview:16000
        anthropic/claude-sonnet-4:10000
"""

import argparse
import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime

import aiohttp
from tqdm import tqdm

# Default grading model (fast and cheap)
DEFAULT_GRADER_MODEL = "openai/gpt-4.1-nano"


@dataclass
class TaskKey:
    """Unique identifier for a single evaluation task."""
    prompt_id: str
    llm: str
    sample_idx: int

    def to_tuple(self) -> tuple:
        return (self.prompt_id, self.llm, self.sample_idx)

    def to_dict(self) -> dict:
        return {"prompt_id": self.prompt_id, "llm": self.llm, "sample_idx": self.sample_idx}


@dataclass
class TaskResult:
    """Result of a single evaluation task with optional grading."""
    key: TaskKey
    prompt: str
    content: str | None = None
    thinking: str | None = None
    tokens_completion: int | None = None
    completion_tokens_details: dict | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str | None = None
    # Grading fields
    criteria_scores: list[float] | None = None  # Score per criterion (0.0 or 1.0)
    criteria_explanations: list[str] | None = None  # Explanation per criterion
    overall_score: float | None = None  # Weighted average

    def is_graded(self) -> bool:
        return self.overall_score is not None

    def to_dict(self) -> dict:
        return {
            **self.key.to_dict(),
            "prompt": self.prompt,
            "content": self.content,
            "thinking": self.thinking,
            "tokens_completion": self.tokens_completion,
            "completion_tokens_details": self.completion_tokens_details,
            "timestamp": self.timestamp,
            "error": self.error,
            "criteria_scores": self.criteria_scores,
            "criteria_explanations": self.criteria_explanations,
            "overall_score": self.overall_score,
        }


@dataclass
class Progress:
    """Tracks completed results with serialization support."""
    results: dict[tuple, TaskResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_result(self, result: TaskResult):
        self.results[result.key.to_tuple()] = result

    def get_result(self, key: TaskKey) -> TaskResult | None:
        return self.results.get(key.to_tuple())

    def has_result(self, key: TaskKey) -> bool:
        return key.to_tuple() in self.results

    def to_dict(self) -> dict:
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results.values()],
        }

    def to_grouped_results(self) -> dict:
        """Group results by (prompt_id, llm) for compatibility."""
        grouped: dict[tuple, dict] = {}
        for result in self.results.values():
            group_key = (result.key.prompt_id, result.key.llm)
            if group_key not in grouped:
                grouped[group_key] = {
                    "prompt_id": result.key.prompt_id,
                    "prompt": result.prompt,
                    "llm": result.key.llm,
                    "output": [],
                    "thinking": [],
                    "tokens_completion": [],
                    "completion_tokens_details": [],
                    "criteria_scores": [],
                    "overall_score": [],
                    "timestamp": result.timestamp,
                }
            g = grouped[group_key]
            while len(g["output"]) < result.key.sample_idx:
                g["output"].append(None)
                g["thinking"].append(None)
                g["tokens_completion"].append(None)
                g["completion_tokens_details"].append(None)
                g["criteria_scores"].append(None)
                g["overall_score"].append(None)
            if len(g["output"]) == result.key.sample_idx:
                g["output"].append(result.content)
                g["thinking"].append(result.thinking)
                g["tokens_completion"].append(result.tokens_completion)
                g["completion_tokens_details"].append(result.completion_tokens_details)
                g["criteria_scores"].append(result.criteria_scores)
                g["overall_score"].append(result.overall_score)
            else:
                g["output"][result.key.sample_idx] = result.content
                g["thinking"][result.key.sample_idx] = result.thinking
                g["tokens_completion"][result.key.sample_idx] = result.tokens_completion
                g["completion_tokens_details"][result.key.sample_idx] = result.completion_tokens_details
                g["criteria_scores"][result.key.sample_idx] = result.criteria_scores
                g["overall_score"][result.key.sample_idx] = result.overall_score
            g["timestamp"] = max(g["timestamp"], result.timestamp)
        return {"results": list(grouped.values())}

    @classmethod
    def from_dict(cls, data: dict) -> "Progress":
        progress = cls()
        progress.metadata = data.get("metadata", {})
        for r in data.get("results", []):
            key = TaskKey(r["prompt_id"], r["llm"], r["sample_idx"])
            result = TaskResult(
                key=key,
                prompt=r.get("prompt", ""),
                content=r.get("content"),
                thinking=r.get("thinking"),
                tokens_completion=r.get("tokens_completion"),
                completion_tokens_details=r.get("completion_tokens_details"),
                timestamp=r.get("timestamp", datetime.now().isoformat()),
                error=r.get("error"),
                criteria_scores=r.get("criteria_scores"),
                criteria_explanations=r.get("criteria_explanations"),
                overall_score=r.get("overall_score"),
            )
            progress.add_result(result)
        return progress

    @classmethod
    def from_grouped_format(cls, data: dict) -> "Progress":
        """Load from grouped format."""
        progress = cls()
        for r in data.get("results", []):
            prompt_id, llm = r["prompt_id"], r["llm"]
            prompt = r.get("prompt", "")
            outputs = r.get("output", [])
            thinkings = r.get("thinking", [])
            tokens = r.get("tokens_completion", [])
            details = r.get("completion_tokens_details", [])
            crit_scores = r.get("criteria_scores", [])
            overall_scores = r.get("overall_score", [])
            timestamp = r.get("timestamp", datetime.now().isoformat())
            for i, output in enumerate(outputs):
                if output is not None:
                    key = TaskKey(prompt_id, llm, i)
                    result = TaskResult(
                        key=key, prompt=prompt, content=output,
                        thinking=thinkings[i] if i < len(thinkings) else None,
                        tokens_completion=tokens[i] if i < len(tokens) else None,
                        completion_tokens_details=details[i] if i < len(details) else None,
                        criteria_scores=crit_scores[i] if i < len(crit_scores) else None,
                        overall_score=overall_scores[i] if i < len(overall_scores) else None,
                        timestamp=timestamp,
                    )
                    progress.add_result(result)
        return progress


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def descramble(data: bytes, key: bytes = b"MisguidedAttention2025") -> bytes:
    return bytes(data[i] ^ key[i % len(key)] for i in range(len(data)))


def load_dataset(path: str) -> dict:
    if path.endswith(".scr"):
        with open(path, "rb") as f:
            scrambled = f.read()
        return json.loads(descramble(scrambled).decode("utf-8"))
    return load_json(path)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_progress(path: str) -> Progress:
    if not os.path.exists(path):
        return Progress()
    try:
        data = load_json(path)
        if data.get("results") and len(data["results"]) > 0:
            first = data["results"][0]
            if "sample_idx" in first:
                return Progress.from_dict(data)
            return Progress.from_grouped_format(data)
        return Progress()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load progress file: {e}")
        return Progress()


def extract_thinking(text: str) -> tuple[str, str | None]:
    """Extract content within <think></think> tags."""
    if not text or "<think>" not in text:
        return text, None
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    segments = pattern.findall(text)
    thinking = "\n".join(segments) if segments else None
    return pattern.sub("", text).strip(), thinking


def parse_model_spec(model_spec: str) -> tuple[str, dict | None]:
    """
    Parse model specification with optional reasoning control suffix.

    Format: model_id[:effort_or_tokens_or_false]
    - Effort levels: none, minimal, low, medium, high, xhigh (for OpenAI models)
    - Token count: integer (for Gemini/Anthropic models)
    - false: Disable thinking (for models with thinking on by default)

    Examples:
        "openai/gpt-4o" -> ("openai/gpt-4o", None)
        "openai/o3-mini:high" -> ("openai/o3-mini", {"effort": "high"})
        "google/gemini-3-pro:16000" -> ("google/gemini-3-pro", {"max_tokens": 16000})
        "z-ai/glm-4.7:false" -> ("z-ai/glm-4.7", {"disabled": True})
    """
    valid_efforts = {"none", "minimal", "low", "medium", "high", "xhigh"}

    if ":" in model_spec:
        parts = model_spec.rsplit(":", 1)
        suffix = parts[1].lower()

        # Check if it's an effort level (check first, before disabled)
        if suffix in valid_efforts:
            return parts[0], {"effort": suffix}

        # Check if it's a token count
        if suffix.isdigit():
            return parts[0], {"max_tokens": int(suffix)}

        # Check if it's disabling thinking
        if suffix in {"false", "off", "disabled"}:
            return parts[0], {"disabled": True}

    return model_spec, None


class LLMClient:
    """OpenRouter API client with sensible defaults."""

    def __init__(self, session: aiohttp.ClientSession, max_retries: int = 3, provider_sort: str | None = None):
        self.session = session
        self.max_retries = max_retries
        self.provider_sort = provider_sort
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            print("ERROR: OPENROUTER_API_KEY environment variable not set")
            print("Set it with: export OPENROUTER_API_KEY='your-key'")
            raise SystemExit(1)

    async def query(self, prompt: str, model: str, reasoning_config: dict | None = None) -> dict | None:
        """Query OpenRouter with model defaults."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "",
            "X-Title": "MA_Eval",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": f"Please answer the following question: {prompt}\nAnswer:"}],
            "include_reasoning": True,
        }

        # Reasoning control: effort (OpenAI), max_tokens (Gemini/Anthropic), or disabled
        if reasoning_config:
            if reasoning_config.get("disabled"):
                # Disable thinking/reasoning
                data["reasoning"] = {"enabled": False}
            else:
                data["reasoning"] = reasoning_config

        # Provider sorting (throughput = optimized providers first)
        if self.provider_sort:
            data["provider"] = {"sort": self.provider_sort}

        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as resp:
                    result = await resp.json()

                    if "error" in result:
                        err = result["error"]
                        err_msg = err.get("message", err) if isinstance(err, dict) else err
                        if attempt < self.max_retries - 1:
                            wait = (2 ** attempt) * 2
                            tqdm.write(f"API error: {err_msg}. Retry {attempt+1}/{self.max_retries} in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        tqdm.write(f"API error (final): {err_msg}")
                        return None

                    if "choices" not in result or not result["choices"]:
                        return None

                    msg = result["choices"][0]["message"]
                    content = msg.get("content", "")
                    thinking = None

                    # Extract thinking from various sources
                    if content and "<think>" in content:
                        content, thinking = extract_thinking(content)
                    else:
                        thinking = msg.get("reasoning") or msg.get("reasoning_content")
                        if not thinking and "thinking" in result["choices"][0]:
                            thinking = result["choices"][0]["thinking"]

                    response = {"content": content, "thinking": thinking}

                    if result.get("usage"):
                        usage = result["usage"]
                        response["tokens_completion"] = usage.get("completion_tokens")
                        response["completion_tokens_details"] = usage.get("completion_tokens_details")

                    return response

            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    tqdm.write(f"Timeout. Retry {attempt+1}/{self.max_retries}...")
                    await asyncio.sleep((2 ** attempt) * 2)
                else:
                    tqdm.write(f"Timeout after {self.max_retries} attempts")
                    return None
            except Exception as e:
                if attempt < self.max_retries - 1:
                    tqdm.write(f"Error: {e}. Retry {attempt+1}/{self.max_retries}...")
                    await asyncio.sleep((2 ** attempt) * 2)
                else:
                    tqdm.write(f"Failed after {self.max_retries} attempts: {e}")
                    return None
        return None

    async def grade(self, prompt: str, response: str, criteria: list[str], model: str) -> dict | None:
        """Grade a response against criteria using LLM judge."""
        grading_prompt = f"""You are evaluating an AI model's response against specific criteria.

ORIGINAL QUESTION:
{prompt}

MODEL'S RESPONSE:
{response}

CRITERIA TO EVALUATE:
{chr(10).join(f"{i+1}. {c}" for i, c in enumerate(criteria))}

For each criterion, determine if the response meets it (1) or not (0).
Respond with ONLY valid JSON in this exact format:
{{"scores": [<list of 0 or 1 for each criterion>], "explanations": [<brief explanation for each>]}}"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "",
            "X-Title": "MA_Eval_Grader",
        }

        data = {
            "model": model,
            "messages": [{"role": "user", "content": grading_prompt}],
            "temperature": 0,
        }

        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                ) as resp:
                    result = await resp.json()

                    if "error" in result:
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep((2 ** attempt) * 2)
                            continue
                        return None

                    if "choices" not in result or not result["choices"]:
                        return None

                    content = result["choices"][0]["message"].get("content", "")
                    # Extract JSON from response (handle markdown code blocks)
                    json_match = re.search(r'\{[^{}]*"scores"[^{}]*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    # Try parsing entire content as JSON
                    return json.loads(content)

            except (json.JSONDecodeError, asyncio.TimeoutError, Exception) as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2 ** attempt) * 2)
                else:
                    tqdm.write(f"Grading failed: {e}")
                    return None
        return None


async def run_single_task(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    key: TaskKey,
    prompt: str,
    model: str,
    reasoning_config: dict | None = None,
) -> TaskResult:
    """Execute a single evaluation task."""
    async with semaphore:
        try:
            response = await client.query(prompt, model, reasoning_config)
            if response is None:
                return TaskResult(key=key, prompt=prompt, error="Failed to get response")
            return TaskResult(
                key=key,
                prompt=prompt,
                content=response.get("content"),
                thinking=response.get("thinking"),
                tokens_completion=response.get("tokens_completion"),
                completion_tokens_details=response.get("completion_tokens_details"),
            )
        except Exception as e:
            return TaskResult(key=key, prompt=prompt, error=str(e))


async def grade_result(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    result: TaskResult,
    criteria: list[str],
    weights: list[float],
    grader_model: str,
) -> TaskResult:
    """Grade a result against criteria."""
    async with semaphore:
        if result.error or not result.content:
            # Can't grade errors or empty responses
            result.criteria_scores = [0.0] * len(criteria)
            result.criteria_explanations = ["No response to grade"] * len(criteria)
            result.overall_score = 0.0
            return result

        grade_result = await client.grade(result.prompt, result.content, criteria, grader_model)

        if grade_result and "scores" in grade_result:
            scores = [float(s) for s in grade_result["scores"]]
            # Pad or truncate to match criteria length
            while len(scores) < len(criteria):
                scores.append(0.0)
            scores = scores[:len(criteria)]

            explanations = grade_result.get("explanations", [])
            while len(explanations) < len(criteria):
                explanations.append("")
            explanations = explanations[:len(criteria)]

            result.criteria_scores = scores
            result.criteria_explanations = explanations

            # Calculate weighted score
            total_weight = sum(weights)
            if total_weight > 0:
                result.overall_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
            else:
                result.overall_score = sum(scores) / len(scores) if scores else 0.0
        else:
            # Grading failed - score as 0
            result.criteria_scores = [0.0] * len(criteria)
            result.criteria_explanations = ["Grading failed"] * len(criteria)
            result.overall_score = 0.0

        return result


class ScoreTracker:
    """Track running score statistics."""
    def __init__(self):
        self.total_score = 0.0
        self.count = 0

    def add(self, score: float | None):
        if score is not None:
            self.total_score += score
            self.count += 1

    @property
    def average(self) -> float:
        return self.total_score / self.count if self.count > 0 else 0.0


async def run_harness(args: argparse.Namespace):
    """Main harness loop with concurrent grading."""
    dataset = load_dataset(args.dataset)
    prompts = dataset.get("prompts", dataset.get("results", []))

    # Build prompt lookup for criteria
    prompt_lookup: dict[str, dict] = {}
    for p in prompts:
        pid = p.get("prompt_id", p.get("id", "unknown"))
        prompt_lookup[pid] = p

    progress = load_progress(args.output)
    print(f"Loaded {len(progress.results)} existing results from {args.output}")

    # Find ungraded results that need grading
    ungraded: list[TaskResult] = []
    for result in progress.results.values():
        if result.content and not result.is_graded() and not result.error:
            ungraded.append(result)

    if ungraded:
        print(f"Found {len(ungraded)} ungraded responses to grade")

    # Build task queue for new responses
    tasks_to_run: list[tuple[TaskKey, str, str, dict | None]] = []
    for model_spec in args.models:
        model, reasoning_config = parse_model_spec(model_spec)
        model_name = model.split("/")[-1] if "/" in model else model
        if reasoning_config:
            if reasoning_config.get("disabled"):
                cfg_str = "off"
            else:
                cfg_str = reasoning_config.get("effort") or str(reasoning_config.get("max_tokens", ""))
            model_name = f"{model_name}:{cfg_str}"
        for prompt_data in prompts[:args.limit] if args.limit > 0 else prompts:
            prompt_id = prompt_data.get("prompt_id", prompt_data.get("id", "unknown"))
            prompt_text = prompt_data.get("prompt", prompt_data.get("text", ""))
            for sample_idx in range(args.samples):
                key = TaskKey(prompt_id, model_name, sample_idx)
                if not progress.has_result(key):
                    tasks_to_run.append((key, prompt_text, model, reasoning_config))

    total_query_tasks = len(tasks_to_run)
    total_grade_tasks = len(ungraded) + total_query_tasks  # Each query also needs grading

    print(f"Tasks: {total_query_tasks} queries + {len(ungraded)} pending grades")

    if total_query_tasks == 0 and len(ungraded) == 0:
        print("All tasks already completed!")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = LLMClient(session, max_retries=args.max_retries, provider_sort=args.provider_sort)

        # Track pending tasks by type
        query_pending: set[asyncio.Task] = set()
        grade_pending: set[asyncio.Task] = set()
        task_iter = iter(tasks_to_run)

        score_tracker = ScoreTracker()

        # Count already graded for score calculation
        for result in progress.results.values():
            if result.is_graded():
                score_tracker.add(result.overall_score)

        def get_pbar_desc() -> str:
            return f"Eval [score: {score_tracker.average:.1%}]"

        def feed_query_tasks(current_pending: set):
            nonlocal task_iter
            while len(current_pending) + len(grade_pending) < args.concurrency:
                try:
                    key, prompt_text, model, reasoning_config = next(task_iter)
                    task = asyncio.create_task(
                        run_single_task(client, semaphore, key, prompt_text, model, reasoning_config)
                    )
                    task.task_type = "query"  # type: ignore
                    task.task_key = key  # type: ignore
                    current_pending.add(task)
                except StopIteration:
                    break

        def feed_grade_tasks(results_to_grade: list[TaskResult]):
            for result in results_to_grade:
                if len(query_pending) + len(grade_pending) >= args.concurrency:
                    break
                prompt_data = prompt_lookup.get(result.key.prompt_id, {})
                criteria = prompt_data.get("criteria", [])
                weights = prompt_data.get("weight", [1.0] * len(criteria))
                if not criteria:
                    # No criteria to grade against
                    result.overall_score = None
                    continue

                task = asyncio.create_task(
                    grade_result(client, semaphore, result, criteria, weights, args.grader)
                )
                task.task_type = "grade"  # type: ignore
                task.task_key = result.key  # type: ignore
                grade_pending.add(task)

        # Start grading ungraded results first
        feed_grade_tasks(ungraded)
        ungraded_iter = iter([])  # Already fed

        # Start query tasks
        feed_query_tasks(query_pending)

        total_tasks = total_query_tasks + total_grade_tasks
        completed = 0
        pbar = tqdm(total=total_tasks, desc=get_pbar_desc(), unit="task")

        if args.debug:
            tqdm.write(f"Initial: {len(query_pending)} queries, {len(grade_pending)} grades pending")

        while query_pending or grade_pending:
            all_pending = query_pending | grade_pending
            done, _ = await asyncio.wait(all_pending, return_when=asyncio.FIRST_COMPLETED)

            results_to_grade: list[TaskResult] = []
            batch_completed = 0

            # Process all immediately available completed tasks
            while True:
                for task in done:
                    task_type = getattr(task, "task_type", "unknown")

                    if task_type == "query":
                        query_pending.discard(task)
                        result = task.result()
                        progress.add_result(result)
                        batch_completed += 1

                        if args.debug:
                            status = "OK" if result.error is None else f"ERR: {result.error}"
                            tqdm.write(f"Q: {result.key.prompt_id} @ {result.key.llm}#{result.key.sample_idx}: {status}")

                        # Queue for grading if successful
                        if result.content and not result.error:
                            results_to_grade.append(result)

                    elif task_type == "grade":
                        grade_pending.discard(task)
                        result = task.result()
                        progress.add_result(result)  # Update with grade
                        batch_completed += 1
                        score_tracker.add(result.overall_score)

                        if args.debug:
                            score_str = f"{result.overall_score:.0%}" if result.overall_score is not None else "N/A"
                            tqdm.write(f"G: {result.key.prompt_id} @ {result.key.llm}#{result.key.sample_idx}: {score_str}")

                # Check for more immediately done tasks
                all_pending = query_pending | grade_pending
                if all_pending:
                    newly_done, _ = await asyncio.wait(all_pending, timeout=0, return_when=asyncio.FIRST_COMPLETED)
                    if newly_done:
                        done = newly_done
                        continue
                break

            # Update progress bar
            pbar.update(batch_completed)
            pbar.set_description(get_pbar_desc())

            # Save progress
            save_json(progress.to_dict(), args.output)

            # Feed new grade tasks from completed queries
            feed_grade_tasks(results_to_grade)

            # Feed new query tasks
            feed_query_tasks(query_pending)

            if args.debug:
                tqdm.write(f"Pending: {len(query_pending)} queries, {len(grade_pending)} grades")

        pbar.close()

    print(f"\nCompleted. Final score: {score_tracker.average:.1%} ({score_tracker.count} graded)")

    # Save final results
    results_file = getattr(args, 'results_file', args.output.replace("_progress.json", "_results.json"))
    save_json(progress.to_grouped_results(), results_file)
    print(f"Saved results to {results_file}")

    # Delete progress file on successful completion
    if os.path.exists(args.output) and args.output != results_file:
        os.remove(args.output)
        print(f"Cleaned up {args.output}")


def export_partial_results(progress_file: str, results_file: str):
    """Export progress file as results, scoring ungraded items as 0."""
    if not os.path.exists(progress_file):
        print(f"Error: {progress_file} not found")
        return False

    progress = load_progress(progress_file)
    print(f"Loaded {len(progress.results)} results from {progress_file}")

    # Score ungraded results as 0
    ungraded_count = 0
    for result in progress.results.values():
        if not result.is_graded():
            result.overall_score = 0.0
            result.criteria_scores = [0.0]
            result.criteria_explanations = ["Ungraded - scored as 0"]
            ungraded_count += 1

    if ungraded_count:
        print(f"Scored {ungraded_count} ungraded results as 0")

    # Calculate final score
    scores = [r.overall_score for r in progress.results.values() if r.overall_score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"Final score: {avg_score:.1%} ({len(scores)} results)")

    # Save as results file
    save_json(progress.to_grouped_results(), results_file)
    print(f"Saved: {results_file}")
    return True


def get_model_prefix(model_spec: str) -> str:
    """Generate filename prefix from model spec (e.g., 'openai/o3-mini:high' -> 'o3-mini_high')."""
    model, reasoning_config = parse_model_spec(model_spec)
    # Get just the model name (after last /)
    name = model.split("/")[-1] if "/" in model else model
    # Add reasoning suffix if present
    if reasoning_config:
        if reasoning_config.get("disabled"):
            suffix = "off"
        else:
            suffix = reasoning_config.get("effort") or str(reasoning_config.get("max_tokens", ""))
        name = f"{name}_{suffix}"
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Async LLM evaluation harness with grading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harness.py --models google/gemini-2.5-pro
  python harness.py --long --models openai/gpt-4o anthropic/claude-sonnet-4 --samples 3
  python harness.py --models openai/o3-mini:high openai/o3-mini:low
  python harness.py --models meta-llama/llama-3.1-70b-instruct --provider-sort throughput

Output files:
  ./{model}_progress.json      - Intermediate progress (deleted on completion)
  results/{model}_results.json - Final grouped results
        """,
    )
    parser.add_argument("--dataset", default="data/misguided_attention_v4.scr", help="Path to dataset (.json or .scr)")
    parser.add_argument("--long", action="store_true", help="Use long dataset (data/misguided_attention_v4_long.scr)")
    parser.add_argument("--models", nargs="+", help="Model IDs (e.g., google/gemini-2.5-pro)")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    parser.add_argument("--samples", type=int, default=1, help="Samples per prompt-model pair")
    parser.add_argument("--limit", type=int, default=0, help="Limit prompts (0=all)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=4, help="Max retries per request")
    parser.add_argument("--provider-sort", choices=["price", "throughput", "latency"],
                        help="Sort providers (throughput = optimized providers first)")
    parser.add_argument("--grader", default=DEFAULT_GRADER_MODEL, help="Model for grading responses")
    parser.add_argument("--debug", action="store_true", help="Debug output")
    parser.add_argument("--export-partial", metavar="PROGRESS_FILE",
                        help="Export a progress file as results, scoring ungraded as 0")

    args = parser.parse_args()

    # Handle export-partial mode
    if args.export_partial:
        progress_file = args.export_partial
        # Results go in output_dir
        basename = os.path.basename(progress_file).replace("_progress.json", "_results.json")
        results_file = os.path.join(args.output_dir, basename)
        os.makedirs(args.output_dir, exist_ok=True)
        export_partial_results(progress_file, results_file)
        return

    if args.long:
        args.dataset = "data/misguided_attention_v4_long.scr"

    # Require --models for normal operation
    if not args.models:
        parser.error("--models is required")

    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # Run each model separately with its own output files
    for model_spec in args.models:
        prefix = get_model_prefix(model_spec)
        progress_file = f"{prefix}_progress.json"  # Progress in current dir
        results_file = os.path.join(args.output_dir, f"{prefix}_results.json")

        # Create a copy of args for this model
        model_args = argparse.Namespace(**vars(args))
        model_args.models = [model_spec]
        model_args.output = progress_file
        model_args.results_file = results_file

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_spec}")
        print(f"Progress: {progress_file}")
        print(f"Results: {results_file}")
        print(f"{'='*60}\n")

        asyncio.run(run_harness(model_args))


if __name__ == "__main__":
    main()
