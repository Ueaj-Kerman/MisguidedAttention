#!/usr/bin/env python3
"""
Async evaluation harness with progress tracking and resumable state.

Uses OpenRouter with sensible defaults. No config files needed.

Usage:
    python harness.py --dataset data/misguided_attention_v4.scr \
                      --models google/gemini-2.5-pro openai/gpt-4o \
                      --samples 3 --concurrency 20

Model syntax supports reasoning effort suffix:
    model_id:effort  where effort is low/medium/high/max

    Examples:
        openai/o3-mini:high
        openai/gpt-5:max
        anthropic/claude-sonnet-4:low
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
    """Result of a single evaluation task."""
    key: TaskKey
    prompt: str
    content: str | None = None
    thinking: str | None = None
    tokens_completion: int | None = None
    completion_tokens_details: dict | None = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error: str | None = None

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
        }


@dataclass
class Progress:
    """Tracks completed results with serialization support."""
    results: dict[tuple, TaskResult] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def add_result(self, result: TaskResult):
        self.results[result.key.to_tuple()] = result

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
                    "timestamp": result.timestamp,
                }
            g = grouped[group_key]
            while len(g["output"]) < result.key.sample_idx:
                g["output"].append(None)
                g["thinking"].append(None)
                g["tokens_completion"].append(None)
                g["completion_tokens_details"].append(None)
            if len(g["output"]) == result.key.sample_idx:
                g["output"].append(result.content)
                g["thinking"].append(result.thinking)
                g["tokens_completion"].append(result.tokens_completion)
                g["completion_tokens_details"].append(result.completion_tokens_details)
            else:
                g["output"][result.key.sample_idx] = result.content
                g["thinking"][result.key.sample_idx] = result.thinking
                g["tokens_completion"][result.key.sample_idx] = result.tokens_completion
                g["completion_tokens_details"][result.key.sample_idx] = result.completion_tokens_details
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
            timestamp = r.get("timestamp", datetime.now().isoformat())
            for i, output in enumerate(outputs):
                if output is not None:
                    key = TaskKey(prompt_id, llm, i)
                    result = TaskResult(
                        key=key, prompt=prompt, content=output,
                        thinking=thinkings[i] if i < len(thinkings) else None,
                        tokens_completion=tokens[i] if i < len(tokens) else None,
                        completion_tokens_details=details[i] if i < len(details) else None,
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


def parse_model_spec(model_spec: str) -> tuple[str, str | None]:
    """
    Parse model specification with optional reasoning effort suffix.

    Format: model_id[:effort]
    Effort levels: low, medium, high, max

    Examples:
        "openai/gpt-4o" -> ("openai/gpt-4o", None)
        "openai/o3-mini:high" -> ("openai/o3-mini", "high")
        "anthropic/claude-sonnet-4:max" -> ("anthropic/claude-sonnet-4", "max")
    """
    valid_efforts = {"low", "medium", "high", "max"}

    if ":" in model_spec:
        # Check if last part after : is an effort level
        parts = model_spec.rsplit(":", 1)
        if parts[1].lower() in valid_efforts:
            return parts[0], parts[1].lower()

    return model_spec, None


class LLMClient:
    """OpenRouter API client with sensible defaults."""

    def __init__(self, session: aiohttp.ClientSession, max_retries: int = 3, provider_sort: str | None = None):
        self.session = session
        self.max_retries = max_retries
        self.provider_sort = provider_sort
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")

    async def query(self, prompt: str, model: str, reasoning_effort: str | None = None) -> dict | None:
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

        # Reasoning effort control (for o-series, gpt-5, claude, etc.)
        if reasoning_effort:
            data["reasoning"] = {"effort": reasoning_effort}

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
                        if attempt < self.max_retries - 1:
                            wait = (2 ** attempt) * 2
                            await asyncio.sleep(wait)
                            continue
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

            except Exception as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2 ** attempt) * 2)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return None
        return None


async def run_single_task(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    key: TaskKey,
    prompt: str,
    model: str,
    reasoning_effort: str | None = None,
) -> TaskResult:
    """Execute a single evaluation task."""
    async with semaphore:
        try:
            response = await client.query(prompt, model, reasoning_effort)
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


async def run_harness(args: argparse.Namespace):
    """Main harness loop."""
    dataset = load_dataset(args.dataset)
    prompts = dataset.get("prompts", dataset.get("results", []))

    progress = load_progress(args.output)
    print(f"Loaded {len(progress.results)} existing results from {args.output}")

    # Build task queue
    tasks_to_run: list[tuple[TaskKey, str, str, str | None]] = []
    for model_spec in args.models:
        # Parse model:effort syntax
        model, reasoning_effort = parse_model_spec(model_spec)
        # Use model ID as display name (last part after /, plus effort if specified)
        model_name = model.split("/")[-1] if "/" in model else model
        if reasoning_effort:
            model_name = f"{model_name}:{reasoning_effort}"
        for prompt_data in prompts[:args.limit] if args.limit > 0 else prompts:
            prompt_id = prompt_data.get("prompt_id", prompt_data.get("id", "unknown"))
            prompt_text = prompt_data.get("prompt", prompt_data.get("text", ""))
            for sample_idx in range(args.samples):
                key = TaskKey(prompt_id, model_name, sample_idx)
                if not progress.has_result(key):
                    tasks_to_run.append((key, prompt_text, model, reasoning_effort))

    total_tasks = len(tasks_to_run)
    print(f"Total tasks to run: {total_tasks}")
    if not tasks_to_run:
        print("All tasks already completed!")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = LLMClient(session, max_retries=args.max_retries, provider_sort=args.provider_sort)

        pending: set[asyncio.Task] = set()
        task_iter = iter(tasks_to_run)

        def feed_tasks(current_pending: set):
            nonlocal task_iter
            while len(current_pending) < args.concurrency:
                try:
                    key, prompt_text, model, reasoning_effort = next(task_iter)
                    task = asyncio.create_task(
                        run_single_task(client, semaphore, key, prompt_text, model, reasoning_effort)
                    )
                    task.task_key = key  # type: ignore
                    current_pending.add(task)
                except StopIteration:
                    break

        feed_tasks(pending)
        pbar = tqdm(total=total_tasks, desc="Evaluating", unit="task")

        if args.debug:
            tqdm.write(f"Initial pending: {len(pending)}")

        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            batch_completed = 0
            while True:
                for task in done:
                    result = task.result()
                    progress.add_result(result)
                    batch_completed += 1
                    if args.debug:
                        status = "OK" if result.error is None else f"ERR: {result.error}"
                        tqdm.write(f"{result.key.prompt_id} @ {result.key.llm}#{result.key.sample_idx}: {status}")

                if pending:
                    newly_done, pending = await asyncio.wait(pending, timeout=0, return_when=asyncio.FIRST_COMPLETED)
                    if newly_done:
                        done = newly_done
                        continue
                break

            pbar.update(batch_completed)
            save_json(progress.to_dict(), args.output)
            feed_tasks(pending)

            if args.debug:
                tqdm.write(f"After feed: {len(pending)} pending")

        pbar.close()

    print(f"\nCompleted {total_tasks} tasks")
    save_json(progress.to_dict(), args.output)

    grouped_path = args.output.replace(".json", "_grouped.json")
    save_json(progress.to_grouped_results(), grouped_path)
    print(f"Saved grouped format to {grouped_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Async LLM evaluation harness using OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python harness.py --dataset data/prompts.scr --models google/gemini-2.5-pro
  python harness.py --dataset data/prompts.json --models openai/gpt-4o anthropic/claude-sonnet-4 --samples 3
  python harness.py --dataset data/prompts.scr --models openai/o3-mini:high openai/o3-mini:low
  python harness.py --dataset data/prompts.scr --models meta-llama/llama-3.1-70b-instruct --provider-sort throughput
        """,
    )
    parser.add_argument("--dataset", required=True, help="Path to dataset (.json or .scr)")
    parser.add_argument("--models", required=True, nargs="+", help="Model IDs (e.g., google/gemini-2.5-pro)")
    parser.add_argument("--output", default="progress.json", help="Output file path")
    parser.add_argument("--samples", type=int, default=1, help="Samples per prompt-model pair")
    parser.add_argument("--limit", type=int, default=0, help="Limit prompts (0=all)")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retries per request")
    parser.add_argument("--provider-sort", choices=["price", "throughput", "latency"],
                        help="Sort providers (throughput = optimized providers first)")
    parser.add_argument("--debug", action="store_true", help="Debug output")

    args = parser.parse_args()
    asyncio.run(run_harness(args))


if __name__ == "__main__":
    main()
