#!/usr/bin/env python3
"""
Async evaluation harness with progress tracking and resumable state.

Key features:
- Progress tracked per (prompt_id, llm, sample_idx) tuple
- State serialized to JSON, reloadable for resume
- Configurable concurrency with proper async handling
- Drains ALL completed requests before serializing to avoid race conditions
"""

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import aiohttp
from tqdm import tqdm

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


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
    """
    Tracks all completed results and allows serialization/deserialization.
    Results are stored per unique TaskKey.
    """
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
        """Group results by (prompt_id, llm) for compatibility with old format."""
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
            # Ensure arrays are the right length (fill gaps with None)
            while len(g["output"]) < result.key.sample_idx:
                g["output"].append(None)
                g["thinking"].append(None)
                g["tokens_completion"].append(None)
                g["completion_tokens_details"].append(None)
            # Append or set at index
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
        """Load from the grouped format (old harness output)."""
        progress = cls()
        for r in data.get("results", []):
            prompt_id = r["prompt_id"]
            llm = r["llm"]
            prompt = r.get("prompt", "")
            outputs = r.get("output", [])
            thinkings = r.get("thinking", [])
            tokens = r.get("tokens_completion", [])
            details = r.get("completion_tokens_details", [])
            timestamp = r.get("timestamp", datetime.now().isoformat())
            for i, output in enumerate(outputs):
                if output is not None:  # Only count completed samples
                    key = TaskKey(prompt_id, llm, i)
                    result = TaskResult(
                        key=key,
                        prompt=prompt,
                        content=output,
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
    """XOR descramble data with key."""
    return bytes(data[i] ^ key[i % len(key)] for i in range(len(data)))


def load_dataset(path: str) -> dict:
    """Load dataset, auto-detecting scrambled (.scr) files."""
    if path.endswith(".scr"):
        with open(path, "rb") as f:
            scrambled = f.read()
        decrypted = descramble(scrambled)
        return json.loads(decrypted.decode("utf-8"))
    return load_json(path)


def save_json(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_progress(path: str) -> Progress:
    """Load progress from file, auto-detecting format."""
    if not os.path.exists(path):
        return Progress()
    try:
        data = load_json(path)
        # Detect format: new format has sample_idx in results
        if data.get("results") and len(data["results"]) > 0:
            first = data["results"][0]
            if "sample_idx" in first:
                return Progress.from_dict(data)
            else:
                return Progress.from_grouped_format(data)
        return Progress()
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Could not load progress file: {e}")
        return Progress()


def extract_thinking_from_response(text: str) -> tuple[str, str | None]:
    """Extract content within <think></think> tags."""
    if not text or "<think>" not in text:
        return text, None
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    segments = pattern.findall(text)
    thinking = "\n".join(segments) if segments else None
    cleaned = pattern.sub("", text).strip()
    return cleaned, thinking


class LLMClient:
    """Handles API calls to various LLM providers."""

    def __init__(self, session: aiohttp.ClientSession, max_retries: int = 3):
        self.session = session
        self.max_retries = max_retries
        self._load_api_keys()

    def _load_api_keys(self):
        self.openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.gemini_key = os.environ.get("GEMINI_API_KEY")
        self.nous_key = os.environ.get("NOUS_API_KEY")
        if not self.openrouter_key:
            self.openrouter_key = self.openai_key

    async def query(
        self,
        prompt: str,
        llm_config: dict,
        temperature_override: float = -1,
        extract_thinking: bool = False,
    ) -> dict | None:
        """Query an LLM and return response dict with content, thinking, tokens."""
        model = llm_config["model"].lower()
        prompt_text = f"Please answer the following question: {prompt}\nAnswer:"

        def prepare_messages(text: str, config: dict) -> list:
            msgs = []
            if config.get("system_prompt"):
                msgs.append({"role": "system", "content": config["system_prompt"]})
            msgs.append({"role": "user", "content": text})
            return msgs

        # Direct Gemini API (only for non-OpenRouter models, i.e. no provider prefix)
        if ("gemini" in model and "/" not in llm_config["model"]):
            return await self._query_gemini(prompt_text, llm_config, temperature_override)

        # Determine endpoint
        if "hermes" in model:
            return await self._query_nous(prompt_text, llm_config, temperature_override, prepare_messages)
        elif "d33pseek" in model:
            return await self._query_deepseek(prompt_text, llm_config, temperature_override, prepare_messages)
        else:
            return await self._query_openrouter(prompt_text, llm_config, temperature_override, prepare_messages)

    async def _query_gemini(self, prompt: str, config: dict, temp_override: float) -> dict | None:
        if not HAS_GENAI:
            print("google-generativeai not installed")
            return None
        if not self.gemini_key:
            print("GEMINI_API_KEY not set")
            return None

        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel(config["model"])
        await asyncio.sleep(6)  # Rate limiting

        try:
            temp = temp_override if temp_override > 0 else config.get("temperature", 1.0)
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=temp),
            )
            content, thinking = extract_thinking_from_response(response.text)
            return {"content": content, "thinking": thinking}
        except Exception as e:
            print(f"Gemini error: {e}")
            return None

    async def _query_openrouter(self, prompt: str, config: dict, temp_override: float, prepare_messages) -> dict | None:
        if not self.openrouter_key:
            print("OPENROUTER_API_KEY not set")
            return None

        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "HTTP-Referer": "",
            "X-Title": "MA_Eval",
        }
        temp = temp_override if temp_override > 0 else config.get("temperature", 1.0)
        data = {
            "model": config["model"],
            "messages": prepare_messages(prompt, config),
            "temperature": temp,
            "max_tokens": config.get("max_tokens", 4000),
            "top_p": config.get("top_p", 1),
            "min_p": config.get("min_p", 0),
            "top_k": config.get("top_k", 0),
            "frequency_penalty": config.get("frequency_penalty", 0),
            "presence_penalty": config.get("presence_penalty", 0),
            "usage": {"include": True},
        }
        if config.get("provider"):
            data["provider"] = {"order": [config["provider"]]}
        if "reasoning" in config:
            data["reasoning"] = config["reasoning"]
        else:
            data["include_reasoning"] = True

        return await self._make_request(
            "https://openrouter.ai/api/v1/chat/completions",
            headers,
            data,
            include_usage=True,
        )

    async def _query_deepseek(self, prompt: str, config: dict, temp_override: float, prepare_messages) -> dict | None:
        if not self.deepseek_key:
            print("DEEPSEEK_API_KEY not set")
            return None

        headers = {
            "Authorization": f"Bearer {self.deepseek_key}",
            "Content-Type": "application/json",
        }
        temp = temp_override if temp_override > 0 else config.get("temperature", 1.0)
        data = {
            "model": config["model"],
            "messages": prepare_messages(prompt, config),
            "temperature": temp,
        }
        return await self._make_request("https://api.deepseek.com/v1/chat/completions", headers, data)

    async def _query_nous(self, prompt: str, config: dict, temp_override: float, prepare_messages) -> dict | None:
        if not self.nous_key:
            print("NOUS_API_KEY not set")
            return None

        headers = {
            "Authorization": f"Bearer {self.nous_key}",
            "Content-Type": "application/json",
        }
        temp = temp_override if temp_override > 0 else config.get("temperature", 1.0)
        data = {
            "model": config["model"],
            "messages": prepare_messages(prompt, config),
            "temperature": temp,
            "max_tokens": config.get("max_tokens", 4000),
            "usage": True,
        }
        return await self._make_request(
            "https://inference-api.nousresearch.com/v1/chat/completions",
            headers,
            data,
            include_usage=True,
        )

    async def _make_request(
        self,
        url: str,
        headers: dict,
        data: dict,
        include_usage: bool = False,
    ) -> dict | None:
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, headers=headers, json=data) as resp:
                    resp.raise_for_status()
                    result = await resp.json()

                    if "error" in result:
                        if attempt < self.max_retries - 1:
                            wait = (2**attempt) * 2
                            print(f"API error: {result['error']}. Retry in {wait}s...")
                            await asyncio.sleep(wait)
                            continue
                        return None

                    if "choices" not in result or not result["choices"]:
                        print(f"Unexpected response: {result}")
                        return None

                    msg = result["choices"][0]["message"]
                    content = msg.get("content", "")
                    thinking = None

                    if content and "<think>" in content:
                        content, thinking = extract_thinking_from_response(content)
                    else:
                        thinking = msg.get("reasoning") or msg.get("reasoning_content")
                        if not thinking and "thinking" in result["choices"][0]:
                            thinking = result["choices"][0]["thinking"]

                    response = {"content": content, "thinking": thinking}

                    if include_usage and result.get("usage"):
                        usage = result["usage"]
                        response["tokens_completion"] = usage.get("completion_tokens")
                        response["completion_tokens_details"] = usage.get("completion_tokens_details")

                    return response

            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait = (2**attempt) * 2
                    print(f"Request error ({type(e).__name__}): {e}. Retry in {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    print(f"Failed after {self.max_retries} attempts: {e}")
                    return None
        return None


async def run_single_task(
    client: LLMClient,
    semaphore: asyncio.Semaphore,
    key: TaskKey,
    prompt: str,
    llm_config: dict,
    args: argparse.Namespace,
) -> TaskResult:
    """Execute a single evaluation task."""
    async with semaphore:
        try:
            response = await client.query(
                prompt,
                llm_config,
                temperature_override=args.temp,
                extract_thinking=args.think,
            )
            if response is None:
                return TaskResult(
                    key=key,
                    prompt=prompt,
                    error="Failed to get response",
                )
            return TaskResult(
                key=key,
                prompt=prompt,
                content=response.get("content"),
                thinking=response.get("thinking"),
                tokens_completion=response.get("tokens_completion"),
                completion_tokens_details=response.get("completion_tokens_details"),
            )
        except Exception as e:
            return TaskResult(
                key=key,
                prompt=prompt,
                error=str(e),
            )


async def run_harness(args: argparse.Namespace):
    """Main harness loop with progress tracking."""
    # Load config and dataset
    config = load_json(args.config)
    dataset = load_dataset(args.dataset)
    prompts = dataset.get("prompts", dataset.get("results", []))

    # Load existing progress
    progress = load_progress(args.output)
    print(f"Loaded {len(progress.results)} existing results from {args.output}")

    # Build task queue
    tasks_to_run: list[tuple[TaskKey, str, dict]] = []
    for llm_config in config["llms"]:
        llm_name = llm_config["name"]
        for prompt_data in prompts[:args.limit] if args.limit > 0 else prompts:
            prompt_id = prompt_data.get("prompt_id", prompt_data.get("id", "unknown"))
            prompt_text = prompt_data.get("prompt", prompt_data.get("text", ""))
            for sample_idx in range(args.samples):
                key = TaskKey(prompt_id, llm_name, sample_idx)
                if not progress.has_result(key):
                    tasks_to_run.append((key, prompt_text, llm_config))

    total_tasks = len(tasks_to_run)
    print(f"Total tasks to run: {total_tasks}")
    if not tasks_to_run:
        print("All tasks already completed!")
        return

    # Create semaphore and session
    semaphore = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = LLMClient(session, max_retries=args.max_retries)

        # Track pending tasks
        pending: set[asyncio.Task] = set()
        task_iter = iter(tasks_to_run)

        def feed_tasks(current_pending: set):
            """Add tasks up to concurrency limit."""
            nonlocal task_iter
            while len(current_pending) < args.concurrency:
                try:
                    key, prompt_text, llm_config = next(task_iter)
                    task = asyncio.create_task(
                        run_single_task(client, semaphore, key, prompt_text, llm_config, args)
                    )
                    task.task_key = key  # type: ignore
                    current_pending.add(task)
                except StopIteration:
                    break

        # Initial feed
        feed_tasks(pending)

        # Progress bar
        pbar = tqdm(total=total_tasks, desc="Evaluating", unit="task")

        if args.debug:
            tqdm.write(f"Initial pending: {len(pending)}")

        while pending:
            # Wait for at least one task to complete
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

            # Drain ALL completed tasks before serializing
            batch_completed = 0
            while True:
                for task in done:
                    result = task.result()
                    progress.add_result(result)
                    batch_completed += 1
                    if args.debug:
                        status = "OK" if result.error is None else f"ERR: {result.error}"
                        tqdm.write(f"{result.key.prompt_id} @ {result.key.llm}#{result.key.sample_idx}: {status}")

                # Check if more completed while we processed
                if pending:
                    newly_done, pending = await asyncio.wait(pending, timeout=0, return_when=asyncio.FIRST_COMPLETED)
                    if newly_done:
                        done = newly_done
                        continue
                break

            # Update progress bar and serialize
            pbar.update(batch_completed)
            save_json(progress.to_dict(), args.output)

            # Feed more tasks - pass current pending set
            feed_tasks(pending)
            if args.debug:
                tqdm.write(f"After feed: {len(pending)} pending")

        pbar.close()

    # Final save in grouped format for compatibility
    print(f"\nCompleted {total_tasks} tasks")
    save_json(progress.to_dict(), args.output)

    # Also save in grouped format
    grouped_path = args.output.replace(".json", "_grouped.json")
    save_json(progress.to_grouped_results(), grouped_path)
    print(f"Saved grouped format to {grouped_path}")


def main():
    parser = argparse.ArgumentParser(description="Async LLM evaluation harness with progress tracking")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSON file")
    parser.add_argument("--output", default="progress.json", help="Path to progress/output JSON file")
    parser.add_argument("--config", required=True, help="Path to LLM configuration JSON file")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per prompt-LLM pair")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of prompts (0=all)")
    parser.add_argument("--temp", type=float, default=-1, help="Override temperature (-1=use config)")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum concurrent requests")
    parser.add_argument("--max-retries", type=int, default=8, help="Max retries per request")
    parser.add_argument("--think", action="store_true", help="Extract <think> tags from responses")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()
    asyncio.run(run_harness(args))


if __name__ == "__main__":
    main()
