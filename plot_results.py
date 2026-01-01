#!/usr/bin/env python3
"""
Plot overall scores with variance for all models in results directory.
Outputs both light and dark theme versions with pastel colors.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Pastel color palettes by provider (multiple shades)
COLOR_PALETTES = {
    "openai": ["#98fb98", "#77dd77", "#5bc85b", "#3cb371", "#2e8b57"],      # Greens
    "anthropic": ["#ffd699", "#ffb347", "#ff9933", "#e68a00", "#cc7a00"],   # Oranges
    "google": ["#b8d4e8", "#89cff0", "#5eb3e4", "#3399cc", "#267399"],      # Blues
    "deepseek": ["#b8b8e8", "#9999d6", "#7a7ac4", "#5c5cb2", "#4b4ba0"],    # Indigo
    "xai": ["#606060", "#4a4a4a", "#3a3a3a", "#2a2a2a", "#1a1a1a"],         # Black/dark gray
    "kimi": ["#d8b8e8", "#c99bd8", "#ba7ec8", "#ab61b8", "#9c44a8"],        # Purple
    "zai": ["#e0c0f0", "#d1a3e1", "#c286d2", "#b369c3", "#a44cb4"],         # Purple (similar to kimi)
    "arcee": ["#fff9b0", "#fff176", "#ffeb3b", "#fdd835", "#f9a825"],       # Yellow
    "default": ["#d0d0d0", "#b0b0b0", "#909090", "#707070", "#505050"],     # Grays
}


def get_provider(model_name: str) -> str:
    """Determine provider from model name."""
    name_lower = model_name.lower()
    if any(x in name_lower for x in ["gpt", "o3", "o1", "openai"]):
        return "openai"
    if any(x in name_lower for x in ["claude", "anthropic"]):
        return "anthropic"
    if any(x in name_lower for x in ["gemini", "google"]):
        return "google"
    if any(x in name_lower for x in ["deepseek"]):
        return "deepseek"
    if any(x in name_lower for x in ["grok", "xai"]):
        return "xai"
    if any(x in name_lower for x in ["kimi", "moonshot"]):
        return "kimi"
    if any(x in name_lower for x in ["glm", "z.ai", "zai"]):
        return "zai"
    if any(x in name_lower for x in ["arcee", "trinity"]):
        return "arcee"
    return "default"


def load_results(results_dir: str) -> dict[str, dict]:
    """Load all results files and compute stats."""
    models = {}

    for f in Path(results_dir).glob("*_results.json"):
        with open(f) as fp:
            data = json.load(fp)

        # Extract model name from filename (remove _results.json)
        model_name = f.stem.replace("_results", "")

        # Collect all scores across all prompts
        all_scores = []
        for result in data.get("results", []):
            scores = result.get("overall_score", [])
            for s in scores:
                if s is not None:
                    all_scores.append(s)

        if all_scores:
            n = len(all_scores)
            std = np.std(all_scores)
            models[model_name] = {
                "mean": np.mean(all_scores),
                "std": std,
                "sem": std / np.sqrt(n),  # Standard error of mean
                "n": n,
                "provider": get_provider(model_name),
            }

    return models


def assign_colors(sorted_models: list) -> list[str]:
    """Assign different shades to models within the same provider."""
    # Count models per provider
    provider_counts: dict[str, int] = {}
    provider_indices: dict[str, int] = {}

    for _, stats in sorted_models:
        provider = stats["provider"]
        provider_counts[provider] = provider_counts.get(provider, 0) + 1

    colors = []
    for _, stats in sorted_models:
        provider = stats["provider"]
        palette = COLOR_PALETTES.get(provider, COLOR_PALETTES["default"])
        idx = provider_indices.get(provider, 0)
        # Cycle through shades
        color = palette[idx % len(palette)]
        provider_indices[provider] = idx + 1
        colors.append(color)

    return colors


def plot_results(models: dict, theme: str, output_path: str):
    """Create bar plot with error bars."""
    if theme == "dark":
        plt.style.use("dark_background")
        edge_color = "#ffffff"
        grid_color = "#404040"
        text_color = "#ffffff"
    else:
        plt.style.use("default")
        edge_color = "#333333"
        grid_color = "#cccccc"
        text_color = "#333333"

    # Sort by mean score descending
    sorted_models = sorted(models.items(), key=lambda x: x[1]["mean"], reverse=True)
    names = [m[0] for m in sorted_models]
    means = [m[1]["mean"] for m in sorted_models]
    sems = [m[1]["sem"] for m in sorted_models]  # Use SEM instead of STD
    colors = assign_colors(sorted_models)

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))

    y_pos = np.arange(len(names))

    # Horizontal bar chart with error bars matching theme
    bars = ax.barh(y_pos, means, xerr=sems, capsize=4, color=colors,
                   edgecolor=edge_color, linewidth=0.5, alpha=0.85,
                   error_kw={"ecolor": edge_color, "capthick": 1.5})

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()  # Highest score at top
    ax.set_xlabel("Overall Score", fontsize=12)
    ax.set_xlim(0, 1.05)
    ax.set_title("Model Performance (mean ± SEM)", fontsize=14, fontweight="bold")

    # Add score labels on bars
    for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
        label = f"{mean:.1%}"
        x_pos = min(mean + sem + 0.02, 0.95)
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
                va="center", ha="left", fontsize=9, color=text_color)

    # Grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, color=grid_color)
    ax.set_axisbelow(True)

    # Legend for providers (use middle shade from each palette)
    from matplotlib.patches import Patch
    providers_in_plot = set(m[1]["provider"] for m in sorted_models)
    provider_labels = {
        "openai": "OpenAI", "anthropic": "Anthropic", "google": "Google",
        "deepseek": "DeepSeek", "xai": "xAI", "kimi": "Kimi", "zai": "Z.ai",
        "arcee": "Arcee"
    }
    legend_elements = [Patch(facecolor=COLOR_PALETTES[p][1], edgecolor=edge_color, label=provider_labels.get(p, p))
                       for p in ["openai", "anthropic", "google", "deepseek", "xai", "kimi", "zai", "arcee"]
                       if p in providers_in_plot]
    if legend_elements:
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs"

    if not os.path.isdir(results_dir):
        print(f"Error: {results_dir} not found")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    models = load_results(results_dir)

    if not models:
        print("No results found")
        sys.exit(1)

    print(f"Found {len(models)} models:")
    for name, stats in sorted(models.items(), key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {name}: {stats['mean']:.1%} ± {stats['sem']:.1%} (n={stats['n']})")

    # Generate both themes
    plot_results(models, "light", os.path.join(output_dir, "scores_light.png"))
    plot_results(models, "dark", os.path.join(output_dir, "scores_dark.png"))


if __name__ == "__main__":
    main()
