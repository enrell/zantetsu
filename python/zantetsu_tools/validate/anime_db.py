#!/usr/bin/env python3
"""
AnimeDB Validator - Validates heuristic parser output against AnimeDB API.

This script:
1. Runs the heuristic parser on test cases
2. Validates extracted titles against AnimeDB API
3. Identifies cases where parsing could be improved
4. Generates validated training data

Usage:
    python tools/validate/anime_db.py --input data/training/nyaa_titles_5000_raw.txt
    python tools/validate/anime_db.py --benchmark
"""

import json
import time
from pathlib import Path
from typing import Optional

from ..common.anime_db import AnimeDbClient
from ..common.jsonl import write_jsonl
from ..common.paths import REPO_ROOT, TARGET_RELEASE_DIR


CLIENT = AnimeDbClient(user_agent="ZantetsuValidator/1.0")


def call_zantetsu_parser(filename: str, mode: str = "heuristic") -> dict:
    """Call the Rust zantetsu parser binary."""
    import subprocess

    try:
        result = subprocess.run(
            [str(TARGET_RELEASE_DIR / "zantetsu-parse"), mode],
            input=filename,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {"error": result.stderr}
    except Exception as e:
        return {"error": str(e)}


def search_anime(title: str, source: str = "both") -> Optional[dict]:
    """Search for anime in AnimeDB API."""
    return CLIENT.search_realtime(title, source=source, limit=5)


def fuzzy_match_title(extracted: str, api_result: dict) -> float:
    """Calculate fuzzy match score between extracted title and API result."""
    if not api_result:
        return 0.0

    # Get all titles from API result
    api_titles = []
    if api_result.get("title"):
        api_titles.append(api_result["title"].lower())
    if api_result.get("romaji"):
        api_titles.append(api_result["romaji"].lower())
    if api_result.get("english"):
        api_titles.append(api_result["english"].lower())

    extracted_lower = extracted.lower()

    # Exact match
    for api_title in api_titles:
        if extracted_lower == api_title:
            return 1.0

    # Substring match
    for api_title in api_titles:
        if extracted_lower in api_title or api_title in extracted_lower:
            return 0.8

    # Word overlap (Jaccard similarity)
    for api_title in api_titles:
        extracted_words = set(extracted_lower.split())
        api_words = set(api_title.split())

        if not extracted_words or not api_words:
            continue

        intersection = extracted_words & api_words
        union = extracted_words | api_words

        if union:
            jaccard = len(intersection) / len(union)
            if jaccard > 0.5:
                return jaccard

    return 0.0


def validate_filename(filename: str, expected: Optional[dict] = None) -> dict:
    """Validate a filename parsing against AnimeDB API."""

    # Step 1: Parse with heuristic parser
    parsed = call_zantetsu_parser(filename, "heuristic")

    # Check for actual errors (not null error field)
    if "error" in parsed and parsed["error"]:
        return {"filename": filename, "error": parsed["error"], "valid": False}

    result = {
        "filename": filename,
        "parsed": parsed,
        "valid": False,
        "title_match": 0.0,
        "api_result": None,
        "issues": [],
    }

    # Step 2: Extract title
    title = parsed.get("title")
    if not title:
        result["issues"].append("No title extracted")
        return result

    # Step 3: Search in AnimeDB
    api_result = search_anime(title)
    result["api_result"] = api_result

    if not api_result:
        result["issues"].append(f"Title not found in AnimeDB: {title}")
        return result

    # Step 4: Calculate match score
    match_score = fuzzy_match_title(title, api_result)
    result["title_match"] = match_score

    # Step 5: Determine validity
    if match_score >= 0.5:
        result["valid"] = True
    else:
        result["issues"].append(f"Low title match score: {match_score:.2f}")

    # Step 6: Validate other fields against expected
    if expected:
        for field in ["group", "episode", "resolution", "video_codec", "source"]:
            if field in expected:
                parsed_field = parsed.get(field)
                expected_field = expected[field]

                # Handle episode which can be dict or int
                if field == "episode" and isinstance(parsed_field, dict):
                    parsed_field = parsed_field.get("Single") or parsed_field.get(
                        "Range"
                    )

                if str(parsed_field) != str(expected_field):
                    result["issues"].append(
                        f"{field} mismatch: got {parsed_field}, expected {expected_field}"
                    )

    return result


def validate_bulk(filenames: list[str], rate_limit: float = 0.5) -> list[dict]:
    """Validate multiple filenames against AnimeDB API."""
    results = []

    print(f"Validating {len(filenames)} filenames against AnimeDB API...")

    for i, filename in enumerate(filenames):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(filenames)}")

        result = validate_filename(filename)
        results.append(result)

        # Rate limiting (500ms between requests)
        time.sleep(rate_limit)

    return results


def run_benchmark() -> dict:
    """Run validation on benchmark test cases."""

    # Load test cases from benchmark
    benchmark_file = REPO_ROOT / "data" / "training" / "silver_dataset.jsonl"

    if not benchmark_file.exists():
        print(f"Benchmark file not found: {benchmark_file}")
        return {}

    filenames = []
    with open(benchmark_file) as f:
        for line in f:
            data = json.loads(line)
            # Reconstruct filename from tokens
            filename = " ".join(data.get("tokens", []))
            filenames.append(filename)

    # Run validation
    results = validate_bulk(filenames[:50])  # Sample 50 for speed

    # Analyze results
    valid_count = sum(1 for r in results if r["valid"])
    total_count = len(results)
    avg_match = (
        sum(r["title_match"] for r in results) / total_count if total_count > 0 else 0
    )

    # Count issues
    issue_counts = {}
    for r in results:
        for issue in r["issues"]:
            issue_type = issue.split(":")[0] if ":" in issue else issue
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    benchmark_results = {
        "total": total_count,
        "valid": valid_count,
        "accuracy": valid_count / total_count if total_count > 0 else 0,
        "avg_match_score": avg_match,
        "issue_counts": issue_counts,
    }

    print(f"\n{'=' * 60}")
    print("ANIMEDB VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {total_count}")
    print(f"Valid parses: {valid_count}")
    print(f"Accuracy: {benchmark_results['accuracy']:.2%}")
    print(f"Average match score: {avg_match:.3f}")
    print(f"\nIssue Distribution:")
    for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count}")

    return benchmark_results


def generate_validated_dataset(
    raw_titles_file: str, output_file: str, max_samples: int = 10000
) -> int:
    """Generate validated training dataset using AnimeDB API."""

    raw_path = Path(raw_titles_file)
    output_path = Path(output_file)

    if not raw_path.exists():
        print(f"Input file not found: {raw_path}")
        return 0

    # Read raw titles
    with open(raw_path) as f:
        raw_titles = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(raw_titles)} raw titles")

    validated_samples = []
    seen = set()

    print(f"Validating against AnimeDB API (max {max_samples} samples)...")

    for i, title in enumerate(raw_titles[:max_samples]):
        if i % 50 == 0:
            print(f"  Progress: {i}/{min(len(raw_titles), max_samples)}")

        if title in seen:
            continue
        seen.add(title)

        # Search in AnimeDB
        api_result = search_anime(title)

        if api_result and api_result.get("score", 0) > 0.3:
            # Good match found - add as validated sample
            validated_samples.append(
                {
                    "title": title,
                    "anilist_id": api_result.get("id"),
                    "api_title": api_result.get("title"),
                    "api_score": api_result.get("score"),
                    "validation": "api_match",
                }
            )

        # Rate limiting
        time.sleep(0.3)

    # Save validated dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_path, validated_samples)

    print(f"\nGenerated {len(validated_samples)} validated samples")
    print(f"Saved to {output_path}")

    return len(validated_samples)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate parser output against AnimeDB"
    )
    parser.add_argument("--input", help="Input file with raw titles")
    parser.add_argument(
        "--output", default="data/training/validated_dataset.jsonl", help="Output file"
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Run benchmark validation"
    )
    parser.add_argument(
        "--max-samples", type=int, default=10000, help="Max samples to validate"
    )
    parser.add_argument("--test", help="Test a single filename")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.test:
        result = validate_filename(args.test)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.input:
        generate_validated_dataset(args.input, args.output, args.max_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
