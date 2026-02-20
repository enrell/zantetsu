#!/usr/bin/env python3
"""Validate Gemini output has exactly N numbered lines without comments."""

import sys
import re

NUM_LINES = 73  # Adjust based on input file


def validate_output(filepath: str) -> tuple[bool, str]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]

    if len(lines) != NUM_LINES:
        return False, f"Expected {NUM_LINES} lines, got {len(lines)}"

    expected_nums = set(range(1, NUM_LINES + 1))
    found_nums = set()

    for line in lines:
        match = re.match(r"^\((\d+)\)\s+(.+)$", line)
        if not match:
            return False, f"Invalid line format: {line[:50]}..."

        num = int(match.group(1))
        title = match.group(2).strip()

        if num < 1 or num > NUM_LINES:
            return False, f"Line number out of range: {num}"

        if num in found_nums:
            return False, f"Duplicate line number: {num}"

        if not title:
            return False, f"Empty title for line: {num}"

        found_nums.add(num)

    if found_nums != expected_nums:
        missing = expected_nums - found_nums
        return False, f"Missing line numbers: {sorted(missing)[:10]}..."

    return (
        True,
        f"Valid: {NUM_LINES} lines, all numbers 1-{NUM_LINES} present, no duplicates",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_gemini_output.py <output_file>")
        sys.exit(1)

    valid, msg = validate_output(sys.argv[1])
    print(msg)
    sys.exit(0 if valid else 1)
