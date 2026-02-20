#!/usr/bin/env python3
"""Generate 50K synthetic training examples from Kitsu titles - fixed BIO tagging."""

import random
import re
import os
import sys

TARGET_SAMPLES = 50000

RELEASE_GROUPS = [
    "[HorribleSubs]",
    "[SubsPlease]",
    "[Erai-raws]",
    "[ASW]",
    "[DKB]",
    "[ToonsHub]",
    "[Judas]",
    "[DB]",
    "[SSA]",
    "[AnimeKaizoku]",
    "[Raws]",
    "[BD]",
    "[BluRay]",
    "[CrackU]",
    "[Moozzi2]",
    "[Reinforce]",
    "[Vivid]",
    "[Tenrai-Sensei]",
    "[FUA]",
    "[GHOST]",
    "[AGB]",
]

QUALITY_TAGS = [
    "1080p",
    "720p",
    "480p",
    "2160p",
    "4K",
    "BD",
    "BluRay",
    "WEB-DL",
    "WEBRip",
    "HDRip",
    "HEVC",
    "x265",
    "x264",
    "HEVC10",
    "AVCbit",
    "8",
    "10bit",
    "AAC",
    "FLAC",
    "Opus",
    "Multi-Subs",
    "Multi-Audio",
    "Dual-Audio",
]

EPISODE_PATTERNS = [
    "S01E{:02d}",
    "S02E{:02d}",
    "S03E{:02d}",
    "E{:02d}",
    "Ep {:02d}",
    "Episode {:02d}",
    "ep {:02d}",
    " - {:02d}",
    " [{:02d}]",
]

DELIMITERS = [" - ", " ", "_", ".", "-"]

ADDITIONAL_TAGS = [
    "[Isekai]",
    "[Shounen]",
    "[Action]",
    "[Romance]",
    "[Comedy]",
    "[Drama]",
    "[Fantasy]",
    "[Sci-Fi]",
    "[Multi-Subs]",
    "[Weekly]",
    "[Batch]",
    "[Complete]",
]


def clean_title(title: str) -> str:
    title = re.sub(r'[<>:"/\\|?*]', "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def tokenize(filename: str) -> list[tuple[str, int, int]]:
    """Tokenize preserving delimiters."""
    tokens = []
    i = 0
    while i < len(filename):
        # Check for bracket groups
        if filename[i] == "[":
            j = i + 1
            while j < len(filename) and filename[j] != "]":
                j += 1
            tokens.append((filename[i : j + 1], i, j + 1))
            i = j + 1
        elif filename[i] == "(":
            j = i + 1
            while j < len(filename) and filename[j] != ")":
                j += 1
            tokens.append((filename[i : j + 1], i, j + 1))
            i = j + 1
        # Check for whitespace
        elif filename[i].isspace():
            j = i
            while j < len(filename) and filename[j].isspace():
                j += 1
            tokens.append((filename[i:j], i, j))
            i = j
        # Single delimiter chars
        elif filename[i] in ".,;-_":
            tokens.append((filename[i], i, i + 1))
            i += 1
        # Regular token
        else:
            j = i
            while j < len(filename) and filename[j] not in " []()[]().,;-_ \t\n":
                j += 1
            if j > i:
                tokens.append((filename[i:j], i, j))
                i = j
            else:
                i += 1
    return tokens


def generate_filename(title: str) -> tuple[str, str]:
    title = clean_title(title)
    fmt = random.randint(1, 8)
    parts = []

    if fmt == 1:
        group = random.choice(RELEASE_GROUPS)
        ep = random.choice(EPISODE_PATTERNS).format(random.randint(1, 24))
        qual = random.choice(QUALITY_TAGS)
        parts = [group, " ", title, " - ", ep, " [", qual, "]"]
    elif fmt == 2:
        ep = random.choice(EPISODE_PATTERNS).format(random.randint(1, 24))
        qual = random.choice(QUALITY_TAGS)
        parts = [title, " ", ep, " [", qual, "]"]
    elif fmt == 3:
        group = random.choice(RELEASE_GROUPS)
        ep = random.choice(EPISODE_PATTERNS).format(random.randint(1, 24))
        qual = random.choice(QUALITY_TAGS)
        parts = [group, " ", title, " ", ep, " ", qual]
    elif fmt == 4:
        ep = random.choice(EPISODE_PATTERNS).format(random.randint(1, 24))
        parts = [title, " ", ep]
    elif fmt == 5:
        ep = random.choice(EPISODE_PATTERNS).format(random.randint(1, 24))
        qual = random.choice(QUALITY_TAGS)
        group = random.choice(RELEASE_GROUPS)
        parts = [title, " - ", ep, " - ", qual, " ", group]
    elif fmt == 6:
        group = random.choice(RELEASE_GROUPS)
        parts = [group, " ", title, " - Complete Batch"]
    elif fmt == 7:
        ep = "S01E{:02d}".format(random.randint(1, 24))
        qual = random.choice(QUALITY_TAGS)
        group = random.choice(RELEASE_GROUPS).strip("[]()")
        parts = [title, ".", ep, ".", qual, ".", group]
    else:
        ep = random.choice(EPISODE_PATTERNS).format(random.randint(1, 24))
        parts = [title, " Episode ", ep]

    if random.random() < 0.3:
        tag = random.choice(ADDITIONAL_TAGS)
        parts.insert(-1, " " + tag)

    filename = "".join(parts)
    return filename, title


def create_bio_tags(filename: str, title: str) -> str:
    tokens = tokenize(filename)
    if not tokens:
        return None

    title_lower = title.lower()
    filename_lower = filename.lower()

    # Find where title appears
    title_start = filename_lower.find(title_lower)
    if title_start == -1:
        return None

    title_end = title_start + len(title)

    result = []
    for token, start, end in tokens:
        # Skip whitespace tokens
        if token.strip() == "":
            continue

        # Check if token is within title span
        if start >= title_start and end <= title_end:
            if start == title_start:
                result.append(f"{token}\tB-TITLE")
            else:
                result.append(f"{token}\tI-TITLE")
        else:
            result.append(f"{token}\tO")

    return "\n".join(result)


# Load Kitsu titles
print("Loading Kitsu titles...", file=sys.stderr)

import subprocess

result = subprocess.run(
    [
        "psql",
        os.environ.get("DATABASE_URL", "postgresql://root:root@localhost:5432/root"),
        "-t",
        "-c",
        "SELECT titles->'en_jp' FROM anime WHERE titles ? 'en_jp' LIMIT 20000;",
    ],
    capture_output=True,
    text=True,
)

titles = [line.strip() for line in result.stdout.split("\n") if line.strip()]

result2 = subprocess.run(
    [
        "psql",
        os.environ.get("DATABASE_URL", "postgresql://root:root@localhost:5432/root"),
        "-t",
        "-c",
        "SELECT canonical_title FROM anime WHERE canonical_title IS NOT NULL LIMIT 20000;",
    ],
    capture_output=True,
    text=True,
)

canonical = [line.strip() for line in result2.stdout.split("\n") if line.strip()]
titles.extend(canonical)
titles = list(set([t for t in titles if t and len(t) > 1]))

print(f"Total unique titles: {len(titles)}", file=sys.stderr)

output = []
attempts = 0
max_attempts = TARGET_SAMPLES * 3

print(f"Generating {TARGET_SAMPLES} samples...", file=sys.stderr)

while len(output) < TARGET_SAMPLES * 3 and attempts < max_attempts:
    attempts += 1
    title = random.choice(titles)
    if not title:
        continue

    filename, expected_title = generate_filename(title)
    bio = create_bio_tags(filename, expected_title)

    if bio and "B-TITLE" in bio:
        output.append(f"# {len(output) // 3 + 1}: {filename} → {expected_title}")
        output.append(bio)
        output.append("")

        if len(output) // 3 % 5000 == 0:
            print(f"Progress: {len(output) // 3} / {TARGET_SAMPLES}", file=sys.stderr)

with open("data/training/bio_train_50k.txt", "w") as f:
    f.write("\n".join(output))

print(f"Generated {len(output) // 3} training examples", file=sys.stderr)

# Stats
labels = []
for line in output:
    if "\t" in line and not line.startswith("#"):
        parts = line.rstrip().split("\t")
        if len(parts) == 2:
            labels.append(parts[1])

from collections import Counter

c = Counter(labels)
print(
    f"Label counts: B-TITLE={c.get('B-TITLE', 0)}, I-TITLE={c.get('I-TITLE', 0)}, O={c.get('O', 0)}",
    file=sys.stderr,
)
