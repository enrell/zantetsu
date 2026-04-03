#!/usr/bin/env python3
"""
RAD (Reinforcement Learning with Augmented Data) Training Data Generator

Fetches anime titles from AnimeDB API and generates synthetic training data
with RAD augmentations for the character-level CNN parser.

Usage:
    python tools/generate_rad_data.py --output data/training/rad_dataset.jsonl
    python tools/generate_rad_data.py --titles-only --output data/training/anime_titles.json
"""

import json
import random
import urllib.request
import urllib.parse
import time
import sys
from pathlib import Path
from typing import Optional

# AnimeDB API base URL
ANIMEDB_BASE = "http://localhost:8081"

# Filename patterns with typical structures
PATTERNS = [
    # Pattern 1: [Group] Title - Episode (Resolution) [CRC32].ext
    lambda t, g, e, r, c, ext: f"[{g}] {t} - {e:02d} ({r}) [{c}].{ext}",
    # Pattern 2: [Group] Title - Episode [Resolution][Codec].ext
    lambda t, g, e, r, c, ext: f"[{g}] {t} - {e:02d} [{r}][{c}].{ext}",
    # Pattern 3: [Group] Title S##E## (Resolution).ext
    lambda t, g, e, r, c, ext: f"[{g}] {t} S01E{e:02d} ({r}).{ext}",
    # Pattern 4: [Group] Title.S##E##.Resolution.Codec.ext
    lambda t, g, e, r, c, ext: f"[{g}] {t.replace(' ', '.')} S02E{e:02d} {r} {c}.mkv",
    # Pattern 5: [Group] Title - Episode [1080p][HEVC][AAC][CRC32].mkv
    lambda t, g, e, r, c, ext: f"[{g}] {t} - {e:02d} [{r}][HEVC][AAC][{c}].{ext}",
    # Pattern 6: Title.Episode.Resolution.Codec.Source.ext
    lambda t, g, e, r, c, ext: f"{t.replace(' ', '.')} {e:02d} {r} WEB-DL AAC {c}.mkv",
    # Pattern 7: [Group] Title - Season X - Episode [Resolution].ext
    lambda t, g, e, r, c, ext: f"[{g}] {t} - Season 2 - {e:02d} [{r}].{ext}",
    # Pattern 8: [Group] Title - Episodev2 (Resolution).ext
    lambda t, g, e, r, c, ext: f"[{g}] {t} - {e:02d}v2 ({r}).{ext}",
    # Pattern 9: [Group] Title - E## (Resolution) [Multi-Sub].ext
    lambda t, g, e, r, c, ext: f"[{g}] {t} - E{e:02d} ({r}) [Multi-Sub].{ext}",
    # Pattern 10: [Group] Title - 01-12 (Resolution) [Batch].ext
    lambda t, g, e, r, c, ext: f"[{g}] {t} - {e:02d}-{e + 11:02d} ({r}) [Batch].{ext}",
]

# Common groups
GROUPS = [
    "SubsPlease",
    "Erai-raws",
    "Judas",
    "Coalgirls",
    "Commie",
    "HorribleSubs",
    "PuyaSubs",
    "Moozzi2",
    "Final8",
    "AnimeTime",
    "AnimeRG",
    "AnimeKaizoku",
    "EveTaku",
    "FFFansubs",
    "Doki",
    "UTW",
    "Asenshi",
    "gg",
    "Aidoru",
    "SubsCafe",
    "HoloSubs",
    "OniiSubs",
    "AnimeSubs",
    "Saizen",
    "Tormained",
]

# Common resolutions
RESOLUTIONS = ["1080p", "720p", "480p", "2160p", "1080i", "720i"]

# Common video codecs
VIDEO_CODECS = ["H264", "x264", "HEVC", "x265", "AV1", "VP9"]

# Common audio codecs
AUDIO_CODECS = ["AAC", "FLAC", "Opus", "AC3", "DTS", "MP3"]

# Common extensions
EXTENSIONS = ["mkv", "mp4", "avi"]

# Common sources
SOURCES = ["WEB-DL", "BluRay", "BD", "HDTV", "DVD", "WEBRip"]


def fetch_anime_titles(page: int = 1, page_size: int = 100) -> list[dict]:
    """Fetch anime titles from AnimeDB API."""
    try:
        url = (
            f"{ANIMEDB_BASE}/anilist/media?page={page}&page_size={page_size}&type=ANIME"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Zantetsu/1.0"})

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        return data.get("data", [])
    except Exception as e:
        print(f"  Error fetching page {page}: {e}", file=sys.stderr)
        return []


def fetch_all_titles(max_pages: int = 50) -> list[dict]:
    """Fetch all anime titles from multiple pages."""
    all_titles = []
    seen_ids = set()

    print("Fetching anime titles from AnimeDB API...")
    for page in range(1, max_pages + 1):
        titles = fetch_anime_titles(page, page_size=500)
        if not titles:
            break

        for title in titles:
            if title["id"] not in seen_ids:
                seen_ids.add(title["id"])
                all_titles.append(title)

        print(f"  Fetched page {page}: {len(titles)} titles (total: {len(all_titles)})")
        time.sleep(0.5)  # Respect rate limit

    return all_titles


def generate_crc32() -> str:
    """Generate a random CRC32 hash."""
    return "".join(random.choices("0123456789ABCDEF", k=8))


def clean_title(title) -> str:
    """Clean anime title for filename generation."""
    import re

    # Handle case where title is a dict (API returns structured titles)
    if isinstance(title, dict):
        title = title.get("english") or title.get("romaji") or str(title)
    elif not isinstance(title, str):
        title = str(title)

    # Remove HTML tags
    title = re.sub(r"<[^>]+>", "", title)

    # Replace special characters
    title = title.replace(":", " -")
    title = title.replace("!", "")
    title = title.replace("?", "")
    title = title.replace("…", "")
    title = title.replace("\u2018", "'").replace("\u2019", "'")
    title = title.replace("\u201c", '"').replace("\u201d", '"')

    # Normalize whitespace
    title = " ".join(title.split())

    return title.strip()


def generate_filename(
    anime: dict, pattern_idx: Optional[int] = None
) -> tuple[str, dict]:
    """Generate a synthetic filename with metadata labels."""

    # Get title from anime dict - handle different API response structures
    title_obj = anime.get("title", {})

    if isinstance(title_obj, dict):
        # Title is a nested dict with romaji/english/native
        title = (
            title_obj.get("english")
            or title_obj.get("romaji")
            or title_obj.get("native", "Unknown")
        )
    elif isinstance(title_obj, str):
        # Title is already a string
        title = title_obj
    else:
        # Try other fields
        title = (
            anime.get("english")
            or anime.get("romaji")
            or anime.get("native")
            or "Unknown"
        )

    title = clean_title(title)

    # Skip very short or very long titles
    if len(title) < 3 or len(title) > 100:
        title = "Unknown Anime"

    # Random metadata
    group = random.choice(GROUPS)
    episode = random.randint(1, 26)  # Most anime have 12-26 episodes per season
    resolution = random.choice(RESOLUTIONS)
    codec = random.choice(VIDEO_CODECS)
    source = random.choice(SOURCES)
    ext = random.choice(EXTENSIONS)
    crc32 = generate_crc32()

    # Select pattern
    if pattern_idx is None:
        pattern_idx = random.randint(0, len(PATTERNS) - 1)

    pattern = PATTERNS[pattern_idx]

    try:
        filename = pattern(title, group, episode, resolution, codec, ext)
    except Exception:
        # Fallback to simple pattern
        filename = f"[{group}] {title} - {episode:02d} ({resolution}) [{crc32}].{ext}"

    return filename, {
        "title": title,
        "group": group,
        "episode": str(episode),
        "resolution": resolution,
        "video_codec": codec,
        "source": source,
        "crc32": crc32,
        "extension": ext,
        "anilist_id": anime.get("id"),
    }


def apply_rad_augmentation(filename: str, metadata: dict) -> list[tuple[str, dict]]:
    """Apply RAD augmentations to generate multiple views of the same input."""
    augmented = []

    # View 1: Character substitution (common OCR errors)
    char_subs = {
        "0": ["O", "o"],
        "1": ["l", "I", "i"],
        "5": ["S", "s"],
        "8": ["B"],
    }
    if random.random() < 0.3:
        for orig, subs in char_subs.items():
            if orig in filename:
                sub = random.choice(subs)
                aug_filename = filename.replace(orig, sub, 1)
                augmented.append((aug_filename, metadata.copy()))

    # View 2: Random segment masking
    if random.random() < 0.4:
        parts = filename.split(".")
        if len(parts) > 2:
            mask_idx = random.randint(1, len(parts) - 2)
            masked_parts = parts[:mask_idx] + ["..."] + parts[mask_idx + 1 :]
            augmented.append((".".join(masked_parts), metadata.copy()))

    # View 3: Random noise injection (add random tags)
    if random.random() < 0.3:
        noise_tags = [
            "[Hi10P]",
            "[Dual Audio]",
            "[Multiple Subtitle]",
            "[RAW]",
            "[VOSTFR]",
        ]
        noise = random.choice(noise_tags)
        aug_filename = filename.replace(".mkv", f" {noise}.mkv")
        augmented.append((aug_filename, metadata.copy()))

    # View 4: Case variation
    if random.random() < 0.2:
        aug_filename = filename.lower()
        augmented.append((aug_filename, metadata.copy()))

    # View 5: Spacing variation
    if random.random() < 0.3:
        aug_filename = filename.replace(" ", "  ")  # Double spaces
        augmented.append((aug_filename, metadata.copy()))

    return augmented


def generate_bio_tags(filename: str, metadata: dict) -> list[tuple[str, str]]:
    """Generate BIO tags for each character in the filename."""
    tags = ["O"] * len(filename)

    def mark_span(start: int, end: int, entity: str):
        if start >= len(tags) or end > len(tags):
            return
        tags[start] = f"B-{entity}"
        for i in range(start + 1, end):
            if i < len(tags):
                tags[i] = f"I-{entity}"

    # Mark group
    group = metadata.get("group", "")
    if group and f"[{group}]" in filename:
        start = filename.find(f"[{group}]")
        mark_span(start + 1, start + 1 + len(group), "GROUP")

    # Mark title
    title = metadata.get("title", "")
    if title:
        # Find title in filename (try different variations)
        title_variants = [
            title,
            title.replace(" ", "."),
            title.replace(" ", "_"),
            title.replace(" - ", " "),
        ]
        for variant in title_variants:
            if variant in filename:
                start = filename.find(variant)
                mark_span(start, start + len(variant), "TITLE")
                break

    # Mark episode
    episode = metadata.get("episode", "")
    if episode:
        ep_patterns = [
            f" - {episode}",
            f"E{episode}",
            f"EP{episode}",
            f".{episode}.",
        ]
        for pattern in ep_patterns:
            if pattern in filename:
                start = filename.find(pattern)
                end = start + len(pattern)
                mark_span(start, end, "EPISODE")
                break

    # Mark resolution
    resolution = metadata.get("resolution", "")
    if resolution and resolution in filename:
        start = filename.find(resolution)
        mark_span(start, start + len(resolution), "RESOLUTION")

    # Mark video codec
    vcodec = metadata.get("video_codec", "")
    if vcodec:
        for variant in [vcodec, vcodec.upper(), vcodec.lower()]:
            if variant in filename:
                start = filename.find(variant)
                mark_span(start, start + len(variant), "VCODEC")
                break

    # Mark source
    source = metadata.get("source", "")
    if source:
        for variant in [
            source,
            source.upper(),
            source.lower(),
            source.replace("-", ""),
        ]:
            if variant in filename:
                start = filename.find(variant)
                mark_span(start, start + len(variant), "SOURCE")
                break

    # Mark CRC32
    crc32 = metadata.get("crc32", "")
    if crc32 and f"[{crc32}]" in filename:
        start = filename.find(f"[{crc32}]")
        mark_span(start + 1, start + 1 + len(crc32), "CRC32")

    # Mark extension
    ext = metadata.get("extension", "")
    if ext and filename.endswith(f".{ext}"):
        start = filename.rfind(f".{ext}")
        mark_span(start + 1, start + 1 + len(ext), "EXTENSION")

    # Convert to token-level format
    tokens = []
    current_token = ""
    current_tag = None

    for char, tag in zip(filename, tags):
        if tag.startswith("B-"):
            # Save previous token
            if current_token:
                tokens.append((current_token, current_tag or "O"))
            current_token = char
            current_tag = tag[2:]  # Remove B- prefix
        elif tag.startswith("I-"):
            current_token += char
        else:
            if current_token:
                tokens.append((current_token, current_tag or "O"))
                current_token = ""
                current_tag = None

    if current_token:
        tokens.append((current_token, current_tag or "O"))

    return tags, tokens


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate RAD training data")
    parser.add_argument(
        "--output", default="data/training/rad_dataset.jsonl", help="Output file path"
    )
    parser.add_argument(
        "--max-pages", type=int, default=50, help="Max API pages to fetch"
    )
    parser.add_argument(
        "--samples-per-anime", type=int, default=5, help="Samples per anime title"
    )
    parser.add_argument(
        "--titles-only",
        action="store_true",
        help="Only save titles, don't generate data",
    )
    parser.add_argument(
        "--augmentations", type=int, default=2, help="RAD augmentations per sample"
    )

    args = parser.parse_args()

    # Fetch anime titles
    titles = fetch_all_titles(args.max_pages)
    print(f"\nFetched {len(titles)} unique anime titles")

    if args.titles_only:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(titles, f, indent=2)
        print(f"Saved titles to {output_path}")
        return

    # Generate training data
    print(
        f"Generating training data with {args.augmentations} augmentations per sample..."
    )

    dataset = []
    seen_filenames = set()

    for i, anime in enumerate(titles):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(titles)} anime")

        for _ in range(args.samples_per_anime):
            filename, metadata = generate_filename(anime)

            if filename in seen_filenames:
                continue

            seen_filenames.add(filename)

            # Generate BIO tags
            char_tags, token_tags = generate_bio_tags(filename, metadata)

            dataset.append(
                {
                    "filename": filename,
                    "tokens": [t for t, _ in token_tags],
                    "ner_tags": [tag for _, tag in token_tags],
                    "char_tags": char_tags,
                    "metadata": metadata,
                    "augmentation": "none",
                }
            )

            # Apply RAD augmentations
            if args.augmentations > 0:
                augmented = apply_rad_augmentation(filename, metadata)
                for aug_idx, (aug_filename, aug_meta) in enumerate(
                    augmented[: args.augmentations]
                ):
                    if aug_filename not in seen_filenames:
                        seen_filenames.add(aug_filename)
                        aug_char_tags, aug_token_tags = generate_bio_tags(
                            aug_filename, aug_meta
                        )
                        dataset.append(
                            {
                                "filename": aug_filename,
                                "tokens": [t for t, _ in aug_token_tags],
                                "ner_tags": [tag for _, tag in aug_token_tags],
                                "char_tags": aug_char_tags,
                                "metadata": aug_meta,
                                "augmentation": f"rad_{aug_idx}",
                            }
                        )

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nGenerated {len(dataset)} training samples")
    print(f"Saved to {output_path}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Unique titles: {len(titles)}")
    print(
        f"  Augmented samples: {len([d for d in dataset if d['augmentation'] != 'none'])}"
    )

    # Count entity types
    entity_counts = {}
    for item in dataset:
        for tag in item["ner_tags"]:
            if tag != "O":
                entity_counts[tag] = entity_counts.get(tag, 0) + 1

    print("\nEntity Distribution:")
    for entity, count in sorted(entity_counts.items()):
        print(f"  {entity}: {count}")


if __name__ == "__main__":
    main()
