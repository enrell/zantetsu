#!/usr/bin/env python3
"""
Comparative benchmark script for torrent title parsers.
Compares: PTT (Python), RTN, Zantetsu Heuristic, Zantetsu Neural CRF

Outputs detailed results to JSON for analysis.
"""

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..common.paths import MODEL_DIR, REPO_ROOT, TARGET_RELEASE_DIR

try:
    from PTT import parse_title as ptt_parse
except ImportError:
    print("ERROR: PTT not installed. Run: pip install parsett")
    sys.exit(1)

try:
    from RTN import parse as rtn_parse
    from RTN.models import SettingsModel, DefaultRanking
except ImportError:
    print("ERROR: RTN not installed. Run: pip install rank-torrent-name")
    sys.exit(1)

try:
    import torch
    from ..training.char_cnn import CharCnnCrf, IDX_TO_TAG, MAX_LEN, char_encode

    CHAR_CNN_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    CharCnnCrf = None
    IDX_TO_TAG = {}
    MAX_LEN = 256
    char_encode = None
    CHAR_CNN_IMPORT_ERROR = exc


ZANTETSU_BINARY_BASENAME = "zantetsu-parse"
REGRESSION_DATA = (
    REPO_ROOT / "data" / "regression" / "tricky_filenames.jsonl"
)
OUTPUT_FILE = REPO_ROOT / "data" / "benchmarks" / "parser_comparison.json"
CHAR_CNN_CHECKPOINT_CANDIDATES = [
    MODEL_DIR / "best_char_cnn_crf.pt",
    MODEL_DIR / "char_cnn_crf.pt",
]


_char_cnn_parser = None


def resolve_zantetsu_binary() -> Path | None:
    """Resolve benchmark binary path across platforms."""
    candidates = [TARGET_RELEASE_DIR / ZANTETSU_BINARY_BASENAME]

    if sys.platform.startswith("win"):
        candidates.insert(0, TARGET_RELEASE_DIR / f"{ZANTETSU_BINARY_BASENAME}.exe")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def resolve_char_cnn_checkpoint() -> Path | None:
    """Resolve the trained char-CNN checkpoint path."""
    for candidate in CHAR_CNN_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def clean_entity_text(text: str, entity_type: str) -> str:
    """Normalize character-span text into parser-friendly values."""
    text = text.strip()
    if entity_type in {"TITLE", "GROUP"}:
        text = text.strip("[](){} -_.")
        text = text.replace("_", " ").replace(".", " ")
        text = re.sub(r"\s+", " ", text).strip()
    elif entity_type == "EXTENSION":
        text = text.strip("[](){} ").lstrip(".").strip()
    else:
        text = text.strip("[](){} -_.")
    return text


def parse_episode_value(text: str) -> dict[str, Any] | None:
    """Convert raw episode text into the benchmark's episode representation."""
    numbers = [int(value) for value in re.findall(r"\d+", text)]
    if not numbers:
        return None

    version_match = re.search(r"v(\d+)", text, re.IGNORECASE)
    if version_match:
        return {
            "Version": {
                "episode": numbers[0],
                "version": int(version_match.group(1)),
            }
        }

    if "-" in text and len(numbers) >= 2:
        return {"Range": numbers[:2]}

    if any(separator in text for separator in [",", "+", "&"]) and len(numbers) > 1:
        return {"Multi": numbers}

    return {"Single": numbers[0]}


def collect_char_cnn_entities(filename: str, tag_ids: list[int]) -> list[tuple[str, str]]:
    """Convert decoded character tags into entity strings."""
    truncated = filename[: MAX_LEN - 2]
    entities = []
    index = 0

    while index < len(truncated):
        tag_name = IDX_TO_TAG.get(tag_ids[index], "O")
        if tag_name == "O":
            index += 1
            continue

        if tag_name.startswith("B-"):
            entity_type = tag_name[2:]
            end_index = index + 1
            inside_tag = f"I-{entity_type}"
            while end_index < len(truncated) and IDX_TO_TAG.get(tag_ids[end_index], "O") == inside_tag:
                end_index += 1
        elif tag_name.startswith("I-"):
            entity_type = tag_name[2:]
            end_index = index + 1
            inside_tag = f"I-{entity_type}"
            while end_index < len(truncated) and IDX_TO_TAG.get(tag_ids[end_index], "O") == inside_tag:
                end_index += 1
        else:
            entity_type = tag_name
            end_index = index + 1
            while end_index < len(truncated) and IDX_TO_TAG.get(tag_ids[end_index], "O") == tag_name:
                end_index += 1

        entity_text = clean_entity_text(truncated[index:end_index], entity_type)
        if entity_text:
            entities.append((entity_type, entity_text))
        index = end_index

    return entities


class CharCnnBenchmarkParser:
    """Inference wrapper around the trained character CNN checkpoint."""

    def __init__(self, checkpoint_path: Path):
        self.device = torch.device("cpu")
        self.model = CharCnnCrf(use_crf=True).to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def parse(self, filename: str) -> dict[str, Any]:
        with torch.inference_mode():
            input_tensor = torch.tensor(
                [char_encode(filename)],
                dtype=torch.long,
                device=self.device,
            )
            tag_ids = self.model.decode(input_tensor)[0]

        entities = collect_char_cnn_entities(filename, tag_ids)
        title_chunks: list[str] = []
        group_chunks: list[str] = []

        result = {
            "title": None,
            "group": None,
            "season": None,
            "episode": None,
            "resolution": None,
            "video_codec": None,
            "audio_codec": None,
            "source": None,
            "year": None,
            "crc32": None,
            "extension": None,
            "version": None,
        }

        for entity_type, entity_text in entities:
            if entity_type == "TITLE":
                title_chunks.append(entity_text)
            elif entity_type == "GROUP":
                group_chunks.append(entity_text)
            elif entity_type == "SEASON" and result["season"] is None:
                match = re.search(r"\d+", entity_text)
                if match:
                    result["season"] = int(match.group(0))
            elif entity_type == "EPISODE" and result["episode"] is None:
                result["episode"] = parse_episode_value(entity_text)
            elif entity_type == "RESOLUTION" and result["resolution"] is None:
                result["resolution"] = convert_resolution(entity_text)
            elif entity_type == "VCODEC" and result["video_codec"] is None:
                result["video_codec"] = convert_codec(entity_text)
            elif entity_type == "ACODEC" and result["audio_codec"] is None:
                result["audio_codec"] = convert_audio([entity_text])
            elif entity_type == "SOURCE" and result["source"] is None:
                result["source"] = convert_source(entity_text)
            elif entity_type == "YEAR" and result["year"] is None:
                match = re.search(r"\d{4}", entity_text)
                if match:
                    result["year"] = int(match.group(0))
            elif entity_type == "CRC32" and result["crc32"] is None:
                match = re.search(r"[A-Fa-f0-9]{8}", entity_text)
                if match:
                    result["crc32"] = match.group(0).upper()
            elif entity_type == "EXTENSION" and result["extension"] is None:
                result["extension"] = entity_text.lower() or None
            elif entity_type == "VERSION" and result["version"] is None:
                match = re.search(r"\d+", entity_text)
                if match:
                    result["version"] = int(match.group(0))

        if title_chunks:
            result["title"] = re.sub(r"\s+", " ", " ".join(title_chunks)).strip()
        if group_chunks:
            result["group"] = re.sub(r"\s+", " ", " ".join(group_chunks)).strip()

        return result


def get_char_cnn_parser() -> CharCnnBenchmarkParser:
    """Load the char-CNN benchmark parser once."""
    global _char_cnn_parser
    if _char_cnn_parser is None:
        checkpoint_path = resolve_char_cnn_checkpoint()
        if torch is None or CharCnnCrf is None or char_encode is None:
            raise RuntimeError(f"char-CNN dependencies unavailable: {CHAR_CNN_IMPORT_ERROR}")
        if checkpoint_path is None:
            raise RuntimeError("char-CNN checkpoint not found")
        _char_cnn_parser = CharCnnBenchmarkParser(checkpoint_path)
    return _char_cnn_parser


def parse_with_char_cnn(filename: str) -> dict[str, Any]:
    """Parse using the trained char-CNN checkpoint."""
    return get_char_cnn_parser().parse(filename)


def parse_with_ptt(filename: str) -> dict[str, Any]:
    """Parse using PTT (Python Torrent Title parser)."""
    result = ptt_parse(filename)

    def convert_episode(e: list[int]) -> dict:
        if len(e) == 1:
            return {"Single": e[0]}
        return {"Multi": e}

    def convert_season(s: list[int]) -> int | None:
        return s[0] if s else None

    return {
        "title": result.get("title"),
        "group": result.get("group"),
        "season": convert_season(result.get("seasons")),
        "episode": convert_episode(result.get("episodes", []))
        if result.get("episodes")
        else None,
        "resolution": convert_resolution(result.get("resolution")),
        "video_codec": convert_codec(result.get("codec")),
        "audio_codec": convert_audio(result.get("audio")),
        "source": convert_source(result.get("quality")),
        "year": result.get("year"),
        "crc32": None,
        "extension": result.get("extension"),
        "version": None,
    }


def parse_with_rtn(filename: str) -> dict[str, Any]:
    """Parse using RTN (Rank Torrent Name)."""
    result = rtn_parse(filename)

    # RTN returns ParsedData directly, not wrapped
    data = result

    def convert_episode(e: list[int]) -> dict:
        if len(e) == 1:
            return {"Single": e[0]}
        return {"Multi": e}

    def convert_season(s: list[int]) -> int | None:
        return s[0] if s else None

    return {
        "title": data.parsed_title,
        "group": data.group,  # RTN extracts group
        "season": convert_season(data.seasons),
        "episode": convert_episode(data.episodes) if data.episodes else None,
        "resolution": convert_resolution(data.resolution),
        "video_codec": convert_codec(data.codec[0] if data.codec else None),
        "audio_codec": convert_audio(data.audio),
        "source": convert_source(data.quality),
        "year": data.year[0]
        if isinstance(data.year, list) and data.year
        else (data.year if isinstance(data.year, int) else None),
        "crc32": data.episode_code,  # RTN extracts CRC in episode_code
        "extension": data.extension,
        "version": None,
    }


def convert_resolution(res: str | None) -> str | None:
    """Convert resolution to Zantetsu format."""
    if not res:
        return None
    res = res.lower().replace("p", "p").replace("i", "i")
    if "2160" in res or "4k" in res:
        return "UHD2160"
    elif "1080" in res:
        return "FHD1080"
    elif "720" in res:
        return "HD720"
    elif "480" in res or "576" in res:
        return "SD480"
    return None


def convert_codec(codec: str | None) -> str | None:
    """Convert codec to Zantetsu format."""
    if not codec:
        return None
    codec = codec.lower()
    if codec in ("hevc", "h265", "x265"):
        return "HEVC"
    elif codec in ("avc", "h264", "x264"):
        return "H264"
    elif codec in ("av1",):
        return "AV1"
    elif codec in ("vp9",):
        return "VP9"
    return None


def convert_audio(audio: list[str] | None) -> str | None:
    """Convert audio to Zantetsu format."""
    if not audio:
        return None
    audio_str = audio[0].lower() if audio else None
    if not audio_str:
        return None
    if "flac" in audio_str:
        return "FLAC"
    elif "aac" in audio_str:
        return "AAC"
    elif "opus" in audio_str:
        return "Opus"
    elif "dts" in audio_str:
        return "DTS"
    elif "truehd" in audio_str:
        return "TrueHD"
    elif "ac3" in audio_str:
        return "AC3"
    elif "mp3" in audio_str:
        return "MP3"
    return None


def convert_source(quality: str | None) -> str | None:
    """Convert quality to Zantetsu format."""
    if not quality:
        return None
    quality = quality.lower()
    if "bluray" in quality or "blu-ray" in quality:
        return "BluRay"
    elif "webdl" in quality or "web-dl" in quality:
        return "WebDL"
    elif "webrip" in quality or "web-rip" in quality:
        return "WebRip"
    elif "hdtv" in quality:
        return "HDTV"
    elif "dvd" in quality:
        return "DVD"
    return None


def parse_with_zantetsu(
    filename: str, zantetsu_binary: Path, mode: str = "heuristic"
) -> dict[str, Any]:
    """Parse using Zantetsu binary."""
    result = subprocess.run(
        [str(zantetsu_binary), mode],
        input=filename,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        details = stderr or stdout or "unknown zantetsu process error"
        raise RuntimeError(f"zantetsu {mode} failed: {details}")

    data = json.loads(result.stdout.strip())
    return {
        "title": data.get("title"),
        "group": data.get("group"),
        "season": data.get("season"),
        "episode": data.get("episode"),
        "resolution": data.get("resolution"),
        "video_codec": data.get("video_codec"),
        "audio_codec": data.get("audio_codec"),
        "source": data.get("source"),
        "year": data.get("year"),
        "crc32": data.get("crc32"),
        "extension": data.get("extension"),
        "version": data.get("version"),
    }


def normalize_title(title: str | None) -> str:
    """Normalize title for comparison."""
    if not title:
        return ""
    title = title.lower()
    title = re.sub(r"[.\-_]", " ", title)
    title = re.sub(r"\s+", " ", title)
    return title.strip()


def field_score(expected: Any, actual: Any, field: str) -> tuple[float, str]:
    """Calculate score for a single field. Returns (score, reason)."""
    if expected is None and actual is None:
        return 1.0, "both_none"
    if expected is None and actual is not None:
        return 0.0, f"expected_none_got_{actual}"
    if expected is not None and actual is None:
        return 0.0, "expected_got_none"

    # Special handling for title (fuzzy match)
    if field == "title":
        expected_norm = normalize_title(expected)
        actual_norm = normalize_title(actual)
        if expected_norm == actual_norm:
            return 1.0, "exact_match"
        if expected_norm in actual_norm or actual_norm in expected_norm:
            return 0.7, "partial_match"
        return 0.0, f"no_match: expected='{expected_norm}' got='{actual_norm}'"

    # Exact match for other fields
    if str(expected).lower() == str(actual).lower():
        return 1.0, "exact_match"
    return 0.0, f"no_match: expected={expected} got={actual}"


def calculate_total_score(
    expected: dict[str, Any], actual: dict[str, Any]
) -> tuple[float, dict[str, tuple[float, str]]]:
    """Calculate total score and field-by-field results."""
    fields = [
        "title",
        "group",
        "season",
        "episode",
        "resolution",
        "video_codec",
        "audio_codec",
        "source",
        "year",
        "crc32",
        "extension",
    ]

    total = 0.0
    field_results = {}
    for field in fields:
        score, reason = field_score(expected.get(field), actual.get(field), field)
        field_results[field] = (score, reason)
        total += score

    return total / len(fields), field_results


def main():
    # Check for optional Zantetsu binary
    zantetsu_binary = resolve_zantetsu_binary()
    zantetsu_available = zantetsu_binary is not None
    if not zantetsu_available:
        print(
            "WARNING: Zantetsu binary not found in "
            f"{TARGET_RELEASE_DIR} (expected {ZANTETSU_BINARY_BASENAME}[.exe])"
        )
        print("         Zantetsu benchmarks will be skipped.")
        print("         To enable: cargo build --release -p benchmark-compare")
    else:
        print(f"Found Zantetsu binary: {zantetsu_binary}")

    char_cnn_checkpoint = resolve_char_cnn_checkpoint()
    char_cnn_available = (
        char_cnn_checkpoint is not None
        and torch is not None
        and CharCnnCrf is not None
        and char_encode is not None
    )
    if char_cnn_available:
        print(f"Found char-CNN checkpoint: {char_cnn_checkpoint}")
    else:
        reason = (
            str(CHAR_CNN_IMPORT_ERROR)
            if CHAR_CNN_IMPORT_ERROR is not None
            else "checkpoint not available"
        )
        print(f"WARNING: Char-CNN benchmark unavailable: {reason}")

    if not REGRESSION_DATA.exists():
        print(f"ERROR: Regression data not found at {REGRESSION_DATA}")
        sys.exit(1)

    # Load test cases
    test_cases = []
    with open(REGRESSION_DATA) as f:
        for line in f:
            data = json.loads(line)
            test_cases.append(data)

    print(f"Loaded {len(test_cases)} test cases")
    print("=" * 60)

    # Results storage
    results = {
        "ptt": {"scores": [], "errors": 0, "field_wins": {}},
        "rtn": {"scores": [], "errors": 0, "field_wins": {}},
        "zantetsu_heuristic": {"scores": [], "errors": 0, "field_wins": {}},
        "zantetsu_auto": {"scores": [], "errors": 0, "field_wins": {}},
        "zantetsu_neural": {"scores": [], "errors": 0, "field_wins": {}},
        "zantetsu_char_cnn": {"scores": [], "errors": 0, "field_wins": {}},
    }

    fields = [
        "title",
        "group",
        "season",
        "episode",
        "resolution",
        "video_codec",
        "audio_codec",
        "source",
        "year",
        "crc32",
        "extension",
    ]
    for parser in results:
        results[parser]["field_wins"] = {f: 0 for f in fields}

    # Detailed results for JSON output
    detailed_results = []

    # Run benchmarks
    for i, case in enumerate(test_cases):
        filename = case["input"]
        expected = case["expected"]

        case_result = {"input": filename, "expected": expected, "parsers": {}}

        # PTT
        try:
            actual_ptt = parse_with_ptt(filename)
            score_ptt, field_results_ptt = calculate_total_score(expected, actual_ptt)
            results["ptt"]["scores"].append(score_ptt)
            case_result["parsers"]["ptt"] = {
                "result": actual_ptt,
                "score": score_ptt,
                "fields": field_results_ptt,
            }
            for f in fields:
                if field_results_ptt[f][0] == 1.0:
                    results["ptt"]["field_wins"][f] += 1
        except Exception as e:
            results["ptt"]["errors"] += 1
            case_result["parsers"]["ptt"] = {"error": str(e)}

        # RTN
        try:
            actual_rtn = parse_with_rtn(filename)
            score_rtn, field_results_rtn = calculate_total_score(expected, actual_rtn)
            results["rtn"]["scores"].append(score_rtn)
            case_result["parsers"]["rtn"] = {
                "result": actual_rtn,
                "score": score_rtn,
                "fields": field_results_rtn,
            }
            for f in fields:
                if field_results_rtn[f][0] == 1.0:
                    results["rtn"]["field_wins"][f] += 1
        except Exception as e:
            results["rtn"]["errors"] += 1
            case_result["parsers"]["rtn"] = {"error": str(e)}

        # Zantetsu Heuristic (if available)
        if zantetsu_available:
            try:
                actual_zh = parse_with_zantetsu(filename, zantetsu_binary, "heuristic")
                score_zh, field_results_zh = calculate_total_score(expected, actual_zh)
                results["zantetsu_heuristic"]["scores"].append(score_zh)
                case_result["parsers"]["zantetsu_heuristic"] = {
                    "result": actual_zh,
                    "score": score_zh,
                    "fields": field_results_zh,
                }
                for f in fields:
                    if field_results_zh[f][0] == 1.0:
                        results["zantetsu_heuristic"]["field_wins"][f] += 1
            except Exception as e:
                results["zantetsu_heuristic"]["errors"] += 1
                case_result["parsers"]["zantetsu_heuristic"] = {"error": str(e)}
        else:
            case_result["parsers"]["zantetsu_heuristic"] = {
                "skipped": "binary not available"
            }

        # Zantetsu Neural (if available)
        if zantetsu_available:
            try:
                actual_zn = parse_with_zantetsu(filename, zantetsu_binary, "neural")
                score_zn, field_results_zn = calculate_total_score(expected, actual_zn)
                results["zantetsu_neural"]["scores"].append(score_zn)
                case_result["parsers"]["zantetsu_neural"] = {
                    "result": actual_zn,
                    "score": score_zn,
                    "fields": field_results_zn,
                }
                for f in fields:
                    if field_results_zn[f][0] == 1.0:
                        results["zantetsu_neural"]["field_wins"][f] += 1
            except Exception as e:
                results["zantetsu_neural"]["errors"] += 1
                case_result["parsers"]["zantetsu_neural"] = {"error": str(e)}
        else:
            case_result["parsers"]["zantetsu_neural"] = {
                "skipped": "binary not available"
            }

        # Zantetsu Character CNN (if available)
        if char_cnn_available:
            try:
                actual_zc = parse_with_char_cnn(filename)
                score_zc, field_results_zc = calculate_total_score(expected, actual_zc)
                results["zantetsu_char_cnn"]["scores"].append(score_zc)
                case_result["parsers"]["zantetsu_char_cnn"] = {
                    "result": actual_zc,
                    "score": score_zc,
                    "fields": field_results_zc,
                }
                for f in fields:
                    if field_results_zc[f][0] == 1.0:
                        results["zantetsu_char_cnn"]["field_wins"][f] += 1
            except Exception as e:
                results["zantetsu_char_cnn"]["errors"] += 1
                case_result["parsers"]["zantetsu_char_cnn"] = {"error": str(e)}
        else:
            case_result["parsers"]["zantetsu_char_cnn"] = {
                "skipped": "char-CNN checkpoint not available"
            }

        # Zantetsu Auto (if available)
        if zantetsu_available:
            try:
                actual_za = parse_with_zantetsu(filename, zantetsu_binary, "auto")
                score_za, field_results_za = calculate_total_score(expected, actual_za)
                results["zantetsu_auto"]["scores"].append(score_za)
                case_result["parsers"]["zantetsu_auto"] = {
                    "result": actual_za,
                    "score": score_za,
                    "fields": field_results_za,
                }
                for f in fields:
                    if field_results_za[f][0] == 1.0:
                        results["zantetsu_auto"]["field_wins"][f] += 1
            except Exception as e:
                results["zantetsu_auto"]["errors"] += 1
                case_result["parsers"]["zantetsu_auto"] = {"error": str(e)}
        else:
            case_result["parsers"]["zantetsu_auto"] = {
                "skipped": "binary not available"
            }

        detailed_results.append(case_result)

        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(test_cases)}...")

    # Save detailed results to JSON
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_count": len(test_cases),
            "fields_evaluated": fields,
        },
        "summary": {},
        "field_analysis": {},
        "detailed_results": detailed_results,
    }

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    for name, data in results.items():
        scores = data["scores"]
        errors = data["errors"]
        field_wins = data["field_wins"]

        if scores:
            avg = sum(scores) / len(scores)
            min_s = min(scores)
            max_s = max(scores)

            sorted_scores = sorted(scores)
            p50 = sorted_scores[len(sorted_scores) // 2]
            p90 = sorted_scores[int(len(sorted_scores) * 0.9)]
            p95 = sorted_scores[int(len(sorted_scores) * 0.95)]

            perfect = sum(1 for s in scores if s >= 0.99)

            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  Samples:    {len(scores)}")
            print(f"  Errors:     {errors}")
            print(f"  Average:    {avg:.4f}")
            print(f"  Min:        {min_s:.4f}")
            print(f"  Max:        {max_s:.4f}")
            print(f"  P50:        {p50:.4f}")
            print(f"  P90:        {p90:.4f}")
            print(f"  P95:        {p95:.4f}")
            print(f"  Perfect:   {perfect} ({100 * perfect / len(scores):.1f}%)")

            output_data["summary"][name] = {
                "samples": len(scores),
                "errors": errors,
                "average": avg,
                "min": min_s,
                "max": max_s,
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "perfect": perfect,
                "perfect_pct": 100 * perfect / len(scores),
            }
        else:
            print(f"\n{name.upper().replace('_', ' ')}:")
            print(f"  No successful parses")
            output_data["summary"][name] = {"samples": 0, "errors": errors}

    # Field analysis
    print("\n" + "=" * 60)
    print("FIELD-BY-FIELD ANALYSIS")
    print("=" * 60)

    for field in fields:
        print(f"\n{field}:")
        for name, data in results.items():
            wins = data["field_wins"][field]
            pct = 100 * wins / len(test_cases) if test_cases else 0
            print(f"  {name:20s}: {wins:3d}/{len(test_cases)} ({pct:5.1f}%)")
            output_data["field_analysis"][field] = output_data["field_analysis"].get(
                field, {}
            )
            output_data["field_analysis"][field][name] = {"wins": wins, "pct": pct}

    # Save to JSON
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n\nDetailed results saved to: {OUTPUT_FILE}")

    # Winner determination
    print("\n" + "=" * 60)
    print("WINNER")
    print("=" * 60)

    best = max(
        results.items(),
        key=lambda x: (
            sum(x[1]["scores"]) / len(x[1]["scores"]) if x[1]["scores"] else 0
        ),
    )
    print(f"Best parser: {best[0].replace('_', ' ').title()}")


if __name__ == "__main__":
    main()
