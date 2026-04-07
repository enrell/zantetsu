#!/usr/bin/env python3
"""Structural tokenizer and feature extraction for hybrid filename tagging."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


OPEN_BRACKETS = {"[": "]", "(": ")", "{": "}"}
CLOSE_BRACKETS = {value: key for key, value in OPEN_BRACKETS.items()}

RESOLUTION_TERMS = {
    "480p",
    "480i",
    "720p",
    "720i",
    "1080p",
    "1080i",
    "2160p",
    "2160i",
    "4k",
}

SOURCE_TERMS = {
    "bluray",
    "blu-ray",
    "bdrip",
    "bd",
    "web-dl",
    "webdl",
    "webrip",
    "web-rip",
    "dvd",
    "dvdrip",
    "dvd-rip",
    "hdtv",
    "remux",
}

CODEC_TERMS = {
    "x264",
    "x265",
    "h264",
    "h.264",
    "h265",
    "h.265",
    "hevc",
    "av1",
    "vp9",
}

AUDIO_TERMS = {
    "aac",
    "ac3",
    "dts",
    "eac3",
    "flac",
    "mp3",
    "opus",
    "truehd",
    "vorbis",
    "dual-audio",
    "multi-audio",
}

LANGUAGE_TERMS = {
    "eng",
    "english",
    "jpn",
    "japanese",
    "spa",
    "spanish",
    "ger",
    "german",
    "vostfr",
    "raw",
}

SUBTITLE_TERMS = {
    "sub",
    "subs",
    "subbed",
    "softsub",
    "hardsub",
    "multi-sub",
    "multi-subs",
}

QUALITY_TERMS = {
    "hd",
    "sd",
    "uhd",
    "hq",
    "hi10p",
    "10bit",
    "8bit",
}

CONTAINER_TERMS = {"mkv", "mp4", "avi", "m2ts", "ts"}

# Lexicon categories for token-level feature extraction.
# Index 0 means the token is not a known metadata keyword.
LEXICON_CATEGORIES = ["NONE", "RESOLUTION", "CODEC", "AUDIO", "CONTAINER", "SOURCE", "SUB", "LANG", "QUALITY"]

LEXICON_LOOKUP: dict[str, int] = {
    **{term: 1 for term in RESOLUTION_TERMS},
    **{term: 2 for term in CODEC_TERMS},
    **{term: 3 for term in AUDIO_TERMS},
    **{term: 4 for term in CONTAINER_TERMS},
    **{term: 5 for term in SOURCE_TERMS},
    **{term: 6 for term in SUBTITLE_TERMS},
    **{term: 7 for term in LANGUAGE_TERMS},
    **{term: 8 for term in QUALITY_TERMS},
}

COMPOSITE_TOKEN_RE = re.compile(
    r"""(?ix)
    \d{1,4}-\d{1,4}
    |s\d{1,2}e\d{1,4}(?:v\d+)?
    |\d{1,4}v\d+
    |web-dl|webdl|web-rip|webrip|blu-ray|bluray|bdrip|dvd-rip|dvdrip
    |dual-audio|multi-audio|multi-subs?|hi10p|10bit|8bit
    |h\.?26[45]|x26[45]|hevc|av1|vp9|flac|aac|opus|ac3|dts|mp3|truehd|eac3|vorbis
    |[A-Za-z0-9]+(?:'[A-Za-z0-9]+)*
    |[._+\-]
    """
)

SEASON_EPISODE_RE = re.compile(r"(?i)^(s\d{1,2})(e\d{1,4})(v\d+)?$")
VERSIONED_EPISODE_RE = re.compile(r"^(\d{1,4})(v\d+)$", re.IGNORECASE)
ROMAN_NUMERAL_RE = re.compile(r"^(?=[ivxlcdm]+$)m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)


@dataclass(frozen=True)
class HybridToken:
    text: str
    normalized: str
    start: int
    end: int
    index: int
    inside_brackets: bool
    bracket_kind: str | None
    shape: str

    def to_feature_dict(self) -> dict[str, Any]:
        return {
            "token": self.text,
            "lower": self.normalized,
            "shape": self.shape,
            "inside_brackets": self.inside_brackets,
            "bracket_kind": self.bracket_kind,
            "is_resolution": self.normalized in RESOLUTION_TERMS,
            "is_source": self.normalized in SOURCE_TERMS,
            "is_codec": self.normalized in CODEC_TERMS,
            "is_audio": self.normalized in AUDIO_TERMS,
            "is_language": self.normalized in LANGUAGE_TERMS,
            "is_subtitle": self.normalized in SUBTITLE_TERMS,
            "is_quality": self.normalized in QUALITY_TERMS,
            "is_container": self.normalized in CONTAINER_TERMS,
            "contains_dash": "-" in self.text,
            "is_numeric": self.normalized.isdigit(),
            "is_alnum": self.normalized.isalnum(),
            "is_roman_numeral": is_roman_numeral(self.normalized),
            "prefix2": self.normalized[:2],
            "prefix4": self.normalized[:4],
            "suffix2": self.normalized[-2:],
            "suffix4": self.normalized[-4:],
        }


def normalize_token(token: str) -> str:
    return token.strip().lower()


def is_roman_numeral(text: str) -> bool:
    return bool(text) and bool(ROMAN_NUMERAL_RE.fullmatch(text))


def token_shape(token: str) -> str:
    if not token:
        return "EMPTY"
    if token.isdigit():
        return "NUM"
    if re.fullmatch(r"\d{1,4}-\d{1,4}", token):
        return "RANGE"
    if is_roman_numeral(token):
        return "ROMAN"
    if token.isalpha() and token.isupper():
        return "ALL_CAPS"
    if token[:1].isupper() and token[1:].islower():
        return "TITLE"
    if "-" in token:
        return "DASHED"
    if any(character.isdigit() for character in token) and any(
        character.isalpha() for character in token
    ):
        return "ALNUM"
    return "MIXED"


def _position_bucket(index: int, total: int) -> str:
    if total <= 1:
        return "singleton"
    if index <= 1:
        return "start"
    if index >= total - 2:
        return "end"
    return "middle"


def _make_token(
    text: str,
    start: int,
    end: int,
    index: int,
    bracket_stack: list[str],
) -> HybridToken:
    normalized = normalize_token(text)
    bracket_kind = bracket_stack[-1] if bracket_stack else None
    return HybridToken(
        text=text,
        normalized=normalized,
        start=start,
        end=end,
        index=index,
        inside_brackets=bool(bracket_stack),
        bracket_kind=bracket_kind,
        shape=token_shape(text),
    )


def _split_composite_token(text: str, start: int, end: int) -> list[tuple[str, int, int]]:
    raw = text[start:end]
    season_episode = SEASON_EPISODE_RE.fullmatch(raw)
    if season_episode:
        season_text, episode_text, version_text = season_episode.groups()
        cursor = start
        pieces = []
        pieces.append((season_text, cursor, cursor + len(season_text)))
        cursor += len(season_text)
        pieces.append((episode_text, cursor, cursor + len(episode_text)))
        cursor += len(episode_text)
        if version_text:
            pieces.append((version_text, cursor, cursor + len(version_text)))
        return pieces

    versioned_episode = VERSIONED_EPISODE_RE.fullmatch(raw)
    if versioned_episode:
        episode_text, version_text = versioned_episode.groups()
        cursor = start
        return [
            (episode_text, cursor, cursor + len(episode_text)),
            (
                version_text,
                cursor + len(episode_text),
                cursor + len(episode_text) + len(version_text),
            ),
        ]

    return [(raw, start, end)]


def tokenize_filename(text: str) -> list[HybridToken]:
    tokens: list[HybridToken] = []
    bracket_stack: list[str] = []
    cursor = 0
    index = 0

    while cursor < len(text):
        character = text[cursor]

        if character.isspace():
            cursor += 1
            continue

        if character in OPEN_BRACKETS:
            tokens.append(_make_token(character, cursor, cursor + 1, index, bracket_stack))
            bracket_stack.append(character)
            cursor += 1
            index += 1
            continue

        if character in CLOSE_BRACKETS:
            tokens.append(_make_token(character, cursor, cursor + 1, index, bracket_stack))
            opener = CLOSE_BRACKETS[character]
            if bracket_stack and bracket_stack[-1] == opener:
                bracket_stack.pop()
            cursor += 1
            index += 1
            continue

        match = COMPOSITE_TOKEN_RE.match(text, cursor)
        if match is None:
            pieces = [(character, cursor, cursor + 1)]
            cursor += 1
        else:
            cursor = match.end()
            pieces = _split_composite_token(text, match.start(), match.end())

        for piece_text, piece_start, piece_end in pieces:
            if not piece_text.strip():
                continue
            tokens.append(
                _make_token(piece_text, piece_start, piece_end, index, bracket_stack)
            )
            index += 1

    return tokens


def build_context_features(tokens: list[HybridToken]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    total = len(tokens)

    for index, token in enumerate(tokens):
        token_features = token.to_feature_dict()
        token_features.update(
            {
                "previous_token": tokens[index - 1].normalized if index > 0 else "<BOS>",
                "next_token": tokens[index + 1].normalized if index + 1 < total else "<EOS>",
                "position_bucket": _position_bucket(index, total),
            }
        )
        features.append(token_features)

    return features


def build_context_features_from_texts(token_texts: list[str]) -> list[dict[str, Any]]:
    """Rebuild token features from already tokenized text without re-tokenizing the filename."""
    rebuilt_tokens: list[HybridToken] = []
    bracket_stack: list[str] = []
    cursor = 0

    for index, token_text in enumerate(token_texts):
        rebuilt_tokens.append(
            HybridToken(
                text=token_text,
                normalized=normalize_token(token_text),
                start=cursor,
                end=cursor + len(token_text),
                index=index,
                inside_brackets=bool(bracket_stack),
                bracket_kind=bracket_stack[-1] if bracket_stack else None,
                shape=token_shape(token_text),
            )
        )

        if token_text in OPEN_BRACKETS:
            bracket_stack.append(token_text)
        elif token_text in CLOSE_BRACKETS:
            opener = CLOSE_BRACKETS[token_text]
            if bracket_stack and bracket_stack[-1] == opener:
                bracket_stack.pop()

        cursor += len(token_text)

    return build_context_features(rebuilt_tokens)
