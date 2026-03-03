use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::collections::HashSet;

use serde::Serialize;
use zantetsu_core::parser::tokenizer::Token;
use zantetsu_core::parser::{BioTag, HeuristicParser, Tokenizer};
use zantetsu_core::{ParseResult, Resolution};

#[derive(Serialize)]
struct BioSample {
    tokens: Vec<String>,
    ner_tags: Vec<String>,
}

fn strip_prefix(line: &str) -> String {
    let mut cleaned = line.trim();
    if cleaned.starts_with('(') {
        if let Some(pos) = cleaned.find(')') {
            cleaned = &cleaned[pos + 1..];
        }
    }
    cleaned.trim().to_string()
}

fn normalize_words(text: &str) -> HashSet<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

fn mark_sequence(tags: &mut [BioTag], idx: usize, begin: BioTag, inside: BioTag, active: &mut bool) {
    tags[idx] = if *active { inside } else { begin };
    *active = true;
}

// Very simple alignment for weak supervision
fn align_tags(tokens: &[Token], result: &ParseResult) -> Vec<BioTag> {
    let mut tags = vec![BioTag::Outside; tokens.len()];

    let title_words = result
        .title
        .as_deref()
        .map(normalize_words)
        .unwrap_or_default();
    let group_words = result
        .group
        .as_deref()
        .map(normalize_words)
        .unwrap_or_default();

    let episode_words: HashSet<String> = result
        .episode
        .as_ref()
        .map(|ep| ep.to_string().to_lowercase())
        .into_iter()
        .flat_map(|s| normalize_words(&s))
        .collect();
    let season_words: HashSet<String> = result
        .season
        .map(|s| s.to_string())
        .into_iter()
        .collect();
    let year_words: HashSet<String> = result
        .year
        .map(|y| y.to_string())
        .into_iter()
        .collect();
    let crc_words: HashSet<String> = result
        .crc32
        .as_deref()
        .map(|s| s.to_lowercase())
        .into_iter()
        .collect();
    let extension_words: HashSet<String> = result
        .extension
        .as_deref()
        .map(|s| s.to_lowercase())
        .into_iter()
        .collect();
    let version_words: HashSet<String> = result
        .version
        .map(|v| format!("v{}", v))
        .into_iter()
        .collect();

    let resolution_words: HashSet<String> = match result.resolution {
        Some(Resolution::UHD2160) => ["2160", "2160p", "4k"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        Some(Resolution::FHD1080) => ["1080", "1080p"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        Some(Resolution::HD720) => ["720", "720p"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        Some(Resolution::SD480) => ["480", "480p", "576", "576p"]
            .into_iter()
            .map(str::to_string)
            .collect(),
        None => HashSet::new(),
    };

    let mut in_group = false;
    let mut in_title = false;
    let mut in_episode = false;
    let mut in_season = false;

    for (i, token) in tokens.iter().enumerate() {
        let t_text = token.text.as_str();

        if t_text.is_empty() {
            in_group = false;
            in_title = false;
            in_episode = false;
            in_season = false;
            continue;
        }

        // Group matching (exact token membership, near head)
        if !group_words.is_empty() && i < 6 && group_words.contains(t_text) {
            mark_sequence(&mut tags, i, BioTag::BeginGroup, BioTag::InsideGroup, &mut in_group);
            in_title = false;
            continue;
        } else {
            in_group = false;
        }

        // Title matching
        if !title_words.is_empty() && title_words.contains(t_text) {
            mark_sequence(&mut tags, i, BioTag::BeginTitle, BioTag::InsideTitle, &mut in_title);
            in_episode = false;
            in_season = false;
            continue;
        } else {
            in_title = false;
        }

        if !episode_words.is_empty() && episode_words.contains(t_text) {
            mark_sequence(
                &mut tags,
                i,
                BioTag::BeginEpisode,
                BioTag::InsideEpisode,
                &mut in_episode,
            );
            continue;
        } else {
            in_episode = false;
        }

        if !season_words.is_empty() && season_words.contains(t_text) {
            mark_sequence(
                &mut tags,
                i,
                BioTag::BeginSeason,
                BioTag::InsideSeason,
                &mut in_season,
            );
            continue;
        } else {
            in_season = false;
        }

        // Resolution matching
        if resolution_words.contains(t_text) {
            tags[i] = BioTag::Resolution;
            continue;
        }

        if year_words.contains(t_text) {
            tags[i] = BioTag::Year;
            continue;
        }

        if crc_words.contains(t_text) {
            tags[i] = BioTag::Crc32;
            continue;
        }

        if extension_words.contains(t_text) {
            tags[i] = BioTag::Extension;
            continue;
        }

        if version_words.contains(t_text) {
            tags[i] = BioTag::Version;
            continue;
        }
    }

    tags
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let input_path = "data/training/nyaa_titles_5000_raw.txt";
    let output_path = "data/training/silver_dataset.jsonl";

    if !Path::new(input_path).exists() {
        println!("Input file not found at {}", input_path);
        return Ok(());
    }

    let file = File::open(input_path)?;
    let reader = BufReader::new(file);

    let mut out_file = File::create(output_path)?;

    let parser = HeuristicParser::new()?;
    let tokenizer = Tokenizer::new();

    let mut success_count = 0;
    let mut seen_inputs: HashSet<String> = HashSet::new();

    const MIN_CONFIDENCE: f32 = 0.20;

    for line in reader.lines() {
        let line = line?;
        let cleaned = strip_prefix(&line);
        if cleaned.is_empty() {
            continue;
        }

        if !seen_inputs.insert(cleaned.clone()) {
            continue;
        }

        if let Ok(result) = parser.parse(&cleaned) {
            if result.confidence >= MIN_CONFIDENCE
                && (result.title.is_some() || result.group.is_some())
            {
                let tokens: Vec<Token> = tokenizer
                    .tokenize(&cleaned)
                    .into_iter()
                    .filter(|t| !t.text.is_empty())
                    .collect();
                if tokens.len() < 3 {
                    continue;
                }

                let tags = align_tags(&tokens, &result);
                if tags.iter().all(|t| *t == BioTag::Outside) {
                    continue;
                }

                let sample = BioSample {
                    tokens: tokens.into_iter().map(|t| t.text).collect(),
                    ner_tags: tags.into_iter().map(|t| t.to_string()).collect(),
                };

                let json = serde_json::to_string(&sample)?;
                writeln!(out_file, "{}", json)?;
                success_count += 1;
            }
        }
    }

    println!("Successfully bootstrapped {} samples to {}", success_count, output_path);
    Ok(())
}
