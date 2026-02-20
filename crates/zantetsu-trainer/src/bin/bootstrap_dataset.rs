use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::Serialize;
use zantetsu_core::parser::tokenizer::Token;
use zantetsu_core::parser::{BioTag, HeuristicParser, Tokenizer};
use zantetsu_core::ParseResult;

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

// Very simple alignment for weak supervision
fn align_tags(tokens: &[Token], result: &ParseResult) -> Vec<BioTag> {
    let mut tags = vec![BioTag::Outside; tokens.len()];
    
    // Convert extracted fields to normalization space
    let title = result.title.as_deref().unwrap_or("").to_lowercase();
    let group = result.group.as_deref().unwrap_or("").to_lowercase();

    let mut in_group = false;
    let mut in_title = false;

    for (i, token) in tokens.iter().enumerate() {
        let t_text = &token.text;

        // Group matching (usually exact or bounds)
        if !group.is_empty() && group.contains(t_text) && i < 5 {
            tags[i] = if in_group { BioTag::InsideGroup } else { BioTag::BeginGroup };
            in_group = true;
            continue;
        } else {
            in_group = false;
        }

        // Title matching
        if !title.is_empty() && title.contains(t_text) {
            tags[i] = if in_title { BioTag::InsideTitle } else { BioTag::BeginTitle };
            in_title = true;
            continue;
        } else {
            in_title = false;
        }

        // Resolution matching
        if let Some(res) = &result.resolution {
            let res_str = format!("{:?}", res).to_lowercase();
            if res_str.contains(t_text) || (t_text == "1080p") || (t_text == "720p") || (t_text == "2160p") || (t_text == "480p") {
                tags[i] = BioTag::Resolution;
                continue;
            }
        }

        // Catch-all mapping could go here. For SFT, partial labels are okay
        // if we use a CrossEntropy loss with ignore_index for unknown stuff,
        // but for now we emit what we can.
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

    for line in reader.lines() {
        let line = line?;
        let cleaned = strip_prefix(&line);
        if cleaned.is_empty() {
            continue;
        }

        if let Ok(result) = parser.parse(&cleaned) {
            // Only output high confidence parses
            if result.confidence >= 0.6 {
                let tokens = tokenizer.tokenize(&cleaned);
                let tags = align_tags(&tokens, &result);

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
