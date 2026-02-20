use serde::{Deserialize, Serialize};
use std::io::{self, BufRead};

#[derive(Debug, Serialize, Deserialize)]
pub struct ParseOutput {
    pub input: String,
    pub title: Option<String>,
    pub group: Option<String>,
    pub season: Option<u32>,
    pub episode: Option<String>,
    pub resolution: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub source: Option<String>,
    pub year: Option<u16>,
    pub crc32: Option<String>,
    pub extension: Option<String>,
    pub version: Option<u8>,
    pub confidence: f32,
    pub mode: String,
    pub error: Option<String>,
}

fn episode_to_string(ep: &zantetsu_core::types::EpisodeSpec) -> String {
    use zantetsu_core::types::EpisodeSpec::*;
    match ep {
        Single(n) => format!("Single({})", n),
        Range(s, e) => format!("Range({},{})", s, e),
        Multi(v) => format!(
            "Multi({})",
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        ),
        Version { episode, version } => format!("Version({},v{})", episode, version),
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("heuristic");

    let stdin = io::stdin();

    if mode == "neural" {
        let mut parser =
            zantetsu_core::parser::NeuralParser::new().expect("Failed to create neural parser");
        parser.init_model().expect("Failed to load safetensors");

        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let result = parser.parse(line);

            let output = match result {
                Ok(r) => ParseOutput {
                    input: r.input.clone(),
                    title: r.title.clone(),
                    group: r.group.clone(),
                    season: r.season,
                    episode: r.episode.as_ref().map(episode_to_string),
                    resolution: r.resolution.as_ref().map(|x| format!("{:?}", x)),
                    video_codec: r.video_codec.as_ref().map(|x| format!("{:?}", x)),
                    audio_codec: r.audio_codec.as_ref().map(|x| format!("{:?}", x)),
                    source: r.source.as_ref().map(|x| format!("{:?}", x)),
                    year: r.year,
                    crc32: r.crc32.clone(),
                    extension: r.extension.clone(),
                    version: r.version,
                    confidence: r.confidence,
                    mode: "neural".to_string(),
                    error: None,
                },
                Err(e) => ParseOutput {
                    input: line.to_string(),
                    title: None,
                    group: None,
                    season: None,
                    episode: None,
                    resolution: None,
                    video_codec: None,
                    audio_codec: None,
                    source: None,
                    year: None,
                    crc32: None,
                    extension: None,
                    version: None,
                    confidence: 0.0,
                    mode: "neural".to_string(),
                    error: Some(e.to_string()),
                },
            };

            println!("{}", serde_json::to_string(&output).unwrap());
        }
    } else {
        let parser = zantetsu_core::parser::HeuristicParser::new()
            .expect("Failed to create heuristic parser");

        for line in stdin.lock().lines() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let result = parser.parse(line);

            let output = match result {
                Ok(r) => ParseOutput {
                    input: r.input.clone(),
                    title: r.title.clone(),
                    group: r.group.clone(),
                    season: r.season,
                    episode: r.episode.as_ref().map(episode_to_string),
                    resolution: r.resolution.as_ref().map(|x| format!("{:?}", x)),
                    video_codec: r.video_codec.as_ref().map(|x| format!("{:?}", x)),
                    audio_codec: r.audio_codec.as_ref().map(|x| format!("{:?}", x)),
                    source: r.source.as_ref().map(|x| format!("{:?}", x)),
                    year: r.year,
                    crc32: r.crc32.clone(),
                    extension: r.extension.clone(),
                    version: r.version,
                    confidence: r.confidence,
                    mode: "heuristic".to_string(),
                    error: None,
                },
                Err(e) => ParseOutput {
                    input: line.to_string(),
                    title: None,
                    group: None,
                    season: None,
                    episode: None,
                    resolution: None,
                    video_codec: None,
                    audio_codec: None,
                    source: None,
                    year: None,
                    crc32: None,
                    extension: None,
                    version: None,
                    confidence: 0.0,
                    mode: "heuristic".to_string(),
                    error: Some(e.to_string()),
                },
            };

            println!("{}", serde_json::to_string(&output).unwrap());
        }
    }

    Ok(())
}
