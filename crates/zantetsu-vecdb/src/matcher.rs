use crate::error::{MatchResult, MatcherError};
use dirs::data_dir;
use flate2::read::GzDecoder;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use strsim::jaro_winkler;
use tracing::debug;

const DEFAULT_TIMEOUT_SECS: u64 = 10;

/// Which backend produced a canonical title match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchProvider {
    /// Match derived from the local Kitsu SQL dump.
    KitsuDump,
    /// Match derived from a remote GraphQL endpoint.
    RemoteEndpoint,
}

/// External and internal identifiers for a matched anime.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimeIds {
    /// Kitsu anime id.
    pub kitsu: Option<u32>,
    /// AniList-compatible media id.
    pub anilist: Option<u32>,
    /// MyAnimeList anime id.
    pub mal: Option<u32>,
}

/// The best-scoring match for a query title.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimeTitleMatch {
    /// Which backend produced this result.
    pub provider: MatchProvider,
    /// Canonical title chosen for the anime.
    pub canonical_title: String,
    /// The specific alias that scored highest for the query.
    pub matched_title: String,
    /// Similarity score in the range `[0.0, 1.0]`.
    pub score: f32,
    /// Known IDs for the matched anime.
    pub ids: AnimeIds,
    /// Known aliases for the anime, canonical title first.
    pub titles: Vec<String>,
}

/// Backend choice for canonical title matching.
#[derive(Debug, Clone)]
pub enum MatchSource {
    /// Use a local Kitsu SQL dump located at `dump_path`.
    KitsuDump {
        /// Path to either the dump directory or a specific `latest.sql` / `latest.sql.gz` file.
        dump_path: PathBuf,
    },
    /// Use a remote GraphQL endpoint.
    RemoteEndpoint {
        /// Remote endpoint URL.
        endpoint: String,
    },
}

impl MatchSource {
    /// Use the local Kitsu dump at the provided path.
    #[must_use]
    pub fn kitsu_dump(dump_path: impl Into<PathBuf>) -> Self {
        Self::KitsuDump {
            dump_path: dump_path.into(),
        }
    }

    /// Use a remote GraphQL endpoint.
    #[must_use]
    pub fn remote_endpoint(endpoint: impl Into<String>) -> Self {
        Self::RemoteEndpoint {
            endpoint: endpoint.into(),
        }
    }
}

/// Returns the default dump directory used by `kitsu-sync`.
#[must_use]
pub fn default_kitsu_dump_dir() -> PathBuf {
    data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("zantetsu")
        .join("kitsu-dumps")
}

/// Canonical title matcher backed by either a local Kitsu dump or a remote GraphQL endpoint.
pub struct TitleMatcher {
    backend: MatcherBackend,
}

enum MatcherBackend {
    Kitsu(KitsuTitleMatcher),
    RemoteEndpoint(RemoteTitleMatcher),
}

impl TitleMatcher {
    /// Create a matcher from the selected source.
    pub fn new(source: MatchSource) -> MatchResult<Self> {
        let backend = match source {
            MatchSource::KitsuDump { dump_path } => {
                MatcherBackend::Kitsu(KitsuTitleMatcher::from_dump(dump_path)?)
            }
            MatchSource::RemoteEndpoint { endpoint } => {
                MatcherBackend::RemoteEndpoint(RemoteTitleMatcher::new(endpoint)?)
            }
        };

        Ok(Self { backend })
    }

    /// Create a matcher from a local Kitsu dump path.
    pub fn from_kitsu_dump(dump_path: impl Into<PathBuf>) -> MatchResult<Self> {
        Self::new(MatchSource::kitsu_dump(dump_path))
    }

    /// Create a matcher using a remote GraphQL endpoint.
    pub fn from_remote_endpoint(endpoint: impl Into<String>) -> MatchResult<Self> {
        Self::new(MatchSource::remote_endpoint(endpoint))
    }

    /// Return the best available match for the provided title.
    pub fn match_title(&self, title: &str) -> MatchResult<Option<AnimeTitleMatch>> {
        Ok(self.search_titles(title, 1)?.into_iter().next())
    }

    /// Search for the best matches for the provided title.
    pub fn search_titles(&self, title: &str, limit: usize) -> MatchResult<Vec<AnimeTitleMatch>> {
        let query = normalize_title(title);
        if query.is_empty() {
            return Err(MatcherError::EmptyQuery);
        }

        let limit = limit.max(1);
        match &self.backend {
            MatcherBackend::Kitsu(matcher) => Ok(score_entries(
                MatchProvider::KitsuDump,
                &matcher.entries,
                title,
                &query,
                limit,
            )),
            MatcherBackend::RemoteEndpoint(matcher) => matcher.search_titles(title, &query, limit),
        }
    }
}

struct KitsuTitleMatcher {
    entries: Vec<CatalogEntry>,
}

impl KitsuTitleMatcher {
    fn from_dump(dump_path: PathBuf) -> MatchResult<Self> {
        let resolved = resolve_dump_path(&dump_path)?;
        debug!("loading Kitsu dump from {}", resolved.display());
        let reader = open_dump_reader(&resolved)?;
        let entries = parse_kitsu_dump(reader)?;
        Ok(Self { entries })
    }
}

struct RemoteTitleMatcher {
    client: Client,
    endpoint: String,
}

impl RemoteTitleMatcher {
    fn new(endpoint: String) -> MatchResult<Self> {
        let endpoint = endpoint.trim().to_string();
        if endpoint.is_empty() {
            return Err(MatcherError::InvalidResponse(
                "remote endpoint URL cannot be empty".into(),
            ));
        }

        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .user_agent(format!("zantetsu/{}", env!("CARGO_PKG_VERSION")))
            .build()?;
        Ok(Self { client, endpoint })
    }

    fn search_titles(
        &self,
        raw_query: &str,
        normalized_query: &str,
        limit: usize,
    ) -> MatchResult<Vec<AnimeTitleMatch>> {
        let payload = RemoteGraphQlRequest {
            query: REMOTE_GRAPHQL_QUERY,
            variables: RemoteGraphQlVariables {
                search: raw_query,
                per_page: limit.max(5) as i64,
            },
        };

        let response = self
            .client
            .post(&self.endpoint)
            .json(&payload)
            .send()?
            .error_for_status()?;

        let envelope: RemoteGraphQlEnvelope = response.json()?;
        if let Some(errors) = envelope.errors {
            let message = errors
                .into_iter()
                .map(|error| error.message)
                .collect::<Vec<_>>()
                .join("; ");
            return Err(MatcherError::GraphQl(message));
        }

        let media = envelope
            .data
            .ok_or_else(|| MatcherError::InvalidResponse("missing data".into()))?
            .page
            .ok_or_else(|| MatcherError::InvalidResponse("missing page".into()))?
            .media;

        let entries = media
            .into_iter()
            .map(CatalogEntry::from_remote_media)
            .collect::<Vec<_>>();

        Ok(score_entries(
            MatchProvider::RemoteEndpoint,
            &entries,
            raw_query,
            normalized_query,
            limit,
        ))
    }
}

#[derive(Debug, Clone)]
struct CatalogEntry {
    canonical_title: String,
    titles: Vec<String>,
    normalized_titles: Vec<String>,
    ids: AnimeIds,
}

impl CatalogEntry {
    fn new(canonical_title: String, titles: Vec<String>, ids: AnimeIds) -> Option<Self> {
        let titles = dedupe_titles(&canonical_title, titles);
        if titles.is_empty() {
            return None;
        }

        let canonical_title = titles
            .first()
            .cloned()
            .unwrap_or_else(|| canonical_title.trim().to_string());

        let normalized_titles = titles.iter().map(|title| normalize_title(title)).collect();
        Some(Self {
            canonical_title,
            titles,
            normalized_titles,
            ids,
        })
    }

    fn from_remote_media(media: RemoteMedia) -> Self {
        let mut titles = Vec::new();
        if let Some(title) = media.title.user_preferred.clone() {
            titles.push(title);
        }
        if let Some(title) = media.title.romaji.clone() {
            titles.push(title);
        }
        if let Some(title) = media.title.english.clone() {
            titles.push(title);
        }
        if let Some(title) = media.title.native.clone() {
            titles.push(title);
        }
        titles.extend(media.synonyms.unwrap_or_default());

        let canonical_title = media
            .title
            .user_preferred
            .clone()
            .or(media.title.romaji.clone())
            .or(media.title.english.clone())
            .or(media.title.native.clone())
            .unwrap_or_else(|| format!("Remote {}", media.id));

        Self::new(
            canonical_title,
            titles,
            AnimeIds {
                kitsu: None,
                anilist: Some(media.id),
                mal: media.id_mal,
            },
        )
        .expect("remote media should always produce at least one title")
    }
}

#[derive(Debug, Default)]
struct RawAnimeRecord {
    canonical_title: Option<String>,
    titles: Vec<String>,
    slug: Option<String>,
}

#[derive(Debug, Clone)]
enum DumpSection {
    None,
    Anime(AnimeColumns),
    Mappings(MappingColumns),
}

#[derive(Debug, Clone)]
struct AnimeColumns {
    id: usize,
    canonical_title: Option<usize>,
    titles: Option<usize>,
    abbreviated_titles: Option<usize>,
    slug: Option<usize>,
}

impl AnimeColumns {
    fn try_from_columns(columns: &[String]) -> MatchResult<Self> {
        let id = find_column_index(columns, "id")
            .ok_or_else(|| MatcherError::InvalidDump("anime COPY is missing id column".into()))?;
        Ok(Self {
            id,
            canonical_title: find_column_index(columns, "canonical_title"),
            titles: find_column_index(columns, "titles"),
            abbreviated_titles: find_column_index(columns, "abbreviated_titles"),
            slug: find_column_index(columns, "slug"),
        })
    }
}

#[derive(Debug, Clone)]
struct MappingColumns {
    item_id: usize,
    item_type: Option<usize>,
    external_site: usize,
    external_id: usize,
}

impl MappingColumns {
    fn try_from_columns(columns: &[String]) -> MatchResult<Self> {
        let item_id = find_column_index(columns, "item_id").ok_or_else(|| {
            MatcherError::InvalidDump("mappings COPY is missing item_id column".into())
        })?;
        let external_site = find_column_index(columns, "external_site").ok_or_else(|| {
            MatcherError::InvalidDump("mappings COPY is missing external_site column".into())
        })?;
        let external_id = find_column_index(columns, "external_id").ok_or_else(|| {
            MatcherError::InvalidDump("mappings COPY is missing external_id column".into())
        })?;

        Ok(Self {
            item_id,
            item_type: find_column_index(columns, "item_type"),
            external_site,
            external_id,
        })
    }
}

fn score_entries(
    provider: MatchProvider,
    entries: &[CatalogEntry],
    raw_query: &str,
    normalized_query: &str,
    limit: usize,
) -> Vec<AnimeTitleMatch> {
    let mut scored = entries
        .iter()
        .filter_map(|entry| score_entry(provider, entry, raw_query, normalized_query))
        .collect::<Vec<_>>();

    scored.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.canonical_title.cmp(&right.canonical_title))
    });
    scored.truncate(limit);
    scored
}

fn score_entry(
    provider: MatchProvider,
    entry: &CatalogEntry,
    _raw_query: &str,
    normalized_query: &str,
) -> Option<AnimeTitleMatch> {
    let mut best_title = None;
    let mut best_score = 0.0_f32;

    for (index, normalized_title) in entry.normalized_titles.iter().enumerate() {
        let score = similarity_score(normalized_query, normalized_title);
        if score > best_score {
            best_score = score;
            best_title = entry.titles.get(index).cloned();
        }
    }

    best_title.map(|matched_title| AnimeTitleMatch {
        provider,
        canonical_title: entry.canonical_title.clone(),
        matched_title,
        score: best_score,
        ids: entry.ids.clone(),
        titles: entry.titles.clone(),
    })
}

fn similarity_score(query: &str, candidate: &str) -> f32 {
    if query.is_empty() || candidate.is_empty() {
        return 0.0;
    }
    if query == candidate {
        return 1.0;
    }

    let jaro = jaro_winkler(query, candidate) as f32;
    let token_overlap = token_overlap_score(query, candidate);
    let contains = if query.contains(candidate) || candidate.contains(query) {
        1.0
    } else {
        0.0
    };
    let prefix = if query.starts_with(candidate) || candidate.starts_with(query) {
        1.0
    } else {
        0.0
    };
    let len_ratio = query.len().min(candidate.len()) as f32 / query.len().max(candidate.len()) as f32;

    let mut score =
        0.55 * jaro + 0.20 * token_overlap + 0.15 * contains + 0.10 * len_ratio + 0.05 * prefix;
    if token_overlap == 0.0 && contains == 0.0 && jaro < 0.90 {
        score *= 0.75;
    }

    score.clamp(0.0, 1.0)
}

fn token_overlap_score(left: &str, right: &str) -> f32 {
    let left_tokens = left.split_whitespace().collect::<HashSet<_>>();
    let right_tokens = right.split_whitespace().collect::<HashSet<_>>();
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }

    let shared = left_tokens.intersection(&right_tokens).count() as f32;
    (2.0 * shared) / (left_tokens.len() as f32 + right_tokens.len() as f32)
}

fn normalize_title(title: &str) -> String {
    let mut normalized = String::with_capacity(title.len());
    let mut last_was_space = true;

    for ch in title.chars() {
        let mapped = match ch {
            '×' | '✕' | '✖' => 'x',
            '&' => ' ',
            _ => ch.to_ascii_lowercase(),
        };

        if mapped.is_alphanumeric() {
            normalized.push(mapped);
            last_was_space = false;
        } else if mapped.is_whitespace() || matches!(mapped, '-' | '_' | '.' | ':' | '/' | '\\') {
            if !last_was_space {
                normalized.push(' ');
                last_was_space = true;
            }
        }
    }

    normalized.trim().to_string()
}

fn dedupe_titles(canonical_title: &str, titles: Vec<String>) -> Vec<String> {
    let mut deduped = Vec::new();
    let mut seen = HashSet::new();

    let mut push_title = |title: String| {
        let cleaned = title.trim();
        if cleaned.is_empty() {
            return;
        }
        let key = normalize_title(cleaned);
        if key.is_empty() || !seen.insert(key) {
            return;
        }
        deduped.push(cleaned.to_string());
    };

    push_title(canonical_title.to_string());
    for title in titles {
        push_title(title);
    }

    deduped
}

fn resolve_dump_path(path: &Path) -> MatchResult<PathBuf> {
    if path.is_file() {
        return Ok(path.to_path_buf());
    }

    if path.is_dir() {
        let sql = path.join("latest.sql");
        if sql.is_file() {
            return Ok(sql);
        }

        let gzip = path.join("latest.sql.gz");
        if gzip.is_file() {
            return Ok(gzip);
        }

        return Err(MatcherError::InvalidDumpPath(format!(
            "directory {} does not contain latest.sql or latest.sql.gz",
            path.display()
        )));
    }

    Err(MatcherError::InvalidDumpPath(format!(
        "{} does not exist",
        path.display()
    )))
}

fn open_dump_reader(path: &Path) -> MatchResult<Box<dyn BufRead>> {
    let file = File::open(path)?;
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        return Ok(Box::new(BufReader::new(GzDecoder::new(file))));
    }

    Ok(Box::new(BufReader::new(file)))
}

fn parse_kitsu_dump<R: BufRead>(reader: R) -> MatchResult<Vec<CatalogEntry>> {
    let mut anime_by_id = HashMap::<u32, RawAnimeRecord>::new();
    let mut ids_by_anime = HashMap::<u32, AnimeIds>::new();
    let mut section = DumpSection::None;

    for line in reader.lines() {
        let line = line?;
        if let Some((table, columns)) = parse_copy_statement(&line) {
            section = match table.as_str() {
                "anime" => DumpSection::Anime(AnimeColumns::try_from_columns(&columns)?),
                "mappings" => DumpSection::Mappings(MappingColumns::try_from_columns(&columns)?),
                _ => DumpSection::None,
            };
            continue;
        }

        if line == "\\." {
            section = DumpSection::None;
            continue;
        }

        match &section {
            DumpSection::Anime(columns) => parse_anime_row(&line, columns, &mut anime_by_id)?,
            DumpSection::Mappings(columns) => {
                parse_mapping_row(&line, columns, &mut ids_by_anime)?;
            }
            DumpSection::None => {}
        }
    }

    if anime_by_id.is_empty() {
        return Err(MatcherError::InvalidDump(
            "anime COPY section was not found".into(),
        ));
    }

    let mut entries = anime_by_id
        .into_iter()
        .filter_map(|(kitsu_id, record)| {
            let fallback_slug = record.slug.as_deref().map(slug_to_title);
            let canonical_title = record
                .canonical_title
                .clone()
                .or_else(|| record.titles.first().cloned())
                .or(fallback_slug)?;

            let mut titles = record.titles;
            if let Some(slug) = record.slug {
                titles.push(slug_to_title(&slug));
            }

            let mut ids = ids_by_anime.remove(&kitsu_id).unwrap_or_default();
            ids.kitsu = Some(kitsu_id);
            CatalogEntry::new(canonical_title, titles, ids)
        })
        .collect::<Vec<_>>();

    entries.sort_by(|left, right| left.canonical_title.cmp(&right.canonical_title));
    Ok(entries)
}

fn parse_copy_statement(line: &str) -> Option<(String, Vec<String>)> {
    let rest = line.strip_prefix("COPY ")?;
    let rest = rest.strip_suffix(" FROM stdin;")?;
    let open = rest.find('(')?;
    let close = rest.rfind(')')?;
    if close <= open {
        return None;
    }

    let table = normalize_identifier(&rest[..open]);
    let columns = rest[open + 1..close]
        .split(',')
        .map(normalize_identifier)
        .collect::<Vec<_>>();
    Some((table, columns))
}

fn parse_anime_row(
    line: &str,
    columns: &AnimeColumns,
    anime_by_id: &mut HashMap<u32, RawAnimeRecord>,
) -> MatchResult<()> {
    let fields = split_copy_row(line);
    let id = field_value(&fields, columns.id)
        .ok_or_else(|| MatcherError::InvalidDump("anime row is missing id".into()))?
        .parse::<u32>()
        .map_err(|_| MatcherError::InvalidDump("anime id is not numeric".into()))?;

    let record = anime_by_id.entry(id).or_default();

    if let Some(index) = columns.canonical_title {
        if let Some(value) = field_value(&fields, index) {
            record.canonical_title = Some(value.to_string());
        }
    }

    if let Some(index) = columns.titles {
        if let Some(value) = field_value(&fields, index) {
            record.titles.extend(parse_json_titles(value));
        }
    }

    if let Some(index) = columns.abbreviated_titles {
        if let Some(value) = field_value(&fields, index) {
            record.titles.extend(parse_json_titles(value));
        }
    }

    if let Some(index) = columns.slug {
        if let Some(value) = field_value(&fields, index) {
            record.slug = Some(value.to_string());
        }
    }

    Ok(())
}

fn parse_mapping_row(
    line: &str,
    columns: &MappingColumns,
    ids_by_anime: &mut HashMap<u32, AnimeIds>,
) -> MatchResult<()> {
    let fields = split_copy_row(line);
    if let Some(index) = columns.item_type {
        if !field_value(&fields, index)
            .is_some_and(|value| value.eq_ignore_ascii_case("anime"))
        {
            return Ok(());
        }
    }

    let item_id = match field_value(&fields, columns.item_id)
        .and_then(|value| value.parse::<u32>().ok())
    {
        Some(id) => id,
        None => return Ok(()),
    };

    let external_site = match field_value(&fields, columns.external_site) {
        Some(site) => site.to_ascii_lowercase(),
        None => return Ok(()),
    };
    let external_id = match field_value(&fields, columns.external_id)
        .and_then(|value| value.parse::<u32>().ok())
    {
        Some(id) => id,
        None => return Ok(()),
    };

    let ids = ids_by_anime.entry(item_id).or_default();
    if external_site.contains("anilist") {
        ids.anilist = Some(external_id);
    } else if external_site.contains("myanimelist") || external_site == "mal" {
        ids.mal = Some(external_id);
    }

    Ok(())
}

fn split_copy_row(line: &str) -> Vec<Option<String>> {
    line.split('\t').map(parse_copy_field).collect()
}

fn parse_copy_field(field: &str) -> Option<String> {
    if field == "\\N" {
        return None;
    }

    let mut output = String::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('t') => output.push('\t'),
                Some('n') => output.push('\n'),
                Some('r') => output.push('\r'),
                Some('b') => output.push('\u{0008}'),
                Some('f') => output.push('\u{000C}'),
                Some('\\') => output.push('\\'),
                Some(other) => output.push(other),
                None => output.push('\\'),
            }
        } else {
            output.push(ch);
        }
    }

    Some(output)
}

fn field_value(fields: &[Option<String>], index: usize) -> Option<&str> {
    fields.get(index).and_then(|field| field.as_deref())
}

fn normalize_identifier(identifier: &str) -> String {
    identifier
        .split('.')
        .next_back()
        .unwrap_or(identifier)
        .trim()
        .trim_matches('"')
        .to_ascii_lowercase()
}

fn find_column_index(columns: &[String], target: &str) -> Option<usize> {
    columns
        .iter()
        .position(|column| column.eq_ignore_ascii_case(target))
}

fn parse_json_titles(raw: &str) -> Vec<String> {
    let parsed = match serde_json::from_str::<Value>(raw) {
        Ok(value) => value,
        Err(_) => return Vec::new(),
    };

    match parsed {
        Value::Object(map) => map
            .into_values()
            .filter_map(|value| value.as_str().map(ToOwned::to_owned))
            .collect(),
        Value::Array(items) => items
            .into_iter()
            .filter_map(|value| value.as_str().map(ToOwned::to_owned))
            .collect(),
        Value::String(title) => vec![title],
        _ => Vec::new(),
    }
}

fn slug_to_title(slug: &str) -> String {
    slug.replace('-', " ").trim().to_string()
}

const REMOTE_GRAPHQL_QUERY: &str = r#"
query($search: String!, $perPage: Int!) {
  Page(page: 1, perPage: $perPage) {
    media(search: $search, type: ANIME) {
      id
      idMal
      title {
        romaji
        english
        native
        userPreferred
      }
      synonyms
    }
  }
}
"#;

#[derive(Debug, Serialize)]
struct RemoteGraphQlRequest<'a> {
    query: &'a str,
    variables: RemoteGraphQlVariables<'a>,
}

#[derive(Debug, Serialize)]
struct RemoteGraphQlVariables<'a> {
    search: &'a str,
    #[serde(rename = "perPage")]
    per_page: i64,
}

#[derive(Debug, Deserialize)]
struct RemoteGraphQlEnvelope {
    data: Option<RemoteGraphQlData>,
    errors: Option<Vec<RemoteGraphQlError>>,
}

#[derive(Debug, Deserialize)]
struct RemoteGraphQlData {
    #[serde(rename = "Page")]
    page: Option<RemoteGraphQlPage>,
}

#[derive(Debug, Deserialize)]
struct RemoteGraphQlPage {
    media: Vec<RemoteMedia>,
}

#[derive(Debug, Deserialize)]
struct RemoteMedia {
    id: u32,
    #[serde(rename = "idMal")]
    id_mal: Option<u32>,
    title: RemoteTitle,
    synonyms: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct RemoteTitle {
    romaji: Option<String>,
    english: Option<String>,
    native: Option<String>,
    #[serde(rename = "userPreferred")]
    user_preferred: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RemoteGraphQlError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn matches_titles_from_plain_sql_dump() {
        let temp_dir = unique_temp_dir();
        let dump_path = temp_dir.join("latest.sql");
        fs::write(&dump_path, sample_dump()).unwrap();

        let matcher = TitleMatcher::from_kitsu_dump(&temp_dir).unwrap();
        let matched = matcher.match_title("spy x family").unwrap().unwrap();

        assert_eq!(matched.provider, MatchProvider::KitsuDump);
        assert_eq!(matched.canonical_title, "Spy x Family");
        assert_eq!(matched.ids.kitsu, Some(1));
        assert_eq!(matched.ids.anilist, Some(777));
        assert_eq!(matched.ids.mal, Some(12345));
        assert!(matched.score > 0.90);

        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn reads_gzipped_dump_files() {
        let temp_dir = unique_temp_dir();
        let dump_path = temp_dir.join("latest.sql.gz");
        write_gzip(&dump_path, sample_dump().as_bytes());

        let matcher = TitleMatcher::from_kitsu_dump(&temp_dir).unwrap();
        let matches = matcher.search_titles("frieren", 2).unwrap();

        assert_eq!(matches[0].canonical_title, "Sousou no Frieren");
        assert_eq!(matches[0].ids.anilist, Some(888));

        fs::remove_dir_all(temp_dir).unwrap();
    }

    #[test]
    fn normalizes_titles_before_scoring() {
        assert_eq!(normalize_title("SPY×FAMILY"), "spyxfamily");
        assert!(similarity_score("spy x family", "spy family") > 0.75);
        assert!(similarity_score("jujutsu kaisen", "bleach") < 0.50);
    }

    #[test]
    fn parses_json_title_values() {
        let titles = parse_json_titles(r#"{"en":"Frieren","en_jp":"Sousou no Frieren"}"#);
        assert_eq!(titles.len(), 2);
        assert!(titles.contains(&"Frieren".to_string()));
        assert!(titles.contains(&"Sousou no Frieren".to_string()));
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "zantetsu-vecdb-test-{}-{}",
            std::process::id(),
            unique
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn write_gzip(path: &Path, bytes: &[u8]) {
        let file = File::create(path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(bytes).unwrap();
        encoder.finish().unwrap();
    }

    fn sample_dump() -> String {
        [
            "-- sample kitsu dump",
            "COPY public.anime (id, slug, titles, canonical_title, abbreviated_titles) FROM stdin;",
            "1\tspy-x-family\t{\"en\":\"Spy x Family\",\"en_jp\":\"SPY×FAMILY\",\"ja_jp\":\"スパイファミリー\"}\tSpy x Family\t[\"Spy Family\"]",
            "2\tsousou-no-frieren\t{\"en\":\"Frieren: Beyond Journey's End\",\"en_jp\":\"Sousou no Frieren\"}\tSousou no Frieren\t[\"Frieren\"]",
            "\\.",
            "COPY public.mappings (item_id, item_type, external_site, external_id) FROM stdin;",
            "1\tAnime\tanilist/anime\t777",
            "1\tAnime\tmyanimelist/anime\t12345",
            "2\tAnime\tanilist/anime\t888",
            "\\.",
            "",
        ]
        .join("\n")
    }
}
