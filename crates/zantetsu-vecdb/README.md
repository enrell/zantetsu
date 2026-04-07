# zantetsu-vecdb

Canonical title matching for parsed anime names.

## Features

- **Local Kitsu Dumps**: Read `latest.sql` or `latest.sql.gz` directly from the `kitsu-sync` dump directory
- **Remote Endpoint**: Query a remote GraphQL endpoint when the client prefers live API data
- **Fuzzy Matching**: Score aliases locally so the crate API stays consistent across both backends
- **Canonical IDs**: Return Kitsu, AniList-compatible, and MAL ids when they are available

## Usage

```rust
use zantetsu_vecdb::{MatchSource, TitleMatcher};

let matcher = TitleMatcher::new(MatchSource::kitsu_dump(
    "/home/user/.local/share/zantetsu/kitsu-dumps",
))
.unwrap();

let best = matcher.match_title("spy x family").unwrap().unwrap();
assert_eq!(best.canonical_title, "Spy x Family");
```

```rust
use zantetsu_vecdb::{MatchSource, TitleMatcher};

let matcher = TitleMatcher::new(MatchSource::remote_endpoint("https://graphql.anilist.co")).unwrap();
let best = matcher.match_title("Sousou no Frieren").unwrap().unwrap();
println!("{} ({:.2})", best.canonical_title, best.score);
```

## License

MIT
