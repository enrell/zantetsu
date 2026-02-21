# Zantetsu

Fast anime metadata parser - extracts title, episode, resolution, codecs from filenames.

## Features

- **Fast parsing** - Optimized regex patterns for instant results
- **Native module support** - Uses Rust native addon when available, falls back to pure JavaScript
- **TypeScript support** - Full type definitions included
- **Batch parsing** - Parse multiple filenames at once

## Installation

```bash
npm install zantetsu
```

## Usage

```typescript
import { parse, HeuristicParser } from 'zantetsu';

// Parse a single filename
const result = parse('[SubsPlease] Spy x Family - 01 (1080p).mkv');

console.log(result);
// {
//   input: '[SubsPlease] Spy x Family - 01 (1080p).mkv',
//   title: 'Spy x Family',
//   episode: { type: 'single', episode: 1 },
//   resolution: 'FHD1080',
//   group: 'SubsPlease',
//   extension: 'mkv',
//   confidence: 0.8,
//   parse_mode: 'Light',
//   ...
// }
```

### Class-based API

```typescript
import { HeuristicParser } from 'zantetsu';

const parser = new HeuristicParser();
const result = parser.parse('[Coalgirls] Clannad (1920x1080 Blu-Ray FLAC) [1234ABCD]/[Coalgirls] Clannad - 01 (1920x1080 Blu-Ray FLAC) [1234ABCD].mkv');

console.log(result.title);     // 'Clannad'
console.log(result.episode);   // { type: 'single', episode: 1 }
console.log(result.resolution); // 'FHD1080'
console.log(result.source);    // 'BluRay'
console.log(result.video_codec); // 'AVC'
console.log(result.audio_codec); // 'FLAC'
```

### Batch parsing

```typescript
import { parseBatch } from 'zantetsu';

const filenames = [
  '[SubsPlease] Spy x Family - 01 (1080p).mkv',
  '[Coalgirls] Clannad - 02 (720p).mkv',
  'One Punch Man S02E03 1080p WEBRip x264.mkv'
];

const results = parseBatch(filenames);
```

## API

### `parse(input: string): ParseResult`

Parse a single filename.

### `parseBatch(inputs: string[]): ParseResult[]`

Parse multiple filenames.

### `HeuristicParser`

Class-based parser with optional configuration.

## Types

```typescript
interface ParseResult {
  input: string;
  title: string | null;
  group: string | null;
  episode: EpisodeSpec | null;
  season: number | null;
  resolution: Resolution | null;
  video_codec: VideoCodec | null;
  audio_codec: AudioCodec | null;
  source: MediaSource | null;
  year: number | null;
  crc32: string | null;
  extension: string | null;
  version: number | null;
  confidence: number;
  parse_mode: ParseMode;
}

type Resolution = 'UHD2160' | 'FHD1080' | 'HD720' | 'SD480';
type VideoCodec = 'AVC' | 'HEVC' | 'VP9' | 'AV1';
type AudioCodec = 'AAC' | 'FLAC' | 'MP3' | 'AC3' | 'DTS';
type MediaSource = 'BluRay' | 'WEB' | 'DVD' | 'TV' | 'HDTV';
type ParseMode = 'Light' | 'Medium' | 'Strict';
```

## Requirements

- Node.js >= 18.0.0

## License

MIT
