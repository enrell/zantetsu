/**
 * Zantetsu - Fast anime metadata parser
 *
 * Type definitions for the heuristic parser bindings
 */

/**
 * Episode specification types
 */
export type EpisodeSpec =
  | { type: 'single'; episode: number }
  | { type: 'range'; start: number; end: number }
  | { type: 'multi'; episodes: number[] }
  | { type: 'versioned'; episode: number; version: number };

/**
 * Video resolution
 */
export type Resolution = 'SD480' | 'HD720' | 'FHD1080' | 'UHD2160';

/**
 * Video codec
 */
export type VideoCodec = 'H264' | 'HEVC' | 'AV1' | 'VP9' | 'MPEG4';

/**
 * Audio codec
 */
export type AudioCodec = 'FLAC' | 'AAC' | 'Opus' | 'AC3' | 'DTS' | 'MP3' | 'Vorbis' | 'TrueHD' | 'EAAC';

/**
 * Media source
 */
export type MediaSource = 'BluRayRemux' | 'BluRay' | 'WebDL' | 'WebRip' | 'HDTV' | 'DVD' | 'LaserDisc' | 'VHS';

/**
 * Parse mode
 */
export type ParseMode = 'Full' | 'Light' | 'Auto';

/**
 * Options for creating a HeuristicParser
 */
export interface HeuristicParserOptions {
  /** Enable debug logging (default: false) */
  debug?: boolean;
}

/**
 * Parse result from the heuristic parser
 */
export interface ParseResult {
  /** Original input string */
  input: string;
  /** Extracted anime title (normalized) */
  title: string | null;
  /** Release group name (e.g., "SubsPlease", "Erai-raws") */
  group: string | null;
  /** Episode specification */
  episode: EpisodeSpec | null;
  /** Season number */
  season: number | null;
  /** Video resolution */
  resolution: Resolution | null;
  /** Video codec */
  video_codec: VideoCodec | null;
  /** Audio codec */
  audio_codec: AudioCodec | null;
  /** Media source */
  source: MediaSource | null;
  /** Release year */
  year: number | null;
  /** CRC32 checksum (hex string) */
  crc32: string | null;
  /** File extension (without leading dot) */
  extension: string | null;
  /** Release version (e.g., v2 = 2) */
  version: number | null;
  /** Confidence score in [0.0, 1.0] */
  confidence: number;
  /** Parse mode used */
  parse_mode: ParseMode;
}
