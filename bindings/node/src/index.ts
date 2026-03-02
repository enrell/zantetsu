/**
 * Zantetsu - Fast anime metadata parser
 *
 * JavaScript/TypeScript bindings for the heuristic parser engine.
 * Provides both a class-based API and convenience functions.
 * Includes a pure JavaScript fallback when native module is unavailable.
 */

// Import types from the declaration file
import type { 
  HeuristicParserOptions, 
  ParseResult, 
  EpisodeSpec,
  Resolution,
  VideoCodec,
  AudioCodec,
  MediaSource,
  ParseMode
} from './types.js';

// Track if native module is available
let useNative = false;
let nativeModule: unknown = null;
let initialized = false;

/**
 * Initialize the native module (done automatically on first use)
 */
function initNative(): void {
  if (initialized) return;
  initialized = true;

  try {
    // Try to load the native addon
    // eslint-disable-next-line @typescript-eslint/no-require-imports
    nativeModule = require('@zan/heuristic-node');
    useNative = true;
  } catch {
    // Use JavaScript fallback
    nativeModule = null;
    useNative = false;
  }
}

/**
 * JavaScript-only fallback parser using regex patterns
 * Matches the HeuristicParser from the Rust crate
 */
class JsHeuristicParser {
  // Resolution pattern
  private readonly reResolution = /1080p|720p|480p|2160p/i;
  // Group tag at the start
  private readonly reGroup = /^\[([^\]]+)\]/;
  // Match episode after dash, space, or dot - but not after year
  private readonly reEpisode = /(?:[\s\-.])(?:[Ee]p?\.?)?\s*(\d{1,4})(?:\b|v\d|[^\d])/;
  private readonly reEpisodeV = /(?:[\s\-.])(?:[Ee]p?\.?)?\s*(\d{1,4})v(\d+)/i;
  private readonly reEpisodeRange = /(?:[\s\-.])(?:[Ee]p?\.?)?\s*(\d{1,4})\s*[-~]\s*(\d{1,4})/i;
  private readonly reSeason = /(?:^|[\s\-])S(\d+)/i;
  private readonly reYear = /\((\d{4})\)/;
  private readonly reExtension = /\.(\w+)$/;
  private readonly reCrc32 = /\[([A-Fa-f0-9]{8})\]/;
  
  // Video codec patterns (matching Rust implementation)
  private readonly reVideoCodec = /\b(x\.?264|x\.?265|h\.?264|h\.?265|hevc|av1|vp9|mpeg4|xvid)\b/i;
  
  // Audio codec patterns (matching Rust implementation)
  private readonly reAudioCodec = /\b(flac|aac|opus|ac3|dts|truehd|mp3|vorbis|ogg|e-?aac\+?)\b/i;
  
  // Media source patterns (matching Rust implementation)
  private readonly reSource = /\b(blu-?ray|web-?dl|webrip|web-?rip|hdtv|dvd|laserdisc|vhs)\b/i;

  parse(input: string): ParseResult {
    const trimmed = input.trim();
    if (!trimmed) {
      throw new Error('input is empty or whitespace-only');
    }

    const result: ParseResult = {
      input: trimmed,
      title: null,
      group: null,
      episode: null,
      season: null,
      resolution: null,
      video_codec: null,
      audio_codec: null,
      source: null,
      year: null,
      crc32: null,
      extension: null,
      version: null,
      confidence: 0,
      parse_mode: 'Light',
    };

    // Extract basic fields
    result.group = this.extractGroup(trimmed);
    result.extension = this.extractExtension(trimmed);
    result.crc32 = this.extractCrc32(trimmed);
    result.resolution = this.extractResolution(trimmed);
    result.season = this.extractSeason(trimmed);
    result.year = this.extractYear(trimmed);
    result.episode = this.extractEpisode(trimmed);
    result.video_codec = this.extractVideoCodec(trimmed);
    result.audio_codec = this.extractAudioCodec(trimmed);
    result.source = this.extractSource(trimmed);

    // Extract title
    result.title = this.extractTitle(trimmed, result);

    // Compute confidence
    result.confidence = this.computeConfidence(result);

    return result;
  }

  private extractGroup(input: string): string | null {
    const match = this.reGroup.exec(input);
    return match ? match[1].trim() : null;
  }

  private extractExtension(input: string): string | null {
    const match = this.reExtension.exec(input);
    return match ? match[1].toLowerCase() : null;
  }

  private extractCrc32(input: string): string | null {
    const match = this.reCrc32.exec(input);
    return match ? match[1].toUpperCase() : null;
  }

  private extractResolution(input: string): Resolution | null {
    if (input.includes('2160p') || input.includes('2160i')) return 'UHD2160';
    if (input.includes('1080p') || input.includes('1080i')) return 'FHD1080';
    if (input.includes('720p') || input.includes('720i')) return 'HD720';
    if (input.includes('480p') || input.includes('480i')) return 'SD480';
    return null;
  }

  private extractSeason(input: string): number | null {
    const match = this.reSeason.exec(input);
    return match ? parseInt(match[1], 10) : null;
  }

  private extractYear(input: string): number | null {
    const match = this.reYear.exec(input);
    return match ? parseInt(match[1], 10) : null;
  }

  private extractEpisode(input: string): EpisodeSpec | null {
    // Try versioned episode
    let match = this.reEpisodeV.exec(input);
    if (match) {
      return { type: 'versioned', episode: parseInt(match[1], 10), version: parseInt(match[2], 10) };
    }

    // Try episode range
    match = this.reEpisodeRange.exec(input);
    if (match) {
      const start = parseInt(match[1], 10);
      const end = parseInt(match[2], 10);
      if (start < end) {
        return { type: 'range', start, end };
      }
    }

    // Try single episode
    match = this.reEpisode.exec(input);
    if (match) {
      return { type: 'single', episode: parseInt(match[1], 10) };
    }

    return null;
  }

  private extractVideoCodec(input: string): VideoCodec | null {
    const match = this.reVideoCodec.exec(input);
    if (!match) return null;
    
    const codec = match[1].toLowerCase();
    if (codec === 'x264' || codec === 'x.264' || codec === 'h264' || codec === 'h.264') return 'H264';
    if (codec === 'x265' || codec === 'x.265' || codec === 'h265' || codec === 'h.265' || codec === 'hevc') return 'HEVC';
    if (codec === 'av1') return 'AV1';
    if (codec === 'vp9') return 'VP9';
    if (codec === 'mpeg4' || codec === 'xvid') return 'MPEG4';
    return null;
  }

  private extractAudioCodec(input: string): AudioCodec | null {
    const match = this.reAudioCodec.exec(input);
    if (!match) return null;
    
    const codec = match[1].toLowerCase();
    if (codec === 'flac') return 'FLAC';
    if (codec === 'aac') return 'AAC';
    if (codec === 'opus') return 'Opus';
    if (codec === 'ac3') return 'AC3';
    if (codec.startsWith('dts')) return 'DTS';
    if (codec.includes('truehd')) return 'TrueHD';
    if (codec === 'mp3') return 'MP3';
    if (codec === 'vorbis' || codec === 'ogg') return 'Vorbis';
    if (codec.startsWith('e-aac') || codec.startsWith('eaac')) return 'EAAC';
    return null;
  }

  private extractSource(input: string): MediaSource | null {
    const match = this.reSource.exec(input);
    if (!match) return null;
    
    const source = match[1].toLowerCase().replace(/[- ]/g, '');
    if (source.includes('remux')) return 'BluRayRemux';
    if (source.includes('blu') || source === 'bd') return 'BluRay';
    if (source === 'webdl') return 'WebDL';
    if (source === 'webrip') return 'WebRip';
    if (source === 'hdtv') return 'HDTV';
    if (source.startsWith('dvd')) return 'DVD';
    if (source === 'laserdisc' || source === 'ld') return 'LaserDisc';
    if (source === 'vhs') return 'VHS';
    return null;
  }

  private extractTitle(input: string, result: ParseResult): string | null {
    let work = input;

    // Remove group tag
    if (result.group) {
      const groupIdx = work.indexOf(']');
      if (groupIdx !== -1) {
        work = work.substring(groupIdx + 1);
      }
    }

    // Remove extension
    if (result.extension) {
      const extPos = work.lastIndexOf(`.${result.extension}`);
      if (extPos !== -1) {
        work = work.substring(0, extPos);
      }
    }

    // Replace metadata tokens with null byte sentinel (like Rust implementation)
    // Take text BEFORE the first sentinel as the title
    const sentinel = '\x00';
    
    // Replace episode info with sentinel
    work = work.replace(this.reEpisodeV, sentinel);
    work = work.replace(this.reEpisodeRange, sentinel);
    work = work.replace(this.reEpisode, sentinel);
    
    // Replace other metadata with sentinel
    work = work.replace(this.reSeason, sentinel);
    work = work.replace(this.reYear, sentinel);
    work = work.replace(this.reResolution, sentinel);
    work = work.replace(this.reVideoCodec, sentinel);
    work = work.replace(this.reAudioCodec, sentinel);
    work = work.replace(this.reSource, sentinel);
    work = work.replace(this.reCrc32, sentinel);

    // Take text before the first sentinel
    const titleRegion = work.split(sentinel)[0];

    // Remove bracketed content from title region
    let cleaned = titleRegion.replace(/\[[^\]]*\]/g, ' ');
    cleaned = cleaned.replace(/\([^\)]*\)/g, ' ');

    // Clean up: replace dots, underscores with spaces; normalize whitespace
    cleaned = cleaned
      .replace(/[._]/g, ' ')
      .replace(/-/g, ' ')
      .split(/\s+/)
      .filter(Boolean)
      .join(' ')
      .trim();

    return cleaned || null;
  }

  private computeConfidence(result: ParseResult): number {
    let fieldsPresent = 0;
    let fieldsTotal = 7;

    if (result.title) {
      fieldsPresent += 2;
      fieldsTotal += 1;
    }
    if (result.group) fieldsPresent += 1;
    if (result.episode) fieldsPresent += 1;
    if (result.resolution) fieldsPresent += 1;
    if (result.video_codec) fieldsPresent += 1;
    if (result.audio_codec) fieldsPresent += 1;
    if (result.source) fieldsPresent += 1;

    return Math.min(fieldsPresent / fieldsTotal, 1.0);
  }
}

// Singleton instances
let jsParser: JsHeuristicParser | null = null;
let nativeParser: unknown = null;

function getParser(): { parse: (input: string) => unknown } {
  initNative();

  if (useNative) {
    if (!nativeParser) {
      // eslint-disable-next-line new-cap
      nativeParser = new (nativeModule as { HeuristicParser: new () => { parse: (s: string) => unknown } }).HeuristicParser();
    }
    return nativeParser as { parse: (s: string) => unknown };
  }

  if (!jsParser) {
    jsParser = new JsHeuristicParser();
  }
  return jsParser;
}

/**
 * Convert native episode spec to typed EpisodeSpec
 */
function convertEpisodeSpec(native: unknown): EpisodeSpec | null {
  if (!native) return null;

  if (typeof native === 'number') {
    return { type: 'single', episode: native };
  }

  const n = native as Record<string, unknown>;

  if (typeof n.type === 'number') {
    switch (n.type) {
      case 0:
        return { type: 'single', episode: n.episode as number };
      case 1:
        return { type: 'range', start: n.start as number, end: n.end as number };
      case 2:
        return { type: 'multi', episodes: n.episodes as number[] };
      case 3:
        return { type: 'versioned', episode: n.episode as number, version: n.version as number };
    }
  }

  if (n.episode !== undefined && n.version !== undefined) {
    return { type: 'versioned', episode: n.episode as number, version: n.version as number };
  }
  if (n.start !== undefined && n.end !== undefined) {
    return { type: 'range', start: n.start as number, end: n.end as number };
  }
  if (n.episodes !== undefined) {
    return { type: 'multi', episodes: n.episodes as number[] };
  }

  return null;
}

/**
 * Convert native parse result to typed ParseResult
 */
function convertResult(native: unknown): ParseResult {
  const n = native as Record<string, unknown>;
  return {
    input: n.input as string,
    title: n.title as string | null,
    group: n.group as string | null,
    episode: convertEpisodeSpec(n.episode),
    season: n.season as number | null,
    resolution: n.resolution as Resolution | null,
    video_codec: n.video_codec as VideoCodec | null,
    audio_codec: n.audio_codec as AudioCodec | null,
    source: n.source as MediaSource | null,
    year: n.year as number | null,
    crc32: n.crc32 as string | null,
    extension: n.extension as string | null,
    version: n.version as number | null,
    confidence: n.confidence as number,
    parse_mode: n.parse_mode as ParseMode,
  };
}

/**
 * HeuristicParser - Fast regex-based anime filename parser
 *
 * Uses optimized regex patterns and scene naming rules for
 * instant parsing with zero ML overhead.
 */
export class HeuristicParser {
  private parser: { parse: (input: string) => unknown };

  constructor(_options?: HeuristicParserOptions) {
    this.parser = getParser();
  }

  parse(input: string): ParseResult {
    if (typeof input !== 'string' || !input.trim()) {
      throw new Error('Input must be a non-empty string');
    }

    const result = this.parser.parse(input);

    if (!useNative) {
      return result as ParseResult;
    }

    return convertResult(result);
  }

  parseBatch(inputs: string[]): ParseResult[] {
    if (!Array.isArray(inputs)) {
      throw new Error('Input must be an array of strings');
    }

    return inputs.map(input => this.parse(input));
  }
}

// Default parser instance for convenience functions
let defaultParser: HeuristicParser | null = null;

function getDefaultParser(): HeuristicParser {
  if (!defaultParser) {
    defaultParser = new HeuristicParser();
  }
  return defaultParser;
}

/**
 * Parse a single filename using the default parser
 */
export function parse(input: string): ParseResult {
  return getDefaultParser().parse(input);
}

/**
 * Parse multiple filenames using the default parser
 */
export function parseBatch(inputs: string[]): ParseResult[] {
  return getDefaultParser().parseBatch(inputs);
}

/**
 * Check if native module is being used
 */
export function isUsingNativeModule(): boolean {
  initNative();
  return useNative;
}

// Export types
export type { HeuristicParserOptions, ParseResult, EpisodeSpec, Resolution, VideoCodec, AudioCodec, MediaSource, ParseMode } from './types.js';
