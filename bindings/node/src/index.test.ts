/**
 * Zantetsu - Jest test suite for heuristic parser
 */

import { HeuristicParser, parse, parseBatch } from '../src/index.js';

describe('HeuristicParser', () => {
  let parser: HeuristicParser;

  beforeAll(() => {
    parser = new HeuristicParser();
  });

  describe('constructor', () => {
    it('should create a parser instance', () => {
      expect(parser).toBeDefined();
    });

    it('should accept options', () => {
      const parserWithOptions = new HeuristicParser({ debug: true });
      expect(parserWithOptions).toBeDefined();
    });
  });

  describe('parse()', () => {
    it('should parse SubsPlease format correctly', () => {
      const result = parser.parse('[SubsPlease] Jujutsu Kaisen - 24 (1080p) [A1B2C3D4].mkv');

      expect(result.title).toBe('Jujutsu Kaisen');
      expect(result.group).toBe('SubsPlease');
      expect(result.episode).toEqual({ type: 'single', episode: 24 });
      expect(result.resolution).toBe('FHD1080');
      expect(result.crc32).toBe('A1B2C3D4');
      expect(result.extension).toBe('mkv');
      expect(result.confidence).toBeGreaterThan(0.5);
    });

    it('should parse Erai-raws versioned episode format', () => {
      const result = parser.parse(
        '[Erai-raws] Shingeki no Kyojin - The Final Season - 28v2 [1080p][HEVC].mkv'
      );

      expect(result.group).toBe('Erai-raws');
      expect(result.episode).toEqual({ type: 'versioned', episode: 28, version: 2 });
      expect(result.resolution).toBe('FHD1080');
      expect(result.video_codec).toBe('HEVC');
    });

    it('should parse batch episode range', () => {
      const result = parser.parse('[Judas] Golden Kamuy S3 - 01-12 (1080p) [Batch]');

      expect(result.group).toBe('Judas');
      expect(result.season).toBe(3);
      expect(result.episode).toEqual({ type: 'range', start: 1, end: 12 });
      expect(result.resolution).toBe('FHD1080');
    });

    it('should parse dot-separated format', () => {
      const result = parser.parse('One.Piece.1084.VOSTFR.1080p.WEB.x264-AAC.mkv');

      expect(result.title).toBe('One Piece');
      expect(result.episode).toEqual({ type: 'single', episode: 1084 });
      expect(result.resolution).toBe('FHD1080');
      expect(result.video_codec).toBe('H264');
      expect(result.audio_codec).toBe('AAC');
    });

    it('should handle various resolutions', () => {
      const res480 = parser.parse('[Test] Show - 01 (480p).mkv');
      expect(res480.resolution).toBe('SD480');

      const res720 = parser.parse('[Test] Show - 01 (720p).mkv');
      expect(res720.resolution).toBe('HD720');

      const res1080 = parser.parse('[Test] Show - 01 (1080p).mkv');
      expect(res1080.resolution).toBe('FHD1080');

      const res2160 = parser.parse('[Test] Show - 01 (2160p).mkv');
      expect(res2160.resolution).toBe('UHD2160');
    });

    it('should extract video codecs', () => {
      expect(parser.parse('[G] Title - 01 [x264].mkv').video_codec).toBe('H264');
      expect(parser.parse('[G] Title - 01 [x265].mkv').video_codec).toBe('HEVC');
      expect(parser.parse('[G] Title - 01 [HEVC].mkv').video_codec).toBe('HEVC');
      expect(parser.parse('[G] Title - 01 [AV1].mkv').video_codec).toBe('AV1');
      expect(parser.parse('[G] Title - 01 [VP9].mkv').video_codec).toBe('VP9');
    });

    it('should extract audio codecs', () => {
      expect(parser.parse('[G] Title - 01 [FLAC].mkv').audio_codec).toBe('FLAC');
      expect(parser.parse('[G] Title - 01 [AAC].mkv').audio_codec).toBe('AAC');
      expect(parser.parse('[G] Title - 01 [Opus].mkv').audio_codec).toBe('Opus');
      expect(parser.parse('[G] Title - 01 [AC3].mkv').audio_codec).toBe('AC3');
    });

    it('should extract media sources', () => {
      expect(parser.parse('[G] Title - 01 Blu-ray 1080p.mkv').source).toBe('BluRay');
      expect(parser.parse('[G] Title - 01 WEB-DL 1080p.mkv').source).toBe('WebDL');
      expect(parser.parse('[G] Title - 01 HDTV 720p.mkv').source).toBe('HDTV');
    });

    it('should extract year', () => {
      const result = parser.parse('[Group] Title (2024) - 01 (1080p).mkv');
      expect(result.year).toBe(2024);
    });

    it('should compute confidence based on extracted fields', () => {
      // Minimal parse - only title
      const minimal = parser.parse('Some Random Title.mkv');
      expect(minimal.confidence).toBeLessThan(0.5);

      // Rich parse - many fields
      const rich = parser.parse(
        '[SubsPlease] Jujutsu Kaisen - 24 (1080p) [H264] [AAC] [A1B2C3D4].mkv'
      );
      expect(rich.confidence).toBeGreaterThan(0.7);
    });

    it('should throw on empty input', () => {
      expect(() => parser.parse('')).toThrow();
      expect(() => parser.parse('   ')).toThrow();
    });

    it('should return correct parse mode', () => {
      const result = parser.parse('[Test] Title - 01 (1080p).mkv');
      expect(result.parse_mode).toBe('Light');
    });
  });

  describe('parseBatch()', () => {
    it('should parse multiple inputs', () => {
      const inputs = [
        '[SubsPlease] Jujutsu Kaisen - 24 (1080p).mkv',
        '[Erai-raws] One Piece - 1000 [1080p].mkv',
        'Anime.Title.01.720p.WEB.x264.mkv',
      ];

      const results = parser.parseBatch(inputs);

      expect(results).toHaveLength(3);
      expect(results[0].title).toBe('Jujutsu Kaisen');
      expect(results[1].title).toBe('One Piece');
      expect(results[2].title).toBe('Anime Title');
    });
  });
});

describe('Convenience functions', () => {
  describe('parse()', () => {
    it('should work as a convenience function', () => {
      const result = parse('[Test] Anime - 01 (1080p).mkv');
      expect(result.title).toBe('Anime');
      expect(result.episode).toEqual({ type: 'single', episode: 1 });
    });
  });

  describe('parseBatch()', () => {
    it('should work as a convenience function', () => {
      const results = parseBatch(['[Test] A - 01.mkv', '[Test] B - 02.mkv']);
      expect(results).toHaveLength(2);
    });
  });
});
