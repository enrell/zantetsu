/**
 * AGENTS.md Coverage Test
 * 
 * This test verifies that the AGENTS.md file exists and contains
 * all required sections for agentic coding agents.
 */

import * as fs from 'fs';
import * as path from 'path';

describe('AGENTS.md Coverage', () => {
  // AGENTS.md is in the project root (parent of bindings/node)
  const agentsMdPath = path.join(process.cwd(), '..', '..', 'AGENTS.md');
  let content: string;

  beforeAll(() => {
    // Read the AGENTS.md file
    content = fs.readFileSync(agentsMdPath, 'utf-8');
  });

  describe('File Existence', () => {
    it('should exist at project root', () => {
      expect(fs.existsSync(agentsMdPath)).toBe(true);
    });
  });

  describe('Required Sections', () => {
    it('should contain Build/Lint/Test Commands section', () => {
      expect(content).toMatch(/Build.*Lint.*Test.*Commands/i);
    });

    it('should contain Rust test commands', () => {
      expect(content).toMatch(/cargo test/);
    });

    it('should contain commands for running a single test', () => {
      expect(content).toMatch(/single test/);
    });

    it('should contain Node.js test commands', () => {
      expect(content).toMatch(/npm test/);
    });

    it('should contain Clippy lint command', () => {
      expect(content).toMatch(/cargo clippy/);
    });

    it('should contain Format command', () => {
      expect(content).toMatch(/cargo fmt/);
    });
  });

  describe('Code Style Guidelines', () => {
    it('should contain code style guidelines section', () => {
      expect(content).toMatch(/Code Style Guidelines/i);
    });

    it('should contain naming conventions', () => {
      expect(content).toMatch(/Naming.*Conventions/i);
    });

    it('should contain error handling guidelines', () => {
      expect(content).toMatch(/Error.*Handling/i);
    });

    it('should contain testing guidelines', () => {
      expect(content).toMatch(/Testing/i);
    });

    it('should contain import ordering guidelines', () => {
      expect(content).toMatch(/Imports/i);
    });

    it('should contain TypeScript guidelines', () => {
      expect(content).toMatch(/TypeScript/i);
    });
  });

  describe('Project Structure', () => {
    it('should contain project structure overview', () => {
      expect(content).toMatch(/Project.*Structure/i);
    });

    it('should mention zantetsu-core crate', () => {
      expect(content).toMatch(/zantetsu-core/);
    });

    it('should mention crates directory', () => {
      expect(content).toMatch(/crates\//);
    });
  });

  describe('Development Tasks', () => {
    it('should contain common development tasks', () => {
      expect(content).toMatch(/Common.*Development.*Tasks/i);
    });

    it('should contain instructions for adding a new crate', () => {
      expect(content).toMatch(/Adding.*New.*Crate/i);
    });
  });

  describe('Technical Accuracy', () => {
    it('should mention thiserror for error handling', () => {
      expect(content).toMatch(/thiserror/);
    });

    it('should mention serde for serialization', () => {
      expect(content).toMatch(/serde/);
    });

    it('should mention tracing for logging', () => {
      expect(content).toMatch(/tracing/);
    });

    it('should specify Rust edition 2024', () => {
      expect(content).toMatch(/Edition.*2024/);
    });

    it('should specify minimum Rust version 1.85', () => {
      expect(content).toMatch(/1\.85/);
    });
  });

  describe('Line Count', () => {
    it('should be a reasonable length for developer documentation', () => {
      const lines = content.split('\n').length;
      // AGENTS.md should be comprehensive but not excessive
      expect(lines).toBeGreaterThan(100);
      expect(lines).toBeLessThan(600);
    });
  });
});
