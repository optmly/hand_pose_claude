# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions use a zero-padded four-digit scheme starting from `0001`.

## [Unreleased]

## [0002] - 2026-05-21

### Added
- Project-scoped Claude Code hook in `.claude/settings.json`. Fires before
  `git commit` and `git push` and reminds the agent to bump `VERSION`,
  prepend a `CHANGELOG.md` entry, and rewrite `README.md` to contain only
  the current setup/run instructions.

### Changed
- `README.md` rewritten to hold only the latest setup and run information.
  Project history and scaffolding details now live exclusively in
  `CHANGELOG.md`.

## [0001] - 2026-05-21

### Added
- Initial project scaffolding.
- `README.md` describing the project.
- `CHANGELOG.md` for tracking versioned changes.
- `VERSION` file recording the current release identifier.
- `.gitignore` covering common Python, editor, and OS artifacts.
