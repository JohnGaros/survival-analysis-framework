# .claude/agents/

## Purpose

Specialized Claude Code agents for automated task execution.

## Available Agents

- `test-runner.md` - Automatically runs unit tests after code changes
- `integration-tester.md` - Validates end-to-end pipeline execution

## Usage

Claude Code invokes these agents automatically when:
- Making code changes to critical modules (data.py, models.py, etc.)
- Before committing changes
- When explicitly requested by user

## Creating New Agents

Each agent is defined in a markdown file with:
- Purpose and scope
- When to invoke
- What to validate
- Success criteria

See existing agents for examples.
