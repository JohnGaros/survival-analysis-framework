# Claude Code Agents

This directory contains specialized AI subagents for the survival analysis framework.

## What are Agents?

Agents are specialized AI assistants that Claude Code can delegate tasks to. Each agent:
- Has a specific purpose and expertise area
- Uses its own context window separate from the main conversation
- Can be configured with specific tools it's allowed to use
- Includes a custom system prompt that guides its behavior

## Available Agents

### code-reviewer

Expert code reviewer specializing in survival analysis, machine learning pipelines, and Python best practices.

**Usage:**
```
> Use the code-reviewer agent to review my recent changes
```

**What it does:**
- Automatically reviews code after modifications
- Checks for survival analysis correctness (risk scores, structured arrays, time grids)
- Validates testing coverage and quality
- Ensures security and best practices
- Verifies project-specific standards (config usage, MLflow tracking, etc.)
- Provides prioritized feedback (Critical, Warnings, Suggestions)

**When it's invoked:**
- PROACTIVELY after any code changes
- Immediately after writing or modifying Python code
- Can be explicitly requested for review

**Tools available:**
- Read, Grep, Glob, Bash - for examining code and context
- mcp__ide__getDiagnostics - for checking linting/type errors

**Review categories:**
1. **🔴 Critical Issues** - Must fix (security, correctness, breaking changes)
2. **🟡 Warnings** - Should fix (quality, best practices, potential bugs)
3. **🟢 Suggestions** - Consider improving (refactoring, performance, clarity)

**Survival analysis specific checks:**
- Structured survival arrays with correct dtype
- Risk score convention (higher risk = higher hazard)
- IPCW usage in metrics
- Fold-aware time grid constraints
- Model wrapper interface compliance
- Preprocessing pipeline correctness

**Project standards enforced:**
- Google-style docstrings with type hints
- >80% test coverage requirement
- Configuration centralization via config.py
- Versioned outputs with versioned_name()
- MLflow tracking for experiments
- Context file updates (CHANGELOG.md, CLAUDE.md)

## Managing Agents

### View available agents
```
/agents
```

This opens an interactive menu where you can:
- View all available agents
- Create new agents
- Edit existing agents
- Delete agents
- Manage tool permissions

### Explicitly invoke an agent
```
> Use the [agent-name] agent to [task description]
```

### Let Claude decide
Claude Code will automatically delegate to appropriate agents based on the task.

## Best Practices

1. **Proactive usage**: Agents marked "MUST BE USED" or "use PROACTIVELY" will be invoked automatically
2. **Focused purpose**: Each agent has a single, clear responsibility
3. **Tool restrictions**: Agents only have access to tools they need
4. **Version control**: Check agents into git so team members can use them

## Creating New Agents

Agents are Markdown files with YAML frontmatter:

```markdown
---
name: agent-name
description: When this agent should be used (include "PROACTIVELY" for automatic use)
tools: Read, Write, Bash  # Optional - omit to inherit all tools
model: inherit  # Optional - use 'sonnet', 'opus', 'haiku', or 'inherit'
---

Your agent's system prompt goes here...
```

See `.claude/agents/code-reviewer.md` for a complete example.

## Related Documentation

- [Claude Code Subagents Docs](https://docs.claude.com/en/docs/claude-code/subagents)
- [Skills](./../skills/README.md) - Reusable workflows for specific tasks
- [Slash Commands](https://docs.claude.com/en/docs/claude-code/slash-commands)
