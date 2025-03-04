# Contributing to Hebo AI

Thank you for your interest in contributing to Hebo AI! We're excited to have you join our community of contributors. This document provides guidelines and steps to help you contribute effectively.

## Getting Started

### Prerequisites

- [Python](https://www.python.org/downloads/) (version specified in `.python-version`)
- [uv](https://github.com/astral-sh/uv) for managing virtual environments and dependencies
- Familiarity with Git and GitHub
- Understanding of our tech stack: Django, FastAPI, Alpine.js, Tailwind CSS, and HTMX

### Development Environment Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/hebo.git
   cd hebo
   ```
3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/heboai/hebo.git
   ```
4. Create a virtual environment and install dependencies using uv:
   ```bash
   uv venv
   uv pip install -e .
   ```

## Contribution Workflow

### 1. Finding an Issue to Work On

- Browse our [issue tracker](https://github.com/heboai/hebo/issues) for open issues
- Look for issues labeled `good first issue` if you're new to the project
- Comment on the issue to express your interest before starting work

### 2. Creating a Branch

- Ensure your local `develop` branch is up to date:
  ```bash
  git checkout develop
  git pull upstream develop
  ```
- Create a new branch with a descriptive name that references the issue number:
  ```bash
  git checkout -b feature/issue-123-brief-description
  ```
  or
  ```bash
  git checkout -b fix/issue-123-brief-description
  ```

### 3. Making Changes

- Follow the coding style and conventions used throughout the project
- Write clear, commented, and testable code
- Include appropriate tests for your changes
- Update documentation as needed

### 4. Committing Your Changes

- Make focused, logical commits with clear messages:
  ```bash
  git commit -m "Fix #123: Brief description of the change"
  ```
- Reference the issue number in commit messages when applicable

### 5. Submitting a Pull Request

- Push your branch to your fork:
  ```bash
  git push origin your-branch-name
  ```
- Go to the [original repository](https://github.com/heboai/hebo) on GitHub
- Click "Compare & pull request"
- Set the base branch to `develop` (not `main`)
- Provide a clear title and description for your PR, including what changes you made and why
- Reference the issue number using "Fixes #123" or "Addresses #123" in the PR description
- Submit the pull request

### 6. Code Review

- Be responsive to feedback and questions
- Make requested changes promptly
- Keep the PR updated if conflicts arise with the target branch

## Development Guidelines

### Code Style

- Follow PEP 8 standards for Python code
- Use consistent naming conventions throughout the project
- Use meaningful variable and function names

## Getting Help

If you need help at any point, you can:
- Ask questions in the [Discord community](https://discord.gg/cCJtXZRU5p)
- Reach out to maintainers
- Comment on the relevant issue for specific guidance

Thank you for contributing to Hebo AI! 