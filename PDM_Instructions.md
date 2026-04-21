# PBC v2.3 - PDM Setup and Usage

This repository now uses PDM for dependency management and reproducible environments.

## 1) Install PDM

On macOS (Homebrew):

```bash
brew install pdm
```

On Linux:

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

On Windows (Scoop):

```bash
scoop install pdm
```

Or with pip:

```bash
pip install pdm
```

Official docs: https://pdm-project.org/latest/

## 2) First-time project setup

From the repository root (folder containing `pyproject.toml`):

```bash
pdm install
```

This creates/updates:

- The local PDM environment for this project
- `pdm.lock` with fully-resolved dependency versions

If `pdm.lock` changes, commit it so teammates/platforms get reproducible installs.

## 3) Run PBC with PDM

Run the Gradio app:

```bash
pdm run gradio
```

Open the notebook:

```bash
pdm run notebook
```

Run any Python command inside the managed environment:

```bash
pdm run python PBC2_3.py
```

## 4) Dependency management

Add a dependency:

```bash
pdm add <package>
```

Add a development dependency:

```bash
pdm add -dG dev <package>
```

Remove a dependency:

```bash
pdm remove <package>
```

Refresh lockfile:

```bash
pdm lock
```

Install exactly from lockfile:

```bash
pdm sync
```

## 5) Useful maintenance commands

Show environment/dependencies:

```bash
pdm info
pdm list
```

Run a one-off command:

```bash
pdm run <command>
```

