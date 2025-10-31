# Repository Guidelines

## Project Structure & Modules
- `ETL/`: ETL code.
  - `etl_extract.py`: Bronze layer (downloads Zillow via HuggingFace, FRED CSV). Writes to `data/raw/{zillow,macro}`.
  - `etl_transform_enhanced.py`: Silver transforms for Zillow and FRED. Writes to `data/silver/...` and unified FRED monthly parquet.
  - `etl_pipeline.py`: Combined pipeline utilities (advanced usage).
- `scripts/`: Analysis utilities (e.g., `macro_analysis.py`).
- `data/`: Generated artifacts only; ignored by Git (`.gitignore`).
- `requirements.txt`, `README.md`.

## Build, Test, and Development
- Create venv and install deps:
  - `python -m venv .venv` then `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
  - `pip install -r requirements.txt`
- Extract (bronze):
  - `python ETL/etl_extract.py --fred-start 2015-01-01 --fred-end 2025-12-31`
  - Optional: set `HF_TOKEN` for HuggingFace access (`huggingface_hub`).
- Transform (silver):
  - `python ETL/etl_transform_enhanced.py`
- Optional analysis:
  - `python scripts/macro_analysis.py` (expects silver FRED unified parquet).

## Coding Style & Naming
- Python 3.x, PEP 8, 4-space indentation; prefer type hints and docstrings.
- Filenames, functions, variables: `snake_case`; constants: `UPPER_SNAKE`.
- Use `pathlib.Path` for file paths, f-strings for formatting, and the existing `log(...)` helper for messages.
- Keep transforms in Polars; avoid unnecessary Pandas back-and-forth.

## Testing Guidelines
- No formal test suite yet. Validate via smoke runs:
  - Confirm manifests: `data/raw/zillow/manifest.json`, `data/raw/macro/manifest.json`.
  - Verify silver outputs exist and are non-empty (e.g., `data/silver/fred/unified_monthly.parquet`).
  - For quick checks, limit date ranges in extract flags.
- If adding tests, place under `tests/` and use `pytest`; name files `test_*.py`.

## Commit & PR Guidelines
- Commits: imperative, concise subject (≤72 chars), scoped when helpful (e.g., `etl: fix zillow date cast`).
- PRs should include:
  - Summary, rationale, and affected modules/paths.
  - Repro steps and commands used (incl. flags, env vars).
  - Sample output paths and, if applicable, screenshots of figures from `data/gold/fred/`.
- Do not commit data or secrets. `data/` is ignored; keep tokens in env (e.g., `HF_TOKEN`).

