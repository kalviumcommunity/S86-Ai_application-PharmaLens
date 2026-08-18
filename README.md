# PharmaLens Assistant Workspace

This repository is set up as a clean, reproducible Python workspace for building an AI assistant.

## Layout

- `data/` source documents and other local inputs
- `src/` application code
- `prompts/` reusable prompt templates
- `outputs/` logs, generated answers, and evaluation artifacts
- `.env` local secrets and runtime settings
- `.env.example` template for required environment variables

## Quick Start

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

2. Install dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy the environment template and fill in real values.

   ```powershell
   copy .env.example .env
   ```

4. Run the starter app.

   ```powershell
   python -m src.main
   ```

## Reproducibility Check

A teammate should be able to clone this repo, create `.venv`, install `requirements.txt`, copy `.env.example` to `.env`, and run the app without editing code.
