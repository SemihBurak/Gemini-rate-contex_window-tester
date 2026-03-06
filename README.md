# Gemini API Rate Limit & Context Window Tester

A tool for testing and detecting Google Gemini API rate limits and context window boundaries. Available as both a Streamlit web UI and a CLI application.

## Features

- **Text Generation** — Send prompts and monitor token usage against context window limits
- **Chat Mode** — Multi-turn conversations with cumulative token tracking
- **Rate Limit Stress Test** — Send rapid requests to trigger and observe rate limiting behavior
- **Model Info** — View token limits and list all available Gemini models
- **Exponential Backoff** — Automatic retry with backoff on rate limit errors (429)
- **Context Window Warnings** — Visual alerts when token usage approaches the limit (60% threshold)

## Supported Models

- `gemma-3-1b-it` (15,000 TPM free tier)
- `gemini-2.5-flash` (250,000 TPM free tier)

## Prerequisites

- **Python 3.9+** — If you don't have Python installed, download it from [python.org](https://www.python.org/downloads/) and follow the installer instructions for your OS.
  - **Windows:** Make sure to check "Add Python to PATH" during installation.
  - **macOS:** You can also install via `brew install python3`.
  - **Linux:** Use your package manager, e.g. `sudo apt install python3 python3-pip`.
- **pip** — Comes bundled with Python. Verify with `pip --version` or `pip3 --version`.

## Setup

1. **Clone the repository** and create a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS / Linux
   .venv\Scripts\activate           # Windows
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure your API key** by creating a `.env` file:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Streamlit Web UI

```bash
streamlit run app.py
```

### CLI

```bash
python cli.py
```

## Pre-built Example Prompts

Both interfaces include example prompts at various token sizes (~2K to ~250K tokens) for quickly testing context window limits.

## Requirements

- Python 3.x
- `google-genai`
- `python-dotenv`
- `streamlit`
