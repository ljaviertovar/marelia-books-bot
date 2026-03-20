# Marelia Books

Single-purpose Telegram bot service that adds books to one fixed Notion Reading List database.

## Scope
- Accepts only:
  - Text: `/addbook <title>`
  - Photo with caption: `/scanbook`
- Works with one fixed Notion database and one fixed Notion template (from env vars).
- Ignores chats not included in `ALLOWED_CHAT_IDS`.

## Tech
- Python 3.11+
- FastAPI
- httpx
- pydantic
- Telegram Bot API
- Gemini API (vision)
- Notion API

## Setup
1. Create and activate a Python 3.11+ environment.
2. Install dependencies:

```bash
pip install -e .
pip install -e .[dev]
```

## Environment variables
- `TELEGRAM_BOT_TOKEN`
- `GEMINI_API_KEY`
- `NOTION_API_KEY`
- `NOTION_DATABASE_ID`
- `NOTION_TEMPLATE_ID`
- `ALLOWED_CHAT_IDS` (comma-separated list, e.g. `12345,67890`)
- `DRY_RUN` (`true`/`false`)

Example:

```bash
export TELEGRAM_BOT_TOKEN="..."
export GEMINI_API_KEY="..."
export NOTION_API_KEY="..."
export NOTION_DATABASE_ID="..."
export NOTION_TEMPLATE_ID="..."
export ALLOWED_CHAT_IDS="123456789"
export DRY_RUN="false"
```

## Run locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```bash
curl http://localhost:8000/health
```

## Telegram webhook setup
Set your webhook to the service endpoint:

```bash
curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://YOUR_DOMAIN/telegram/webhook"}'
```

## DRY_RUN
When `DRY_RUN=true`, the bot performs the full flow (parse, vision, metadata, duplicate detection) but does not create/update Notion pages.

## Behavior examples
1. Text command:
   - Input: `/addbook Dune`
   - Behavior: resolve metadata, check duplicate in fixed Notion DB, create with fixed template if missing, or fill allowed missing fields if existing.

2. Image command:
   - Input: photo + caption `/scanbook`
   - Behavior: run Gemini vision extraction JSON schema, confidence gate:
     - `>= 0.85`: continue automatically
     - `0.60 - 0.84`: ask user to confirm via `/addbook <title>`
     - `< 0.60`: ask for clearer image

3. Existing book match:
   - Match rule: normalized title + normalized author.
   - If exists: only fills missing `Author`, `Book Series`, `Cover`, `Category`, `Reading Type`, `Type`, `Link`.
   - Never changes `Order to read`, `Score`, `Start Date`, `Finish Date`, `Status`.

## Tests

```bash
pytest -q
```
