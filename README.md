# STAT1301 Notes Assistant

A web chat agent that answers questions using only the STAT1301 course notes.
It finds the most relevant passages in `data/notes.txt` (BM25 search) and sends
them to Claude, which answers and cites the sections it used.

## Files
| File | Purpose |
|---|---|
| `app.py` | The agent: loads/chunks notes, searches them, calls Claude, serves the web page |
| `templates/index.html` | Chat interface (renders Markdown and maths) |
| `data/notes.txt` | The course notes (swap in another text file to reuse for another course) |
| `requirements.txt` | Python packages |
| `render.yaml` | Render deployment settings |
| `.env.example` | Environment variables you need |

## Environment variables
| Name | Required | Default | Meaning |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | yes | – | Your key from console.anthropic.com |
| `ANTHROPIC_MODEL` | no | `claude-sonnet-5` | Use `claude-haiku-4-5-20251001` for cheaper answers |
| `APP_PASSWORD` | no | empty | If set, users must enter this access code |
| `TOP_K` | no | 8 | Number of note excerpts sent per question |

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                  # Windows: set ANTHROPIC_API_KEY=sk-ant-...
python app.py                                         # open http://localhost:5000
```

## Deploy on Render
1. Push this folder to a GitHub repository.
2. Render dashboard → **New → Blueprint** → pick the repo (uses `render.yaml`).
3. Paste your `ANTHROPIC_API_KEY` (and optional `APP_PASSWORD`) when asked → **Apply**.
4. Open the `https://<name>.onrender.com` URL once the deploy is live.
