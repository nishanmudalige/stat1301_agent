"""
STAT1301 Notes Assistant
------------------------
A small web app that answers questions using ONLY the STAT1301 course notes.

How it works (Retrieval-Augmented Generation, "RAG"):
  1. At start-up the notes (data/notes.txt) are split into overlapping chunks,
     each tagged with the section it came from (e.g. "7.2 One-sample t-test").
  2. A BM25 keyword index is built over those chunks (fast, no extra API needed).
  3. For each question, the most relevant chunks are retrieved and sent to Claude
     together with the question. Claude answers from those excerpts and cites
     the sections it used.
"""

import os
import re
import hmac

from flask import Flask, jsonify, render_template, request
from rank_bm25 import BM25Okapi
import anthropic

# ---------------------------------------------------------------- settings ---
NOTES_PATH = os.environ.get("NOTES_PATH", "data/notes.txt")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
TOP_K = int(os.environ.get("TOP_K", "8"))              # chunks sent to Claude
CHUNK_CHARS = int(os.environ.get("CHUNK_CHARS", "1800"))
OVERLAP_CHARS = int(os.environ.get("OVERLAP_CHARS", "300"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")      # optional access code
MAX_QUESTION_CHARS = 2000
MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are a friendly, precise tutor for the university course \
STAT1301 "Advanced Analysis of Scientific Data".

Answer the student's question using ONLY the excerpts from the course notes \
provided in <notes>. Rules:
- If the excerpts contain the answer, explain it clearly, step by step where helpful, \
using the notation and terminology of the notes.
- Cite the section(s) you used in square brackets, e.g. [7.2 One-sample t-test].
- If the excerpts do not contain enough information, say so plainly \
("The notes don't cover this in the sections I found") rather than guessing. \
You may then suggest which topic to look up.
- Write maths in plain text or simple LaTeX between $...$ signs.
- Keep answers focused; use short paragraphs or lists."""

# ------------------------------------------------------- load & chunk notes ---
# Section headings look like "7.2 One-sample t-test" or "5 Descriptive Statistics"
HEADING_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\s+([A-Z][^\n]{2,80})$")


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        lines = f.read().replace("\f", "\n").split("\n")

    # Skip the table of contents (lines with dot leaders ". . . .").
    # A line only counts as a new heading if its number comes right after the
    # current one (e.g. 7.2 -> 7.3 or 7.7 -> 8), which filters out page
    # headers like "20 Understanding Randomness" and numbered R code lines.
    sections, current_title, buf = [], "Front matter", []
    cur = (0, 0)
    for line in lines:
        s = line.strip()
        if ". . ." in s:
            continue
        m = HEADING_RE.match(s)
        num = None
        if m and len(s) < 90 and not s.endswith(".") and not s[-1].isdigit():
            parts = [int(p) for p in m.group(1).split(".")]
            num = (parts[0], parts[1] if len(parts) > 1 else 0)
            is_next = (num[0] == cur[0] and num[1] == cur[1] + 1) or (
                num[0] == cur[0] + 1 and num[1] in (0, 1)
            )
            if not is_next:
                num = None
        if num:
            cur = num
            if buf:
                sections.append((current_title, "\n".join(buf)))
            current_title, buf = f"{m.group(1)} {m.group(2).strip()}", [s]
        else:
            buf.append(line)
    if buf:
        sections.append((current_title, "\n".join(buf)))

    # Split each section into overlapping windows
    chunks = []
    for title, text in sections:
        text = text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + CHUNK_CHARS, len(text))
            # try to end on a line break so we don't cut words/formulas in half
            if end < len(text):
                nl = text.rfind("\n", start + CHUNK_CHARS // 2, end)
                if nl != -1:
                    end = nl
            chunks.append({"section": title, "text": text[start:end].strip()})
            if end >= len(text):
                break
            start = max(end - OVERLAP_CHARS, start + 1)
    return chunks


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an and are as at be by for from how i in is it of on or that the this to "
    "what when where which who why with do does can you me explain tell about".split()
)


def tokenize(text):
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


CHUNKS = load_chunks(NOTES_PATH)
BM25 = BM25Okapi([tokenize(c["section"] + " " + c["text"]) for c in CHUNKS])
print(f"Loaded {len(CHUNKS)} chunks from {NOTES_PATH}", flush=True)


def retrieve(query, k=TOP_K):
    scores = BM25.get_scores(tokenize(query))
    ranked = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)
    return [CHUNKS[i] for i in ranked[:k] if scores[i] > 0]


# ------------------------------------------------------------------ web app ---
app = Flask(__name__)
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment


@app.get("/")
def index():
    return render_template("index.html", needs_password=bool(APP_PASSWORD))


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(CHUNKS), "model": MODEL}


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}

    if APP_PASSWORD and not hmac.compare_digest(
        str(data.get("password", "")), APP_PASSWORD
    ):
        return jsonify(error="Wrong or missing access code."), 401

    question = str(data.get("question", "")).strip()[:MAX_QUESTION_CHARS]
    if not question:
        return jsonify(error="Please type a question."), 400

    # Previous turns (so follow-ups like "give an example" work)
    history = []
    for turn in (data.get("history") or [])[-MAX_HISTORY_TURNS * 2 :]:
        role, content = turn.get("role"), str(turn.get("content", ""))[:4000]
        if role in ("user", "assistant") and content:
            history.append({"role": role, "content": content})
    while history and history[0]["role"] != "user":
        history.pop(0)

    # Retrieve with the question plus the last user question for context
    last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
    excerpts = retrieve(question + " " + last_user[:300])
    notes_block = "\n\n".join(
        f'<excerpt section="{c["section"]}">\n{c["text"]}\n</excerpt>' for c in excerpts
    ) or "(no matching excerpts found)"

    messages = history + [
        {"role": "user", "content": f"<notes>\n{notes_block}\n</notes>\n\nQuestion: {question}"}
    ]

    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=SYSTEM_PROMPT, messages=messages
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")
    except anthropic.APIError as e:
        app.logger.exception("Anthropic API error")
        return jsonify(error=f"The AI service returned an error: {e.__class__.__name__}"), 502

    sources = list(dict.fromkeys(c["section"] for c in excerpts))
    return jsonify(answer=answer, sources=sources)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
