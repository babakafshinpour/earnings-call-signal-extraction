"""
Scrape earnings call transcripts from Motley Fool.

Usage:
    python scrape_fool.py --url https://www.fool.com/earnings/call-transcripts/...
    python scrape_fool.py --urls urls.txt

Output:
    data/raw/{ticker}_{fiscal_period}.json with structure:
    {
        "ticker": "MSFT",
        "company": "Microsoft",
        "fiscal_period": "Q3-2026",
        "call_date": "2026-04-29",
        "source_url": "...",
        "scraped_at": "2026-05-29T...",
        "participants": [{"name": "...", "title": "...", "role": "ceo"}, ...],
        "segments": [
            {"section": "prepared_remarks", "speaker": "Satya Nadella",
             "role": "ceo", "text": "..."},
            ...
        ]
    }
"""
import argparse
import json
import re
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

USER_AGENT = "earnings-research-scraper/0.2 (portfolio project; contact: your@email)"
RATE_LIMIT_SEC = 1.0
CACHE_DIR = Path("data/cache")
OUT_DIR = Path("data/raw")


@dataclass
class Participant:
    name: str
    title: str
    role: str


@dataclass
class Segment:
    section: str           # "prepared_remarks" or "qa"
    speaker: str
    role: str
    text: str


@dataclass
class Transcript:
    ticker: str
    company: str
    fiscal_period: str
    call_date: str
    source_url: str
    scraped_at: str
    participants: list = field(default_factory=list)
    segments: list = field(default_factory=list)


# ---------------- HTTP ----------------

def fetch(url: str) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = re.sub(r"[^a-zA-Z0-9]", "_", urlparse(url).path)[-100:]
    cache_path = CACHE_DIR / f"{cache_key}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    time.sleep(RATE_LIMIT_SEC)
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    r.raise_for_status()
    cache_path.write_text(r.text, encoding="utf-8")
    return r.text


# ---------------- Role inference ----------------

ROLE_PATTERNS = [
    (re.compile(r"\bchief executive officer\b|\bceo\b", re.I), "ceo"),
    (re.compile(r"\bchief financial officer\b|\bcfo\b", re.I), "cfo"),
    (re.compile(r"\bpresident\b|\bchairman\b|\bchief\s+\w+\s+officer\b|"
                r"\bgeneral counsel\b|\bcorporate secretary\b|"
                r"\binvestor relations\b|\bhead of\b", re.I), "other_exec"),
    (re.compile(r"\banalyst\b|\bequity research\b|\bcapital markets\b|"
                r"\bsecurities\b|\bbernstein\b|\bmorgan stanley\b|"
                r"\bgoldman sachs\b|\bjefferies\b|\bevercore\b|\bubs\b|"
                r"\brbc\b", re.I), "analyst"),
    (re.compile(r"\boperator\b", re.I), "operator"),
]


def infer_role(title_or_context: str) -> str:
    for pat, role in ROLE_PATTERNS:
        if pat.search(title_or_context):
            return role
    return "unknown"


# ---------------- Parse: participants ----------------

def parse_participants(soup: BeautifulSoup) -> list[Participant]:
    """
    The 'Call participants' section is a <ul> following the
    'Call participants' heading. Each <li> is 'Title — Name' or 'Title -- Name'.
    """
    participants: list[Participant] = []
    heading = soup.find(lambda t: t.name in ("h2", "h3")
                        and "call participants" in t.get_text(strip=True).lower())
    if not heading:
        return participants

    ul = heading.find_next("ul")
    if not ul:
        return participants

    for li in ul.find_all("li"):
        text = li.get_text(" ", strip=True)
        # Split on em-dash, en-dash, or double-hyphen
        parts = re.split(r"\s+[—–]\s+|\s+--\s+", text, maxsplit=1)
        if len(parts) != 2:
            continue
        title, name = parts[0].strip(), parts[1].strip()
        participants.append(Participant(
            name=name, title=title, role=infer_role(title)
        ))
    return participants


# ---------------- Parse: transcript body ----------------

# Phrase-level QA triggers. These are weak signals — many calls mention
# "questions" during prepared remarks. We only trust them once we've also
# seen the operator speak at least once (housekeeping), and we require the
# operator to actually introduce an analyst as the strong signal.
QA_TRANSITION_PATTERNS = [
    re.compile(r"open the call (now )?for questions", re.I),
    re.compile(r"begin the question[- ]and[- ]answer session", re.I),
    re.compile(r"we will (now )?(begin|take) the q[\.&]a", re.I),
]

# Strong signal: operator introducing the first analyst question.
OPERATOR_FIRST_QUESTION = re.compile(
    r"(first|next) question (comes |is )?from|"
    r"line of .+ with .+\.|"
    r"please go ahead",
    re.I,
)


def find_transcript_root(soup: BeautifulSoup):
    """
    Find the heading 'Full Conference Call Transcript' (or close variant).
    Returns the heading itself; we then walk forward through siblings.
    """
    heading = soup.find(lambda t: t.name in ("h2", "h3")
                        and "conference call transcript" in t.get_text(strip=True).lower())
    if heading:
        return heading
    # Fallback: main article body
    return (soup.find("article")
            or soup.find("div", class_=re.compile("article-body|tailwind-article-body"))
            or soup)


def iter_transcript_paragraphs(root):
    """
    Yield <p> tags that come *after* the 'Full Conference Call Transcript'
    heading, until we hit the 'Read Next' / 'Stocks Mentioned' / disclaimer.
    """
    if root.name in ("h2", "h3"):
        for sib in root.find_all_next():
            if sib.name in ("h2", "h3"):
                t = sib.get_text(strip=True).lower()
                if "read next" in t or "stocks mentioned" in t or "premium" in t:
                    break
            if sib.name == "p":
                yield sib
    else:
        for p in root.find_all("p"):
            yield p


def extract_speaker_and_text(p: Tag):
    """
    If the paragraph begins with <strong>Name:</strong>, return (name, rest_of_text).
    Otherwise return (None, full_text).
    """
    first_child = None
    for c in p.children:
        if isinstance(c, NavigableString) and not c.strip():
            continue
        first_child = c
        break

    if isinstance(first_child, Tag) and first_child.name in ("strong", "b"):
        label = first_child.get_text(strip=True)
        if label.endswith(":") and len(label) < 120:
            speaker_name = label[:-1].strip()
            remaining = []
            seen_strong = False
            for c in p.children:
                if not seen_strong:
                    if c is first_child:
                        seen_strong = True
                    continue
                if isinstance(c, NavigableString):
                    remaining.append(str(c))
                else:
                    remaining.append(c.get_text(" ", strip=False))
            return speaker_name, " ".join("".join(remaining).split()).strip()

    return None, p.get_text(" ", strip=True)


def parse_transcript_body(soup: BeautifulSoup,
                          participants: list):
    name_to_role = {p.name: p.role for p in participants}

    root = find_transcript_root(soup)
    if root is None:
        return []

    segments = []
    current_section = "prepared_remarks"
    current_speaker = None
    current_role = "unknown"
    current_buffer = []
    seen_operator = False

    def flush():
        if current_buffer and current_speaker:
            segments.append(Segment(
                section=current_section,
                speaker=current_speaker,
                role=current_role,
                text=" ".join(current_buffer).strip(),
            ))

    for p in iter_transcript_paragraphs(root):
        full_text = p.get_text(" ", strip=True)
        if not full_text:
            continue

        speaker, text = extract_speaker_and_text(p)

        # Strong QA trigger: operator paragraph that introduces an analyst.
        # This is far more reliable than free-text phrase matching, since
        # "questions" gets mentioned during prepared remarks frequently.
        strong_trigger = False
        if current_section == "prepared_remarks" and speaker is not None:
            speaker_role = name_to_role.get(speaker) or infer_role(speaker)
            if speaker_role == "operator" and OPERATOR_FIRST_QUESTION.search(text or full_text):
                strong_trigger = True

        # Weak phrase trigger: only honored once we've already seen the
        # operator speak (which means housekeeping is done and we're at
        # the natural QA boundary). Tracked via seen_operator flag below.
        weak_trigger = (current_section == "prepared_remarks"
                        and seen_operator
                        and any(pat.search(full_text) for pat in QA_TRANSITION_PATTERNS))

        if speaker is not None:
            # Flush previous segment under OLD section before any changes.
            flush()
            current_buffer = []
            if strong_trigger or weak_trigger:
                current_section = "qa"
            current_speaker = speaker
            current_role = name_to_role.get(speaker) or infer_role(speaker)
            if current_role == "operator":
                seen_operator = True
            if text:
                current_buffer.append(text)
        else:
            # Continuation paragraph.
            if weak_trigger:
                flush()
                current_buffer = []
                current_section = "qa"
            if current_speaker:
                current_buffer.append(text)

    flush()
    return segments


# ---------------- Parse: metadata ----------------

def parse_metadata(soup: BeautifulSoup, url: str) -> dict:
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    ticker_match = re.search(r"\(([A-Z\.]{1,6})\)", title)
    period_match = re.search(r"(Q[1-4])\s+(\d{4})", title, re.I)
    company = title.split("(")[0].strip() if "(" in title else ""
    ticker = ticker_match.group(1) if ticker_match else ""
    fiscal_period = (f"{period_match.group(1).upper()}-{period_match.group(2)}"
                     if period_match else "")

    call_date = ""
    date_heading = soup.find(lambda t: t.name in ("h2", "h3")
                             and t.get_text(strip=True).lower() == "date")
    if date_heading:
        nxt = date_heading.find_next(["p", "div"])
        if nxt:
            txt = nxt.get_text(" ", strip=True)
            m = re.search(r"([A-Z][a-z]+,?\s+[A-Z][a-z]+\.?\s+\d{1,2},\s+\d{4})", txt)
            if m:
                for fmt in ("%A, %b. %d, %Y", "%A, %B %d, %Y"):
                    try:
                        call_date = datetime.strptime(m.group(1), fmt).date().isoformat()
                        break
                    except ValueError:
                        continue
    if not call_date:
        m = re.search(r"([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", soup.get_text())
        if m:
            for fmt in ("%b %d, %Y", "%B %d, %Y"):
                try:
                    call_date = datetime.strptime(m.group(1), fmt).date().isoformat()
                    break
                except ValueError:
                    continue

    return {
        "ticker": ticker,
        "company": company,
        "fiscal_period": fiscal_period,
        "call_date": call_date,
    }


# ---------------- Main parse ----------------

def parse_fool_transcript(html: str, source_url: str) -> Transcript:
    soup = BeautifulSoup(html, "html.parser")
    meta = parse_metadata(soup, source_url)
    participants = parse_participants(soup)
    segments = parse_transcript_body(soup, participants)

    return Transcript(
        ticker=meta["ticker"],
        company=meta["company"],
        fiscal_period=meta["fiscal_period"],
        call_date=meta["call_date"],
        source_url=source_url,
        scraped_at=datetime.now(timezone.utc).isoformat(),
        participants=[asdict(p) for p in participants],
        segments=[asdict(s) for s in segments],
    )


# ---------------- CLI ----------------

def save(transcript: Transcript) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = f"{transcript.ticker}_{transcript.fiscal_period}.json"
    out_path = OUT_DIR / fname
    out_path.write_text(json.dumps(asdict(transcript), indent=2), encoding="utf-8")
    return out_path


def scrape_one(url: str) -> Path:
    html = fetch(url)
    t = parse_fool_transcript(html, url)
    out = save(t)
    n_prep = sum(1 for s in t.segments if s["section"] == "prepared_remarks")
    n_qa = sum(1 for s in t.segments if s["section"] == "qa")
    print(f"  saved {out}")
    print(f"    ticker={t.ticker}  period={t.fiscal_period}  date={t.call_date}")
    print(f"    {len(t.participants)} participants, "
          f"{len(t.segments)} segments ({n_prep} prepared, {n_qa} Q&A)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Single transcript URL")
    ap.add_argument("--urls", help="Path to file with one URL per line")
    args = ap.parse_args()

    urls = []
    if args.url:
        urls.append(args.url)
    if args.urls:
        urls.extend([u.strip() for u in Path(args.urls).read_text().splitlines() if u.strip()])

    if not urls:
        ap.error("provide --url or --urls")

    for u in urls:
        print(f"scraping {u}")
        try:
            scrape_one(u)
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
