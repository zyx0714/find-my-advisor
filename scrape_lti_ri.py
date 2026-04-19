#!/usr/bin/env python3
"""Scraper for CMU LTI and RI faculty."""
import html
import json
import re
import subprocess
import sys
import time
from urllib.parse import urljoin

UNIVERSITY = "Carnegie Mellon University"


def fetch(url: str, retries: int = 2) -> str:
    for attempt in range(retries + 1):
        try:
            raw = subprocess.check_output(
                ["curl", "-Ls", "--max-time", "15", "-A", "Mozilla/5.0", url],
            )
            return raw.decode("utf-8", errors="replace")
        except subprocess.CalledProcessError:
            if attempt == retries:
                return ""
            time.sleep(2)
    return ""


def strip_html(raw: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_scholar(page: str) -> str:
    m = re.search(r"https?://scholar\.google\.[^\s\"'<>)]+", page, re.IGNORECASE)
    if not m:
        return ""
    return html.unescape(m.group(0)).rstrip("\"'.,);")


# ─── LTI ──────────────────────────────────────────────────────────────────────

LTI_BASE = "https://lti.cmu.edu/people/faculty/"


def scrape_lti() -> list[dict]:
    print("[LTI] Fetching faculty list...", flush=True)
    index_html = fetch(LTI_BASE)
    pairs = re.findall(
        r"<h[23][^>]*>\s*<a[^>]+href=\"([^\"]+)\"[^>]*>(.*?)</a>",
        index_html, re.DOTALL,
    )
    faculty = []
    for rel_url, raw_name in pairs:
        name = re.sub(r"\s+", " ", strip_html(raw_name)).strip()
        if not name or "School" in name or len(name) < 3:
            continue
        profile_url = urljoin(LTI_BASE, rel_url)
        faculty.append((name, profile_url))

    print(f"[LTI] Found {len(faculty)} faculty. Fetching profiles...", flush=True)
    records = []
    for i, (name, profile_url) in enumerate(faculty):
        print(f"  [{i+1}/{len(faculty)}] {name}", flush=True)
        page = fetch(profile_url)
        scholar = extract_scholar(page)
        title = _extract_lti_title(page, name)

        records.append({
            "name": name,
            "title": title,
            "university": UNIVERSITY,
            "department": "Language Technologies Institute",
            "homepage_url": profile_url,
            "google_scholar_url": scholar,
            "homepage_content": page,
        })
        time.sleep(0.3)
    return records


def _extract_lti_title(page: str, name: str = "") -> str:
    # Try structured title tags first
    for pattern in [
        r'<(?:div|p|span)[^>]*class="[^"]*(?:title|position|role)[^"]*"[^>]*>(.*?)</(?:div|p|span)>',
        r'<(?:div|p|span)[^>]*id="[^"]*(?:title|position)[^"]*"[^>]*>(.*?)</(?:div|p|span)>',
    ]:
        m = re.search(pattern, page, re.DOTALL | re.IGNORECASE)
        if m:
            t = strip_html(m.group(1)).strip()
            if t and t != name and len(t) < 120:
                return t

    # Scan stripped page text for name then title
    text = strip_html(page)
    if name:
        idx = text.find(name)
        if idx >= 0:
            after = text[idx + len(name):idx + len(name) + 300]
            for line in re.split(r"[|\n]", after):
                line = line.strip()
                if line and line != name and len(line) < 100 and not line.startswith("http"):
                    return line

    # Fallback: keyword regex
    m = re.search(
        r"\b((?:Associate |Assistant |Research |Adjunct |Emeritus |Teaching )*Professor"
        r"|(?:Senior |Principal )?Research(?:er| Scientist|Fellow)?)\b",
        text[:5000], re.IGNORECASE,
    )
    return m.group(0).strip() if m else ""


# ─── RI ───────────────────────────────────────────────────────────────────────

RI_FACULTY_BASE = "https://www.ri.cmu.edu/people/faculty/"


def _fetch_ri_cards(page_num: int) -> list[str]:
    url = RI_FACULTY_BASE if page_num == 1 else f"{RI_FACULTY_BASE}page/{page_num}/"
    page = fetch(url)
    return re.findall(r"<div class='ri_person'>(.*?)</div>\s*</div>", page, re.DOTALL)


def scrape_ri() -> list[dict]:
    print("[RI] Fetching faculty list (3 pages)...", flush=True)
    all_cards: list[str] = []
    for pg in range(1, 10):
        cards = _fetch_ri_cards(pg)
        if not cards:
            break
        print(f"  Page {pg}: {len(cards)} cards", flush=True)
        all_cards.extend(cards)

    faculty = []
    seen: set[str] = set()
    for card in all_cards:
        link_m = re.search(r"href='(https://www\.ri\.cmu\.edu/ri-faculty/[^']+)'", card)
        name_m = re.search(r"href='[^']+'>\s*([^<]+)\s*</a>", card)
        if not link_m or not name_m:
            continue
        name = name_m.group(1).strip()
        if name in seen:
            continue
        seen.add(name)
        faculty.append({
            "name": name,
            "profile_url": link_m.group(1).strip(),
        })

    print(f"[RI] Found {len(faculty)} faculty. Fetching profiles...", flush=True)
    records = []
    for i, info in enumerate(faculty):
        name = info["name"]
        profile_url = info["profile_url"]
        print(f"  [{i+1}/{len(faculty)}] {name}", flush=True)

        profile_page = fetch(profile_url)
        scholar = extract_scholar(profile_page)
        title = _extract_ri_title(profile_page, name)

        records.append({
            "name": name,
            "title": title,
            "university": UNIVERSITY,
            "department": "Robotics Institute",
            "homepage_url": profile_url,
            "google_scholar_url": scholar,
            "homepage_content": profile_page,
        })
        time.sleep(0.3)
    return records


# ─── Main ─────────────────────────────────────────────────────────────────────

def _extract_ri_title(page: str, name: str = "") -> str:
    text = strip_html(page)
    title_re = re.compile(
        r"\b((?:Associate |Assistant |Research |Adjunct |Emeritus |Teaching |Systems )*Professor"
        r"|(?:Senior |Principal |Systems )?Scientist"
        r"|(?:Senior |Principal )?Research(?:er| Scientist| Engineer| Faculty)?)\b",
        re.IGNORECASE,
    )
    # Search from after the second occurrence of name (first is page title, second is profile heading)
    if name:
        idx1 = text.find(name)
        if idx1 >= 0:
            idx2 = text.find(name, idx1 + 1)
            start = idx2 if idx2 >= 0 else idx1
            m = title_re.search(text, start, start + 500)
            if m:
                return m.group(0).strip()
    m = title_re.search(text[500:15000])
    return m.group(0).strip() if m else ""


def save_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}", flush=True)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "both"

    if target in ("lti", "both"):
        lti = scrape_lti()
        save_jsonl(lti, "lti_faculty.jsonl")

    if target in ("ri", "both"):
        ri = scrape_ri()
        save_jsonl(ri, "ri_faculty.jsonl")
