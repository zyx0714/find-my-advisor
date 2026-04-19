#!/usr/bin/env python3
import html
import json
import re
import subprocess
from typing import Any
from urllib.parse import quote_plus


INDEX_URLS = [
    "https://ml.cmu.edu/peopleindexes/core-faculty-index.v1.json",
]

DIRECTORY_URL = "https://www.cs.cmu.edu/directory/api/v1/all.json"

UNIVERSITY = "Carnegie Mellon University"
DEFAULT_DEPARTMENT = "Machine Learning Department"


def fetch_json(url: str) -> dict[str, Any]:
    output = subprocess.check_output(["curl", "-Ls", url], text=True)
    return json.loads(output)


def fetch_text(url: str) -> str:
    return subprocess.check_output(["curl", "-Ls", url], text=True)


def fetch_page_cached(url: str, page_cache: dict[str, str]) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    if url in page_cache:
        return page_cache[url]
    try:
        page_cache[url] = fetch_text(url)
    except Exception:
        page_cache[url] = ""
    return page_cache[url]


def fetch_directory_map() -> dict[str, dict[str, Any]]:
    payload = fetch_json(DIRECTORY_URL)
    result: dict[str, dict[str, Any]] = {}
    for person in payload.get("data", []):
        if not isinstance(person, dict):
            continue
        person_id = normalize_text(person.get("id"))
        if person_id:
            result[person_id] = person
    return result


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return html.unescape(value).strip()
    if isinstance(value, list):
        parts = [html.unescape(str(item)).strip() for item in value if str(item).strip() and str(item).strip().lower() != "none"]
        return "; ".join(parts)
    return html.unescape(str(value)).strip()


def pick_homepage(person: dict[str, Any]) -> str:
    social = person.get("soc")
    if isinstance(social, list):
        for item in social:
            if not isinstance(item, dict):
                continue
            url = normalize_text(item.get("url"))
            if url and url.lower().startswith(("http://", "https://")):
                return url
    return normalize_text(person.get("href"))


def pick_google_scholar(person: dict[str, Any]) -> str:
    social = person.get("soc")
    if isinstance(social, list):
        for item in social:
            if not isinstance(item, dict):
                continue
            url = normalize_text(item.get("url"))
            if "scholar.google" in url.lower():
                return url
    for key in ("google_scholar", "googleScholar", "scholar"):
        value = normalize_text(person.get(key))
        if "scholar.google" in value.lower():
            return value
    return ""


def extract_scholar_url_from_page(url: str, page_cache: dict[str, str]) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    page = fetch_page_cached(url, page_cache)
    if not page:
        return ""

    match = re.search(r'https?://scholar\.google\.[^"\'\s<>)]*', page, flags=re.IGNORECASE)
    if not match:
        return ""

    scholar_url = html.unescape(match.group(0)).strip()
    scholar_url = scholar_url.rstrip('"\'.,);')
    return scholar_url


def build_scholar_search_url(name: str) -> str:
    name = normalize_text(name)
    if not name:
        return ""
    query = f'"{name}" "Carnegie Mellon"'
    return f"https://scholar.google.com/scholar?q={quote_plus(query)}"


def extract_name_from_profile(profile_url: str) -> str:
    if not profile_url.startswith(("http://", "https://")):
        return ""
    try:
        page = fetch_text(profile_url)
    except Exception:
        return ""

    patterns = [
        r'<meta\s+property="og:title"\s+content="([^"]+)"',
        r'<title>\s*([^<]+)\s*</title>',
    ]
    for pattern in patterns:
        match = re.search(pattern, page, flags=re.IGNORECASE)
        if not match:
            continue
        title_text = html.unescape(match.group(1)).strip()
        if " - " in title_text:
            candidate = title_text.split(" - ", 1)[0].strip()
        else:
            candidate = title_text
        if candidate.lower() in {"machine learning", "machine learning department"}:
            continue
        if candidate and len(candidate) <= 120:
            return candidate
    return ""


def extract_name_from_image_url(image_url: str) -> str:
    image_url = normalize_text(image_url)
    if not image_url:
        return ""
    filename = image_url.rsplit("/", 1)[-1]
    filename = re.sub(r"\.[a-zA-Z0-9]+$", "", filename)
    filename = re.sub(r"[-_]?(cropped|min|sq|web|web2|web3|small|medium|large|headshot|relaxed|profile|photo|final|new|copy)\b", "", filename, flags=re.IGNORECASE)
    tokens = re.findall(r"[A-Za-z]+", filename)
    if len(tokens) < 2:
        return ""
    return " ".join(token.capitalize() for token in tokens)


def to_record(
    person: dict[str, Any],
    directory_person: dict[str, Any] | None,
    name_cache: dict[str, str],
    scholar_cache: dict[str, str],
    page_cache: dict[str, str],
) -> dict[str, str]:
    homepage = pick_homepage(person)
    profile_url = normalize_text(person.get("href"))
    name = normalize_text(person.get("n"))
    if not name and directory_person:
        name = normalize_text(directory_person.get("n"))
    if not name:
        if profile_url:
            if profile_url not in name_cache:
                name_cache[profile_url] = extract_name_from_profile(profile_url)
            name = name_cache.get(profile_url, "")
    if not name:
        name = extract_name_from_image_url(normalize_text(person.get("img")))

    scholar = pick_google_scholar(person)
    if not scholar and profile_url:
        if profile_url not in scholar_cache:
            scholar_cache[profile_url] = extract_scholar_url_from_page(profile_url, page_cache)
        scholar = scholar_cache.get(profile_url, "")
    if not scholar and homepage and homepage != profile_url:
        if homepage not in scholar_cache:
            scholar_cache[homepage] = extract_scholar_url_from_page(homepage, page_cache)
        scholar = scholar_cache.get(homepage, "")
    if not scholar:
        scholar = build_scholar_search_url(name)

    homepage_content = fetch_page_cached(homepage, page_cache) if homepage else ""
    if not homepage_content and profile_url and profile_url != homepage:
        homepage_content = fetch_page_cached(profile_url, page_cache)

    return {
        "name": name,
        "title": normalize_text(person.get("titl")),
        "university": UNIVERSITY,
        "department": DEFAULT_DEPARTMENT,
        "homepage_url": homepage,
        "homepage_content": homepage_content,
        "google_scholar_url": scholar,
    }


def merge_records(base: dict[str, str], extra: dict[str, str]) -> dict[str, str]:
    merged = dict(base)
    for key, value in extra.items():
        if not merged.get(key) and value:
            merged[key] = value
    return merged


def main() -> None:
    by_id: dict[str, dict[str, str]] = {}
    name_cache: dict[str, str] = {}
    scholar_cache: dict[str, str] = {}
    page_cache: dict[str, str] = {}
    directory_map = fetch_directory_map()
    for url in INDEX_URLS:
        payload = fetch_json(url)
        for person in payload.get("data", []):
            if not isinstance(person, dict):
                continue
            person_id = normalize_text(person.get("id")) or normalize_text(person.get("href"))
            if not person_id:
                continue
            directory_person = directory_map.get(person_id)
            record = to_record(person, directory_person, name_cache, scholar_cache, page_cache)
            existing = by_id.get(person_id)
            if existing is None:
                by_id[person_id] = record
            else:
                by_id[person_id] = merge_records(existing, record)

    output = sorted(by_id.values(), key=lambda item: item.get("name", ""))
    with open("cmu_mld_professors.json", "w", encoding="utf-8") as fp:
        json.dump(output, fp, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output)} records to cmu_mld_professors.json")


if __name__ == "__main__":
    main()