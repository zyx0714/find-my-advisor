import json
import re
import os

_BASE = os.path.join(os.path.dirname(__file__), "..")

_DEPT_FILES = {
    "mld": ("cmu_mld_professors.json", "json"),
    "csd": ("csd_faculty_final.jsonl", "jsonl"),
    "lti": ("lti_faculty.jsonl", "jsonl"),
    "ri":  ("ri_faculty.jsonl", "jsonl"),
}


def load_professors(department: str = "mld") -> list[dict]:
    key = department.lower()
    if key not in _DEPT_FILES:
        raise ValueError(f"Unknown department '{department}'. Choose from: {list(_DEPT_FILES)}")
    filename, fmt = _DEPT_FILES[key]
    path = os.path.join(_BASE, filename)
    with open(path, encoding="utf-8") as f:
        if fmt == "jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def professor_index_text(p: dict, max_chars: int = 3000) -> str:
    parts = [
        p.get("name", ""),
        p.get("title", ""),
        strip_html(p.get("homepage_content", ""))[:max_chars],
    ]
    return ". ".join(x for x in parts if x)
