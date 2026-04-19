import json
import os
import re

import anthropic
import httpx

from .professor_loader import strip_html

_TOOLS = [
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch the current live content of a webpage. "
            "Use this to find latest research directions, open PhD positions, "
            "recent publications, or lab news for a professor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"}
            },
            "required": ["url"],
        },
    }
]

_SYSTEM = """You are an expert academic advisor matching assistant for CMU's Machine Learning Department.
Given a student's research interests and a professor's profile, you will:
1. Assess research alignment on a 0-10 scale
2. Determine if the professor appears to be accepting new PhD students
3. Extract 3-4 keyword tags for the professor's research areas
4. Write a compelling 2-3 sentence match rationale for the student
5. Note one key finding from your research

You may use the fetch_webpage tool to retrieve the latest information from the professor's homepage or lab page if needed.

Always end with a JSON block in this exact format (no trailing commas):
```json
{
  "score": <int 0-10>,
  "keywords": ["tag1", "tag2", "tag3"],
  "accepting_students": <true|false|null>,
  "match_rationale": "<2-3 sentences>",
  "key_finding": "<one sentence>"
}
```"""


class ResearchAgent:
    def __init__(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key or api_key == "your_key_here":
            raise ValueError("ANTHROPIC_API_KEY is not set in environment / .env")
        base_url = os.environ.get("ANTHROPIC_BASE_URL") or None
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = anthropic.AsyncAnthropic(**kwargs)

    async def analyze_professor(self, professor: dict, user_query: str) -> dict:
        name = professor["name"]
        title = professor.get("title", "")
        homepage_url = professor.get("homepage_url", "")
        scholar_url = professor.get("google_scholar_url", "")
        similarity = professor.get("similarity_score", 0.0)

        clean_content = strip_html(professor.get("homepage_content", ""))[:3000]

        user_msg = f"""Student's research interests:
{user_query}

---
Professor: {name} ({title})
CMU Machine Learning Department
Homepage: {homepage_url}
Google Scholar: {scholar_url}

Homepage content snapshot (may be outdated):
{clean_content}

---
Please analyze this professor as a potential PhD advisor match. You may fetch their homepage for more current information if needed. End with the JSON summary block."""

        messages = [{"role": "user", "content": user_msg}]

        for step in range(3):
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=_SYSTEM,
                tools=_TOOLS,
                messages=messages,
            )
            print(f"  [{name}] step={step} stop_reason={response.stop_reason}", flush=True)

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        url = block.input.get("url", "")
                        print(f"  [{name}] TOOL CALL fetch_webpage({url})", flush=True)
                        fetched = await _fetch_page(url)
                        print(f"  [{name}] fetched {len(fetched)} chars", flush=True)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": fetched[:3000],
                            }
                        )
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                break

        full_text = " ".join(
            block.text for block in response.content if hasattr(block, "text")
        )

        parsed = _parse_json(full_text)
        raw_score = parsed.get("score", 5)
        try:
            normalized_score = min(1.0, max(0.0, float(raw_score) / 10.0))
        except (TypeError, ValueError):
            normalized_score = similarity

        return {
            "name": name,
            "title": title,
            "homepage_url": homepage_url,
            "google_scholar_url": scholar_url,
            "department": professor.get("department", "Machine Learning Department"),
            "university": professor.get("university", "Carnegie Mellon University"),
            "similarity_score": similarity,
            "score": normalized_score,
            "keywords": parsed.get("keywords", []),
            "accepting_students": parsed.get("accepting_students"),
            "match_rationale": parsed.get("match_rationale", ""),
            "key_finding": parsed.get("key_finding", ""),
        }


async def _fetch_page(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return "Invalid URL"
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FindMyAdvisor/1.0)"},
        ) as client:
            resp = await client.get(url)
            return strip_html(resp.text)
    except Exception as exc:
        return f"Could not fetch {url}: {exc}"


def _parse_json(text: str) -> dict:
    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to finding any JSON object containing "score"
    m = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
