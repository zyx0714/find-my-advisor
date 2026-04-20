import json
import os
import re

import anthropic
import httpx

from .professor_loader import strip_html

_TOOLS = [
    {
        "name": "web_search",
        "description": (
            "Search the web for information about a professor. "
            "Use this to find: whether they are recruiting PhD students, recent grants/funding, "
            "recent publications, lab news, or any other current information. "
            "Returns top search result titles, URLs, and snippets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'Graham Neubig CMU recruiting PhD students 2025'"}
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_webpage",
        "description": (
            "Fetch the full content of a specific webpage URL. "
            "Use this after web_search to read a promising page in detail, "
            "or to fetch the professor's homepage, lab page, or Google Scholar page directly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL to fetch"}
            },
            "required": ["url"],
        },
    },
]

_SYSTEM = """You are an expert academic advisor matching assistant.
Given a student's research interests and a professor's profile, you will:
1. Assess research alignment on a 0-10 scale
2. Determine if the professor appears to be accepting new PhD students
3. Extract 3-4 keyword tags for the professor's research areas
4. Write a compelling 2-3 sentence match rationale for the student
5. Note one key finding from your research (ideally something specific like a recent paper, grant, or recruiting status)

You have two tools:
- web_search: search for current information (use this first to find recruiting status, recent funding, new papers)
- fetch_webpage: fetch a specific URL in full detail

Strategy: use web_search to discover current information you wouldn't find in the static profile snapshot, then fetch_webpage if you need to read a specific page more carefully.

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
Homepage: {homepage_url}
Google Scholar: {scholar_url}

Homepage content snapshot (may be outdated):
{clean_content}

---
Please analyze this professor as a potential PhD advisor match.
Use web_search to find current recruiting status, recent papers, or funding information.
Then fetch a page if you need more detail.
End with the JSON summary block."""

        messages = [{"role": "user", "content": user_msg}]

        for step in range(4):
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
                        if block.name == "web_search":
                            query = block.input.get("query", "")
                            print(f"  [{name}] TOOL CALL web_search({query!r})", flush=True)
                            result = await _web_search(query)
                            print(f"  [{name}] search returned {len(result)} chars", flush=True)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            })
                        elif block.name == "fetch_webpage":
                            url = block.input.get("url", "")
                            print(f"  [{name}] TOOL CALL fetch_webpage({url})", flush=True)
                            fetched = await _fetch_page(url)
                            print(f"  [{name}] fetched {len(fetched)} chars", flush=True)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": fetched[:4000],
                            })
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


async def _web_search(query: str, num_results: int = 5) -> str:
    """Search DuckDuckGo and return top results as text. No API key needed."""
    search_url = "https://html.duckduckgo.com/html/"
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FindMyAdvisor/1.0)"},
        ) as client:
            resp = await client.post(search_url, data={"q": query})
            html = resp.text

        # Extract result titles, URLs, snippets from DuckDuckGo HTML
        results = []
        # Each result is in a div with class 'result'
        blocks = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</(?:a|span)>', html, re.DOTALL)

        for i, (url, title) in enumerate(blocks[:num_results]):
            title_clean = re.sub(r'<[^>]+>', '', title).strip()
            snippet_clean = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
            results.append(f"[{i+1}] {title_clean}\nURL: {url}\n{snippet_clean}")

        if not results:
            return f"No results found for: {query}"
        return "\n\n".join(results)

    except Exception as exc:
        return f"Search failed: {exc}"


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
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[^{}]*"score"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
