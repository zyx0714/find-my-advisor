import asyncio
import io
import json
import os
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from .embedder import Embedder
from .professor_loader import load_professors
from .research_agent import ResearchAgent

# dept_key -> (professors, embedder)
_indexes: dict[str, tuple[list[dict], Embedder]] = {}
_jobs: dict[str, asyncio.Queue] = {}

_DEPARTMENTS = ["mld", "csd", "lti", "ri"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = Embedder()  # load model once
    for dept in _DEPARTMENTS:
        profs = load_professors(dept)
        print(f"[Startup] Building index for {dept} ({len(profs)} professors)...", flush=True)
        # clone embedder state per dept by sharing the same model
        from .embedder import Embedder as _Emb
        import copy
        dept_emb = copy.copy(embedder)  # share model weights, separate matrix
        dept_emb.professors = []
        dept_emb.matrix = None
        await asyncio.to_thread(dept_emb.build_index, profs)
        _indexes[dept] = (profs, dept_emb)
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    research_statement: str
    cv_text: str = ""
    department: str = "mld"


@app.post("/api/search")
async def start_search(req: SearchRequest):
    dept = req.department.lower()
    if dept not in _indexes:
        return {"error": f"Department '{dept}' not available."}
    job_id = str(uuid.uuid4())
    _jobs[job_id] = asyncio.Queue()
    asyncio.create_task(_run_pipeline(job_id, req))
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
async def stream_results(job_id: str):
    if job_id not in _jobs:
        async def _err():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream")

    async def _generate():
        q = _jobs[job_id]
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event.get("type") in ("done", "error"):
                break
        _jobs.pop(job_id, None)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class TailorRequest(BaseModel):
    professor: dict
    user_query: str
    cv_text: str = ""


@app.post("/api/tailor")
async def tailor_sop(req: TailorRequest):
    suggestions = await _generate_sop_suggestions(req.professor, req.user_query, req.cv_text)
    return {"suggestions": suggestions}


async def _generate_sop_suggestions(professor: dict, user_query: str, cv_text: str) -> list[dict]:
    import os, anthropic, json as _json, re

    client = anthropic.AsyncAnthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
    )
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    prof_name = professor.get("name", "the professor")
    prof_title = professor.get("title", "")
    match_rationale = professor.get("match_rationale", "")
    key_finding = professor.get("key_finding", "")
    keywords = ", ".join(professor.get("keywords") or [])

    has_cv = bool(cv_text.strip())

    if has_cv:
        user_content = f"""Professor: {prof_name} ({prof_title}), CMU MLD
Research keywords: {keywords}
Match rationale: {match_rationale}
Key finding from deep research: {key_finding}

Student's CV / SOP text:
{cv_text[:4000]}

Student's research interests:
{user_query[:1000]}

---
Generate 2-3 specific, actionable SOP revision suggestions that will strengthen the student's application to {prof_name}.

For each suggestion:
1. Quote a SHORT passage from the student's CV/SOP that could be improved (keep original under 30 words)
2. Rewrite that passage to directly echo {prof_name}'s research language and current projects
3. Explain in one sentence WHY this revision improves alignment

Return ONLY a JSON array (no markdown):
[
  {{
    "section": "which section of CV/SOP this is from",
    "original": "the original text from the student's materials",
    "revised": "the improved version that references the professor's work",
    "reason": "one sentence explaining the alignment improvement"
  }}
]"""
    else:
        user_content = f"""Professor: {prof_name} ({prof_title}), CMU MLD
Research keywords: {keywords}
Match rationale: {match_rationale}
Key finding from deep research: {key_finding}

Student's research interests:
{user_query[:1000]}

---
The student hasn't uploaded a CV, so generate 2-3 concrete writing suggestions for their cold-email or SOP targeting {prof_name}.

For each suggestion provide a specific example sentence they should write.

Return ONLY a JSON array (no markdown):
[
  {{
    "section": "Cold Email / SOP",
    "original": "Generic version: [a weak generic phrasing to avoid]",
    "revised": "Strong version: [a specific, compelling phrasing referencing the prof's work]",
    "reason": "one sentence explaining why this lands better"
  }}
]"""

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": user_content}],
    )
    raw = response.content[0].text.strip()

    # strip markdown fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return _json.loads(raw)
    except Exception:
        # fallback: return single item with raw text
        return [{"section": "General", "original": "", "revised": raw, "reason": ""}]


@app.post("/api/parse-file")
async def parse_file(file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or ""
    try:
        if filename.lower().endswith(".pdf"):
            text = _parse_pdf(content)
        else:
            text = content.decode("utf-8", errors="replace")
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return {"text": text, "filename": filename}


def _parse_pdf(data: bytes) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    path = os.path.join(os.path.dirname(__file__), "..", "frontend.html")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


async def _run_pipeline(job_id: str, req: SearchRequest) -> None:
    q = _jobs[job_id]
    dept = req.department.lower()
    profs, embedder = _indexes[dept]
    dept_label = {"mld": "MLD", "csd": "CSD", "lti": "LTI", "ri": "RI"}.get(dept, dept.upper())

    try:
        await q.put(
            {
                "type": "step",
                "icon": "📄",
                "title": "Parsing Research Profile",
                "desc": "Analyzing your research interests and background...",
            }
        )
        await asyncio.sleep(0.3)

        query = req.research_statement
        if req.cv_text:
            query = f"{req.research_statement}\n\n{req.cv_text}"

        await q.put(
            {
                "type": "step",
                "icon": "🔍",
                "title": "Semantic Retrieval",
                "desc": f"Querying BAAI/bge-m3 embeddings against {len(profs)} {dept_label} faculty profiles...",
            }
        )

        top_profs = await asyncio.to_thread(embedder.search, query, 10)

        await q.put(
            {
                "type": "step",
                "icon": "🌐",
                "title": "Deep Research (ReAct Agents)",
                "desc": f"Analyzing top {len(top_profs)} candidates with Claude...",
            }
        )

        agent = ResearchAgent()
        results = []
        for i, prof in enumerate(top_profs):
            await q.put(
                {
                    "type": "analyzing",
                    "name": prof["name"],
                    "index": i + 1,
                    "total": len(top_profs),
                }
            )
            result = await agent.analyze_professor(prof, query)
            results.append(result)

        results.sort(key=lambda x: x.get("score", 0), reverse=True)

        await q.put(
            {
                "type": "step",
                "icon": "✨",
                "title": "Generating Recommendations",
                "desc": "Synthesizing personalized advisor matches...",
            }
        )
        await asyncio.sleep(0.5)

        await q.put({"type": "results", "professors": results})
        await q.put({"type": "done"})

    except Exception as exc:
        await q.put({"type": "error", "message": str(exc)})
