"""FinRAG Universal — FastAPI server (replaces Streamlit app.py)."""
import os
import io
import time
import json
import math
import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ── Lazy-load heavy modules once ─────────────────────────────────────
from core.hybrid_retriever import HybridRetriever
from core.llm_client import GroqClient
from domains import get_domain
from utils.pdf_parser import extract_pdfs, chunk_text
from config import UPLOAD_DIR

# ── App ───────────────────────────────────────────────────────────────
app = FastAPI(title="FinRAG Universal", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons ────────────────────────────────────────────────────────
retriever = None
llm       = None

def get_retriever():
    global retriever
    if retriever is None:
        retriever = HybridRetriever()
    return retriever

def get_llm():
    global llm
    if llm is None:
        llm = GroqClient()
    return llm

# ── Analytics history (rolling window) ───────────────────────────────
_analytics_history: list[float] = []

# ── Static files (frontend) ───────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ═══════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return HTMLResponse(content=index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found. Place index.html in /frontend/</h1>")


@app.get("/api/status")
async def status():
    """Health check + doc count."""
    return {
        "ok": True,
        "docs_indexed": len(get_retriever().vector),
        "graph_nodes":  get_retriever().graph.number_of_nodes(),
        "model":        get_llm().model,
        "domains":      ["financial", "clinical", "cybersecurity", "manufacturing", "others"],
    }


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    domain: str = Form("others"),
):
    """
    Upload one or more PDFs, parse → chunk → index into FAISS + graph.
    Returns processing summary per file.
    """
    results = []
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    for uf in files:
        t0 = time.time()

        # Size guard (10 MB)
        content = await uf.read()
        if len(content) > 10_000_000:
            results.append({
                "file":    uf.filename,
                "status":  "error",
                "message": "File exceeds 10 MB limit",
                "chunks":  0,
            })
            continue

        # Save to disk
        dest = upload_dir / uf.filename
        dest.write_bytes(content)

        # Parse PDF (max 20 pages)
        pages = extract_pdfs([dest], max_pages=20)
        if not pages:
            results.append({
                "file":    uf.filename,
                "status":  "error",
                "message": "No extractable text found",
                "chunks":  0,
            })
            continue

        # Chunk (max 5 pages × 3 chunks)
        chunks = []
        for page in pages[:5]:
            for tc in chunk_text(page["text"], chunk_size=800, overlap=20)[:3]:
                chunks.append({
                    "text":   tc,
                    "source": page["source"],
                    "page":   page["page"],
                })

        # Index
        get_retriever().add_documents(chunks)

        elapsed = time.time() - t0
        results.append({
            "file":    uf.filename,
            "status":  "ok",
            "message": f"Indexed {len(chunks)} chunks in {elapsed:.1f}s",
            "chunks":  len(chunks),
            "pages":   len(pages),
        })

        # Log analytics point
        nodes = get_retriever().graph.number_of_nodes()
        _analytics_history.append(nodes)
        if len(_analytics_history) > 20:
            _analytics_history.pop(0)

    return {
        "files":       results,
        "total_docs":  len(get_retriever().vector),
        "graph_nodes": get_retriever().graph.number_of_nodes(),
    }


@app.post("/api/query")
async def query(body: dict):
    """
    Body: { "query": str, "domain": str, "k": int }
    Returns answer, sources, graph_context, risk warnings.
    """
    q      = (body.get("query") or "").strip()
    domain = body.get("domain", "others")
    k      = int(body.get("k", 5))

    if not q:
        raise HTTPException(status_code=400, detail="query is required")
    if len(get_retriever().vector) == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a PDF first.")

    # Retrieve
    result = get_retriever().retrieve(q, k=k)
    docs   = result["results"]

    if not docs:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # LLM answer
    answer_result = get_llm().answer_with_context(q, docs, result.get("graph_context"))

    # Domain risk analysis (others = no special domain analyzer, use generic)
    if domain == "others":
        warnings = []
    else:
        try:
            domain_analyzer = get_domain(domain)
            warnings = domain_analyzer.detect_anomalies(docs)
        except Exception:
            warnings = []

    # Build sources
    sources = []
    for doc in docs[:5]:
        sources.append({
            "source":    doc.get("source", "unknown"),
            "page":      doc.get("page", "?"),
            "score":     round(doc.get("score", 0.0), 3),
            "snippet":   doc.get("text", "")[:300],
        })

    # Graph data (for visualization)
    graph_data = None
    gc = result.get("graph_context")
    if gc and gc.get("neighbors"):
        graph_data = {
            "entities":    gc["neighbors"][0][:12] if gc["neighbors"] else [],
            "total_nodes": gc.get("subgraph_nodes", 0),
        }

    return {
        "answer":  answer_result["answer"],
        "model":   answer_result.get("model", ""),
        "usage":   answer_result.get("usage", {}),
        "error":   answer_result.get("error"),
        "sources": sources,
        "graph":   graph_data,
        "risks":   warnings[:5],
    }


@app.get("/api/analytics")
async def analytics():
    """
    Return graph analytics: avg centrality, anomalies, density, growth rate.
    """
    G = get_retriever().graph
    n = G.number_of_nodes()
    e = G.number_of_edges()

    # Average degree centrality
    if n > 1:
        deg_sum = sum(d for _, d in G.degree())
        avg_centrality = (deg_sum / n) / (n - 1)
    else:
        avg_centrality = 0.0

    # Relationship density
    max_edges    = n * (n - 1) / 2 if n > 1 else 1
    density      = e / max_edges if max_edges > 0 else 0.0

    # Anomaly nodes: degree > mean + 2*std
    degrees = [d for _, d in G.degree()]
    if degrees:
        mean_d = sum(degrees) / len(degrees)
        variance = sum((d - mean_d)**2 for d in degrees) / len(degrees)
        std_d = math.sqrt(variance)
        anomalies = sum(1 for d in degrees if d > mean_d + 2 * std_d)
    else:
        anomalies = 0

    # Growth rate: pct change in last 2 analytics snapshots
    if len(_analytics_history) >= 2:
        prev = _analytics_history[-2] or 1
        curr = _analytics_history[-1]
        growth = ((curr - prev) / prev) * 100
        growth_str = f"+{growth:.1f}%" if growth >= 0 else f"{growth:.1f}%"
    else:
        growth_str = "0.0%"

    # History: normalized node counts for sparkline
    hist = _analytics_history[-8:] if _analytics_history else [0]
    max_h = max(hist) or 1
    norm_hist = [v / max_h for v in hist]

    return {
        "avg_centrality": round(avg_centrality, 4),
        "anomalies":      anomalies,
        "density":        round(density, 4),
        "growth_rate":    growth_str,
        "nodes":          n,
        "edges":          e,
        "history":        norm_hist,
    }


@app.post("/api/report")
async def report(body: dict):
    """
    Generate a downloadable summary report with all indexed sources.
    Body: { "format": "markdown"|"text"|"json", "domain": str }
    """
    fmt    = body.get("format", "markdown")
    domain = body.get("domain", "others")

    docs_indexed = len(get_retriever().vector)
    if docs_indexed == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload PDFs first.")

    G         = get_retriever().graph
    n_nodes   = G.number_of_nodes()
    n_edges   = G.number_of_edges()

    # Gather unique sources from the vector store
    all_sources: list[dict] = []
    seen_sources: set[str] = set()
    for doc in get_retriever().vector:
        src_key = f"{doc.get('source','unknown')}:p{doc.get('page','?')}"
        if src_key not in seen_sources:
            seen_sources.add(src_key)
            all_sources.append({
                "source":  doc.get("source", "unknown"),
                "page":    doc.get("page", "?"),
                "snippet": doc.get("text", "")[:200],
            })

    # Analytics snapshot
    n = n_nodes
    e = n_edges
    density   = round(e / max(n*(n-1)/2, 1), 4)
    now_str   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if fmt == "json":
        content = json.dumps({
            "report_generated": now_str,
            "domain":           domain,
            "total_documents":  docs_indexed,
            "graph_nodes":      n_nodes,
            "graph_edges":      n_edges,
            "relationship_density": density,
            "sources":          all_sources,
            "summary":          (
                f"This report covers {docs_indexed} indexed document(s) across {n_nodes} knowledge graph nodes "
                f"and {n_edges} edges, with a relationship density of {density}."
            ),
        }, indent=2)

    elif fmt == "markdown":
        sources_md = "\n".join(
            f"- **{s['source']}** (p.{s['page']}): {s['snippet']}…"
            for s in all_sources[:30]
        )
        content = f"""# OmniRAGGraph — Summary Report

**Generated:** {now_str}  
**Domain:** {domain.title()}  
**Documents Indexed:** {docs_indexed}  

---

## Knowledge Graph Metrics

| Metric | Value |
|---|---|
| Graph Nodes | {n_nodes} |
| Graph Edges | {n_edges} |
| Relationship Density | {density} |

---

## Summary

This report covers **{docs_indexed}** indexed document(s) organized into **{n_nodes}** knowledge graph nodes 
with **{n_edges}** relationship edges. The relationship density of **{density}** indicates the 
interconnectedness of the document corpus.

---

## Sources ({len(all_sources)} unique)

{sources_md}

---

*Generated by OmniRAGGraph · GraphRAG + FAISS + Groq LLM*
"""

    else:  # plain text
        sources_txt = "\n".join(
            f"  [{i+1}] {s['source']} (p.{s['page']}): {s['snippet']}…"
            for i, s in enumerate(all_sources[:30])
        )
        content = f"""OMNIRAG GRAPH — SUMMARY REPORT
Generated: {now_str}
Domain:    {domain.upper()}
Documents: {docs_indexed}

KNOWLEDGE GRAPH METRICS
  Nodes:                {n_nodes}
  Edges:                {n_edges}
  Relationship Density: {density}

SUMMARY
  {docs_indexed} indexed document(s) organized into {n_nodes} graph nodes
  with {n_edges} relationship edges (density={density}).

SOURCES ({len(all_sources)} unique)
{sources_txt}

--- Generated by OmniRAGGraph ---
"""

    word_count = len(content.split())
    return {
        "report":        content,
        "format":        fmt,
        "sources_count": len(all_sources),
        "word_count":    word_count,
    }


# ── Entry point ───────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
