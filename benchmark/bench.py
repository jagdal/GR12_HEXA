"""Benchmark des modèles Ollama pour le RAG Givaudan.

Usage (depuis la racine du projet, venv activé) :
    python benchmark/bench.py embed nomic-embed-text mxbai-embed-large ...
    python benchmark/bench.py llm --embed nomic-embed-text qwen2.5:1.5b qwen2.5:3b ...
    options : --delete (supprime chaque modèle après test, sauf ceux de --keep)

Les résultats sont écrits dans benchmark/results/*.json, puis le PDF est généré par
benchmark/make_pdf.py.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmark"))

import numpy as np  # noqa: E402
import ollama  # noqa: E402

import config  # noqa: E402
import ingest  # noqa: E402
from questions import QUESTIONS  # noqa: E402

RESULTS = ROOT / "benchmark" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
TOP_K = 4
client = ollama.Client(host=config.OLLAMA_HOST)

SYSTEM_PROMPT = """Tu es un assistant spécialisé sur l'entreprise Givaudan (arômes et parfums).
Règles :
- Réponds UNIQUEMENT à partir des extraits du dossier Givaudan fournis.
- Si la réponse ne se trouve pas dans les extraits, dis que le dossier ne le précise pas. N'invente jamais de chiffre.
- Recopie les chiffres exactement comme dans les extraits, avec leur unité (millions de CHF, %) et la période (année, semestre).
- Réponds en français, de façon claire et précise, en quelques phrases maximum."""

# Préfixes recommandés par chaque modèle d'embeddings (document, requête)
EMBED_PREFIXES = {
    "nomic-embed-text": ("search_document: ", "search_query: "),
    "mxbai-embed-large": ("", "Represent this sentence for searching relevant passages: "),
    "qwen3-embedding": ("", "Instruct: Given a question, retrieve passages that answer the question\nQuery: "),
    "embeddinggemma": ("title: none | text: ", "task: search result | query: "),
}

REFUSAL = [
    "ne precise pas", "ne le precise pas", "ne mentionne pas", "n'est pas mentionne", "pas d'information",
    "aucune information", "ne contient pas", "ne donne pas", "ne fournit pas", "pas precise", "pas disponible",
    "ne figure pas", "pas indique", "ne sais pas", "n'indique pas", "ne permet pas", "pas mentionne",
    "pas d'indication", "aucune donnee", "pas de donnee", "ne specifie pas", "pas specifie", "ne parle pas",
    "n'est pas precise", "pas fourni", "ne peux pas", "impossible de",
    "ne se trouve pas", "pas trouve", "aucune mention",
]


# --------------------------------------------------------------------------- utilitaires

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm_words(s: str) -> str:
    s = strip_accents(s).lower().replace("’", "'").replace(" ", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s)


def norm_compact(s: str) -> str:
    return re.sub(r"\s+", "", norm_words(s)).replace(".", ",")


def keyword_found(answer: str, kw: str) -> bool:
    if re.search(r"[a-zA-Z]", strip_accents(kw)):
        return re.search(r"\b" + re.escape(norm_words(kw)) + r"\b", norm_words(answer)) is not None
    k = norm_compact(kw)
    a = norm_compact(answer)
    pat = r"(?<![\d,])" + re.escape(k) + r"(?![\d])"
    if re.search(pat, a):
        return True
    return "," not in k and re.search(pat, a.replace(",", "")) is not None  # 7,472 écrit avec séparateur


def is_refusal(answer: str) -> bool:
    a = norm_words(answer)
    return any(p in a for p in REFUSAL)


def score(item: dict, answer: str) -> bool:
    if item["expect"] is None:
        return is_refusal(answer)
    return all(any(keyword_found(answer, alt) for alt in group) for group in item["expect"])


def load_chunks() -> list[dict]:
    text = config.STORY_PATH.read_text(encoding="utf-8")
    _, sections = ingest.split_into_sections(text)
    chunks = []
    for chapter, paragraphs in sections:
        for c in ingest.chunk_paragraphs(paragraphs):
            chunks.append({"chapter": chapter, "text": c})
    return chunks


def ensure_model(name: str) -> float:
    """Télécharge le modèle s'il manque ; retourne sa taille sur disque en Go."""
    have = {m.model: m for m in client.list().models}
    if name not in have and f"{name}:latest" not in have:
        print(f"  téléchargement de {name}...", flush=True)
        client.pull(name)
        have = {m.model: m for m in client.list().models}
    m = have.get(name) or have.get(f"{name}:latest")
    return round(m.size / 1e9, 2)


def delete_model(name: str) -> None:
    try:
        client.delete(name)
        print(f"  {name} supprimé", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  suppression impossible de {name}: {exc}", flush=True)


def prefixes(model: str) -> tuple[str, str]:
    for key, val in EMBED_PREFIXES.items():
        if model.startswith(key):
            return val
    return "", ""


def embed_all(model: str, texts: list[str]) -> np.ndarray:
    vecs = client.embed(model=model, input=texts)["embeddings"]
    arr = np.array(vecs, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def retrieve_index(model: str, chunks: list[dict]):
    doc_p, q_p = prefixes(model)
    docs = embed_all(model, [doc_p + f"{c['chapter']}\n{c['text']}" for c in chunks])

    def search(question: str, k: int = TOP_K) -> list[int]:
        q = embed_all(model, [q_p + question])[0]
        return list(np.argsort(-(docs @ q))[:k])

    return search


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.]+", "_", name)


# --------------------------------------------------------------------------- embeddings

def bench_embed(model: str, delete: bool) -> None:
    print(f"\n=== Embeddings : {model}", flush=True)
    try:
        size = ensure_model(model)
    except Exception as exc:  # noqa: BLE001
        print(f"  indisponible : {exc}", flush=True)
        return
    chunks = load_chunks()
    t0 = time.perf_counter()
    search = retrieve_index(model, chunks)
    index_s = time.perf_counter() - t0

    items = [it for it in QUESTIONS if it["chapter"]]
    hit1 = hitk = 0
    rr = 0.0
    q_times = []
    for it in items:
        t = time.perf_counter()
        ranked = search(it["q"], k=len(chunks))
        q_times.append(time.perf_counter() - t)
        chapters = [chunks[i]["chapter"] for i in ranked]
        pos = next((r for r, ch in enumerate(chapters) if ch.startswith(it["chapter"] + " ")), None)
        if pos is not None:
            hit1 += pos == 0
            hitk += pos < TOP_K
            rr += 1 / (pos + 1)
    n = len(items)
    res = {
        "model": model, "size_gb": size, "n_questions": n, "n_chunks": len(chunks),
        "hit_at_1": round(hit1 / n, 3), "hit_at_k": round(hitk / n, 3), "k": TOP_K,
        "mrr": round(rr / n, 3), "index_seconds": round(index_s, 2),
        "query_ms": round(1000 * float(np.median(q_times)), 1),
    }
    print("  ", res, flush=True)
    (RESULTS / f"embed_{slug(model)}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    if delete:
        delete_model(model)


# --------------------------------------------------------------------------- LLM

def bench_llm(model: str, embed_model: str, delete: bool) -> None:
    print(f"\n=== LLM : {model}", flush=True)
    try:
        size = ensure_model(model)
    except Exception as exc:  # noqa: BLE001
        print(f"  indisponible : {exc}", flush=True)
        return

    chunks = load_chunks()
    search = retrieve_index(embed_model, chunks)
    contexts = {}
    for it in QUESTIONS:
        idx = search(it["q"])
        contexts[it["q"]] = "\n\n---\n\n".join(f"[{chunks[i]['chapter']}]\n{chunks[i]['text']}" for i in idx)

    extra = {"think": False} if model.startswith("qwen3") else {}
    opts = {"temperature": 0, "num_ctx": 4096}

    # Chargement à froid
    t0 = time.perf_counter()
    client.chat(model=model, messages=[{"role": "user", "content": "Bonjour"}], options=opts, **extra)
    load_s = time.perf_counter() - t0
    mem = next((p for p in client.ps().models if p.model.startswith(model)), None)
    ram_gb = round(mem.size / 1e9, 2) if mem else None

    rows = []
    for it in QUESTIONS:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extraits du dossier Givaudan :\n\n{contexts[it['q']]}\n\nQuestion : {it['q']}"},
        ]
        t = time.perf_counter()
        try:
            r = client.chat(model=model, messages=messages, options=opts, **extra)
            answer = r["message"]["content"].strip()
            ev_n, ev_d = r.get("eval_count") or 0, r.get("eval_duration") or 0
            pe_n, pe_d = r.get("prompt_eval_count") or 0, r.get("prompt_eval_duration") or 0
        except Exception as exc:  # noqa: BLE001
            answer, ev_n, ev_d, pe_n, pe_d = f"ERREUR: {exc}", 0, 0, 0, 0
        dt = time.perf_counter() - t
        ok = score(it, answer)
        rows.append({
            "q": it["q"], "answerable": it["expect"] is not None, "ok": ok, "answer": answer,
            "seconds": round(dt, 2), "gen_tokens": ev_n,
            "gen_tps": round(ev_n / (ev_d / 1e9), 1) if ev_d else None,
            "prompt_tps": round(pe_n / (pe_d / 1e9), 1) if pe_d else None,
        })
        print(f"  [{'OK ' if ok else 'NON'}] {dt:5.1f}s  {it['q'][:60]}  ->  {answer[:90]!r}", flush=True)

    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    tps = [r["gen_tps"] for r in rows if r["gen_tps"]]
    ptps = [r["prompt_tps"] for r in rows if r["prompt_tps"]]
    res = {
        "model": model, "embed_model": embed_model, "size_gb": size, "ram_gb": ram_gb,
        "load_seconds": round(load_s, 1),
        "accuracy": round(sum(r["ok"] for r in ans) / len(ans), 3),
        "refusal_rate": round(sum(r["ok"] for r in una) / len(una), 3),
        "overall": round(sum(r["ok"] for r in rows) / len(rows), 3),
        "median_seconds": round(float(np.median([r["seconds"] for r in rows])), 2),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 2),
        "gen_tps": round(float(np.median(tps)), 1) if tps else None,
        "prompt_tps": round(float(np.median(ptps)), 1) if ptps else None,
        "n_answerable": len(ans), "n_unanswerable": len(una), "rows": rows,
    }
    print(f"  => justesse {res['accuracy']:.0%}, refus {res['refusal_rate']:.0%}, "
          f"médiane {res['median_seconds']}s, {res['gen_tps']} tok/s, RAM {ram_gb} Go", flush=True)
    (RESULTS / f"llm_{slug(model)}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    client.generate(model=model, prompt="", keep_alive=0)  # libère la RAM
    if delete:
        delete_model(model)


def main() -> None:
    args = sys.argv[1:]
    mode = args.pop(0)
    delete = "--delete" in args
    keep = set()
    embed_model = config.EMBED_MODEL
    models = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--delete":
            pass
        elif a == "--keep":
            keep.update(args[i + 1].split(","))
            i += 1
        elif a == "--embed":
            embed_model = args[i + 1]
            i += 1
        else:
            models.append(a)
        i += 1
    for m in models:
        d = delete and m not in keep
        if mode == "embed":
            bench_embed(m, d)
        else:
            bench_llm(m, embed_model, d)


if __name__ == "__main__":
    main()
