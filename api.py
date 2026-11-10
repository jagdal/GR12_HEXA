"""API de questions/réponses sur le dossier Givaudan (RAG + Ollama).

Lancement :
    uvicorn api:app --reload
Swagger : http://localhost:8000/docs
Contrat OpenAPI : http://localhost:8000/openapi.json
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import rag

SYSTEM_PROMPT = """Tu es un assistant spécialisé sur l'entreprise Givaudan (arômes et parfums).
Règles :
- Réponds UNIQUEMENT à partir des extraits du dossier Givaudan fournis.
- Si la réponse ne se trouve pas dans les extraits, dis que le dossier ne le précise pas. N'invente jamais de chiffre.
- Recopie les chiffres exactement comme dans les extraits, avec leur unité (millions de CHF, %) et la période (année, semestre).
- Réponds en français, de façon claire et précise, en quelques phrases maximum."""

DESCRIPTION = """
Pose des questions sur Givaudan (histoire, résultats financiers, stratégie, durabilité).
Les réponses sont générées par un LLM Ollama à partir des passages retrouvés dans une base vectorielle (ChromaDB).

## Format

* Échanges en **JSON** (`Content-Type: application/json`), encodage **UTF-8**.
* Pas d'authentification (usage local).
* Les erreurs renvoient toujours un objet `{"detail": "<message>"}`.

## Codes de retour

| Code | Signification |
|------|---------------|
| 200 | Succès |
| 422 | Corps de requête invalide (champ manquant, `question` vide, `top_k` hors de 1–20) |
| 502 | Ollama injoignable ou modèle absent (`/ask` uniquement) |
| 503 | Base vide : lancer `python ingest.py` |
"""

app = FastAPI(
    title="Givaudan RAG API",
    description=DESCRIPTION,
    version="1.0.0",
    openapi_tags=[
        {"name": "système", "description": "État du service."},
        {"name": "questions", "description": "Recherche de passages et réponses du LLM."},
    ],
)


# ---------- Schémas ----------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question posée, en langage naturel (français conseillé). Obligatoire, non vide.",
        examples=["Quel a été le chiffre d'affaires de Givaudan en 2025 ?"],
    )
    top_k: int = Field(
        config.TOP_K,
        ge=1,
        le=20,
        description=f"Nombre de passages à retrouver (entier de 1 à 20, défaut {config.TOP_K}). "
                    "Plus il est grand, plus le contexte donné au LLM est large.",
        examples=[4],
    )


class Source(BaseModel):
    chapter: str = Field(description="Section du dossier dont provient le passage.", examples=["Résultats financiers"])
    text: str = Field(description="Texte intégral du passage retrouvé (≈ 800 caractères).")
    score: float = Field(
        description="Similarité cosinus avec la question, arrondie à 4 décimales (1 = identique). "
                    "Les sources sont triées par score décroissant.",
        examples=[0.8123],
    )


class AskResponse(BaseModel):
    question: str = Field(description="Question reçue, renvoyée telle quelle.")
    answer: str = Field(description="Réponse rédigée par le LLM à partir des sources uniquement.")
    model: str = Field(description="Nom du modèle Ollama ayant généré la réponse.", examples=[config.LLM_MODEL])
    sources: list[Source] = Field(description="Passages utilisés pour répondre (de 1 à `top_k`).")


class HealthResponse(BaseModel):
    chunks: int = Field(description="Nombre de passages en base. 0 = base vide, lancer `ingest.py`.", examples=[42])
    llm_model: str = Field(description="Modèle Ollama utilisé pour générer les réponses.", examples=[config.LLM_MODEL])
    embed_model: str = Field(description="Modèle Ollama utilisé pour les embeddings.", examples=[config.EMBED_MODEL])


class ErrorResponse(BaseModel):
    detail: str = Field(description="Message d'erreur lisible.")


ERR_503 = {503: {"model": ErrorResponse, "description": "Base vide : lancer `python ingest.py`."}}
ERR_502 = {502: {"model": ErrorResponse, "description": "Ollama injoignable ou modèle absent."}}


# ---------- Logique ----------

def retrieve(question: str, top_k: int) -> list[Source]:
    collection = rag.get_collection()
    if collection.count() == 0:
        raise HTTPException(503, "La base est vide : lance d'abord `python ingest.py`.")
    result = collection.query(
        query_embeddings=[rag.embed_query(question)],
        n_results=min(top_k, collection.count()),
    )
    return [
        Source(chapter=meta["chapter"], text=doc, score=round(1 - dist, 4))
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse, tags=["système"], summary="État du service")
def health():
    """Vérifie que la base est chargée et indique les modèles utilisés."""
    return HealthResponse(
        chunks=rag.get_collection().count(),
        llm_model=config.LLM_MODEL,
        embed_model=config.EMBED_MODEL,
    )


@app.post("/search", response_model=list[Source], responses=ERR_503, tags=["questions"],
          summary="Passages pertinents (sans LLM)")
def search(req: AskRequest):
    """Retourne seulement les passages les plus proches de la question, sans appeler le LLM.
    Utile pour déboguer la recherche."""
    return retrieve(req.question, req.top_k)


@app.post("/ask", response_model=AskResponse, responses={**ERR_502, **ERR_503}, tags=["questions"],
          summary="Réponse du LLM + sources")
def ask(req: AskRequest):
    """Retrouve les passages pertinents puis demande au LLM d'y répondre en s'appuyant uniquement sur eux."""
    sources = retrieve(req.question, req.top_k)
    context = "\n\n---\n\n".join(f"[{s.chapter}]\n{s.text}" for s in sources)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extraits du dossier Givaudan :\n\n{context}\n\nQuestion : {req.question}"},
    ]
    try:
        answer = rag.chat(messages)
    except Exception as exc:  # Ollama injoignable, modèle absent...
        raise HTTPException(502, f"Erreur Ollama : {exc}") from exc
    return AskResponse(question=req.question, answer=answer.strip(), model=config.LLM_MODEL, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
