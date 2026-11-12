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
| 503 | Base vide : lancer `python ingest.py` (questions) ou `python -m reco.preparer` (recommandation) |

## Recommandation de fragrances

`POST /recommander` reçoit la description libre d'une boutique, la traduit en 5 scores
(chaleur, fraîcheur, floral, boisé, intensité) avec Ollama, puis renvoie les 3 formules
Givaudan les plus proches et 5 ingrédients Givaudan encore au catalogue.
`POST /recommander/curseurs` fait le même calcul à partir des curseurs, sans LLM.
"""

app = FastAPI(
    title="Givaudan RAG API",
    description=DESCRIPTION,
    version="1.1.0",
    openapi_tags=[
        {"name": "système", "description": "État du service."},
        {"name": "questions", "description": "Recherche de passages et réponses du LLM."},
        {"name": "recommandation", "description": "Recommandation de fragrances pour une boutique."},
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


# ---------- Recommandation de fragrances ----------

def _axe(description: str):
    return Field(..., ge=0, le=10, description=description, examples=[5])


class Curseurs(BaseModel):
    chaleur: int = _axe("Odeur chaude, gourmande, vanillée, ambrée, épicée (0 à 10).")
    fraicheur: int = _axe("Odeur fraîche : agrumes, vert, marin, menthe, propre (0 à 10).")
    floral: int = _axe("Odeur de fleurs (0 à 10).")
    boise: int = _axe("Bois, cèdre, santal, cuir, mousse (0 à 10).")
    intensite: int = _axe("Force de l'odeur : 0 = à peine perceptible, 10 = très puissante.")


class AjustementCurseurs(BaseModel):
    chaleur: int | None = Field(None, ge=0, le=10)
    fraicheur: int | None = Field(None, ge=0, le=10)
    floral: int | None = Field(None, ge=0, le=10)
    boise: int | None = Field(None, ge=0, le=10)
    intensite: int | None = Field(None, ge=0, le=10)


class RecoRequest(BaseModel):
    description: str = Field(
        ..., min_length=3, max_length=1000,
        description="Description libre de l'ambiance voulue, écrite par la boutique.",
        examples=["Salon de coiffure de quartier, une odeur fraîche et propre, un peu florale, pas trop forte"],
    )
    ajustements: AjustementCurseurs | None = Field(
        None, description="Curseurs corrigés par la boutique après une première proposition : "
                          "ils remplacent les scores lus par le LLM sur les axes donnés.")
    top_formules: int = Field(3, ge=1, le=13, description="Nombre de formules renvoyées.")
    top_ingredients: int = Field(5, ge=0, le=20, description="Nombre d'ingrédients Givaudan renvoyés.")


class CurseursRequest(Curseurs):
    top_formules: int = Field(3, ge=1, le=13, description="Nombre de formules renvoyées.")
    top_ingredients: int = Field(5, ge=0, le=20, description="Nombre d'ingrédients Givaudan renvoyés.")


class ProfilFormule(BaseModel):
    chaleur: float
    fraicheur: float
    floral: float
    boise: float
    intensite: float


class FormuleReco(BaseModel):
    rang: int = Field(description="1 = la plus proche.")
    id_formule: int
    nom: str
    application: str = Field(description="Usage d'origine de la formule de démonstration.")
    correspondance_pct: float = Field(description="100 = identique aux scores demandés, 0 = opposé sur tous les axes.")
    profil: ProfilFormule = Field(description="Les 5 scores de la formule (0 à 10, une décimale).")
    famille_dominante: str
    familles_principales: str = Field(description="Part des 3 premières familles dans la formule.")
    part_givaudan_pct: float = Field(description="Part de la formule (hors solvants) faite d'ingrédients Givaudan.")
    ingredients_givaudan_principaux: str
    alerte_catalogue: str | None = Field(description="Ingrédients Givaudan de la formule absents du catalogue 2024.")
    pourquoi: str = Field(description="Explication générée par le code (pas par le LLM).")


class IngredientReco(BaseModel):
    nom: str
    famille: str
    correspondance_pct: float
    utilise_dans_une_formule_recommandee: bool
    prix_indicatif_eur_kg: int | None = Field(description="Prix revendeur au détail, 100 g pur, en EUR/kg. Indicatif.")


class RecoResponse(BaseModel):
    scores_utilises: Curseurs = Field(description="Scores après lecture du texte et ajustements.")
    methode_lecture: str | None = Field(None, description="« llm » ou « mots-clés » (secours si Ollama ne répond pas).")
    formules: list[FormuleReco]
    ingredients: list[IngredientReco]
    modele: dict = Field(description="Version du modèle et poids des axes appris.")
    note_prix: str


def _recommander(scores: dict, top_f: int, top_i: int) -> dict:
    from reco.moteur import recommander
    try:
        return recommander(scores, top_formules=top_f, top_ingredients=top_i)
    except FileNotFoundError as exc:
        raise HTTPException(503, "Données absentes : lance d'abord `python -m reco.preparer`.") from exc


@app.post("/recommander", response_model=RecoResponse, responses=ERR_503, tags=["recommandation"],
          summary="Description libre → 3 formules + ingrédients")
def recommander_description(req: RecoRequest):
    """Lit la description avec Ollama (secours : mots-clés), applique les ajustements, puis recommande."""
    from reco.extraction import extraire
    from reco.moteur import charger_modele

    scores, methode = extraire(req.description, variante=charger_modele().variante_prompt)
    if req.ajustements:
        scores.update({k: v for k, v in req.ajustements.model_dump().items() if v is not None})
    return {**_recommander(scores, req.top_formules, req.top_ingredients), "methode_lecture": methode}


@app.post("/recommander/curseurs", response_model=RecoResponse, responses=ERR_503, tags=["recommandation"],
          summary="Curseurs → 3 formules + ingrédients (sans LLM, instantané)")
def recommander_curseurs(req: CurseursRequest):
    """Recommandation directe à partir des 5 curseurs."""
    scores = req.model_dump(include={"chaleur", "fraicheur", "floral", "boise", "intensite"})
    return _recommander(scores, req.top_formules, req.top_ingredients)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
