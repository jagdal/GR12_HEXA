"""Étape C — Lire la description libre d'une boutique et la traduire en 5 scores.

Deux extracteurs, interchangeables :
    extraire_mots_cles(texte)        règles simples, sans IA : la référence (baseline)
    extraire_llm(texte, variante)    Ollama (qwen2.5:3b), sortie JSON contrôlée

`extraire(texte)` utilise le LLM et retombe sur les mots-clés si Ollama
ne répond pas ou renvoie une réponse invalide : l'API ne plante jamais.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata

import ollama

from . import config
from .config import AXES

# ---------------------------------------------------------------------------
# Extracteur 1 : mots-clés (référence sans IA)
# ---------------------------------------------------------------------------
LEXIQUE = {
    "chaleur": ["chaleur", "chaud", "chaleureu", "cosy", "cocon", "reconfort", "vanill", "caramel", "gourmand",
                "sucre", "gateau", "patisser", "ambre", "epice", "cannelle", "miel", "rhum", "chocolat",
                "accueillant", "muffin", "biscuit", "praline"],
    "fraicheur": ["frais", "fraich", "agrume", "citron", "orange", "pamplemousse", "menthe", "mentho", "marin",
                  "ocean", "mer", "propre", "linge", "energ", "vert", "herbe", "basilic", "aquatique", "pur",
                  "eucalyptus", "sport", "tonique"],
    "floral": ["fleur", "floral", "rose", "jasmin", "muguet", "pivoine", "iris", "violette", "lilas",
               "feminin", "bouquet", "fleuriste", "poudre"],
    "boise": ["bois", "cedre", "santal", "vetiver", "patchouli", "cuir", "mousse", "foret", "tabac",
              "masculin", "viril", "terre"],
}
FORT = ["fort", "intense", "puissant", "marquant", "present", "prononce", "impactant", "sillage", "energique"]
LEGER = ["leger", "discret", "doux", "subtil", "pas trop fort", "delicat", "neutre", "apaisant", "zen", "relax"]


def _sans_accents(texte: str) -> str:
    t = unicodedata.normalize("NFD", texte.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def extraire_mots_cles(texte: str) -> dict[str, int]:
    t = _sans_accents(texte)
    scores = {a: 2 for a in AXES[:4]}
    for axe, mots in LEXIQUE.items():
        n = sum(1 for m in mots if m in t)
        if n:
            scores[axe] = min(10, 6 + 2 * n)
    intensite = 5
    if any(m in t for m in LEGER):
        intensite = 3
    if any(m in t for m in FORT) and "pas trop fort" not in t:
        intensite = 8
    if "tres leger" in t or "tres discret" in t:
        intensite = 2
    scores["intensite"] = intensite
    return scores


# ---------------------------------------------------------------------------
# Extracteur 2 : LLM via Ollama
# ---------------------------------------------------------------------------
CONSIGNE = """Tu es un expert en parfum d'ambiance pour les commerces.
Une boutique décrit l'odeur qu'elle veut. Traduis sa description en 5 notes entières de 0 à 10 :
- chaleur : odeur chaude, gourmande, vanillée, ambrée, épicée, réconfortante
- fraicheur : odeur fraîche, agrumes, verte, marine, mentholée, propre
- floral : fleurs (rose, jasmin, muguet...)
- boise : bois, cèdre, santal, cuir, mousse
- intensite : force de l'odeur (0 = à peine perceptible, 10 = très puissante)
Un axe que la boutique ne mentionne pas reçoit une note basse (0 à 3).
L'intensité vaut 5 si la boutique ne dit rien sur la force de l'odeur.
Réponds uniquement avec le JSON demandé."""

# Exemples tirés du jeu d'ENTRAÎNEMENT (jamais du jeu de test)
EXEMPLES_FEW_SHOT = [
    ("Salon de thé et pâtisserie, on veut une odeur gourmande de gâteau aux fruits rouges qui donne faim",
     {"chaleur": 9, "fraicheur": 1, "floral": 2, "boise": 1, "intensite": 6}),
    ("Barbier pour hommes, odeur boisée, virile, cuir et bois de cèdre, assez intense",
     {"chaleur": 6, "fraicheur": 1, "floral": 0, "boise": 9, "intensite": 8}),
    ("Spa et institut de beauté, ambiance fraîche et zen, agrumes et citron vert, très léger",
     {"chaleur": 1, "fraicheur": 9, "floral": 3, "boise": 1, "intensite": 2}),
]

SCHEMA = {
    "type": "object",
    "properties": {a: {"type": "integer", "minimum": 0, "maximum": 10} for a in AXES},
    "required": AXES,
}

_client = ollama.Client(host=config.OLLAMA_HOST)


def _messages(texte: str, variante: str) -> list[dict]:
    msgs = [{"role": "system", "content": CONSIGNE}]
    if variante == "few_shot":
        for desc, rep in EXEMPLES_FEW_SHOT:
            msgs.append({"role": "user", "content": desc})
            msgs.append({"role": "assistant", "content": json.dumps(rep)})
    msgs.append({"role": "user", "content": texte})
    return msgs


def valider(brut: str) -> dict[str, int]:
    """Contrôle la réponse du LLM : JSON, 5 axes, entiers entre 0 et 10."""
    match = re.search(r"\{.*\}", brut, re.S)
    if not match:
        raise ValueError("pas de JSON dans la réponse")
    data = json.loads(match.group(0))
    scores = {}
    for a in AXES:
        v = float(data[a])
        if not 0 <= v <= 10:
            raise ValueError(f"{a} hors de 0-10 : {v}")
        scores[a] = int(round(v))
    return scores


# Cache disque : la même question au même modèle ne coûte qu'un appel
_CACHE_FILE = config.CACHE_DIR / "llm_cache.json"
_cache: dict | None = None


def _cache_get(cle: str):
    global _cache
    if _cache is None:
        _cache = json.loads(_CACHE_FILE.read_text(encoding="utf-8")) if _CACHE_FILE.exists() else {}
    return _cache.get(cle)


def _cache_set(cle: str, valeur) -> None:
    _cache[cle] = valeur
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _CACHE_FILE.write_text(json.dumps(_cache, ensure_ascii=False, indent=1), encoding="utf-8")


def extraire_llm(texte: str, variante: str = "few_shot", modele: str | None = None) -> dict[str, int]:
    """Appelle Ollama ; lève une exception si la réponse reste invalide après 2 essais."""
    modele = modele or config.RECO_LLM_MODEL
    cle = hashlib.sha1(f"{modele}|{variante}|{texte}".encode()).hexdigest()
    if (cache := _cache_get(cle)) is not None:
        return cache
    derniere_erreur = None
    for _ in range(2):
        try:
            rep = _client.chat(
                model=modele,
                messages=_messages(texte, variante),
                format=SCHEMA,
                options={"temperature": 0, "num_ctx": 2048},
            )
            scores = valider(rep["message"]["content"])
            _cache_set(cle, scores)
            return scores
        except Exception as exc:  # JSON invalide, Ollama injoignable...
            derniere_erreur = exc
    raise RuntimeError(f"Extraction LLM impossible : {derniere_erreur}")


def extraire(texte: str, variante: str = "few_shot") -> tuple[dict[str, int], str]:
    """Scores + méthode utilisée (« llm » ou « mots-clés » en secours)."""
    try:
        return extraire_llm(texte, variante), "llm"
    except Exception:
        return extraire_mots_cles(texte), "mots-clés"
