"""Étape D — Calculer la ressemblance entre la boutique et chaque fragrance.

Distance euclidienne pondérée sur les 5 axes. Les poids des axes sont
appris sur le jeu d'entraînement (reco/entrainer.py) et lus dans
models/modele_reco.json ; à défaut, tous les axes pèsent 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config
from .config import AXES

NOMS_AXES = {"chaleur": "chaleur", "fraicheur": "fraîcheur", "floral": "floral", "boise": "boisé",
             "intensite": "intensité"}


@dataclass
class Modele:
    poids: dict[str, float] = field(default_factory=lambda: {a: 1.0 for a in AXES})
    variante_prompt: str = "few_shot"
    version: str = "non entraîné"


@lru_cache(maxsize=1)
def charger_modele() -> Modele:
    if config.MODEL_PATH.exists():
        m = json.loads(config.MODEL_PATH.read_text(encoding="utf-8"))
        return Modele(poids=m["poids_axes"], variante_prompt=m["extracteur"]["variante_prompt"],
                      version=m["version"])
    return Modele()


@lru_cache(maxsize=1)
def charger_donnees() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    formules = pd.read_csv(config.PROCESSED_DIR / "formules_profils.csv").fillna("")
    ingredients = pd.read_csv(config.PROCESSED_DIR / "ingredients.csv")
    lignes = pd.read_csv(config.PROCESSED_DIR / "formules_lignes.csv")
    # On ne recommande que des ingrédients Givaudan encore vendus
    ingredients = ingredients[ingredients["encore_au_catalogue"]].copy()
    return formules, ingredients, lignes


def distance(u: np.ndarray, v: np.ndarray, poids: np.ndarray) -> np.ndarray:
    """Distance euclidienne pondérée. u : (5,), v : (n, 5) -> (n,)."""
    return np.sqrt(((v - u) ** 2 * poids).sum(axis=1))


def correspondance(d: np.ndarray, poids: np.ndarray) -> np.ndarray:
    """Distance -> % de correspondance. 100 % = identique ; 0 % = opposé sur tous les axes."""
    d_max = math.sqrt((100 * poids).sum())
    return np.clip(100 * (1 - d / d_max), 0, 100)


def _explication(u: dict, f: pd.Series) -> str:
    ecarts = sorted(AXES, key=lambda a: abs(u[a] - f[a]))
    proches = [f"{NOMS_AXES[a]} ({f[a]:.0f} pour {u[a]} demandé)" for a in ecarts[:2]]
    loin = ecarts[-1]
    texte = f"Proche sur {proches[0]} et {proches[1]}."
    if abs(u[loin] - f[loin]) >= 3:
        sens = "plus" if f[loin] > u[loin] else "moins"
        texte += f" Plus éloigné sur {NOMS_AXES[loin]} : la formule est {sens} marquée ({f[loin]:.0f} pour {u[loin]})."
    return texte


def recommander(scores: dict[str, int], top_formules: int = 3, top_ingredients: int = 5,
                modele: Modele | None = None) -> dict:
    """Top des formules et des ingrédients Givaudan les plus proches des scores demandés."""
    modele = modele or charger_modele()
    formules, ingredients, lignes = charger_donnees()
    w = np.array([modele.poids[a] for a in AXES])
    u = np.array([scores[a] for a in AXES], dtype=float)

    d = distance(u, formules[AXES].to_numpy(), w)
    formules = formules.assign(distance=d, correspondance=correspondance(d, w)).sort_values("distance")
    top = formules.head(top_formules)

    resultats_formules = [{
        "rang": i + 1,
        "id_formule": int(f.id_formule),
        "nom": f.nom,
        "application": f.application,
        "correspondance_pct": round(float(f.correspondance), 1),
        "profil": {a: round(float(f[a]), 1) for a in AXES},
        "famille_dominante": f.famille_dominante,
        "familles_principales": f.familles_principales,
        "part_givaudan_pct": float(f.part_givaudan_pct),
        "ingredients_givaudan_principaux": f.ingredients_givaudan_principaux,
        "alerte_catalogue": (f"Contient des ingrédients Givaudan absents du catalogue 2024 : "
                             f"{f.givaudan_hors_catalogue_2024}") if f.givaudan_hors_catalogue_2024 else None,
        "pourquoi": _explication(scores, f),
    } for i, (_, f) in enumerate(top.iterrows())]

    # Ingrédients : même calcul ; à égalité, on préfère ceux déjà utilisés dans une
    # formule recommandée, puis ceux dont on connaît le prix.
    if top_ingredients <= 0:
        ingredients = ingredients.head(0)
    di = distance(u, ingredients[AXES].to_numpy(), w)
    cles_top = set(lignes.loc[lignes["id_formule"].isin(set(top["id_formule"])), "cle"])
    ingr = ingredients.assign(
        distance=di.round(6),
        correspondance=correspondance(di, w),
        dans_top=ingredients["cle"].isin(cles_top),
        a_prix=ingredients["prix_eur_kg_100g"].notna(),
    ).sort_values(["distance", "dans_top", "a_prix", "nom"], ascending=[True, False, False, True])
    resultats_ingredients = [{
        "nom": r.nom,
        "famille": r.famille,
        "correspondance_pct": round(float(r.correspondance), 1),
        "utilise_dans_une_formule_recommandee": bool(r.dans_top),
        "prix_indicatif_eur_kg": None if pd.isna(r.prix_eur_kg_100g) else round(float(r.prix_eur_kg_100g)),
    } for r in ingr.head(top_ingredients).itertuples()]

    return {
        "scores_utilises": scores,
        "formules": resultats_formules,
        "ingredients": resultats_ingredients,
        "modele": {"version": modele.version, "poids_axes": modele.poids},
        "note_prix": f"Prix de détail revendeur pour 100 g pur, convertis en EUR (taux {config.TAUX_DATE}) : "
                     "indicatifs, ce ne sont pas les tarifs Givaudan.",
    }
