"""Outils communs à l'entraînement et au test : jeu de cas, découpage, métriques."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from . import config
from .config import AXES
from .extraction import EXEMPLES_FEW_SHOT
from .moteur import Modele, recommander

TAILLE_TEST = 14


def charger_cas() -> pd.DataFrame:
    cas = pd.read_csv(config.EVAL_DIR / "cas_boutiques.csv", sep=";", encoding="utf-8").fillna("")
    cas["pertinentes"] = cas["formules_pertinentes"].map(lambda s: [int(x) for x in str(s).split("|")])
    cas["dans_prompt"] = cas["description"].isin([d for d, _ in EXEMPLES_FEW_SHOT])
    return cas


def decouper(cas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Jeu de test tiré au hasard (seed fixe), jamais parmi les exemples du prompt."""
    candidats = sorted(cas.loc[~cas["dans_prompt"], "id"])
    test_ids = set(random.Random(config.SEED).sample(candidats, TAILLE_TEST))
    return cas[~cas["id"].isin(test_ids)].copy(), cas[cas["id"].isin(test_ids)].copy()


def attendus(cas: pd.DataFrame) -> list[dict]:
    return [{a: int(r[a]) for a in AXES} for _, r in cas.iterrows()]


def mae(predits: list[dict], vrais: list[dict]) -> dict:
    """Erreur absolue moyenne, par axe et globale (en points sur 10)."""
    p = np.array([[s[a] for a in AXES] for s in predits], dtype=float)
    v = np.array([[s[a] for a in AXES] for s in vrais], dtype=float)
    par_axe = np.abs(p - v).mean(axis=0)
    return {"global": round(float(par_axe.mean()), 2), **{a: round(float(x), 2) for a, x in zip(AXES, par_axe)}}


def classement(scores: dict, modele: Modele) -> list[int]:
    rep = recommander(scores, top_formules=13, top_ingredients=0, modele=modele)
    return [f["id_formule"] for f in rep["formules"]]


def metriques_classement(classements: list[list[int]], pertinentes: list[list[int]]) -> dict:
    """hit@1, hit@3 : part des cas où une formule pertinente est 1re / dans le top 3.
    MRR : moyenne de 1/rang de la première formule pertinente (1 = toujours première)."""
    h1 = h3 = rr = 0.0
    for cl, pert in zip(classements, pertinentes):
        rang = next(i + 1 for i, fid in enumerate(cl) if fid in pert)
        h1 += rang == 1
        h3 += rang <= 3
        rr += 1 / rang
    n = len(classements)
    return {"hit@1": round(h1 / n, 3), "hit@3": round(h3 / n, 3), "mrr": round(rr / n, 3), "n": n}
