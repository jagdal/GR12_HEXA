"""Étape E (1/2) — Entraînement : calibrer le modèle sur le jeu d'entraînement SEULEMENT.

    python -m reco.entrainer

Ce qui est appris :
    1. l'extracteur : mots-clés, LLM sans exemple (zero_shot) ou LLM avec 3 exemples
       (few_shot) -> on garde celui qui se trompe le moins sur les scores (MAE) ;
    2. les poids des 5 axes dans la distance -> recherche sur une grille de
       3^5 = 243 combinaisons, on garde celle qui classe le mieux les formules
       pertinentes (hit@3, puis MRR, puis la plus proche de poids égaux).
Le jeu de test n'est jamais lu ici.
"""

from __future__ import annotations

import itertools
import json
import time
from datetime import date

from . import config
from .config import AXES
from .evaluation import attendus, charger_cas, classement, decouper, mae, metriques_classement
from .extraction import extraire_llm, extraire_mots_cles
from .moteur import Modele

GRILLE = [0.5, 1.0, 2.0]


def extraire_tout(textes: list[str], methode: str) -> list[dict]:
    if methode == "mots_cles":
        return [extraire_mots_cles(t) for t in textes]
    out = []
    for i, t in enumerate(textes, 1):
        out.append(extraire_llm(t, variante=methode))
        print(f"  {methode} {i}/{len(textes)}", end="\r", flush=True)
    print()
    return out


def entrainer() -> dict:
    cas = charger_cas()
    train, test = decouper(cas)
    # Les 3 exemples du prompt sont exclus des mesures (le LLM a vu leur réponse)
    mesure = train[~train["dans_prompt"]]
    textes, vrais = list(mesure["description"]), attendus(mesure)
    print(f"Entraînement : {len(train)} cas ({len(mesure)} mesurés), test mis de côté : {len(test)} cas")

    # 1. Choix de l'extracteur
    t0 = time.time()
    extractions = {m: extraire_tout(textes, m) for m in ["mots_cles", "zero_shot", "few_shot"]}
    resultats_extracteurs = {m: mae(p, vrais) for m, p in extractions.items()}
    for m, r in resultats_extracteurs.items():
        print(f"  MAE {m:10s} : {r['global']:.2f} / 10")
    meilleur = min(["few_shot", "zero_shot", "mots_cles"], key=lambda m: resultats_extracteurs[m]["global"])
    print(f"Extracteur retenu : {meilleur}  ({time.time() - t0:.0f} s)")

    # 2. Poids des axes, appris sur les scores produits par l'extracteur retenu
    predits = extractions[meilleur]
    pertinentes = list(mesure["pertinentes"])
    essais = []
    for combo in itertools.product(GRILLE, repeat=len(AXES)):
        modele = Modele(poids=dict(zip(AXES, combo)), variante_prompt=meilleur)
        m = metriques_classement([classement(s, modele) for s in predits], pertinentes)
        ecart = sum(abs(x - 1) for x in combo)  # régularisation : préférer des poids proches de 1
        essais.append((m["hit@3"], m["mrr"], -ecart, combo, m))
    essais.sort(key=lambda e: e[:3], reverse=True)
    *_, poids, m_train = essais[0]
    uniforme = next(e[4] for e in essais if e[3] == (1.0,) * 5)
    print(f"Poids appris : {dict(zip(AXES, poids))}")
    print(f"  train poids égaux : {uniforme}")
    print(f"  train poids appris : {m_train}")

    modele = {
        "version": f"1.0.0-{date.today().isoformat()}",
        "date_entrainement": date.today().isoformat(),
        "extracteur": {
            "type": "llm" if meilleur != "mots_cles" else "mots_cles",
            "modele_ollama": config.RECO_LLM_MODEL,
            "variante_prompt": meilleur if meilleur != "mots_cles" else "few_shot",
            "temperature": 0,
        },
        "poids_axes": dict(zip(AXES, poids)),
        "decoupage": {"seed": config.SEED, "train_ids": sorted(train["id"].tolist()),
                      "test_ids": sorted(test["id"].tolist())},
        "metriques_entrainement": {
            "mae_extracteurs": resultats_extracteurs,
            "classement_poids_egaux": uniforme,
            "classement_poids_appris": m_train,
        },
    }
    config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.MODEL_PATH.write_text(json.dumps(modele, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modèle écrit : {config.MODEL_PATH.relative_to(config.BASE_DIR)}")
    return modele


if __name__ == "__main__":
    entrainer()
