"""Étape E (2/2) — Test : mesurer le modèle entraîné sur des cas qu'il n'a jamais vus.

    python -m reco.evaluer

Quatre configurations sont comparées sur le même jeu de test (ablation) :
    A. référence : mots-clés + poids égaux     (sans IA)
    B. LLM + poids égaux                        (apport du LLM seul)
    C. LLM + poids appris = MODÈLE FINAL        (apport de l'entraînement)
    D. scores attendus + poids appris           (« oracle » : limite du moteur si le LLM était parfait)
Sorties : reports/evaluation_test.json et reports/evaluation_test.md
"""

from __future__ import annotations

import json
import statistics
import time

from . import config
from .config import AXES
from .evaluation import attendus, charger_cas, classement, decouper, mae, metriques_classement
from .extraction import extraire_llm, extraire_mots_cles
from .moteur import Modele, charger_donnees, charger_modele


def evaluer() -> dict:
    modele_final = charger_modele()
    if modele_final.version == "non entraîné":
        raise SystemExit("Lancez d'abord : python -m reco.entrainer")
    _, test = decouper(charger_cas())
    textes, vrais, pert = list(test["description"]), attendus(test), list(test["pertinentes"])

    llm, durees = [], []
    for i, t in enumerate(textes, 1):
        t0 = time.time()
        llm.append(extraire_llm(t, variante=modele_final.variante_prompt))
        durees.append(time.time() - t0)
        print(f"  LLM {i}/{len(textes)}", end="\r", flush=True)
    print()
    mots = [extraire_mots_cles(t) for t in textes]

    egaux = Modele(variante_prompt=modele_final.variante_prompt)
    configs = {
        "A_mots_cles_poids_egaux": (mots, egaux),
        "B_llm_poids_egaux": (llm, egaux),
        "C_llm_poids_appris_MODELE_FINAL": (llm, modele_final),
        "D_oracle_scores_attendus_poids_appris": (vrais, modele_final),
    }
    resultats, classements = {}, {}
    for nom, (scores, modele) in configs.items():
        cl = [classement(s, modele) for s in scores]
        classements[nom] = cl
        resultats[nom] = {"classement": metriques_classement(cl, pert)}
        if nom.startswith(("A", "B", "C")):
            resultats[nom]["mae_scores"] = mae(scores, vrais)

    formules, _, _ = charger_donnees()
    noms = dict(zip(formules["id_formule"], formules["nom"].str.replace(r" /2 \(Givaudan\)", "", regex=True)))
    final = classements["C_llm_poids_appris_MODELE_FINAL"]
    detail = [{
        "id": int(r.id),
        "description": r.description,
        "scores_attendus": vrais[i],
        "scores_llm": llm[i],
        "top3": [noms[f] for f in final[i][:3]],
        "pertinentes": [noms[f] for f in r.pertinentes],
        "rang_premiere_pertinente": next(k + 1 for k, f in enumerate(final[i]) if f in r.pertinentes),
    } for i, r in enumerate(test.itertuples())]

    rapport = {
        "modele": {"version": modele_final.version, "poids_axes": modele_final.poids,
                   "variante_prompt": modele_final.variante_prompt, "llm": config.RECO_LLM_MODEL},
        "jeu_de_test": {"n": len(test), "ids": sorted(test["id"].tolist())},
        "resultats": resultats,
        "latence_llm_s": {"mediane": round(statistics.median(durees), 1), "max": round(max(durees), 1)},
        "detail_par_cas": detail,
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "evaluation_test.json").write_text(
        json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.REPORTS_DIR / "evaluation_test.md").write_text(_markdown(rapport), encoding="utf-8")
    return rapport


def _markdown(r: dict) -> str:
    lignes = [f"# Évaluation sur le jeu de test ({r['jeu_de_test']['n']} cas)", "",
              f"Modèle {r['modele']['version']} · LLM {r['modele']['llm']} ({r['modele']['variante_prompt']}) · "
              f"poids {r['modele']['poids_axes']}", "",
              "| Configuration | MAE scores (/10) | hit@1 | hit@3 | MRR |", "|---|---|---|---|---|"]
    for nom, v in r["resultats"].items():
        c = v["classement"]
        m = v.get("mae_scores", {}).get("global", "—")
        lignes.append(f"| {nom} | {m} | {c['hit@1']:.0%} | {c['hit@3']:.0%} | {c['mrr']:.2f} |")
    lignes += ["", f"Latence LLM : médiane {r['latence_llm_s']['mediane']} s, max {r['latence_llm_s']['max']} s.", "",
               "| # | Description | Scores LLM | Top 3 | Pertinentes | Rang |", "|---|---|---|---|---|---|"]
    for d in r["detail_par_cas"]:
        s = " ".join(str(d["scores_llm"][a]) for a in AXES)
        lignes.append(f"| {d['id']} | {d['description']} | {s} | {', '.join(d['top3'])} | "
                      f"{', '.join(d['pertinentes'])} | {d['rang_premiere_pertinente']} |")
    return "\n".join(lignes) + "\n"


if __name__ == "__main__":
    rap = evaluer()
    for nom, v in rap["resultats"].items():
        print(nom, v)
    print("latence", rap["latence_llm_s"])
