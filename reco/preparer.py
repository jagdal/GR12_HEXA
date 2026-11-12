"""Étapes A + B : nettoyer les CSV bruts et calculer les profils.

    python -m reco.preparer

Sorties dans data/processed/ :
    ingredients.csv       un ingrédient du catalogue par ligne, avec ses 5 scores et son prix de référence
    formules_lignes.csv   les lignes des formules, avec famille et fabricant corrigés
    formules_profils.csv  une formule par ligne, avec ses 5 scores
    rapport_qualite.json  chaque problème trouvé et ce qui a été corrigé
"""

from __future__ import annotations

import json

from . import config
from .nettoyage import charger_brut, nettoyer_catalogue, nettoyer_formules, nettoyer_prix
from .profils import profils_formules, profils_ingredients


def preparer() -> dict:
    brut = charger_brut()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    ingredients, conflits = nettoyer_catalogue(brut["catalogue"])
    formules, details_formules = nettoyer_formules(brut["formules"], ingredients)
    prix_ref, details_prix = nettoyer_prix(brut["prix"], ingredients)

    ingr = profils_ingredients(ingredients, prix_ref)
    profils = profils_formules(formules)

    ingr.to_csv(config.PROCESSED_DIR / "ingredients.csv", index=False, encoding="utf-8")
    formules.to_csv(config.PROCESSED_DIR / "formules_lignes.csv", index=False, encoding="utf-8")
    profils.to_csv(config.PROCESSED_DIR / "formules_profils.csv", index=False, encoding="utf-8")

    cat = brut["catalogue"]
    rapport = {
        "catalogue": {
            "lignes_brutes": len(cat),
            "ingredients_uniques": len(ingredients),
            "doublons_fusionnes": len(cat) - len(ingredients),
            "familles_contradictoires": len(conflits),
            "detail_conflits": conflits,
            "ingredients_sans_odeur_exclus": int(len(ingredients) - len(ingr)),
            "encore_au_catalogue_2024": int(ingredients["encore_au_catalogue"].sum()),
        },
        "formules": {
            "formules": int(formules["id_formule"].nunique()),
            "lignes": len(formules),
            **details_formules,
        },
        "prix": details_prix,
    }
    with open(config.PROCESSED_DIR / "rapport_qualite.json", "w", encoding="utf-8") as fh:
        json.dump(rapport, fh, ensure_ascii=False, indent=2, default=str)
    return rapport


if __name__ == "__main__":
    r = preparer()
    c, f, p = r["catalogue"], r["formules"], r["prix"]
    print(f"Catalogue : {c['lignes_brutes']} lignes -> {c['ingredients_uniques']} ingrédients uniques "
          f"({c['doublons_fusionnes']} doublons fusionnés, {c['familles_contradictoires']} conflits de famille résolus)")
    print(f"Formules  : {f['formules']} formules, {f['lignes']} lignes ; familles {f['familles_par_source']} ; "
          f"{len(f['fabricants_corriges'])} fabricants corrigés ; sans famille : {f['ingredients_sans_famille']}")
    print(f"Prix      : {p['produits_avec_prix_reference_100g']} produits avec un prix de référence (100 g pur, EUR) ; "
          f"non rapprochés : {p['produits_non_rapproches']}")
