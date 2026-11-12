"""Étape A — Charger et nettoyer les CSV bruts.

Chaque problème de qualité trouvé dans les données a une règle écrite ici,
et chaque correction est comptée dans le rapport de qualité.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from . import config
from .referentiels import ALIAS, FAMILLES_HORS_CATALOGUE, PRIORITE_CATALOGUE, SOLVANTS

FICHIERS = {
    "catalogue": "givaudan_anciens_catalogues.csv",
    "formules": "givaudan_formules_demo.csv",
    "prix": "givaudan_prix_revendeurs.csv",
}

_DILUTION = re.compile(r"\d+(?:[.,]\d+)?\s*%(?:\s*/?\s*[A-Z][A-Z\-]*)?")
_SUFFIXE_LOT = re.compile(r"/\d+$")


def normaliser_nom(nom: str) -> str:
    """Clé de rapprochement d'un nom d'ingrédient.

    « Iso Raldeine 70 » et « Isoraldeine 70 » -> ISORALDEINE70
    « METHYL LAITONE 10%/TEC » et « Methyl Laitone 10%/DPG » -> METHYLLAITONE
    (la dilution ne change pas la molécule ; elle est retirée de la clé)
    """
    s = str(nom).upper().strip()
    s = _DILUTION.sub("", s)
    s = _SUFFIXE_LOT.sub("", s.strip())
    s = re.sub(r"[^A-Z0-9]", "", s)
    return ALIAS.get(s, s)


def charger_brut() -> dict[str, pd.DataFrame]:
    """Lit les CSV (séparateur « ; », UTF-8 avec BOM)."""
    return {
        nom: pd.read_csv(config.RAW_DIR / fichier, sep=";", encoding="utf-8-sig", dtype=str)
        for nom, fichier in FICHIERS.items()
    }


# ---------------------------------------------------------------------------
# Catalogue : un ingrédient = une ligne, une famille
# ---------------------------------------------------------------------------
def nettoyer_catalogue(cat: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    cat = cat.copy()
    cat["cle"] = cat["nom_produit"].map(normaliser_nom)
    cat["priorite"] = cat["catalogue"].map(PRIORITE_CATALOGUE).fillna(9)

    conflits = []
    lignes = []
    for cle, groupe in cat.groupby("cle"):
        groupe = groupe.sort_values("priorite")
        retenue = groupe.iloc[0]
        familles = sorted(groupe["famille_olfactive"].unique())
        if len(familles) > 1:
            conflits.append({
                "ingredient": retenue["nom_produit"],
                "familles": {r.catalogue: r.famille_olfactive for r in groupe.itertuples()},
                "famille_retenue": retenue["famille_olfactive"],
            })
        lignes.append({
            "cle": cle,
            "nom": retenue["nom_produit"],
            "famille": retenue["famille_olfactive"],
            # « oui » dans au moins un catalogue = toujours vendu en 2024
            "encore_au_catalogue": (groupe["encore_au_catalogue_2024"] == "oui").any(),
            "catalogue_retenu": retenue["catalogue"],
            "nb_catalogues": len(groupe),
        })
    return pd.DataFrame(lignes), conflits


# ---------------------------------------------------------------------------
# Formules : chaque ligne reçoit une famille et un fabricant fiables
# ---------------------------------------------------------------------------
def nettoyer_formules(f: pd.DataFrame, ingredients: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    f = f.copy()
    f["pourcentage"] = f["pourcentage"].astype(float)
    f["id_formule"] = f["id_formule"].astype(int)
    f["cle"] = f["ingredient"].map(normaliser_nom)

    par_cle = ingredients.set_index("cle")
    stats = Counter()

    def famille(cle: str) -> tuple[str | None, str]:
        if cle in SOLVANTS:
            return None, "solvant"
        if cle in par_cle.index:
            return par_cle.at[cle, "famille"], "catalogue"
        if cle in FAMILLES_HORS_CATALOGUE:
            return FAMILLES_HORS_CATALOGUE[cle], "table manuelle"
        return None, "inconnue"

    resultat = f["cle"].map(famille)
    f["famille"] = resultat.map(lambda t: t[0])
    f["source_famille"] = resultat.map(lambda t: t[1])
    stats.update(f["source_famille"])

    # Fabricant : un même ingrédient ne peut avoir qu'un fabricant.
    # Règle, dans l'ordre :
    #   1. le fabricant nommé le plus souvent dans les formules (Givaudan, IFF, Firmenich) ;
    #   2. sinon, un produit du catalogue Givaudan encore vendu en 2024 -> Givaudan ;
    #   3. sinon, générique (les anciens catalogues Quest listent aussi des matières
    #      courantes comme le Linalyl Acetate, qui ne sont pas des exclusivités).
    nommes = (
        f[f["fabricant_ingredient"] != "Générique / naturel"]
        .groupby("cle")["fabricant_ingredient"]
        .agg(lambda s: s.value_counts().index[0])
    )

    def fabricant(row) -> str:
        if row.cle in nommes.index:
            return nommes[row.cle]
        if row.cle in par_cle.index and par_cle.at[row.cle, "encore_au_catalogue"]:
            return "Givaudan"
        return row.fabricant_ingredient

    f["fabricant"] = f.apply(fabricant, axis=1)
    corriges = f[f["fabricant"] != f["fabricant_ingredient"]]
    stats["fabricants_corriges"] = len(corriges)
    f["encore_au_catalogue"] = f["cle"].map(par_cle["encore_au_catalogue"]) if len(par_cle) else None

    details = {
        "familles_par_source": {k: int(v) for k, v in stats.items() if k != "fabricants_corriges"},
        "fabricants_corriges": corriges[["id_formule", "ingredient", "fabricant_ingredient", "fabricant"]]
        .to_dict("records"),
        "ingredients_sans_famille": sorted(f.loc[f["source_famille"] == "inconnue", "ingredient"].unique()),
    }
    return f, details


# ---------------------------------------------------------------------------
# Prix : même devise (EUR), même format de référence
# ---------------------------------------------------------------------------
def nettoyer_prix(p: pd.DataFrame, ingredients: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = p.copy()
    p["cle"] = p["produit"].map(normaliser_nom)
    cles = set(ingredients["cle"])
    # Rapprochement par début de nom quand il est unique (« Methyl Tuberate » -> « Methyl Tuberate Pure »)
    def rapprocher(cle: str) -> str | None:
        if cle in cles:
            return cle
        candidats = [c for c in cles if c.startswith(cle) or cle.startswith(c)]
        return candidats[0] if len(candidats) == 1 else None

    p["cle_catalogue"] = p["cle"].map(rapprocher)
    p["prix"] = pd.to_numeric(p["prix"], errors="coerce")
    p["taille_g"] = pd.to_numeric(p["taille_g"], errors="coerce")
    p["prix_eur"] = p["prix"] * p["devise"].map(config.TAUX_EUR)
    p["prix_eur_kg"] = p["prix_eur"] / p["taille_g"] * 1000

    # Le prix au kg dépend énormément du format (Ambrofix : 8 000 $/kg en 1 g,
    # 995 $/kg en 1 kg). On compare donc au même format : 100 g de produit pur.
    ref = p[(p["taille_g"] == 100) & (p["qualite"] == "100 % pur")]
    ref = ref.groupby("cle_catalogue", as_index=False)["prix_eur_kg"].min()
    ref = ref.rename(columns={"prix_eur_kg": "prix_eur_kg_100g"})

    details = {
        "lignes": len(p),
        "devises": p["devise"].value_counts().to_dict(),
        "lignes_sans_taille": int(p["taille_g"].isna().sum()),
        "produits_rapproches_du_catalogue": int(p["cle_catalogue"].notna().groupby(p["produit"]).first().sum()),
        "produits_non_rapproches": sorted(p.loc[p["cle_catalogue"].isna(), "produit"].unique()),
        "produits_avec_prix_reference_100g": len(ref),
        "taux_de_change": config.TAUX_EUR,
        "taux_date": config.TAUX_DATE,
    }
    return ref, details
