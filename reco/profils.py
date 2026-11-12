"""Étape B — Transformer les données en scores sur les 5 axes (feature engineering)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AXES
from .referentiels import FAMILLE_SCORES, FAMILLES_SANS_ODEUR


def vecteur_famille(famille: str | None) -> np.ndarray | None:
    """Scores (0-10) d'une famille olfactive, ou None si elle n'a pas d'odeur ou est inconnue."""
    if famille is None or famille in FAMILLES_SANS_ODEUR or famille not in FAMILLE_SCORES:
        return None
    return np.array(FAMILLE_SCORES[famille], dtype=float)


def profils_ingredients(ingredients: pd.DataFrame, prix_ref: pd.DataFrame) -> pd.DataFrame:
    """Un ingrédient du catalogue = les scores de sa famille."""
    df = ingredients.copy()
    vecteurs = df["famille"].map(vecteur_famille)
    df = df[vecteurs.notna()].copy()
    df[AXES] = np.vstack(vecteurs[vecteurs.notna()].to_numpy())
    df = df.merge(prix_ref, how="left", left_on="cle", right_on="cle_catalogue").drop(columns="cle_catalogue")
    return df


def profils_formules(formules: pd.DataFrame) -> pd.DataFrame:
    """Profil d'une formule = moyenne de ses ingrédients, pondérée par leur pourcentage.

    Seuls les ingrédients odorants avec une famille connue comptent (les solvants
    sont exclus). Le profil brut est ensuite remis à l'échelle 0-10 sur chaque axe,
    pour que les formules occupent toute l'échelle des curseurs.
    """
    lignes = []
    for fid, g in formules.groupby("id_formule"):
        g = g.copy()
        g["vecteur"] = g["famille"].map(vecteur_famille)
        odorants = g[g["source_famille"] != "solvant"]
        connus = odorants[odorants["vecteur"].notna()]
        poids = connus["pourcentage"].to_numpy()
        profil = (np.vstack(connus["vecteur"].to_numpy()) * poids[:, None]).sum(axis=0) / poids.sum()

        par_famille = connus.groupby("famille")["pourcentage"].sum().sort_values(ascending=False)
        givaudan = odorants[odorants["fabricant"] == "Givaudan"].sort_values("pourcentage", ascending=False)
        arretes = givaudan[givaudan["encore_au_catalogue"] == False]  # noqa: E712 (NaN = inconnu, pas « arrêté »)

        lignes.append({
            "id_formule": int(fid),
            "nom": g["nom_formule"].iloc[0],
            "application": g["application"].iloc[0],
            **{f"{a}_brut": round(float(v), 3) for a, v in zip(AXES, profil)},
            "famille_dominante": par_famille.index[0],
            "familles_principales": " · ".join(
                f"{fam} {pct / odorants['pourcentage'].sum() * 100:.0f} %" for fam, pct in par_famille.head(3).items()
            ),
            "part_givaudan_pct": round(givaudan["pourcentage"].sum() / odorants["pourcentage"].sum() * 100, 1),
            "ingredients_givaudan_principaux": " · ".join(givaudan["ingredient"].head(3).str.title()),
            "givaudan_hors_catalogue_2024": " · ".join(arretes["ingredient"].str.title()),
            "couverture_familles_pct": round(poids.sum() / odorants["pourcentage"].sum() * 100, 1),
            "nb_ingredients": len(g),
        })

    df = pd.DataFrame(lignes)
    for a in AXES:
        brut = df[f"{a}_brut"]
        etendue = brut.max() - brut.min()
        df[a] = ((brut - brut.min()) / etendue * 10).round(2) if etendue > 0 else 5.0
    return df
