"""Tests unitaires du modèle de recommandation (sans LLM : rapides et reproductibles).

    python -m pytest tests -q
"""

import numpy as np
import pandas as pd
import pytest

from reco.config import AXES
from reco.extraction import extraire_mots_cles, valider
from reco.moteur import Modele, correspondance, distance, recommander
from reco.nettoyage import charger_brut, nettoyer_catalogue, nettoyer_formules, normaliser_nom
from reco.preparer import preparer
from reco.profils import profils_formules


@pytest.fixture(scope="module")
def donnees():
    preparer()
    brut = charger_brut()
    ingredients, conflits = nettoyer_catalogue(brut["catalogue"])
    formules, details = nettoyer_formules(brut["formules"], ingredients)
    return ingredients, conflits, formules, details


# ---------- A. Nettoyage ----------

@pytest.mark.parametrize("a, b", [
    ("Iso Raldeine 70", "Isoraldeine 70"),
    ("Musk R-1", "Musk R1"),
    ("METHYL LAITONE 10%/TEC", "Methyl Laitone 10%/DPG"),
    ("Di Hydro Beta Ionone", "Dihydro Ionone Beta"),
    ("OAKMOSS GIVCO 214/3", "Oakmoss Givco 214"),
    ("Pharaone 10% DPG", "PHARAONE 10%/DPG"),
])
def test_noms_ecrits_differemment_ont_la_meme_cle(a, b):
    assert normaliser_nom(a) == normaliser_nom(b)


def test_un_ingredient_une_ligne(donnees):
    ingredients, *_ = donnees
    assert ingredients["cle"].is_unique


def test_conflit_de_famille_resolu_par_le_catalogue_prioritaire(donnees):
    ingredients, conflits, *_ = donnees
    undecavertol = ingredients.set_index("cle").loc["UNDECAVERTOL"]
    assert undecavertol["famille"] == "Vert"  # catalogue ingrédients Givaudan > Givaudan-Roure
    assert any(c["ingredient"] == "Undecavertol" for c in conflits)


def test_chaque_ingredient_de_formule_a_une_famille_ou_est_un_solvant(donnees):
    *_, formules, details = donnees
    assert details["ingredients_sans_famille"] == []
    assert set(formules.loc[formules["source_famille"] == "solvant", "cle"]) <= {"DIPROPYLENEGLYCOL", "TRIETHYLCITRATE"}


def test_un_fabricant_par_ingredient(donnees):
    *_, formules, _ = donnees
    assert (formules.groupby("cle")["fabricant"].nunique() == 1).all()
    assert formules.loc[formules["cle"] == "GALAXOLIDE", "fabricant"].unique().tolist() == ["IFF"]


# ---------- B. Profils ----------

def test_profils_sur_echelle_0_10(donnees):
    *_, formules, _ = donnees
    p = profils_formules(formules)
    assert len(p) == 13
    assert ((p[AXES] >= 0) & (p[AXES] <= 10)).all().all()


def test_profils_coherents_avec_les_noms(donnees):
    *_, formules, _ = donnees
    p = profils_formules(formules).set_index("id_formule")
    assert p["fraicheur"].idxmax() == 4    # Key Lime cream : 50 % d'huile de citron
    assert p["boise"].idxmax() == 12       # Masculine : Iso E Super, cèdre, vétiver
    assert p["chaleur"].idxmax() == 1      # Blueberry muffin : vanilline, maltol


# ---------- C. Lecture du texte ----------

def test_validation_json_llm():
    assert valider('Voici : {"chaleur": 2, "fraicheur": 9, "floral": 3, "boise": 1, "intensite": 4}') == \
        {"chaleur": 2, "fraicheur": 9, "floral": 3, "boise": 1, "intensite": 4}
    with pytest.raises(ValueError):
        valider('{"chaleur": 12, "fraicheur": 9, "floral": 3, "boise": 1, "intensite": 4}')
    with pytest.raises(KeyError):
        valider('{"chaleur": 2}')


def test_mots_cles():
    s = extraire_mots_cles("Une odeur fraîche, citron et menthe, très légère")
    assert s["fraicheur"] >= 8 and s["intensite"] <= 3


# ---------- D. Moteur ----------

def test_distance_et_correspondance():
    w = np.ones(5)
    u = np.array([5, 5, 5, 5, 5.0])
    d = distance(u, np.array([[5, 5, 5, 5, 5.0], [0, 0, 0, 0, 0.0]]), w)
    assert d[0] == 0 and d[1] > 0
    assert correspondance(np.array([0.0]), w)[0] == 100


def test_recommandation_boisee():
    rep = recommander({"chaleur": 6, "fraicheur": 1, "floral": 0, "boise": 10, "intensite": 8}, modele=Modele())
    assert rep["formules"][0]["id_formule"] == 12
    assert [f["rang"] for f in rep["formules"]] == [1, 2, 3]
    assert all(i["famille"] for i in rep["ingredients"])


def test_api_curseurs():
    from fastapi.testclient import TestClient
    from api import app

    r = TestClient(app).post("/recommander/curseurs",
                             json={"chaleur": 1, "fraicheur": 10, "floral": 1, "boise": 0, "intensite": 3})
    assert r.status_code == 200
    assert r.json()["formules"][0]["nom"].startswith("Key Lime")
    assert TestClient(app).post("/recommander/curseurs", json={"chaleur": 11}).status_code == 422
