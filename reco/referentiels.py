"""Référentiels écrits à la main par l'équipe : ce sont des CHOIX, pas des données.

Tout ce qui est ici est à faire valider par un parfumeur Givaudan.
Chaque table est documentée : pourquoi elle existe, et comment elle est utilisée.
"""

# ---------------------------------------------------------------------------
# 1. Famille olfactive -> scores sur les 5 axes (0 à 10)
#    Ordre : chaleur, fraicheur, floral, boise, intensite
#    Une famille "Neutre" n'a pas d'odeur : elle est exclue des moyennes.
# ---------------------------------------------------------------------------
FAMILLE_SCORES = {
    "Hespéridé (agrumes)": [1, 10, 1, 0, 3],
    "Vert":                [1, 8, 2, 2, 4],
    "Marin":               [0, 9, 1, 0, 4],
    "Frais":               [0, 9, 1, 1, 3],
    "Menthé":              [0, 10, 0, 0, 5],
    "Camphré":             [1, 7, 0, 3, 7],
    "Aromatique (herbes)": [2, 7, 1, 3, 5],
    "Aldéhydé":            [3, 6, 4, 1, 6],
    "Fruité":              [4, 5, 4, 0, 5],
    "Floral":              [3, 3, 10, 0, 5],
    "Lactonique":          [6, 2, 3, 1, 4],
    "Poudré":              [6, 1, 5, 2, 4],
    "Musqué":              [6, 2, 3, 2, 6],
    "Gras":                [4, 1, 1, 1, 3],
    "Gourmand":            [9, 0, 1, 1, 7],
    "Sucré":               [9, 0, 1, 1, 6],
    "Balsamique":          [8, 0, 2, 3, 7],
    "Épicé":               [8, 1, 1, 3, 7],
    "Ambré":               [9, 0, 1, 4, 8],
    "Mousse (chypre)":     [4, 2, 1, 8, 7],
    "Terreux":             [5, 1, 0, 8, 8],
    "Boisé":               [5, 1, 0, 10, 7],
    "Cuir":                [7, 0, 0, 6, 9],
    "Tabac":               [8, 0, 0, 6, 8],
    "Animal":              [8, 0, 1, 3, 10],
}
FAMILLES_SANS_ODEUR = {"Neutre"}

# ---------------------------------------------------------------------------
# 2. Solvants : sans odeur, ils diluent la formule. Retirés avant tout calcul.
# ---------------------------------------------------------------------------
SOLVANTS = {"DIPROPYLENEGLYCOL", "TRIETHYLCITRATE"}

# ---------------------------------------------------------------------------
# 3. Familles des ingrédients des formules ABSENTS des catalogues Givaudan.
#    106 ingrédients des formules ne sont dans aucun catalogue (matières
#    génériques, huiles essentielles, molécules IFF/Firmenich).
#    Clé = nom normalisé (voir nettoyage.normaliser_nom).
# ---------------------------------------------------------------------------
FAMILLES_HORS_CATALOGUE = {
    # Gourmand / sucré
    "MALTYLISOBUTYRATE": "Gourmand", "ACETYL2PYRIDINE": "Gourmand", "CORYLONEDRIED": "Gourmand",
    "CORYLONEPYRAZINE": "Gourmand", "HOMOFURONOL": "Gourmand", "COUMARINPURE": "Gourmand",
    "MALTOL": "Gourmand", "ETHYLMALTOL": "Gourmand", "ETHYLVANILLIN": "Gourmand", "VANILLIN": "Gourmand",
    "HYDROXYETHYLMETHYLTHIAZOL": "Gourmand", "RHUMACETAL": "Gourmand",
    # Fruité
    "BUCHULEAFOILSOUTHAFRICA": "Fruité", "DAMASCENONE": "Fruité", "ETHYLBUTYRATE": "Fruité",
    "ETHYLCAPROATE": "Fruité", "DAMASCONEALPHA": "Fruité", "DAMASCONEDELTA": "Fruité",
    "RASPBERRYKETONE": "Fruité", "DIMETHYLBENZYLCARBINYLBUTYRATE": "Fruité", "ALLYLCAPROATE": "Fruité",
    "HEXYLACETATE": "Fruité", "FRUCTONE": "Fruité", "ALLYLOENANTHATE": "Fruité", "GRAPELIONE": "Fruité",
    "ETHYLISOBUTYRATE": "Fruité", "DEWFRUITTECAB2927R": "Fruité", "ISOAMYLACETATE": "Fruité",
    "DAVANAOILINDIA": "Fruité", "ETHYLBENZOATE": "Fruité", "BUTYLACETATE": "Fruité",
    "VERDOX": "Fruité", "ETHYLACETOACETATE": "Fruité", "TAGETEOIL": "Fruité",
    # Lactonique
    "OCTALACTONEGAMMA": "Lactonique", "DECALACTONEGAMMA": "Lactonique", "ACETOIN": "Lactonique",
    # Floral
    "LINALOOL": "Floral", "BENZOPHENONE": "Floral", "GERANIOLINTERMEDIATE60": "Floral",
    "GERANIOL980": "Floral", "HEDIONE": "Floral", "HEXYLCINNAMICALDEHYDE": "Floral",
    "CITRONELLYLACETATE": "Floral", "BENZYLACETATE": "Floral", "DIPHENYLOXIDE": "Floral",
    "ANISICALDEHYDE": "Floral", "CITRONELLOL": "Floral", "CITRONELLYLETHOXALATE": "Floral",
    "CITRONELLYLPROPIONATE": "Floral", "TERPINEOLPURE": "Floral", "ROSEOXIDE": "Floral",
    "HELIONAL": "Floral", "PHENYLETHYLALCOHOL": "Floral", "GERANIUMOILEGYPT": "Floral",
    "CANTHOXAL": "Floral", "ROSEOILBULGARY": "Floral",
    # Vert
    "DIMETHYLSULFIDE": "Vert", "HEXENOL3CIS": "Vert", "TRIPLAL": "Vert", "STYRALLYLACETATE": "Vert",
    "HEXENYL3CISACETATE": "Vert", "LIFFAROME": "Vert", "HEXENYL3CISSALICYLATE": "Vert",
    "HEXENYL3CISBUTYRATE": "Vert", "VIRIDINE": "Vert",
    # Hespéridé
    "PARADISAMIDE": "Hespéridé (agrumes)", "MANDARINOILYELLOWITALY": "Hespéridé (agrumes)",
    "LEMONOILITALY": "Hespéridé (agrumes)", "PETITGRAINOILPARAGUAY": "Hespéridé (agrumes)",
    "BERGAMOTOILITALY": "Hespéridé (agrumes)", "ORANGEOILSWEETITALIANTAROCCO": "Hespéridé (agrumes)",
    "GRAPEFRUITOIL": "Hespéridé (agrumes)", "ORANGEOILFLORIDA": "Hespéridé (agrumes)",
    # Menthé / marin / aromatique
    "ISOPULEGOL": "Menthé", "PARAMENTHONE": "Menthé", "CALONE": "Marin",
    "DELTA3CARENE": "Aromatique (herbes)", "MYRCENE90": "Aromatique (herbes)",
    "PINENEALPHA": "Aromatique (herbes)", "TERPENYLACETATE": "Aromatique (herbes)",
    "JUNIPEROIL": "Aromatique (herbes)", "SAGEOFFICINALISOIL": "Aromatique (herbes)",
    "LAVANDINGROSSOOILFRANCE": "Aromatique (herbes)", "ROSEMARYOILNORTHAFRICA": "Aromatique (herbes)",
    "CLARYSAGEOIL": "Aromatique (herbes)", "BASEHERBALUR": "Aromatique (herbes)",
    # Aldéhydé / épicé / balsamique
    "ALDEHYDEC8OCTYLIC": "Aldéhydé", "EUGENOLPURE": "Épicé", "ANETHOLE": "Épicé",
    "CARDAMOMSEEDOIL": "Épicé", "METHYLCINNAMATE": "Balsamique",
    # Musqué (Iso E Super est classé boisé)
    "ISOESUPER": "Boisé", "GALAXOLIDE": "Musqué", "ETHYLENEBRASSYLATE": "Musqué", "THIBETOLIDE": "Musqué",
    # Boisé / terreux
    "PARABUTYLCYCLOHEXANOL": "Boisé", "PTBUTYLCYCLOHEXYLACETATE": "Boisé", "FIRBALSAMOILCANADA": "Boisé",
    "CEDARWOODOILATLAS": "Boisé", "CEDARWOODOILVIRGINIA": "Boisé",
    "VETIVEROILHAITIDRIEDFILTERED": "Boisé", "METHYLCEDRYLKETONE": "Boisé",
    "PATCHOULIOILCOLORLESS": "Terreux",
}

# ---------------------------------------------------------------------------
# 4. Règle de résolution quand un ingrédient a deux familles différentes :
#    on garde le catalogue le plus récent et le plus proche de Givaudan
#    aujourd'hui. Plus le rang est petit, plus le catalogue est prioritaire.
# ---------------------------------------------------------------------------
PRIORITE_CATALOGUE = {
    "Givaudan - catalogue ingrédients": 1,
    "Givaudan-Roure (code GR)": 2,
    "Givaudan - bases spéciales Givco": 3,
    "Quest International - Perfumer's Compendium": 4,
}

# ---------------------------------------------------------------------------
# 5. Noms écrits différemment d'une source à l'autre, que la normalisation
#    automatique ne suffit pas à rapprocher. Clé et valeur = noms normalisés.
# ---------------------------------------------------------------------------
ALIAS = {
    "DIHYDROIONONEBETA": "DIHYDROBETAIONONE",
    "DIHYDROBETAIONONE": "DIHYDROBETAIONONE",
    "2ISOBUTYLQUINOLINE": "ISOBUTYLQUINOLINE2",
    "METHYLTUBERATE": "METHYLTUBERATEPURE",
    "CYCLAMENALDEHYDE": "CYCLAMENALDEHYDEEXTRA",
    "CHOCOVAN": "CHOCOVANFM1197",
    "DEWFRUIT": "DEWFRUITABQ2977",
    "TRIDEEN2NITRILE": "TRIDECENE2NITRILE",
    "ORANGECRYSTALS": "ORANGERCRYSTALS",
}
