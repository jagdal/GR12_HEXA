"""Jeu de questions du benchmark, avec réponses attendues (mots-clés) et chapitre source.

- `expect` : liste de groupes ; chaque groupe est une liste d'alternatives acceptées.
  La réponse est juste si chaque groupe a au moins une alternative présente.
- `chapter` : début du titre du chapitre qui contient la réponse (pour évaluer la recherche).
- Les questions avec `expect=None` n'ont pas de réponse dans le dossier :
  le modèle doit le dire au lieu d'inventer.
"""

QUESTIONS = [
    {"q": "Quel a été le chiffre d'affaires de Givaudan en 2025 ?", "expect": [["7 472", "7472", "7,47"]], "chapter": "Chapitre 5"},
    {"q": "Quelle a été la croissance like-for-like de Givaudan en 2025 ?", "expect": [["5,1"]], "chapter": "Chapitre 5"},
    {"q": "Quel dividende par action Givaudan propose-t-il pour l'exercice 2025 ?", "expect": [["72"]], "chapter": "Chapitre 5"},
    {"q": "Qui est le CEO de Givaudan depuis mars 2026 ?", "expect": [["Stammkoetter"]], "chapter": "Chapitre 3"},
    {"q": "En quelle année Givaudan a-t-elle été fondée ?", "expect": [["1895"]], "chapter": "Chapitre 2"},
    {"q": "Quelle a été la croissance de la parfumerie fine (Fine Fragrance) en 2025 ?", "expect": [["18,3"]], "chapter": "Chapitre 6"},
    {"q": "Quel était le bénéfice net de Givaudan au premier semestre 2026 ?", "expect": [["475"]], "chapter": "Chapitre 9"},
    {"q": "En quelle année Givaudan vise-t-elle le net zéro sur sa chaîne de valeur ?", "expect": [["2045"]], "chapter": "Chapitre 12"},
    {"q": "Qui sont les principaux concurrents de Givaudan ?", "expect": [["firmenich"], ["IFF"], ["Symrise"]], "chapter": "Chapitre 1"},
    {"q": "Quel était le chiffre d'affaires de Givaudan sur les neuf premiers mois de 2026 ?", "expect": [["5 891", "5891"]], "chapter": "Chapitre 10"},
    {"q": "Quel était le ratio dette nette sur EBITDA de Givaudan à fin 2025 ?", "expect": [["2,1"]], "chapter": "Chapitre 5"},
    {"q": "Quel était l'EBITDA de Givaudan en 2025 ?", "expect": [["1 751", "1751", "1 807", "1807"]], "chapter": "Chapitre 5"},
    {"q": "Quel a été le free cash flow de Givaudan en 2025 ?", "expect": [["1 053", "1053"]], "chapter": "Chapitre 5"},
    {"q": "Quel était le chiffre d'affaires de la division Fragrance & Beauty en 2025 ?", "expect": [["3 830", "3830"]], "chapter": "Chapitre 6"},
    {"q": "Quel était le chiffre d'affaires de la division Taste & Wellbeing en 2025 ?", "expect": [["3 642", "3642"]], "chapter": "Chapitre 7"},
    {"q": "Quelle région a le plus progressé dans la division Taste & Wellbeing en 2025 ?", "expect": [["SAMEA", "Asie du Sud"]], "chapter": "Chapitre 7"},
    {"q": "En quelle année Roche a-t-il racheté Givaudan ?", "expect": [["1963"]], "chapter": "Chapitre 2"},
    {"q": "En quelle année Givaudan a-t-elle été séparée de Roche et cotée en bourse ?", "expect": [["2000"]], "chapter": "Chapitre 2"},
    {"q": "Quelle société Givaudan a-t-elle rachetée en 2007 ?", "expect": [["Quest"]], "chapter": "Chapitre 2"},
    {"q": "Dans quelle ville se trouve le siège de Givaudan ?", "expect": [["Vernier"]], "chapter": "Chapitre 1"},
    {"q": "Combien d'employés Givaudan comptait-elle à fin 2025 ?", "expect": [["17 600", "17600", "17 6"]], "chapter": "Chapitre 1"},
    {"q": "Quelle a été la croissance like-for-like de Givaudan au troisième trimestre 2026 ?", "expect": [["8,6"]], "chapter": "Chapitre 10"},
    {"q": "Depuis quelle année toute l'électricité de Givaudan est-elle renouvelable ?", "expect": [["2024"]], "chapter": "Chapitre 12"},
    {"q": "Quel pourcentage des postes de direction est occupé par des femmes chez Givaudan ?", "expect": [["34"]], "chapter": "Chapitre 12"},
    {"q": "Quel est l'objectif de croissance des ventes de la stratégie 2030 de Givaudan ?", "expect": [["4 à 6", "4-6", "4 et 6", "4 % à 6", "4% à 6", "4 a 6"]], "chapter": "Chapitre 11"},
    {"q": "Quelle a été la croissance moyenne des ventes de Givaudan sur la période 2021-2025 ?", "expect": [["6,8"]], "chapter": "Chapitre 11"},
    {"q": "Quel était le chiffre d'affaires de Givaudan en 2024 ?", "expect": [["7 412", "7412"]], "chapter": "Chapitre 4"},
    {"q": "En quelle année Givaudan a-t-elle racheté Naturex ?", "expect": [["2018"]], "chapter": "Chapitre 2"},
    {"q": "Quelle était la dette nette de Givaudan au 30 juin 2026 ?", "expect": [["4 604", "4604"]], "chapter": "Chapitre 9"},
    {"q": "Quelle était la marge EBITDA ajustée de Givaudan au premier semestre 2026 ?", "expect": [["24,3"]], "chapter": "Chapitre 9"},
    {"q": "Quel était le chiffre d'affaires de la région Asie-Pacifique au premier semestre 2026 ?", "expect": [["909"]], "chapter": "Chapitre 9"},
    {"q": "De combien Givaudan a-t-elle réduit ses émissions des scopes 1 et 2 par rapport à 2015 ?", "expect": [["50"]], "chapter": "Chapitre 12"},
    # Questions sans réponse dans le dossier : le modèle doit refuser d'inventer
    {"q": "Quel est le salaire annuel du CEO de Givaudan ?", "expect": None, "chapter": None},
    {"q": "Quel est le cours de l'action Givaudan aujourd'hui ?", "expect": None, "chapter": None},
    {"q": "Combien de brevets Givaudan a-t-elle déposés en 2025 ?", "expect": None, "chapter": None},
    {"q": "Quel sera le chiffre d'affaires de Givaudan en 2027 ?", "expect": None, "chapter": None},
]
