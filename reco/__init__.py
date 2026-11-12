"""Modèle de recommandation de fragrances Givaudan pour boutiques indépendantes.

Étapes (chaque module correspond à une étape du pipeline) :
    nettoyage.py   — A. charger et nettoyer les CSV bruts
    profils.py     — B. transformer familles et formules en scores sur 5 axes
    extraction.py  — C. lire la description de la boutique (Ollama ou mots-clés)
    moteur.py      — D. calculer la ressemblance et le top 3
    entrainer.py / evaluer.py — E. calibrage sur le jeu d'entraînement, mesure sur le jeu de test

Commandes :
    python -m reco.preparer    # A + B : produit data/processed/
    python -m reco.entrainer   # calibre le modèle, écrit models/modele_reco.json
    python -m reco.evaluer     # mesure sur le jeu de test, écrit reports/
"""
