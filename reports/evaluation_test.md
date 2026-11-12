# Évaluation sur le jeu de test (14 cas)

Modèle 1.0.0-2026-11-12 · LLM qwen2.5:3b (few_shot) · poids {'chaleur': 1.0, 'fraicheur': 1.0, 'floral': 0.5, 'boise': 2.0, 'intensite': 2.0}

| Configuration | MAE scores (/10) | hit@1 | hit@3 | MRR |
|---|---|---|---|---|
| A_mots_cles_poids_egaux | 1.4 | 43% | 57% | 0.57 |
| B_llm_poids_egaux | 1.31 | 50% | 57% | 0.61 |
| C_llm_poids_appris_MODELE_FINAL | 1.31 | 43% | 64% | 0.60 |
| D_oracle_scores_attendus_poids_appris | — | 36% | 64% | 0.55 |

Latence LLM : médiane 4.4 s, max 9.7 s.

| # | Description | Scores LLM | Top 3 | Pertinentes | Rang |
|---|---|---|---|---|---|
| 3 | Laverie automatique : odeur de linge propre, douce et fraîche | 1 9 1 1 5 | Key Lime cream, AP Roll-on UR1007, Shampoo HR1009 | Fabric conditioner LR1007, Shower gel GM1006 | 6 |
| 6 | Boutique de lingerie féminine, odeur florale, poudrée et musquée, élégante | 2 1 9 1 6 | Shower gel GM1006, Fabric conditioner LR1007, Vert de figue basilic | Feminine FG1006, Feminine FM1006 | 6 |
| 7 | Fleuriste : je veux que ça sente les fleurs fraîches et le vert, comme un jardin | 1 9 9 1 4 | Shampoo HR1009, Shower gel GM1006, Vert de figue basilic | Vert de figue basilic, Shower gel GM1006 | 2 |
| 11 | Concept store de mode masculine, sobre, boisé et un peu épicé | 4 1 0 9 5 | AP Roll-on UR1007, Masculine MM1006, Feminine FM1006 | Masculine MM1006, AP Roll-on UR1007 | 1 |
| 12 | Hôtel boutique, hall d'accueil, floral élégant et chic, présence discrète | 1 1 9 1 3 | Shower gel GM1006, Shampoo HR1009, Fabric conditioner LR1007 | Feminine FM1006, Feminine FG1006 | 7 |
| 17 | Pharmacie-parapharmacie, odeur propre, neutre, très légère, rassurante | 1 9 1 1 2 | Key Lime cream, Shampoo HR1009, AP Roll-on UR1007 | Shower gel GM1006, Shampoo HR1009, Fabric conditioner LR1007 | 2 |
| 18 | Crêperie bretonne, odeur gourmande, chaleureuse, caramel et vanille | 9 1 2 1 5 | Blueberry muffin, Grenadine mandarin, Mango peach | Blueberry muffin | 1 |
| 19 | Boutique de bougies, odeur cosy et réconfortante, chaude et douce | 9 1 1 2 7 | Blueberry muffin, Mango peach, Feminine FM1006 | Blueberry muffin, Quince rum | 1 |
| 21 | Bijouterie de luxe, parfum floral raffiné, rose et jasmin, avec une note musquée | 2 1 9 1 5 | Shower gel GM1006, Fabric conditioner LR1007, Vert de figue basilic | Feminine FM1006, Feminine FG1006 | 6 |
| 22 | Glacier : fraîcheur fruitée, citron, crème glacée | 1 9 1 1 5 | Key Lime cream, AP Roll-on UR1007, Shampoo HR1009 | Key Lime cream | 1 |
| 25 | Salon esthétique pour femmes, floral fruité, doux et féminin | 3 1 8 1 5 | Fabric conditioner LR1007, Shower gel GM1006, Vert de figue basilic | Feminine FM1006, Feminine FG1006, Grenadine mandarin | 5 |
| 27 | Cabinet de massage, odeur douce et relaxante, florale et musquée, très légère | 2 1 9 1 1 | Shampoo HR1009, Shower gel GM1006, Fabric conditioner LR1007 | Feminine FM1006, Shower gel GM1006 | 2 |
| 34 | Parfumerie de niche, floral boisé musqué, sophistiqué et intense | 2 1 9 7 8 | Feminine FM1006, Feminine FG1006, AP Roll-on UR1007 | Feminine FG1006, Feminine FM1006, Masculine MM1006 | 1 |
| 38 | Bar à jus de fruits, frais, agrumes, mangue | 1 9 0 1 5 | Key Lime cream, AP Roll-on UR1007, Shampoo HR1009 | Mango peach, Key Lime cream | 1 |
