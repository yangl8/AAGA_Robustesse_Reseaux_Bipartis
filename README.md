# AAGA_Robustesse_Reseaux_Bipartis

## Présentation du projet
Ce projet a été réalisé dans le cadre du **Master 2 Informatique – Parcours Science et Technologie du Logiciel (en alternance)** à **Sorbonne Université**, dans le module **AAGA (Analyse d'Algorithmes et Graphes et Applications)**.  

L’objectif est d’analyser la robustesse des réseaux bipartis écologiques (par exemple, les réseaux *plantes–pollinisateurs*) en simulant différentes stratégies de suppression de nœuds.  
Le travail combine modélisation algorithmique, mesure de la robustesse structurelle et évaluation expérimentale.

---

## Auteurs
- Mengxiao LI  
- Liu YANG  
- Xue YANG  

---

## Structure du projet
```
AAGA_Robustesse_Reseaux_Bipartis/
│
├── src/
│   ├── degree_strategy.py          # Stratégie de suppression par degré
│   ├── hits_strategy.py            # Algorithme HITS
│   ├── musrank_strategy.py         # Algorithme MusRank (non linéaire)
│   ├── pageRank.py                 # Algorithme PageRank
│   ├── random_score.py             # Stratégie aléatoire
│   ├── robustesse_bipartite.ipynb  # Notebook principal d’expérimentation
│
├── tests/
│   ├── figure/                     # Figures générées (visualisations, courbes)
│   └── results/                #Les résultats des expériences.
│   ├── network_pollinator.json # Réseau écologique Pollinator
│   └── network_prunus.json     # Réseau écologique Prunus
│
├── .gitignore
├── run_all.sh        # Script Bash pour exécuter automatiquement toutes les expériences
└── README.md
```
---

## Jeux de données
Le projet repose sur deux **réseaux écologiques bipartis réels**, représentant des interactions plantes–pollinisateurs.
Ces réseaux diffèrent par leur taille et leur complexité structurelle.

| Jeu de données | Description                         | Taille du réseau                                             | Nombre d’arêtes |
| -------------- | ----------------------------------- | ------------------------------------------------------------ | --------------- |
| **Prunus**     | Petit réseau *plante–pollinisateur* | 64 nœuds actifs (pollinisateurs) + 5 nœuds passifs (plantes) | 95              |
| **Pollinator** | Grand réseau *plante–pollinisateur* | 677 nœuds actifs + 91 nœuds passifs                          | 1 193           |

Chaque fichier est au format **JSON** et contient deux sections : les **nœuds** et les **liens**.

**Exemple de structure :**

```json
{
  "nodes": [
    {"nodeid": "0", "name": "Celastrina argiolus", "group": "Pollinator", "fill": "#f8db16", "value": 0.0838},
    {"nodeid": "1", "name": "Syritta pipiens", "group": "Pollinator", "fill": "#f8db16", "value": 0.1676},
    {"nodeid": "2", "name": "Apis mellifera", "group": "Pollinator", "fill": "#f8db16", "value": 0.4191}
  ],
  "links": [
    {"source": "0", "target": "68"},
    {"source": "1", "target": "72"}
  ]
}
```

---

## Signification des champs
| Champ | Type | Description |
|-------|------|-------------|
| `nodeid` | string | Identifiant unique du nœud |
| `name` | string | Nom de l'espèce |
| `group` | string | Catégorie du nœud (Pollinator ou Plant) |
| `fill` | string | Couleur utilisée pour la visualisation |
| `value` | float | Poids ou importance du nœud |
| `source / target` | string | Identifiants des deux extrémités d'un lien |



---

## Installation et exécution
---

### Exécution automatique

Pour lancer toutes les expériences d’un seul coup :

```bash
bash run_all.sh
```

### 1. Création d’un environnement virtuel
```
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installation des dépendances
```
pip install -r requirements.txt
```

Si le fichier requirements.txt n’est pas disponible, installez manuellement :
```
pip install numpy pandas matplotlib networkx scipy jupyterlab
```
---

## Lancement des expériences sous JupyterLab

Les tests expérimentaux et les visualisations des résultats sont réalisés dans un notebook Jupyter :
```
src/robustesse_bipartite.ipynb.
```

Ce notebook exécute l’ensemble des simulations de robustesse sur les **deux réseaux écologiques** (*Pollinator et Prunus*), en comparant les différentes stratégies de suppression de nœuds (*Random, Degree, HITS, PageRank, BiPageRank, MusRank*).
Il produit automatiquement les courbes de connectivité (LCC), les courbes d’extinction et les aires d’extinction (EA) pour chaque algorithme.

### Installation de JupyterLab

Si JupyterLab n’est pas encore installé :
```
pip install jupyterlab
```

### Lancement

Démarrer l’environnement JupyterLab :
```
jupyter lab
```

Le navigateur ouvre automatiquement `http://localhost:8888/lab`.
Dans le panneau latéral, ouvrir :
```
src/robustesse_bipartite.ipynb
```

puis exécuter les cellules du notebook pour lancer les simulations sur les deux jeux de données.

Si la page reste vide, copiez manuellement l’URL complète (contenant le token) depuis le terminal et collez-la dans le navigateur.

---

## Algorithmes implémentés

- **Random** — Suppression aléatoire de nœuds
- **Degree** — Suppression selon le degré décroissant
- **HITS** — Algorithme Hub/Authority
- **PageRank** — Classement basé sur le modèle de marche aléatoire
- **BiPageRank** — Version adaptée aux réseaux bipartis
- **MusRank** — Algorithme non linéaire fondé sur la nestedness du réseau

---

## Résultats produits

L'exécution du notebook génère :

- Les courbes LCC (plus grand composant connexe)
- Les courbes d'extinction (espèces secondaires supprimées)
- Les valeurs d'aire d'extinction (EA)
- Les graphiques comparant le temps d'exécution et la complexité

Les résultats et figures sont enregistrés dans :
```
tests/figure/
```

## Références

- Domínguez-García V., Muñoz M. A. (2015). Ranking species in mutualistic networks. Scientific Reports, 5, 8182.

- Kleinberg J. (1999). Authoritative sources in a hyperlinked environment. Journal of the ACM, 46(5), 604–632.

- Brin S., Page L. (1998). The anatomy of a large-scale hypertextual web search engine. Computer Networks, 30(1–7), 107–117.
