#!/bin/bash
echo "=== [0/4] Vérification de l'environnement Python ==="
if ! command -v python3 &> /dev/null
then
    echo "❌ Python3 n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

echo "=== [1/4] Création ou activation de l'environnement virtuel ==="
if [ ! -d ".venv" ]; then
    echo "Aucun environnement virtuel détecté. Création de l'environnement .venv..."
    python3 -m venv .venv
fi
source .venv/bin/activate
echo "✅ Environnement virtuel activé."

echo "=== [2/4] Installation des dépendances ==="
if [ -f "requirements.txt" ]; then
    echo "Fichier requirements.txt détecté. Installation automatique des dépendances..."
    pip install -r requirements.txt
else
    echo "Aucun fichier requirements.txt trouvé. Installation manuelle des bibliothèques de base..."
    pip install numpy pandas matplotlib networkx scipy jupyterlab
fi

echo "=== [3/4] Exécution du notebook et exportation en HTML ==="
time jupyter nbconvert --to html --execute src/robustesse_bipartite.ipynb \
    --output results.html \
    --ExecutePreprocessor.timeout=600 \
    --log-level=INFO

echo "=== [4/4] Expériences terminées ! Le résultat HTML est disponible dans src/results.html ==="

# (Optionnel) Ouvrir automatiquement le fichier HTML dans le navigateur
if command -v open &> /dev/null; then
    open src/results.html
elif command -v xdg-open &> /dev/null; then
    xdg-open src/results.html
fi
