# Tableau de bord - Vélos libre-service à Rouen

Ce projet est un tableau de bord interactif développé avec Dash et Plotly pour analyser l'utilisation des vélos en libre-service de la Métropole Rouen Normandie.

## Réalisé par
- Bafdel Moufdi Zakaria
- Aguida Yahya
- Mejdoub Yanis

## 📊 Fonctionnalités
- Visualisation interactive de la disponibilité des vélos.
- Analyse temporelle (par heure, jour, période...).
- Carte interactive des stations.
- Clustering des stations selon l’utilisation.
- Aide à la décision entre stations.
- Analyse de précision des méthodes d'interpolation (MAE, RMSE).


## 🛠️ Technologies
- Python, Dash, Plotly
- Pandas, NumPy, Scikit-learn
- Folium (pour la carte interactive)

## 📁 Données
Les données utilisées sont issues du service open-data de la Métropole Rouen (API GBFS). Le fichier `data/data.1.csv` contient les observations locales (avec interpolation si nécessaire).

## ▶️ Lancement
```
pip install -r requirements.txt
python app.py
```

## 🚀 Déploiement
Le projet peut être déployé sur Render ou Vercel (via fonctions serverless ou gunicorn). Configuration incluse dans la documentation.

## 📄 Licence
Ce projet est open-source, proposé sous licence MIT.

