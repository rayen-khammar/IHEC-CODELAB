# IHEC-CODELAB 2.0 — Assistant Intelligent de Trading (BVMT)

## Contexte
La Bourse des Valeurs Mobilières de Tunis (BVMT) poursuit sa modernisation dans un contexte financier complexe et volatil. Les investisseurs tunisiens ont besoin d'outils intelligents, sécurisés et conformes au cadre réglementaire (CMF). Le marché présente des spécificités : liquidité variable, sources d'information multilingues (français/arabe), et besoin de surveillance des manipulations.

## Problématique
Concevoir un **Assistant Intelligent de Trading** intégré combinant :
- prévision des prix et de la liquidité,
- analyse de sentiment de marché,
- détection d'anomalies,
- agent de décision et gestion de portefeuille.

## Objectifs principaux
- Aider les investisseurs à prendre des décisions éclairées.
- Détecter des comportements suspects en quasi temps réel.
- Expliquer de manière transparente les recommandations.

## Fonctionnalités cœur
### A. Prévision des prix et de la liquidité (ML/Deep Learning)
- Prédire les prix à court terme (1 à 5 jours) pour les principales valeurs BVMT.
- Anticiper les périodes de faible/forte liquidité.
- Identifier les meilleurs moments d'entrée/sortie.

### B. Analyse de sentiment (NLP)
- Collecter et analyser des actualités financières tunisiennes.
- Classifier le sentiment (positif/négatif/neutre) par valeur.
- Corréler sentiment et mouvements de prix (optionnel).

### C. Détection d'anomalies (Surveillance de marché)
- Identifier les pics de volume et variations anormales.
- Générer des alertes pour protéger l'investisseur.
- Détecter des potentielles manipulations de marché.

### D. Agent de décision augmentée (IA + Interface)
- Recommandations : acheter / vendre / conserver.
- Simulation de portefeuille virtuel.
- Explication claire des recommandations.
- Suivi et optimisation d'un portefeuille multi‑actifs.

## Spécifications techniques (synthèse)
### Module 1 — Prévision
**Objectifs**
- Prix de clôture des 5 prochains jours ouvrables.
- Volume journalier et probabilité de liquidité élevée/faible.

**Livrables attendus**
- Modèle entraîné avec métriques (RMSE, MAE, Directional Accuracy).
- Visualisations prévision vs réel (avec intervalles de confiance).
- API/fonction Python de prévision.
- Pipeline temps réel (bonus).

### Module 2 — Sentiment
- Scraping de 3+ sources d'actualités tunisiennes.
- Score de sentiment quotidien agrégé par entreprise.
- Gestion du multilinguisme (français + arabe).

### Module 3 — Anomalies
- Détection :
  - pics de volume (> 3σ),
  - variations anormales (> 5 % en 1h sans news),
  - patterns d'ordres suspects.
- Livrables :
  - métriques Precision/Recall/F1,
  - interface d'alertes et visualisations.

### Module 4 — Décision & Portefeuille
- Profil utilisateur : conservateur / modéré / agressif.
- Simulation portefeuille (capital virtuel, ROI, Sharpe, Max Drawdown).
- Explainability obligatoire.
- RL (optionnel) ou logique rule‑based (minimum viable).

## Interface utilisateur (Dashboard)
**Pages obligatoires**
1. **Vue d'ensemble du marché**
   - Indices (TUNINDEX), top gagnants/perdants, sentiment global, alertes.
2. **Analyse d'une valeur spécifique**
   - Historique + prévisions 5 jours, sentiment, RSI/MACD, recommandation.
3. **Mon portefeuille**
   - Positions, répartition, performance globale, suggestions.
4. **Surveillance & alertes**
   - Flux temps réel des anomalies, filtres, historique.

## Livrables finaux (hackathon)
### Livrables techniques
- Code source complet (GitHub/GitLab ou ZIP).
- README avec instructions d'installation.
- Requirements.txt (Python) ou package.json (Node).
- Application fonctionnelle (locale ou hébergée) avec URL et identifiants.
- Documentation technique : architecture, choix modèles, métriques, limites.
- Notebooks Jupyter (recommandé) : EDA, entraînement, visualisations.

### Livrables de présentation
- Pitch Deck (10–15 min).
- Vidéo démo (3–5 min).
- Parcours utilisateur complet.
- Cas d'usage : "Je veux investir 5000 TND, que recommandez‑vous ?"

## Scénarios d'usage (User Stories)
### 1) Investisseur débutant
Ahmed (28 ans) obtient un profil "modéré", reçoit un portefeuille diversifié et des explications détaillées pour chaque recommandation.

### 2) Trader averti
Leila (35 ans) reçoit des alertes d'anomalie (volume anormal), vérifie les news, et ajuste sa stratégie selon le sentiment et la prévision de volatilité.

### 3) Régulateur (CMF)
Un inspecteur reçoit une alerte sur une variation suspecte et déclenche une enquête en s'appuyant sur la timeline d'ordres et d'anomalies.

---

**Projet :** Assistant Intelligent de Trading pour la BVMT — IHEC‑CODELAB 2.0
