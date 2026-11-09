# Trade Analysis

🇬🇧 [Read in English](README.md)

Plateforme interactive d'analyse commerciale et économique construite avec Streamlit, alimentée par un pipeline de données en architecture médaillon (Bronze → Silver → Gold) qui fonctionne — volontairement — sur **trois backends de base de données interchangeables** : SQL Server, DuckDB et Databricks.

Sources de données : **UN Comtrade** (flux commerciaux bilatéraux) · **API World Bank** (indicateurs macroéconomiques)

> Ce projet a été construit comme un portfolio de compétences : plusieurs parties du pipeline sont volontairement implémentées plus d'une fois (pandas vs. SQL pur, trois backends de base de données) pour démontrer qu'un même problème peut être résolu de plusieurs façons, plutôt que pour éviter la duplication.

---

## Application en ligne

🔗 _Lien de l'app déployée — à ajouter une fois en ligne sur Streamlit Community Cloud._

La page **Architecture** inclut une démo en direct qui exécute exactement la même requête SQL simultanément sur DuckDB, Databricks et SQL Server, et affiche les résultats côte à côte pour prouver que le pipeline produit des données identiques quel que soit le backend.

---

## Défis techniques réalisés

Quelques problèmes que ce projet a dû résoudre, au-delà de « charger un CSV dans un dashboard » :

- **Extraction API concurrente.** Les données UN Comtrade (~200 déclarants × plusieurs années × 2 flux) sont récupérées via `asyncio` et un sémaphore borné (`--workers` configurable, 4 par défaut) : des dizaines d'appels s'exécutent en parallèle sans bloquer la boucle d'événements, avec un backoff automatique en cas de rate-limiting HTTP 429.
- **Téléchargements reprenables, avec checkpoints.** Chaque paire `(déclarant, flux)` (données commerciales) et chaque indicateur (World Bank) écrit un marqueur dans `.checkpoints/` dès qu'il réussit. Relancer une extraction interrompue ignore tout ce qui est déjà fait au lieu de tout retélécharger ; les échecs sont journalisés dans `failed_reporters.json` pour des relances ciblées.
- **HTTP résilient aux erreurs transitoires.** L'extracteur World Bank utilise une session `requests` équipée de l'adaptateur `Retry` de `urllib3` (backoff exponentiel sur les erreurs 500/502/503/504) plutôt qu'un simple `requests.get`.
- **Un seul DataFrame, trois bases de données.** Une unique abstraction `save_to_db()` (`src/db/loader.py`) écrit le même DataFrame vers SQL Server, DuckDB ou Databricks via une interface de connecteur SQLAlchemy commune, avec un réglage spécifique par backend (DuckDB et Databricks utilisent des insertions multi-lignes avec des plafonds différents ; SQL Server garde volontairement `fast_executemany`, les deux stratégies de batching ne se combinant pas).
- **Deux implémentations ETL, même résultat.** Le pipeline médaillon (bronze → silver → gold) est implémenté une fois en pandas (`src/bronze|silver|gold/`) et une fois en SQL pur (`src/sql/`, natif DuckDB, `CREATE TABLE AS SELECT`). Les deux sont validés pour produire des tables gold identiques (nombre de lignes, intégrité référentielle, vérification de divergence numérique).
- **Deux implémentations d'analyse, mêmes questions.** L'analyse commerce/indicateurs existe à la fois en notebooks pandas (`Analysis/notebooks/`) et en notebooks SQL pur via `jupysql` (`Analysis/notebooks_sql/`, `src/sql/analysis/`) — mêmes questions, mêmes réponses, outil différent.
- **Analyse croisée (« Link »).** L'analyse des flux commerciaux et l'analyse des indicateurs macro sont deux notebooks distincts à l'origine — elles sont combinées dans une troisième couche d'analyse (`Analysis/Analysis_files/Link/`, pages *Link Signals* / *Link Strategy*) qui cherche quels indicateurs World Bank expliquent réellement le comportement commercial (concentration, volatilité, dépendance bilatérale) d'un pays donné.
- **Vérifications de cohérence inter-backend.** Comme les mêmes tables gold existent sur trois moteurs, les comparer numériquement ne peut pas se faire par simple arrondi — l'ordre de sommation en virgule flottante diverge entre moteurs à l'échelle de milliers de milliards de dollars, donc les comparaisons utilisent une tolérance relative (`numpy.allclose`) plutôt qu'un arrondi décimal.

---

## Pages de l'application

| Page | Description |
|---|---|
| Pipeline | Lancer les pipelines bronze / silver / gold depuis l'interface |
| Architecture | Vue d'ensemble de l'architecture ETL + démo en direct multi-backend |
| Trade Overview | Résumé et tendances du commerce mondial |
| Global Trade Dynamic | Évolution des flux commerciaux dans le temps |
| Country Trade Profile | Exports, imports et balance commerciale par pays |
| Cross-Country Comparison | Comparaison de pays côte à côte |
| Bilateral Risk Analysis | Score de risque entre paires de pays |
| Typology Scoring | Classification des pays par profil commercial |
| Indicators Overview | Tableau de bord des indicateurs World Bank |
| Indicator Deep Dive | Analyse approfondie d'un indicateur |
| Country Indicators Profile | Profil complet d'indicateurs par pays |
| Advanced Analysis | Prévisions et modèles statistiques |
| Link Signals | Quels indicateurs expliquent le comportement commercial |
| Link Strategy | Analyse stratégique des liens entre pays |

---

## Backends

| | DuckDB | Databricks | SQL Server |
|---|---|---|---|
| Serveur requis | Non (embarqué) | Oui (warehouse cloud) | Oui |
| `.env` requis | Non | Oui (token) | Oui (auth Windows) |
| Fonctionne sur Streamlit Cloud | Oui | Oui | Non — réseau local uniquement |

DuckDB est le backend par défaut pour le développement local et pour le déploiement public sur Streamlit Cloud. Databricks et SQL Server sont pleinement intégrés et validés, et affichés en direct sur la page Architecture quand ils sont accessibles.

---

## Structure du projet

```
Trade_Analysis/
├── app/
│   ├── Home.py                # Point d'entrée Streamlit
│   └── pages/                 # Pages d'analyse
├── src/
│   ├── bronze/ silver/ gold/  # ETL en pandas (première implémentation)
│   ├── sql/                   # ETL en SQL pur, natif DuckDB (deuxième implémentation)
│   ├── pipelines/              # run_bronze / run_silver / run_gold
│   ├── db/                     # fabrique de connecteurs + loader multi-backend
│   ├── bootstrap/              # Création des schémas
│   └── config/                 # Paramètres, config DB, logging
├── Analysis/
│   ├── notebooks/              # Analyse pandas (Trade, Indicators, Link)
│   └── notebooks_sql/          # Analyse SQL pur (jupysql)
├── data/
│   ├── bronze/ silver/         # CSV brut / nettoyé (ignoré par git, régénérable)
│   ├── gold/                   # CSV du schéma en étoile (suivi)
│   └── resources/              # Fichiers de référence statiques
├── setup.bat / setup.sh        # Installation initiale
└── Makefile                    # Workflow de développement
```

---

## Démarrage rapide

**Windows**
```bat
.\setup.bat
```

**Linux / Mac**
```bash
chmod +x setup.sh && ./setup.sh
```

Le script d'installation gère tout : environnement virtuel, dépendances, schémas de base de données, et propose de lancer le pipeline ou l'application immédiatement.

---

## Installation manuelle

```bash
python -m venv myenv
source myenv/bin/activate          # Linux/Mac
# myenv\Scripts\activate           # Windows

pip install -r requirements.txt

python -c "from src.config.settings import ensure_directories; ensure_directories()"
python -m src.bootstrap.create_schemas --db duckdb   # à ignorer en mode CSV seul

python -m src.pipelines.run_bronze --db duckdb
python -m src.bronze.extract_trade_data --consolidate --db duckdb
python -m src.pipelines.run_silver --db duckdb
python -m src.pipelines.run_gold   --db duckdb

streamlit run app/Home.py
```

---

## Makefile

```bash
make install              # Créer le venv + installer les dépendances
make pipeline DB=duckdb   # Pipeline complet (bronze → silver → gold)
make pipeline             # Idem, mode CSV seul
make bronze   DB=duckdb   # Bronze seul
make silver   DB=duckdb   # Silver seul
make gold     DB=duckdb   # Gold seul
make app                  # Lancer l'application Streamlit
```

---

## Extraction des données commerciales

Les données commerciales sont extraites par lots séparés (volume : ~200 déclarants × plusieurs années), en concurrence, avec reprise par checkpoint :

```bash
# 1. Extraire par année (répéter pour chaque année nécessaire) — concurrence bornée + checkpoints
python -m src.bronze.extract_trade_data --year 2022 --product TOTAL --flows M X --workers 6

# 2. Consolider tous les lots → crée la table bronze trade_data
python -m src.bronze.extract_trade_data --consolidate --db duckdb

# 3. Relancer silver + gold pour intégrer trade_data
python -m src.pipelines.run_silver --module trade_data --db duckdb
python -m src.pipelines.run_gold   --db duckdb
```

Interrompre puis relancer l'étape 1 ignore chaque paire `(déclarant, flux)` déjà marquée comme terminée — seules les paires manquantes sont récupérées.

---

## Déploiement sur Streamlit Cloud

1. Pousser le repo sur GitHub (les CSV de la couche gold et le fichier DuckDB sont suivis ; le bronze/silver brut et le fichier DuckDB dupliqué en SQL pur sont ignorés par git et régénérables localement si besoin)
2. Aller sur [share.streamlit.io](https://share.streamlit.io) → New app → fichier principal `app/Home.py`
3. Dans les **Secrets** de l'app, définir `DB_BACKEND = "duckdb"` pour que l'instance publique utilise le fichier embarqué (SQL Server est local uniquement par conception ; les identifiants Databricks peuvent aussi être ajoutés si tu veux que la démo multi-backend en direct puisse l'atteindre depuis le cloud)

---

## Variables d'environnement

Requises uniquement pour SQL Server ou Databricks. Créer un fichier `.env` à la racine du projet :

```
DB_BACKEND=sqlserver
DB_SERVER=...
DB_NAME=...

# ou, pour Databricks :
DB_BACKEND=databricks
DB_HOST=...
DB_HTTP_PATH=...
DB_TOKEN=...
DB_CATALOG=...
```
