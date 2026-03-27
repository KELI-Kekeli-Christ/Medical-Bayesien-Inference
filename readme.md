# 🩺 Diagnostic Médical Bayésien

Application Streamlit de diagnostic médical intelligent basée sur un moteur bayésien avec sélection optimale des questions par **Information Gain**.

---

## Aperçu

Cette application pose des questions ciblées sur les symptômes d'un patient et met à jour en temps réel les probabilités de chaque maladie grâce au **théorème de Bayes**. À chaque étape, elle choisit la question la plus informative possible pour converger vers un diagnostic le plus rapidement possible.

---

## Fonctionnalités

- **Sélection optimale des symptômes** : chaque question est choisie en maximisant l'Information Gain réel (et non une heuristique approximative par variance)
- **Mise à jour bayésienne** : les probabilités sont recalculées après chaque réponse avec normalisation
- **Option "Ne sais pas"** : permet d'ignorer un symptôme sans biaiser le modèle
- **Visualisation en temps réel** : graphique des probabilités et courbe de réduction d'entropie
- **Seuils configurables** : nombre de questions max et seuil de confiance ajustables dans la sidebar
- **Lissage de Laplace** : évite les probabilités nulles sur les maladies rares

---

## Installation

```bash
# Cloner le projet
git clone https://github.com/votre-repo/diagnostic-bayesien.git
cd diagnostic-bayesien

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py
```

### Dépendances

| Package | Version minimale | Rôle |
|---|---|---|
| streamlit | 1.32.0 | Interface web |
| pandas | 2.0.0 | Chargement et traitement du CSV |
| numpy | 1.24.0 | Calculs numériques |
| plotly | 5.18.0 | Graphiques interactifs |

---

## Dataset

L'application utilise le dataset **Disease Symptom Prediction** disponible sur Kaggle.

- **Lien** : https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
- **Fichier requis** : `Training.csv`
- **Format** : 132 colonnes de symptômes binaires (0/1) + 1 colonne `prognosis`
- **Contenu** : 41 maladies × ~120 symptômes

Une fois téléchargé, chargez le fichier directement via la sidebar de l'application.

---

## Fonctionnement du moteur bayésien

### 1. Apprentissage (`fit`)

Pour chaque maladie `m`, on calcule :

- **Prior** : `P(M=m) = nombre de lignes avec maladie m / total`
- **Vraisemblance** (avec lissage de Laplace) :
  `P(S=s | M=m) = (occurrences de s dans m + 1) / (total lignes m + 2)`

### 2. Sélection du meilleur symptôme (`get_best_symptom`)

À chaque tour, on choisit le symptôme `s*` qui maximise l'**Information Gain** :

```
IG(s) = H(actuelle) - [ P(s=1) × H(post | s=1) + P(s=0) × H(post | s=0) ]
```

Où `H` est l'entropie de Shannon :

```
H = - Σ P(m) × log₂(P(m))
```

C'est un calcul exact, contrairement à l'approximation par variance utilisée dans la version originale.

### 3. Mise à jour (`update_probs`)

Après chaque réponse, application du théorème de Bayes :

```
P(M=m | s=v) ∝ P(M=m) × P(s=v | M=m)
```

Suivi d'une normalisation pour que la somme des probabilités reste égale à 1.

### 4. Arrêt

Le diagnostic s'arrête si l'une de ces conditions est remplie :
- La maladie la plus probable dépasse le **seuil de confiance** (défaut : 95 %)
- Le **nombre maximum de questions** est atteint (défaut : 10)
- Plus aucun symptôme disponible

---

## Structure du projet

```
.
├── app.py              # Application Streamlit principale
├── requirements.txt    # Dépendances Python
├── README.md           # Ce fichier
└── data/
    └── Training.csv    # Dataset Kaggle (à télécharger séparément)
```

---

## Paramètres configurables

Tous les paramètres sont accessibles dans la **sidebar** de l'application sans modifier le code.

| Paramètre | Défaut | Description |
|---|---|---|
| Questions max | 10 | Nombre maximum de symptômes demandés |
| Seuil de confiance | 95 % | Probabilité minimale pour conclure |

---

## Différences avec la version originale

| Aspect | Version originale | Version améliorée |
|---|---|---|
| Sélection des symptômes | Variance des vraisemblances (approximation) | Information Gain exact (entropie réelle) |
| Interface | CLI (`input()`) | Streamlit avec graphiques interactifs |
| Gestion de l'incertitude | Binaire (o/n) | Trois options : Oui / Non / Ne sais pas |
| Reset | Redémarrage du script | Bouton dans la sidebar |
| Visualisation | Aucune | Probabilités en temps réel + courbe d'entropie |
| Robustesse | Division par zéro possible | Protection + lissage de Laplace |

---

Pour tester : https://medical-bayesien-inference.streamlit.app/


[![Watch the video](https://img.youtube.com/vi/rKSPhcrZsso/0.jpg)](https://youtu.be/rKSPhcrZsso)


## Avertissement

> ⚠️ **Cet outil est à but strictement éducatif.** Il ne constitue en aucun cas un avis médical. Ne pas utiliser pour établir un diagnostic réel. Consultez un professionnel de santé pour tout problème médical.