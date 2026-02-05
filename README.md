# LLM Music Generation Project

Ce projet implémente un modèle de langage (LLM) basé sur l'architecture Transformer pour générer de la musique à partir de fichiers MIDI.

## Installation

```bash
pip install -r requirements.txt
```

## Structure du Projet

```
projet-midi/
├── config.py              # Configuration et hyperparamètres
├── midi2tokens.py         # Convertir MIDI → Tokens
├── tokens2midi.py         # Convertir Tokens → MIDI
├── data_preparation.py    # Préparation des données et tokenization
├── model.py              # Architecture du modèle Transformer
├── train.py              # Script d'entraînement
├── generate.py           # Script de génération de musique
├── requirements.txt      # Dépendances
└── data/                 # Données (créé automatiquement)
    ├── tokenizer.pkl
    └── training_data.pkl
```

## Utilisation

### 1. Convertir un fichier MIDI en tokens

```bash
python midi2tokens.py <fichier.mid> <output.txt>
```

Exemple:
```bash
python midi2tokens.py music.mid tokens.txt
```

### 2. Préparer les données pour l'entraînement

```bash
python data_preparation.py
```

Cela va :
- Analyser le fichier `exempleFichierToken.txt`
- Créer le vocabulaire
- Générer les séquences d'entraînement
- Sauvegarder dans `data/`

### 3. Entraîner le modèle

```bash
python train.py
```

Le modèle sera entraîné pendant 50 epochs. Les checkpoints seront sauvegardés dans `models/`.

### 4. Générer de la musique

```bash
python generate.py <length> [temperature] [top_k]
```

Exemple:
```bash
python generate.py 1000 0.8 50
```

Cela génèrera:
- `output/generated_tokens.txt` : Les tokens générés
- `output/generated_music.mid` : Le fichier MIDI jouable

### 5. Convertir des tokens en MIDI

```bash
python tokens2midi.py <tokens.txt> <output.mid> [tempo]
```

Exemple:
```bash
python tokens2midi.py tokens.txt music.mid 120
```

## Architecture du Modèle

Le modèle utilise l'architecture Transformer avec :

- **Embedding Layer** : Convertit les tokens en vecteurs
- **Positional Encoding** : Encodage sinusoïdal des positions
- **4 Transformer Blocks** avec :
  - Multi-head attention (4 têtes)
  - Feed-forward network
  - Layer normalization
  - Residual connections
- **Output Layer** : Prédit le prochain token

### Hyperparamètres (config.py)

- Séquence length : 256
- Embedding dim : 128
- Nombre de têtes : 4
- Nombre de couches : 4
- Batch size : 32
- Learning rate : 0.0001

## Format des Tokens

Chaque note est représentée par 3 tokens :

```
POSITION_<time>   # Position temporelle (en ticks)
NOTE_ON_<pitch>   # Hauteur de la note (0-127)
DURATION_<dur>    # Durée de la note (en ticks)
```

Exemple:
```
POSITION_0
NOTE_ON_60
DURATION_480
```

## Résultats

Après l'entraînement, vous trouverez :

- `models/best_model.pt` : Meilleur modèle
- `models/checkpoint_epoch_X.pt` : Checkpoints périodiques
- `output/training_loss.png` : Graphique des pertes
- `output/generated_music.mid` : Musique générée

## Notes

- Le modèle est entraîné sur un seul instrument (piano)
- La qualité de génération dépend de la quantité et diversité des données
- Vous pouvez ajuster les hyperparamètres dans `config.py`

## Auteur

Projet réalisé dans le cadre du cours sur les LLM pour la musique.
