# Guide d'Installation et Setup - LLM Music Generation

Ce guide explique **étape par étape** comment cloner le projet et entraîner le modèle avec le dataset GrandMidiPiano.

---

## Prérequis

Avant de commencer, assurez-vous d'avoir :

- **Python 3.8+** installé ([Télécharger Python](https://www.python.org/downloads/))
- **Git** installé ([Télécharger Git](https://git-scm.com/downloads))
- **15 GB d'espace disque** disponible (dataset MIDI + modèles)
- **8 GB RAM minimum** (16 GB recommandé)
- **GPU (optionnel)** : Accélère l'entraînement (CUDA compatible)

---

## Étape 1 : Cloner le Projet

```bash
git clone https://github.com/FatineKa/Projet-MIDI-LLM-MusicGenerator.git
cd Projet-MIDI-LLM-MusicGenerator
```

---

## Étape 2 : Installer les Dépendances

```bash
pip install -r requirements.txt
```

Paquets installés :
- `miditoolkit` : Manipulation de fichiers MIDI
- `torch` : PyTorch pour le modèle Transformer
- `numpy` : Calculs numériques
- `matplotlib` : Visualisation des résultats
- `tqdm` : Barres de progression

---

## Étape 3 : Télécharger le Dataset GrandMidiPiano

### Option A : Téléchargement Manuel

1. **Télécharger** le dataset GrandMidiPiano depuis Kaggle :
   - Lien : [GrandMidiPiano on Kaggle](https://www.kaggle.com/datasets/soumikrakshit/classical-music-midi)
   - Ou rechercher "GrandMidiPiano dataset" sur Google

2. **Extraire** le fichier ZIP téléchargé

3. **Déplacer** le dossier dans votre projet :
   ```
   Projet-MIDI-LLM-MusicGenerator/
   └── GrandMidiPiano/
       └── GrandMidiPiano/
           ├── file1.mid
           ├── file2.mid
           └── ... (10,000+ fichiers MIDI)
   ```

### Option B : Téléchargement avec Kaggle API (optionnel)

Si vous avez un compte Kaggle :

```bash
# Installer Kaggle CLI
pip install kaggle

# Télécharger les identifiants depuis kaggle.com/account
# Placer kaggle.json dans ~/.kaggle/

# Télécharger le dataset
kaggle datasets download -d soumikrakshit/classical-music-midi
unzip classical-music-midi.zip -d GrandMidiPiano/
```

### Option C : Utiliser un Autre Dataset MIDI

Si vous n'utilisez pas GrandMidiPiano, vous pouvez utiliser **n'importe quel dataset MIDI** :
1. Créer un dossier (ex: `midi_files/`)
2. Y placer vos fichiers `.mid`
3. Modifier `batch_convert_midi_v2.py` ligne 170 pour pointer vers ce dossier

---

## Étape 4 : Convertir les MIDI en Tokens

Cette étape convertit tous les fichiers MIDI en un seul fichier de tokens.

```bash
python batch_convert_midi_v2.py
```

**Ce script va :**
- Parcourir tous les fichiers MIDI dans `GrandMidiPiano/GrandMidiPiano/`
- Convertir chaque MIDI en tokens (TIME_SHIFT, NOTE_ON, DURATION)
- Créer le fichier `all_midi_tokens_v2.txt` (plusieurs millions de tokens)

**Durée estimée :** 10-30 minutes selon votre machine et le nombre de fichiers.

**Sortie attendue :**
```
Conversion batch MIDI vers tokens (TIME_SHIFT)
Fichiers MIDI trouvés: 10,855

Conversion terminée!
Total de tokens: 15,234,567
Vocabulaire total: 2,345 tokens uniques
Fichier de sortie: all_midi_tokens_v2.txt
```

---

## Étape 5 : Préparer les Données d'Entraînement

```bash
python data_preparation.py
```

**Ce script va :**
- Analyser le fichier de tokens
- Créer le vocabulaire (tokenizer)
- Générer les séquences d'entraînement (fenêtres de 256 tokens)
- Sauvegarder dans `data/tokenizer.pkl` et `data/training_data.pkl`

**Durée estimée :** 5-10 minutes.

---

## Étape 6 : Entraîner le Modèle

```bash
python train.py
```

**L'entraînement va :**
- Charger les données préparées
- Entraîner le modèle Transformer pendant 50 epochs
- Sauvegarder le meilleur modèle dans `models/best_model.pt`
- Créer un graphique de la loss dans `output/training_loss.png`

**Durée estimée :**
- **Avec GPU** : 2-4 heures
- **Sans GPU (CPU)** : 10-20 heures

**Sortie attendue :**
```
Device: cuda (ou cpu)
Architecture du modele:
  Vocabulaire: 2,349 tokens
  Parametres: 1,234,567

Epoch 1/50
Train Loss: 4.2345 | Val Loss: 4.1234
Meilleur modèle sauvegardé!
...
```

---

## Étape 7 : Générer de la Musique

Une fois l'entraînement terminé, générez de nouvelles compositions :

```bash
python generate.py 1000 0.8 50
```

**Paramètres :**
- `1000` : Nombre de tokens à générer
- `0.8` : Temperature (créativité, 0.5-1.0)
- `50` : Top-K sampling (diversité)

**Sortie :**
- `output/generated_tokens.txt` : Tokens générés
- `output/generated_music.mid` : Fichier MIDI jouable

---

## Structure des Dossiers Finale

```
Projet-MIDI-LLM-MusicGenerator/
├── GrandMidiPiano/          # Dataset MIDI (téléchargé par vous)
│   └── GrandMidiPiano/
│       └── *.mid
├── data/                    # Données préparées (générées)
│   ├── tokenizer.pkl
│   └── training_data.pkl
├── models/                  # Modèles entraînés (générés)
│   ├── best_model.pt
│   └── checkpoint_epoch_*.pt
├── output/                  # Résultats (générés)
│   ├── training_loss.png
│   └── generated_music.mid
├── all_midi_tokens_v2.txt   # Tokens combinés (généré)
└── [fichiers Python]
```

---

## Résolution de Problèmes

### Erreur : "No such file or directory: GrandMidiPiano"

**Solution :** Vérifiez que le dataset est bien placé dans le bon dossier.

```bash
# Vérifier la structure
ls GrandMidiPiano/GrandMidiPiano/
# Devrait afficher des fichiers .mid
```

### Erreur : "No module named 'miditoolkit'"

**Solution :**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Mémoire GPU insuffisante

**Solution :** Réduire `BATCH_SIZE` dans `config.py` :
```python
BATCH_SIZE = 16  # Au lieu de 32
```

### Le modèle ne s'améliore pas

**Solution :**
- Vérifier que `data/training_data.pkl` existe
- Augmenter `EPOCHS` dans `config.py` (ex: 100 au lieu de 50)
- Vérifier que vous avez assez de données (au minimum 100,000 tokens)

### Conversion batch trop lente

**Solution :** Limiter le nombre de fichiers pour tester :
```python
# Dans batch_convert_midi_v2.py, ligne 172
MAX_FILES = 100  # Au lieu de None
```

---

## Configuration Système Recommandée

### Minimum
- Python 3.8+
- 8 GB RAM
- 15 GB espace disque

### Recommandé
- Python 3.9+
- 16 GB RAM
- GPU NVIDIA avec CUDA (GTX 1060 ou supérieur)
- 20 GB espace disque

---

## Prochaines Étapes

Après avoir suivi ce guide, vous pouvez :

1. **Expérimenter** avec les hyperparamètres dans `config.py`
2. **Générer** différentes compositions en ajustant la temperature
3. **Entraîner plus longtemps** en augmentant le nombre d'epochs
4. **Utiliser votre propre dataset** MIDI

---

## Aide et Support

Si vous rencontrez des problèmes :
1. Consultez la section "Résolution de Problèmes" ci-dessus
2. Vérifiez que vous avez suivi **toutes les étapes dans l'ordre**
3. Assurez-vous que tous les fichiers requis sont présents

Bon entraînement ! 🎵
