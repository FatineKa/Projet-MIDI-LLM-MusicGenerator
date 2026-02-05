"""
Configuration du projet LLM Music Generation

Ce fichier centralise tous les hyperparamètres et configurations du projet.

Contexte du projet:
Ce projet implémente un modèle de langage (LLM) basé sur l'architecture Transformer
pour générer de la musique. Le modèle apprend les patterns musicaux en traitant 
les fichiers MIDI comme des séquences de tokens, puis génère de nouvelles mélodies.

Workflow:
1. MIDI vers tokens (midi2tokens_v2.py)
2. Préparation des données (data_preparation.py)
3. Entraînement du modèle (train.py)
4. Génération de musique (generate.py)
5. Tokens vers MIDI (tokens2midi_v2.py)
"""

# Chemins de fichiers
DATA_DIR = "data"           # Données préparées (tokenizer et séquences)
MODELS_DIR = "models"       # Modèles entraînés
OUTPUT_DIR = "output"       # Résultats de génération

# Paramètres de tokenization
SEQ_LENGTH = 256    # Longueur des séquences d'entrée
                    # Plus long = plus de contexte musical mais plus de mémoire GPU
                    # 256 tokens représente environ quelques mesures de musique

VOCAB_SIZE = None   # Calculé automatiquement lors de data_preparation.py
                    # Dépend de la complexité des fichiers MIDI (environ 1000-5000)

# Paramètres du modèle Transformer
EMBEDDING_DIM = 128     # Dimension des vecteurs d'embedding
                        # Chaque token est représenté par un vecteur de 128 dimensions

N_HEADS = 4             # Nombre de têtes d'attention
                        # Permet au modèle de regarder différents aspects simultanément
                        # EMBEDDING_DIM doit être divisible par N_HEADS

N_LAYERS = 4            # Nombre de couches Transformer empilées
                        # Plus de couches = modèle plus profond et expressif

HIDDEN_DIM = 512        # Dimension de la couche cachée du feed-forward network
                        # Généralement 2-4x la taille de EMBEDDING_DIM

DROPOUT = 0.1           # Taux de dropout pour la régularisation
                        # Désactive aléatoirement 10% des neurones pendant l'entraînement
                        # Évite le sur-apprentissage (overfitting)

# Paramètres d'entraînement
BATCH_SIZE = 32         # Nombre de séquences traitées simultanément
                        # Plus grand = plus rapide mais plus de mémoire GPU
                        # Réduire si erreur "out of memory"

LEARNING_RATE = 0.0001  # Taux d'apprentissage
                        # Contrôle la taille des mises à jour des poids
                        # Trop grand = instabilité, trop petit = trop lent

EPOCHS = 50             # Nombre d'époques d'entraînement
                        # Une époque = le modèle voit toutes les données une fois

WARMUP_STEPS = 1000     # Nombre de steps de warmup pour le learning rate
                        # Au début, on augmente progressivement le learning rate
                        # Améliore la stabilité d'entraînement

# Tokens spéciaux
PAD_TOKEN = "<PAD>"     # Padding (remplissage des séquences courtes)
START_TOKEN = "<START>" # Début de séquence (optionnel)
END_TOKEN = "<END>"     # Fin de séquence
UNK_TOKEN = "<UNK>"     # Token inconnu

# Paramètres MIDI
TPQ = 480       # Ticks per quarter note (résolution temporelle MIDI standard)
TEMPO = 120     # Tempo par défaut en BPM (beats per minute)

# Paramètres de génération
GENERATION_LENGTH = 1000    # Nombre de tokens à générer
TEMPERATURE = 1.0           # Température de sampling (créativité)
                            # 0.0 = déterministe, 1.0 = équilibré, >1.0 = créatif
TOP_K = 50                  # Top-k sampling (ne considère que les K tokens les plus probables)
