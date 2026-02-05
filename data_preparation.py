"""
Préparation des données pour l'entraînement du LLM

Ce script transforme les tokens musicaux (texte) en données numériques prêtes
pour l'entraînement du modèle Transformer.

Que fait ce script:
1. Tokenizer: Construit un vocabulaire qui associe chaque token unique à un ID numérique
   Exemple: "TIME_SHIFT_0" → 4, "NOTE_ON_60" → 127

2. Séquences d'entraînement: Découpe les tokens en fenêtres de taille fixe
   (256 tokens) pour l'entraînement

3. Sauvegarde: Écrit le tokenizer et les séquences sur disque

Fenêtre glissante (sliding window):
Pour créer les séquences, on utilise une fenêtre glissante:
  Tokens: [A, B, C, D, E, F, G, ...] 
  Séquences:
    Input: [A, B, C] → Target: [B, C, D]
    Input: [B, C, D] → Target: [C, D, E]
    ...

Le modèle apprend à prédire le prochain token à chaque position.

Fichiers créés:
- data/tokenizer.pkl: Vocabulaire (token ←→ ID)
- data/training_data.pkl: Séquences d'entraînement
"""

import os
from collections import Counter
import pickle
import config


class MusicTokenizer:
    """
    Convertit les tokens musicaux (texte) en IDs numériques et vice-versa.
    
    C'est comme un dictionnaire bilingue:
    - token → ID (encode)
    - ID → token (decode)
    """
    
    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.vocab_size = 0
        
        # Ajouter les tokens spéciaux en premier
        self.special_tokens = [
            config.PAD_TOKEN,      # Padding (remplissage)
            config.START_TOKEN,    # Début de séquence
            config.END_TOKEN,      # Fin de séquence
            config.UNK_TOKEN       # Token inconnu
        ]
        
        for token in self.special_tokens:
            self.add_token(token)
    
    def add_token(self, token):
        """Ajoute un token au vocabulaire s'il n'existe pas déjà."""
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token
            self.vocab_size = len(self.token2id)
    
    def build_vocab(self, tokens):
        """Construit le vocabulaire à partir d'une liste de tokens."""
        print(f"Construction du vocabulaire...")
        unique_tokens = set(tokens)
        
        for token in unique_tokens:
            self.add_token(token)
        
        print(f"  Taille du vocabulaire: {self.vocab_size}")
        print(f"  Tokens speciaux: {len(self.special_tokens)}")
        print(f"  Tokens musicaux: {self.vocab_size - len(self.special_tokens)}")
    
    def encode(self, tokens):
        """Convertit tokens → IDs."""
        return [self.token2id.get(token, self.token2id[config.UNK_TOKEN]) 
                for token in tokens]
    
    def decode(self, ids):
        """Convertit IDs → tokens."""
        return [self.id2token.get(id, config.UNK_TOKEN) for id in ids]
    
    def save(self, path):
        """Sauvegarde le tokenizer sur disque."""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2id': self.token2id,
                'id2token': self.id2token,
                'vocab_size': self.vocab_size
            }, f)
        print(f"Tokenizer sauvegarde dans {path}")
    
    def load(self, path):
        """Charge un tokenizer depuis le disque."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.token2id = data['token2id']
            self.id2token = data['id2token']
            self.vocab_size = data['vocab_size']
        print(f"Tokenizer charge depuis {path}")


def load_tokens_from_file(file_path):
    """Charge tous les tokens depuis un fichier texte (un token par ligne)."""
    print(f"Chargement des tokens depuis {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]
    print(f"  Total de tokens: {len(tokens):,}")
    return tokens


def create_sequences(tokens, seq_length):
    """
    Crée des séquences d'entraînement avec fenêtre glissante.
    
    Pour chaque position, on crée:
    - input_seq: tokens d'entrée
    - target_seq: tokens cibles (décalé de 1)
    
    Le modèle apprend à prédire le prochain token à chaque position.
    """
    print(f"Creation des sequences (longueur={seq_length})...")
    
    input_seqs = []
    target_seqs = []
    
    for i in range(0, len(tokens) - seq_length):
        input_seq = tokens[i:i + seq_length]
        target_seq = tokens[i + 1:i + seq_length + 1]
        
        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
    
    print(f"  Sequences creees: {len(input_seqs):,}")
    return input_seqs, target_seqs


def analyze_tokens(tokens):
    """Analyse le vocabulaire et affiche des statistiques."""
    print("\nAnalyse du vocabulaire:")
    
    # Compter les différents types de tokens
    position_tokens = [t for t in tokens if t.startswith("POSITION_") or t.startswith("TIME_SHIFT_")]
    note_tokens = [t for t in tokens if t.startswith("NOTE_ON_")]
    duration_tokens = [t for t in tokens if t.startswith("DURATION_")]
    
    print(f"  TIME_SHIFT/POSITION: {len(position_tokens)} ({len(position_tokens)/len(tokens)*100:.1f}%)")
    print(f"  NOTE_ON: {len(note_tokens)} ({len(note_tokens)/len(tokens)*100:.1f}%)")
    print(f"  DURATION: {len(duration_tokens)} ({len(duration_tokens)/len(tokens)*100:.1f}%)")
    
    # Tokens uniques
    unique_positions = len(set(position_tokens))
    unique_notes = len(set(note_tokens))
    unique_durations = len(set(duration_tokens))
    
    print(f"\n  TIME_SHIFT/POSITION uniques: {unique_positions}")
    print(f"  Notes uniques: {unique_notes}")
    print(f"  Durations uniques: {unique_durations}")
    
    # Top 5 notes
    note_counter = Counter(note_tokens)
    print(f"\n  Top 5 notes:")
    for note, count in note_counter.most_common(5):
        pitch = note.split("_")[2]
        print(f"    {note} (pitch={pitch}): {count}")


def main():
    """Prépare les données pour l'entraînement."""
    
    # Créer le dossier data/
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Charger les tokens
    tokens = load_tokens_from_file("exempleFichierToken.txt")
    
    # Analyser le vocabulaire
    analyze_tokens(tokens)
    
    # Créer le tokenizer
    tokenizer = MusicTokenizer()
    tokenizer.build_vocab(tokens)
    tokenizer.save(os.path.join(config.DATA_DIR, "tokenizer.pkl"))
    
    # Créer les séquences
    input_seqs, target_seqs = create_sequences(tokens, config.SEQ_LENGTH)
    
    # Encoder les séquences (tokens → IDs)
    print("\nEncodage des sequences...")
    input_ids = [tokenizer.encode(seq) for seq in input_seqs]
    target_ids = [tokenizer.encode(seq) for seq in target_seqs]
    
    # Sauvegarder
    data = {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'vocab_size': tokenizer.vocab_size
    }
    
    output_path = os.path.join(config.DATA_DIR, "training_data.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nDonnees d'entrainement sauvegardees dans {output_path}")
    print(f"  Sequences: {len(input_ids)}")
    print(f"  Taille du vocabulaire: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
