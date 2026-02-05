"""
Génération de musique avec le modèle entraîné

Ce script utilise le modèle entraîné pour générer de nouvelles mélodies.

Comment ça marche:
1. Charger le modèle entraîné et le tokenizer
2. Fournir quelques notes de départ (seed)
3. Le modèle prédit le prochain token
4. Ajouter ce token à la séquence
5. Répéter jusqu'à avoir la longueur désirée
6. Convertir les tokens en fichier MIDI

Génération auto-régressive:
Le modèle génère un token à la fois, comme écrire une phrase.

Paramètres de créativité:
- Temperature (0.0-2.0):
  0.0 = déterministe (prévisible)
  1.0 = équilibré (recommandé)
  >1.0 = créatif (surprenant, parfois chaotique)

- Top-k (1-100):
  Ne considère que les k tokens les plus probables
  50 = bon compromis entre qualité et diversité

Fichiers créés:
- output/generated_tokens.txt: Tokens générés
- output/generated_music.mid: Musique MIDI jouable

Usage:
  python generate.py <length> [temperature] [top_k]
"""

import torch
import pickle
import os
import sys

import config
from model import MusicLLM
from data_preparation import MusicTokenizer
from tokens2midi_v2 import tokens_to_midi_v2


def load_model(checkpoint_path, device='cpu'):
    """Charge le modèle entraîné depuis un checkpoint."""
    print(f"Chargement du modele depuis {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint['vocab_size']
    
    model = MusicLLM(
        vocab_size=vocab_size,
        d_model=config.EMBEDDING_DIM,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Modele charge (epoch {checkpoint['epoch'] + 1}, val_loss={checkpoint['val_loss']:.4f})")
    
    return model


def load_tokenizer(tokenizer_path):
    """Charge le tokenizer (vocabulaire)."""
    tokenizer = MusicTokenizer()
    tokenizer.load(tokenizer_path)
    return tokenizer


def generate_music(model, tokenizer, start_tokens, length, temperature=1.0, top_k=50, device='cpu'):
    """Génère de la musique avec le modèle."""
    print(f"\nGeneration:")
    print(f"  Tokens a generer: {length}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    
    # Encoder les tokens de départ
    start_ids = tokenizer.encode(start_tokens)
    
    print(f"\nTokens de depart: {len(start_ids)}")
    for i, token in enumerate(start_tokens[:min(10, len(start_tokens))]):
        print(f"  {token}")
    if len(start_tokens) > 10:
        print(f"  ...")
    
    # Générer
    print(f"\nGeneration en cours...")
    generated_ids = model.generate(
        start_ids,
        max_length=length,
        temperature=temperature,
        top_k=top_k,
        device=device
    )
    
    # Décoder
    generated_tokens = tokenizer.decode(generated_ids)
    
    print(f"Generation terminee! Total: {len(generated_tokens)} tokens")
    
    return generated_tokens


def save_tokens(tokens, output_path):
    """Sauvegarde les tokens dans un fichier."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')
    print(f"Tokens sauvegardes dans {output_path}")


def main():
    """Fonction principale."""
    
    if len(sys.argv) < 2:
        print("Generation de musique avec le modele LLM")
        print("\nUsage: python generate.py <length> [temperature] [top_k]")
        print("\nExemples:")
        print("  python generate.py 1000")
        print("  python generate.py 2000 0.8 50")
        print("  python generate.py 500 1.2 30")
        sys.exit(1)
    
    length = int(sys.argv[1])
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else config.TEMPERATURE
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else config.TOP_K
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Charger le tokenizer
    tokenizer_path = os.path.join(config.DATA_DIR, "tokenizer.pkl")
    if not os.path.exists(tokenizer_path):
        print(f"Erreur: Tokenizer introuvable: {tokenizer_path}")
        print("Veuillez d'abord executer: python data_preparation.py")
        sys.exit(1)
    
    tokenizer = load_tokenizer(tokenizer_path)
    print(f"Tokenizer charge (vocabulaire: {tokenizer.vocab_size} tokens)\n")
    
    # Charger le modèle
    model_path = os.path.join(config.MODELS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        print(f"Erreur: Modele introuvable: {model_path}")
        print("Veuillez d'abord entrainer le modele avec: python train.py")
        sys.exit(1)
    
    model = load_model(model_path, device)
    
    # Tokens de départ (seed)
    # Vous pouvez personnaliser cette séquence
    start_tokens = [
        "TIME_SHIFT_0",
        "NOTE_ON_60",     # Do
        "DURATION_480",
        "TIME_SHIFT_480",
        "NOTE_ON_62",     # Ré
        "DURATION_480",
        "TIME_SHIFT_480",
        "NOTE_ON_64",     # Mi
        "DURATION_480"
    ]
    
    print("\nTokens de depart (seed):")
    for token in start_tokens:
        print(f"  {token}")
    
    # Générer
    generated_tokens = generate_music(
        model, tokenizer, start_tokens,
        length, temperature, top_k, device
    )
    
    # Sauvegarder les tokens
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tokens_path = os.path.join(config.OUTPUT_DIR, "generated_tokens.txt")
    save_tokens(generated_tokens, tokens_path)
    
    # Convertir en MIDI
    print(f"\nConversion en MIDI...")
    midi_path = os.path.join(config.OUTPUT_DIR, "generated_music.mid")
    
    try:
        tokens_to_midi_v2(
            tokens_path,
            midi_path,
            tpq=config.TPQ,
            tempo=int(60_000_000 / config.TEMPO)
        )
    except Exception as e:
        print(f"\nErreur lors de la conversion MIDI: {e}")
        sys.exit(1)
    
    print(f"\nMusique generee avec succes!")
    print(f"  Tokens: {tokens_path}")
    print(f"  MIDI: {midi_path}")
    print(f"\nVous pouvez maintenant ecouter {os.path.basename(midi_path)}")
    print(f"\nConseils:")
    print(f"  - Musique trop previsible → augmentez temperature (ex: 1.2)")
    print(f"  - Musique chaotique → diminuez temperature (ex: 0.7)")


if __name__ == "__main__":
    main()
