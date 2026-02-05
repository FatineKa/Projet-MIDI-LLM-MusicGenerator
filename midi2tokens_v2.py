"""
Conversion MIDI vers tokens (version optimisée avec TIME_SHIFT)

Ce script convertit un fichier MIDI en séquence de tokens textuels qui peuvent
être traités par un modèle de langage.

Pourquoi tokenizer?
Les modèles de langage (comme GPT) travaillent avec des tokens (mots), pas avec
des fichiers binaires MIDI. On doit donc convertir:
MIDI → Tokens → Entraînement LLM → Tokens générés → MIDI

Différence v1 vs v2:
- V1 (POSITION absolue): vocabulaire énorme (millions de positions possibles)
- V2 (TIME_SHIFT relatif): vocabulaire réduit (~300 TIME_SHIFT)
  Avantage: meilleur apprentissage des patterns rythmiques

Format des tokens:
Chaque note = 3 tokens consécutifs:
1. TIME_SHIFT_<delta> : Temps écoulé depuis la note précédente
2. NOTE_ON_<pitch> : Hauteur de la note (0-127, ex: 60=Do central)
3. DURATION_<ticks> : Durée de la note (ex: 480=noire)

Exemple:
  TIME_SHIFT_0      (début, temps 0)
  NOTE_ON_60        (Do)
  DURATION_480      (durée noire)
  TIME_SHIFT_480    (480 ticks plus tard)
  NOTE_ON_64        (Mi)
  DURATION_480      (durée noire)

Polyphonie (accords):
TIME_SHIFT_0 signifie "en même temps"
  TIME_SHIFT_0
  NOTE_ON_60        (Do)
  DURATION_480
  TIME_SHIFT_0      (simultané!)
  NOTE_ON_64        (Mi)
  DURATION_480

Usage:
  python midi2tokens_v2.py <fichier.mid> [output.txt]
"""

import miditoolkit
import sys


def midi_to_tokens_v2(midi_file_path, output_file):
    """
    Convertit un fichier MIDI en tokens avec TIME_SHIFT (delta temporel relatif).
    
    Étapes:
    1. Charger le fichier MIDI
    2. Extraire toutes les notes de tous les instruments
    3. Trier les notes par ordre chronologique
    4. Pour chaque note, calculer le TIME_SHIFT depuis la note précédente
    5. Générer les 3 tokens: TIME_SHIFT, NOTE_ON, DURATION
    6. Écrire les tokens dans un fichier texte
    """
    
    print(f"Lecture du fichier MIDI: {midi_file_path}...")
    midi = miditoolkit.MidiFile(midi_file_path)
    
    # Collecter toutes les notes
    all_notes = []
    
    for inst in midi.instruments:
        if inst.is_drum:
            continue  # Ignorer les percussions
        
        for note in inst.notes:
            all_notes.append({
                'start': note.start,
                'pitch': note.pitch,
                'duration': note.end - note.start
            })
    
    # Trier chronologiquement (crucial pour TIME_SHIFT)
    all_notes.sort(key=lambda x: x['start'])
    
    # Générer les tokens avec TIME_SHIFT
    tokens = []
    current_time = 0
    
    for note in all_notes:
        # Calculer le décalage temporel depuis la dernière note
        time_shift = note['start'] - current_time
        
        # Trois tokens par note
        tokens.append(f"TIME_SHIFT_{time_shift}")
        tokens.append(f"NOTE_ON_{note['pitch']}")
        tokens.append(f"DURATION_{note['duration']}")
        
        # Mettre à jour le temps courant
        current_time = note['start']
    
    # Écrire les tokens (un par ligne)
    print(f"Ecriture de {len(tokens)} tokens dans {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in tokens:
            f.write(token + '\n')
    
    # Statistiques
    note_count = len(all_notes)
    time_shifts = [t for t in tokens if t.startswith("TIME_SHIFT_")]
    unique_shifts = len(set(time_shifts))
    
    print(f"\nStatistiques:")
    print(f"  Notes converties: {note_count:,}")
    print(f"  Tokens generes: {len(tokens):,}")
    print(f"  TIME_SHIFT uniques: {unique_shifts}")
    print(f"\nConversion terminee!")


def main():
    """Point d'entrée principal du script."""
    
    if len(sys.argv) < 2:
        print("Conversion MIDI vers tokens (version TIME_SHIFT)")
        print("\nUsage: python midi2tokens_v2.py <fichier.mid> [output.txt]")
        print("\nExemples:")
        print("  python midi2tokens_v2.py bach.mid tokens.txt")
        print("  python midi2tokens_v2.py chopin.mid")
        sys.exit(1)
    
    midi_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "tokens_v2.txt"
    
    try:
        midi_to_tokens_v2(midi_file, output_file)
    except FileNotFoundError:
        print(f"\nErreur: Le fichier {midi_file} n'existe pas!")
        sys.exit(1)
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
