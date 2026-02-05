"""
Conversion tokens vers MIDI (version optimisée avec TIME_SHIFT)

Ce script fait l'inverse de midi2tokens_v2.py: il reconstruit un fichier MIDI
jouable à partir d'une séquence de tokens textuels.

Workflow complet:
1. MIDI → Tokens (midi2tokens_v2.py)
2. Entraînement du LLM (train.py)
3. Génération de nouveaux tokens (generate.py)
4. Tokens → MIDI (ce script)
5. Écouter la musique générée

Algorithme de reconstruction:
On parcourt les tokens séquentiellement:
1. Lire TIME_SHIFT_X → avancer de X ticks dans le temps
2. Lire NOTE_ON_Y → note de hauteur Y
3. Lire DURATION_Z → la note dure Z ticks
4. Créer la note MIDI: start=temps_courant, end=temps_courant+duration
5. Répéter pour tous les tokens

Exemple:
Input (tokens):
  TIME_SHIFT_0
  NOTE_ON_60
  DURATION_480
  TIME_SHIFT_120
  NOTE_ON_64
  DURATION_480

Output (notes MIDI):
  Note 1: pitch=60, start=0, end=480
  Note 2: pitch=64, start=120, end=600

Paramètres MIDI:
- TPQ (Ticks Per Quarter): résolution temporelle (défaut: 480)
- Tempo: vitesse de lecture (défaut: 120 BPM)

Usage:
  python tokens2midi_v2.py <tokens.txt> [output.mid] [tempo_bpm]
"""

import miditoolkit
import sys
import os


def tokens_to_midi_v2(tokens_file, output_midi, tpq=480, tempo=500000):
    """
    Convertit des tokens TIME_SHIFT en fichier MIDI jouable.
    
    Args:
        tokens_file: Fichier contenant les tokens (une ligne par token)
        output_midi: Nom du fichier MIDI de sortie
        tpq: Ticks per quarter note (résolution, défaut 480)
        tempo: Tempo en microsecondes par beat (défaut 500000 = 120 BPM)
    """
    
    print(f"Lecture des tokens depuis {tokens_file}...")
    with open(tokens_file, 'r', encoding='utf-8') as f:
        tokens = [line.strip() for line in f if line.strip()]
    
    print(f"Tokens lus: {len(tokens):,}")
    
    # Créer un nouveau fichier MIDI
    midi = miditoolkit.MidiFile(ticks_per_beat=tpq)
    midi.tempo_changes.append(miditoolkit.TempoChange(tempo=tempo, time=0))
    
    # Créer un instrument (Piano)
    instrument = miditoolkit.Instrument(
        program=0,       # 0 = Piano acoustique
        is_drum=False,
        name="Piano"
    )
    
    # Reconstruire les notes à partir des tokens
    current_time = 0
    note_count = 0
    i = 0
    
    while i < len(tokens):
        token = tokens[i]
        
        if token.startswith("TIME_SHIFT_"):
            # Extraire le nombre de ticks et avancer le temps
            time_shift = int(token.split("_")[2])
            current_time += time_shift
            i += 1
            
        elif token.startswith("NOTE_ON_"):
            # Une note est toujours suivie d'une DURATION
            if i + 1 < len(tokens):
                next_token = tokens[i + 1]
                
                if next_token.startswith("DURATION_"):
                    # Extraire pitch et duration
                    pitch = int(token.split("_")[2])
                    duration = int(next_token.split("_")[1])
                    
                    # Créer la note MIDI
                    note = miditoolkit.Note(
                        velocity=80,                    # Intensité moyenne
                        pitch=pitch,
                        start=current_time,
                        end=current_time + duration
                    )
                    
                    instrument.notes.append(note)
                    note_count += 1
                    i += 2  # Sauter NOTE_ON + DURATION
                else:
                    print(f"Attention: DURATION manquant apres {token}")
                    i += 1
            else:
                i += 1
        else:
            # Token inconnu, ignorer
            i += 1
    
    # Ajouter l'instrument au MIDI et sauvegarder
    midi.instruments.append(instrument)
    midi.dump(output_midi)
    
    print(f"\nFichier MIDI genere: {output_midi}")
    print(f"Notes creees: {note_count:,}")
    print(f"Duree totale: {current_time} ticks ({current_time/tpq:.1f} beats)")


def main():
    """Point d'entrée principal."""
    
    if len(sys.argv) < 2:
        print("Conversion tokens vers MIDI (version TIME_SHIFT)")
        print("\nUsage: python tokens2midi_v2.py <tokens.txt> [output.mid] [tempo_bpm]")
        print("\nExemples:")
        print("  python tokens2midi_v2.py generated_tokens.txt music.mid")
        print("  python tokens2midi_v2.py generated_tokens.txt music.mid 140")
        sys.exit(1)
    
    tokens_file = sys.argv[1]
    output_midi = sys.argv[2] if len(sys.argv) > 2 else "output_v2.mid"
    
    # Tempo: si fourni en BPM, convertir en microsecondes
    if len(sys.argv) > 3:
        bpm = int(sys.argv[3])
        tempo = int(60_000_000 / bpm)
    else:
        tempo = 500000  # 120 BPM par défaut
    
    if not os.path.exists(tokens_file):
        print(f"\nErreur: Le fichier {tokens_file} n'existe pas!")
        sys.exit(1)
    
    try:
        tokens_to_midi_v2(tokens_file, output_midi, tempo=tempo)
    except Exception as e:
        print(f"\nErreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
