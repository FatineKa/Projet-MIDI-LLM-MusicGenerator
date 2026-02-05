"""
Conversion batch: Tous les MIDI vers tokens

Ce script convertit automatiquement tous les fichiers MIDI d'un dossier en tokens.
Utile pour créer un grand dataset d'entraînement.

Pourquoi ce script:
- midi2tokens_v2.py: convertit un fichier à la fois
- batch_convert_midi_v2.py: convertit des milliers de fichiers automatiquement

Utilisation:
1. Télécharger un dataset MIDI (ex: GrandMidiPiano)
2. Lancer ce script
3. Utiliser all_midi_tokens_v2.txt dans data_preparation.py
4. Entraîner le modèle avec beaucoup plus de données

Format du fichier de sortie:
Un fichier texte avec tous les tokens de tous les MIDI combinés.

Configuration:
Modifiez les variables dans main():
- midi_directory: dossier contenant les MIDI
- output_file: nom du fichier de sortie
- MAX_FILES: None pour tous, ou un nombre pour limiter
"""

import os
import miditoolkit
from pathlib import Path
from tqdm import tqdm


def midi_to_tokens_v2(midi_file_path):
    """
    Convertit un fichier MIDI en tokens (version batch optimisée).
    
    Retourne une liste vide si le fichier est corrompu.
    Le script principal continue avec les autres fichiers.
    """
    try:
        midi = miditoolkit.MidiFile(midi_file_path)
        
        # Collecter toutes les notes
        all_notes = []
        
        for inst in midi.instruments:
            if inst.is_drum:
                continue
            
            for note in inst.notes:
                all_notes.append({
                    'start': note.start,
                    'pitch': note.pitch,
                    'duration': note.end - note.start
                })
        
        # Trier chronologiquement
        all_notes.sort(key=lambda x: x['start'])
        
        # Générer les tokens
       tokens = []
        current_time = 0
        
        for note in all_notes:
            time_shift = note['start'] - current_time
            tokens.append(f"TIME_SHIFT_{time_shift}")
            tokens.append(f"NOTE_ON_{note['pitch']}")
            tokens.append(f"DURATION_{note['duration']}")
            current_time = note['start']
        
        return tokens
        
    except Exception as e:
        return []


def find_all_midi_files(directory):
    """Trouve récursivement tous les fichiers MIDI dans un dossier."""
    midi_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mid') or file.endswith('.midi'):
                full_path = os.path.join(root, file)
                midi_files.append(full_path)
    
    return midi_files


def process_all_midi_files(input_dir, output_file, max_files=None):
    """Traite tous les fichiers MIDI d'un dossier en batch."""
    
    print(f"Conversion batch MIDI vers tokens (TIME_SHIFT)")
    print(f"Dossier source: {input_dir}")
    print(f"Fichier de sortie: {output_file}\n")
    
    # Trouver tous les fichiers MIDI
    print("Recherche des fichiers MIDI...")
    midi_files = find_all_midi_files(input_dir)
    
    if max_files:
        midi_files = midi_files[:max_files]
        print(f"  Limite appliquee: {max_files} fichiers")
    
    print(f"  Fichiers MIDI trouves: {len(midi_files):,}\n")
    
    if len(midi_files) == 0:
        print("Erreur: Aucun fichier MIDI trouve!")
        print(f"Verifiez que {input_dir} contient des fichiers .mid ou .midi")
        return
    
    # Convertir tous les fichiers
    all_tokens = []
    successful_files = 0
    failed_files = 0
    
    print("Conversion en cours...\n")
    
    for midi_file in tqdm(midi_files, desc="Progression", unit="fichier"):
        tokens = midi_to_tokens_v2(midi_file)
        
        if tokens:
            all_tokens.extend(tokens)
            successful_files += 1
        else:
            failed_files += 1
    
    # Sauvegarder
    print(f"\nSauvegarde des tokens...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for token in all_tokens:
            f.write(token + '\n')
    
    # Statistiques
    print(f"\nStatistiques:")
    print(f"  Fichiers traites: {successful_files:,}")
    print(f"  Fichiers en erreur: {failed_files:,}")
    print(f"  Taux de reussite: {successful_files / len(midi_files) * 100:.1f}%")
    print(f"\n  Total de tokens: {len(all_tokens):,}")
    print(f"  Tokens par fichier (moyenne): {len(all_tokens) // successful_files:,}")
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n  Fichier de sortie: {output_file}")
    print(f"  Taille: {file_size_mb:.2f} MB")
    
    # Analyse du vocabulaire
    time_shifts = [t for t in all_tokens if t.startswith("TIME_SHIFT_")]
    notes = [t for t in all_tokens if t.startswith("NOTE_ON_")]
    durations = [t for t in all_tokens if t.startswith("DURATION_")]
    
    unique_shifts = len(set(time_shifts))
    unique_notes = len(set(notes))
    unique_durations = len(set(durations))
    
    print(f"\nAnalyse du vocabulaire:")
    print(f"  TIME_SHIFT: {len(time_shifts):,} tokens, {unique_shifts:,} uniques")
    print(f"  NOTE_ON: {len(notes):,} tokens, {unique_notes:,} uniques")
    print(f"  DURATION: {len(durations):,} tokens, {unique_durations:,} uniques")
    
    total_unique = unique_shifts + unique_notes + unique_durations
    print(f"\n  Vocabulaire total: {total_unique:,} tokens uniques")
    
    print(f"\nConversion terminee!")


def main():
    """Point d'entrée principal."""
    
    # Configuration
    midi_directory = "GrandMidiPiano/GrandMidiPiano"
    output_file = "all_midi_tokens_v2.txt"
    MAX_FILES = None  # None = tous, ou un nombre pour limiter
    
    # Vérifier que le dossier existe
    if not os.path.exists(midi_directory):
        print(f"Erreur: Le dossier {midi_directory} n'existe pas!")
        print("\nSuggestions:")
        print("  1. Verifiez le chemin dans la variable midi_directory")
        print("  2. Assurez-vous d'avoir telecharge le dataset MIDI")
        return
    
    # Lancer la conversion
    process_all_midi_files(midi_directory, output_file, MAX_FILES)
    
    # Prochaines étapes
    print(f"\nProchaines etapes:")
    print(f"1. Modifier data_preparation.py ligne 148:")
    print(f'   tokens = load_tokens_from_file("{output_file}")')
    print(f"2. Lancer: python data_preparation.py")
    print(f"3. Lancer: python train.py")


if __name__ == "__main__":
    main()
