"""
Entraînement du modèle LLM pour la musique

Ce script entraîne le modèle Transformer sur les données musicales préparées.

Processus d'entraînement:
1. Charger les données (séquences input/target)
2. Diviser en train (80%) et validation (20%)
3. Pour chaque epoch:
   - Forward pass: prédire les tokens
   - Calculer la loss (erreur de prédiction)
   - Backward pass: calculer les gradients
   - Mettre à jour les poids du modèle
   - Évaluer sur la validation
   - Sauvegarder si meilleur modèle

Loss (fonction de perte):
Mesure à quel point le modèle se trompe.
Loss élevée = mauvaises prédictions, Loss faible = bonnes prédictions.
Objectif: minimiser la loss.

Overfitting vs Underfitting:
- Overfitting: le modèle mémorise les données (train loss baisse, val loss monte)
- Underfitting: le modèle n'apprend pas assez (les deux restent élevées)

Fichiers créés:
- models/best_model.pt: Meilleur modèle
- models/checkpoint_epoch_X.pt: Checkpoints périodiques
- output/training_loss.png: Graphique de la loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from model import MusicLLM


class MusicDataset(Dataset):
    """Dataset PyTorch pour les séquences musicales."""
    
    def __init__(self, input_ids, target_ids):
        self.input_ids = input_ids
        self.target_ids = target_ids
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'target_ids': torch.tensor(self.target_ids[idx], dtype=torch.long)
        }


def load_data():
    """Charge les données d'entraînement depuis le disque."""
    data_path = os.path.join(config.DATA_DIR, "training_data.pkl")
    
    print(f"Chargement des donnees depuis {path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  Sequences: {len(data['input_ids']):,}")
    print(f"  Vocabulaire: {data['vocab_size']:,}")
    
    return data


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entraîne le modèle sur une epoch."""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(input_ids)
        
        # Calculer la loss
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Évalue le modèle sur les données de validation."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def plot_losses(train_losses, val_losses=None):
    """Visualise l'évolution de la loss pendant l'entraînement."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    if val_losses:
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evolution de la loss pendant l\'entraînement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, 'training_loss.png')
    plt.savefig(output_path, dpi=150)
    print(f"Graphique sauvegarde dans {output_path}")


def main():
    """Fonction principale d'entraînement."""
    
    # Device (GPU si disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Charger les données
   data = load_data()
    
    # Split train/val (80/20)
    split_idx = int(len(data['input_ids']) * 0.8)
    
    train_dataset = MusicDataset(
        data['input_ids'][:split_idx],
        data['target_ids'][:split_idx]
    )
    
    val_dataset = MusicDataset(
        data['input_ids'][split_idx:],
        data['target_ids'][split_idx:]
    )
    
    print(f"Train sequences: {len(train_dataset):,}")
    print(f"Val sequences: {len(val_dataset):,}\n")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Créer le modèle
    model = MusicLLM(
        vocab_size=data['vocab_size'],
        d_model=config.EMBEDDING_DIM,
        n_heads=config.N_HEADS,
        n_layers=config.N_LAYERS,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"Architecture du modele:")
    print(f"  Vocabulaire: {data['vocab_size']:,} tokens")
    print(f"  Embedding dim: {config.EMBEDDING_DIM}")
    print(f"  Attention heads: {config.N_HEADS}")
    print(f"  Layers: {config.N_LAYERS}")
    print(f"  Hidden dim: {config.HIDDEN_DIM}")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametres: {n_params:,}\n")
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Entraînement
    print(f"Entrainement:")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Learning rate: {config.LEARNING_RATE}\n")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(config.MODELS_DIR, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': data['vocab_size']
            }, checkpoint_path)
            print(f"Meilleur modele sauvegarde!")
        
        # Checkpoints périodiques
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config.MODELS_DIR, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'vocab_size': data['vocab_size']
            }, checkpoint_path)
    
    # Visualiser les courbes
    plot_losses(train_losses, val_losses)
    
    print(f"\nEntrainement termine!")
    print(f"Meilleure val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
