"""
Architecture Transformer pour la génération de musique

Ce fichier implémente l'architecture complète du modèle Transformer (style GPT)
pour la génération de musique.

Architecture:
Input (tokens) → Embeddings → Positional Encoding → Transformer Blocks → Output

Composants principaux:
- PositionalEncoding: Ajoute l'information de position dans la séquence
- MultiHeadAttention: Permet au modèle de "regarder" les relations entre tokens
- FeedForward: Transformations non-linéaires
- TransformerBlock: Combine attention + feedforward + normalisation
- MusicLLM: Modèle complet

Masque causal:
Le modèle ne peut pas "tricher" en regardant le futur. À chaque position, 
il ne voit que les tokens précédents (auto-régression).

Génération:
Le modèle génère de la musique token par token, comme compléter une phrase.
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal.
    
    Les Transformers n'ont pas de notion de position par défaut.
    On ajoute donc un pattern sinusoïdal unique pour chaque position.
    """
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Créer la matrice d'encodage positionnel
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    Mécanisme d'attention multi-têtes.
    
    L'attention permet au modèle de "regarder" les autres tokens pour comprendre
    le contexte. Avec plusieurs têtes, chaque tête peut se spécialiser dans
    différents aspects (mélodie, rythme, harmonie, etc.).
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # Projections linéaires et division en têtes
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Calcul de l'attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(output)


class FeedForward(nn.Module):
    """Réseau feed-forward (2 couches linéaires avec ReLU)."""
    
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Bloc Transformer complet: Attention + FeedForward + Normalisation + Résiduelle.
    """
    
    def __init__(self, d_model, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention avec connexion résiduelle
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward avec connexion résiduelle
        fed_forward = self.ff(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x


class MusicLLM(nn.Module):
    """
    Modèle complet: Mini-GPT pour la génération de musique.
    
    Architecture: Embedding → Positional → N × Transformer → Linear Output
    """
    
    def __init__(self, vocab_size, d_model, n_heads, n_layers, hidden_dim, 
                 max_len=5000, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Embedding et positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Stack de blocs Transformer
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        
        # Couche de sortie
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """Passe forward du modèle."""
        batch_size, seq_len = x.shape
        
        # Créer le masque causal si non fourni
        if mask is None:
            mask = self._create_causal_mask(seq_len).to(x.device)
        
        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        # Projection vers le vocabulaire
        logits = self.fc_out(x)
        
        return logits
    
    def _create_causal_mask(self, seq_len):
        """
        Crée un masque causal triangulaire.
        Empêche l'attention de regarder les tokens futurs.
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.unsqueeze(0).unsqueeze(0)
        return mask
    
    def generate(self, start_tokens, max_length, temperature=1.0, top_k=50, device='cpu'):
        """
        Génère une séquence de tokens de manière auto-régressive.
        
        Args:
            start_tokens: Liste d'IDs de tokens pour amorcer
            max_length: Nombre de tokens à générer
            temperature: Créativité (0.0=déterministe, >1.0=aléatoire)
            top_k: Nombre de tokens candidats
            device: 'cpu' ou 'cuda'
        
        Returns:
            Liste complète de tokens générés
        """
        self.eval()
        
        generated = start_tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                input_ids = torch.tensor([generated], dtype=torch.long).to(device)
                
                # Prédire le prochain token
                logits = self.forward(input_ids)
                logits = logits[0, -1, :] / temperature
                
                # Top-k sampling
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    probs = torch.softmax(top_k_logits, dim=-1)
                    next_token_idx = torch.multinomial(probs, 1).item()
                    next_token = top_k_indices[next_token_idx].item()
                else:
                    next_token = torch.argmax(logits).item()
                
                generated.append(next_token)
        
        return generated


if __name__ == "__main__":
    # Test du modèle
    print("Test du modèle MusicLLM\n")
    
    model = MusicLLM(vocab_size=1000, d_model=128, n_heads=4, n_layers=4, hidden_dim=512)
    
    # Test forward
    x = torch.randint(0, 1000, (2, 10))
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test génération
    generated = model.generate([1, 2, 3], max_length=20)
    print(f"\nTokens générés: {generated[:10]}...")
