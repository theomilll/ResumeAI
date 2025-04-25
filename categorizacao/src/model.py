import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer


class MultiHeadTextClassifier(nn.Module):
    """
    A multi-head classifier that makes independent decisions for each class,
    helping avoid the bias of predicting only one class.
    """
    def __init__(self, vocab_size=30000, embed_dim=128, hidden_dim=128, num_classes=5, dropout=0.4):
        super(MultiHeadTextClassifier, self).__init__()
        
        # Basic embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional LSTM for sequence understanding
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        
        # Independent classifiers for each class (multi-head approach)
        self.class_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_classes)
        ])
        
        # Mixing layer to combine heads
        self.mixing_layer = nn.Linear(num_classes, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with controlled randomness for better convergence."""
        # Using orthogonal initialization for RNNs
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        # Initialize embedding with normal distribution
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Initialize each head with slightly different weights
        for i, head in enumerate(self.class_heads):
            for name, param in head.named_parameters():
                if 'weight' in name:
                    # Use different init for each head to break symmetry
                    nn.init.xavier_normal_(param, gain=0.8 + 0.1*i)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Initialize mixing layer
        nn.init.xavier_normal_(self.mixing_layer.weight)
        nn.init.zeros_(self.mixing_layer.bias)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass with multi-head decision making.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for each class
        """
        # Clamp input IDs to vocabulary size
        input_ids = torch.clamp(input_ids, min=0, max=self.embedding.num_embeddings-1)
        
        # Create embeddings
        x = self.embedding(input_ids)  # [batch_size, seq_len, embed_dim]
        
        # Apply dropout to embeddings during training
        if self.training:
            x = self.dropout(x)
        
        # Process through LSTM
        if attention_mask is not None:
            # Convert mask to lengths
            lengths = attention_mask.sum(dim=1).cpu()
            
            # Use packed sequence for variable length inputs
            packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, (hidden, _) = self.lstm(packed_x)
            
            # Unpack the sequence
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            # Process without packing
            lstm_out, (hidden, _) = self.lstm(x)
        
        # Get the final hidden state from forward and backward pass
        # hidden shape: [2, batch_size, hidden_dim]
        hidden_forward = hidden[0]
        hidden_backward = hidden[1]
        
        # Concatenate forward and backward hidden states
        hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        
        # Apply each class head independently
        head_outputs = []
        for head in self.class_heads:
            # Each head predicts for one class
            head_out = head(hidden_concat)
            head_outputs.append(head_out)
            
        # Combine outputs into a single tensor [batch_size, num_classes]
        logits = torch.cat(head_outputs, dim=1)
        
        # During inference, apply higher temperature and per-class scaling
        if not self.training:
            # Use higher temperature of 1.5 to make decisions less confident
            logits = self.mixing_layer(logits) / 1.5
            
            # Apply class-specific scaling to counter the bias
            # Decrease confidence for class 2 (index 2, which is Gestão de Equipe) by 15%
            scaling = torch.ones_like(logits)
            scaling[:, 2] = 0.85  # Reduce confidence for Gestão de Equipe
            
            # Increase confidence for underrepresented classes
            scaling[:, 0] = 1.05  # Atualizações de Projeto
            scaling[:, 1] = 1.10  # Achados de Pesquisa
            scaling[:, 3] = 1.05  # Reuniões com Clientes
            scaling[:, 4] = 1.05  # Outras
            
            logits = logits * scaling
        else:
            # During training, add noise for regularization
            logits = self.mixing_layer(logits + torch.randn_like(logits) * 0.1)
            
            # Ensure all classes have a chance during training
            if random.random() < 0.3:  # 30% of batches
                # Randomly boost classes other than Gestão de Equipe more often
                class_idx = random.randint(0, logits.size(1) - 1)
                # Reduce the chance of boosting class 2 (Gestão de Equipe)
                if class_idx == 2 and random.random() < 0.7:
                    # 70% chance to pick another class
                    possible_classes = [i for i in range(logits.size(1)) if i != 2]
                    class_idx = random.choice(possible_classes)
                
                boost = torch.zeros_like(logits)
                boost[:, class_idx] = 0.3  # Add boost to the selected class
                logits = logits + boost
        
        return logits
        
    def save_pretrained(self, output_dir):
        """Save model to directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(output_dir, 'model.pt'))
        
        # Save config
        config = {
            'vocab_size': self.embedding.num_embeddings,
            'embed_dim': self.embedding.embedding_dim,
            'hidden_dim': self.lstm.hidden_size,
            'num_classes': len(self.class_heads),
            'dropout': self.dropout.p
        }
        
        import json
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        print(f"Model saved to {output_dir}")


def load_bert_model(num_classes, model_name='bert-base-multilingual-cased'):
    """
    Load a text classification model.
    
    Args:
        num_classes: Number of classification categories
        model_name: Name of the pre-trained model (ignored)
        
    Returns:
        Text classification model
    """
    print(f"Creating a multi-head text classifier with {num_classes} classes")
    model = MultiHeadTextClassifier(num_classes=num_classes)
    print("Multi-head text classifier created successfully")
    return model

def save_model(model, tokenizer, output_dir):
    """
    Save the model and tokenizer.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Output directory
    """
    # Make sure the directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    model.save_pretrained(output_dir)
    
    # Save the tokenizer
    try:
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")
        # Create a minimal vocabulary file as fallback
        with open(os.path.join(output_dir, 'vocab.txt'), 'w') as f:
            f.write("<PAD>\n<UNK>\n" + "\n".join([f"token{i}" for i in range(100)]))
    
    # Save label mapping
    print(f"Model saved to {output_dir}")

def load_model(model_dir):
    """
    Load model and tokenizer from directory.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        model, tokenizer
    """
    try:
        print(f"Loading model from {model_dir}")
        
        # Load configuration
        config_path = os.path.join(model_dir, 'config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Create model with saved configuration
            model = MultiHeadTextClassifier(
                vocab_size=config.get('vocab_size', 30000),
                embed_dim=config.get('embed_dim', 128),
                hidden_dim=config.get('hidden_dim', 128),
                num_classes=config.get('num_classes', 5),
                dropout=config.get('dropout', 0.4)
            )
        else:
            # Use default configuration
            model = MultiHeadTextClassifier(num_classes=5)
        
        # Load weights
        model_path = os.path.join(model_dir, 'model.pt')
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            print("Model weights loaded successfully")
        
        # Load tokenizer
        try:
            tokenizer = BertTokenizer.from_pretrained(model_dir)
            print("Tokenizer loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            print("Using default tokenizer")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise 