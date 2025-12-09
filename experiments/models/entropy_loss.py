"""
Entropy Regularization Loss for Chess Behavioral Cloning

This module implements an entropy-regularized cross-entropy loss that encourages
the model to maintain uncertainty/exploration rather than being overconfident.

The loss combines:
1. Standard cross-entropy loss (correctness)
2. Negative entropy penalty (encourages higher entropy/less confident predictions)

L_total = L_ce - λ * H(p)

where H(p) is the entropy of the predicted distribution, and λ controls the
strength of the entropy regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyRegularizedLoss(nn.Module):
    """
    Cross-entropy loss with entropy regularization.
    
    Args:
        eps: Label smoothing factor
        n_predictions: Number of moves to predict
        entropy_weight: Weight for entropy regularization (λ)
                       Higher values encourage more uniform/exploratory distributions
    """
    
    def __init__(self, eps=0.1, n_predictions=1, entropy_weight=0.01):
        super().__init__()
        self.eps = eps
        self.n_predictions = n_predictions
        self.entropy_weight = entropy_weight
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=eps)
    
    def compute_entropy(self, logits):
        """
        Compute entropy of predicted distribution.
        
        H(p) = -Σ p(x) log p(x)
        
        Args:
            logits: (batch_size, vocab_size) unnormalized logits
            
        Returns:
            entropy: (batch_size,) entropy for each prediction
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Compute log probabilities (more numerically stable)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Entropy: H(p) = -Σ p(x) log p(x)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return entropy
    
    def forward(self, predicted, targets, lengths):
        """
        Compute entropy-regularized loss.
        
        Args:
            predicted: (batch_size, n_predictions, vocab_size) predicted logits
            targets: (batch_size, n_predictions) target move indices
            lengths: (batch_size,) number of valid predictions per sequence
            
        Returns:
            total_loss: scalar loss value
            ce_loss: cross-entropy component (for logging)
            entropy: average entropy (for logging)
        """
        batch_size, n_pred, vocab_size = predicted.shape
        
        # Flatten for loss computation
        predicted_flat = predicted.reshape(-1, vocab_size)  # (B*n_pred, V)
        targets_flat = targets.reshape(-1)  # (B*n_pred,)
        
        # Create mask for valid positions
        mask = torch.arange(n_pred, device=lengths.device)[None, :] < lengths[:, None]
        mask_flat = mask.reshape(-1)  # (B*n_pred,)
        
        # Compute cross-entropy loss
        ce_loss_per_token = self.ce_loss(predicted_flat, targets_flat)
        ce_loss_masked = (ce_loss_per_token * mask_flat.float()).sum()
        ce_loss_normalized = ce_loss_masked / lengths.sum()
        
        # Compute entropy
        entropy_per_token = self.compute_entropy(predicted_flat)
        entropy_masked = (entropy_per_token * mask_flat.float()).sum()
        entropy_normalized = entropy_masked / lengths.sum()
        
        # Total loss: minimize CE, maximize entropy (so subtract entropy penalty)
        # L = L_ce - λ * H(p)
        total_loss = ce_loss_normalized - self.entropy_weight * entropy_normalized
        
        return total_loss, ce_loss_normalized, entropy_normalized


class LabelSmoothedCE(nn.Module):
    """
    Standard label-smoothed cross-entropy loss (for baseline comparison).
    """
    
    def __init__(self, eps=0.1, n_predictions=1):
        super().__init__()
        self.eps = eps
        self.n_predictions = n_predictions
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', label_smoothing=eps)
    
    def forward(self, predicted, targets, lengths):
        """
        Compute standard cross-entropy loss.
        
        Args:
            predicted: (batch_size, n_predictions, vocab_size) predicted logits
            targets: (batch_size, n_predictions) target move indices
            lengths: (batch_size,) number of valid predictions per sequence
            
        Returns:
            loss: scalar loss value
        """
        batch_size, n_pred, vocab_size = predicted.shape
        
        # Flatten
        predicted_flat = predicted.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Create mask
        mask = torch.arange(n_pred, device=lengths.device)[None, :] < lengths[:, None]
        mask_flat = mask.reshape(-1)
        
        # Compute loss
        loss_per_token = self.ce_loss(predicted_flat, targets_flat)
        loss_masked = (loss_per_token * mask_flat.float()).sum()
        loss_normalized = loss_masked / lengths.sum()
        
        return loss_normalized
    
    def compute_ce_loss(self, predicted, targets, lengths):
        """Alias for compatibility with validation code."""
        return self.forward(predicted, targets, lengths)
