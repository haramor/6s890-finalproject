"""
Simple Game-Theoretic Loss with Entropy Regularization

This implements a minimal but theoretically sound game-theoretic regularization:
- Encourages the model to spread probability mass over all legal moves
- Based on maximum entropy principle from game theory
- Prevents overfitting to single training moves
"""

import torch
import torch.nn.functional as F
from typing import Optional
import chess


class SimpleGameTheoreticLoss(torch.nn.Module):
    """
    Loss = CE + λ × Entropy_Penalty
    
    Where entropy penalty encourages exploration of legal move space.
    This is based on maximum entropy RL and game-theoretic exploration.
    """

    def __init__(
        self,
        eps: float,
        n_predictions: int,
        gt_weight: float = 0.1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(SimpleGameTheoreticLoss, self).__init__()
        self.eps = eps
        self.gt_weight = gt_weight
        self.device = device
        self.indices = torch.arange(n_predictions).unsqueeze(0).to(device)
        self.indices.requires_grad = False

    def compute_ce_loss(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """Standard cross-entropy loss."""
        predicted = predicted[self.indices < lengths]
        targets = targets[self.indices < lengths]

        target_vector = (
            torch.zeros_like(predicted)
            .scatter(dim=1, index=targets.unsqueeze(1), value=1.0)
            .to(self.device)
        )
        target_vector = target_vector * (1.0 - self.eps) + self.eps / target_vector.size(1)

        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(dim=1)
        return torch.mean(loss)

    def compute_entropy_regularization(
        self,
        predicted: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative entropy (we want HIGH entropy = exploration).
        
        Game-theoretic interpretation:
        - High entropy = more uniform distribution over actions
        - Prevents deterministic overfitting
        - Encourages considering multiple legal options (minimax exploration)
        - Based on maximum entropy principle in game theory
        
        Entropy = -Σ p(a) log p(a)
        We return NEGATIVE entropy so minimizing it increases entropy.
        """
        # Get probabilities
        probs = F.softmax(predicted, dim=-1)
        
        # Compute entropy: -Σ p log p
        log_probs = F.log_softmax(predicted, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        # Return negative (we want to MAXIMIZE entropy)
        return -entropy.mean()

    def forward(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        **kwargs
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            predicted: Model predictions (N, n_predictions, vocab_size)
            targets: Ground truth moves (N, n_predictions)
            lengths: Sequence lengths (N, 1)
            
        Returns:
            Tuple of (total_loss, ce_loss, entropy_penalty) for logging
        """
        # Compute cross-entropy loss
        ce_loss = self.compute_ce_loss(predicted, targets, lengths)

        # Compute entropy regularization (game-theoretic exploration)
        if self.gt_weight > 0:
            # Only on first prediction
            first_pred = predicted[:, 0, :]
            entropy_penalty = self.compute_entropy_regularization(first_pred)
        else:
            entropy_penalty = torch.tensor(0.0, device=self.device)

        # Combined loss
        total_loss = ce_loss + self.gt_weight * entropy_penalty

        return total_loss, ce_loss, -entropy_penalty  # Return positive entropy for logging

    def __del__(self):
        pass


class LabelSmoothedCE(torch.nn.Module):
    """Standard cross-entropy loss for baseline."""

    def __init__(self, eps: float, n_predictions: int):
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.indices = torch.arange(n_predictions).unsqueeze(0).to(device)
        self.indices.requires_grad = False

    def forward(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        predicted = predicted[self.indices < lengths]
        targets = targets[self.indices < lengths]
        device = predicted.device

        target_vector = (
            torch.zeros_like(predicted)
            .scatter(dim=1, index=targets.unsqueeze(1), value=1.0)
            .to(device)
        )
        target_vector = target_vector * (1.0 - self.eps) + self.eps / target_vector.size(1)

        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(dim=1)
        return torch.mean(loss)
