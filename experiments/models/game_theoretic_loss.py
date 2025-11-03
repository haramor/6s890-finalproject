"""
Game-Theoretic Regularization Loss

This module implements a custom loss function that combines:
1. Standard cross-entropy loss (for imitating expert moves)
2. KL-divergence regularization (for alignment with minimax-optimal play)

The hypothesis is that explicitly penalizing deviations from game-theoretic
equilibrium (approximated by Stockfish) will reduce sample complexity.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict
import chess
import chess.engine
from functools import lru_cache


class GameTheoreticLoss(torch.nn.Module):
    """
    Combined loss: L = L_CE + λ * L_KL

    where:
    - L_CE is label-smoothed cross-entropy for imitating expert moves
    - L_KL is KL-divergence between model predictions and Stockfish evaluations
    - λ is the weight balancing the two terms
    """

    def __init__(
        self,
        eps: float,
        n_predictions: int,
        gt_weight: float = 0.1,
        stockfish_path: Optional[str] = None,
        stockfish_depth: int = 15,
        stockfish_time_limit: float = 0.1,
        use_cache: bool = True,
        cache_size: int = 10000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the game-theoretic loss.

        Args:
            eps: Label smoothing coefficient for cross-entropy
            n_predictions: Number of predictions per datapoint
            gt_weight: Weight λ for KL-divergence term
            stockfish_path: Path to Stockfish executable
            stockfish_depth: Search depth for Stockfish
            stockfish_time_limit: Time limit per position (seconds)
            use_cache: Whether to cache Stockfish evaluations
            cache_size: Size of LRU cache for Stockfish evaluations
            device: Device for tensor operations
        """
        super(GameTheoreticLoss, self).__init__()

        self.eps = eps
        self.gt_weight = gt_weight
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.stockfish_time_limit = stockfish_time_limit
        self.use_cache = use_cache
        self.device = device

        # Initialize Stockfish engine if path provided
        self.engine = None
        if stockfish_path and gt_weight > 0:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
                print("Game-theoretic regularization will be disabled")

        # Cache for Stockfish evaluations
        if use_cache:
            self._get_stockfish_distribution = lru_cache(maxsize=cache_size)(
                self._get_stockfish_distribution_uncached
            )
        else:
            self._get_stockfish_distribution = self._get_stockfish_distribution_uncached

        # For efficient indexing
        self.indices = torch.arange(n_predictions).unsqueeze(0).to(device)
        self.indices.requires_grad = False

    def _get_stockfish_distribution_uncached(
        self,
        fen: str,
        legal_moves_uci: tuple
    ) -> Dict[str, float]:
        """
        Get move distribution from Stockfish for a given position.

        Uses multi-PV analysis to get top moves and their evaluations,
        then converts to a probability distribution via softmax over centipawn scores.

        Args:
            fen: Position in FEN notation
            legal_moves_uci: Tuple of legal moves in UCI notation

        Returns:
            Dictionary mapping UCI moves to probabilities
        """
        if self.engine is None:
            # Return uniform distribution if Stockfish not available
            uniform_prob = 1.0 / len(legal_moves_uci)
            return {move: uniform_prob for move in legal_moves_uci}

        try:
            board = chess.Board(fen)

            # Analyze with multi-PV to get top moves
            info = self.engine.analyse(
                board,
                chess.engine.Limit(
                    depth=self.stockfish_depth,
                    time=self.stockfish_time_limit
                ),
                multipv=min(len(list(board.legal_moves)), 5)  # Top 5 moves
            )

            # Extract centipawn scores and moves
            move_scores = {}
            for result in (info if isinstance(info, list) else [info]):
                if "pv" in result and len(result["pv"]) > 0:
                    move = result["pv"][0].uci()
                    score = result.get("score", chess.engine.Score(0))

                    # Convert score to centipawns (from perspective of side to move)
                    if score.is_mate():
                        # Mate scores: very high for winning, very low for losing
                        mate_in = score.mate()
                        cp = 10000 * (1 if mate_in > 0 else -1) / abs(mate_in)
                    else:
                        cp = score.score()

                    move_scores[move] = cp

            # Convert centipawn scores to probabilities via softmax
            # Higher centipawn score = better move = higher probability
            temperature = 100.0  # Temperature for softmax (tune this)
            scores = torch.tensor([move_scores.get(m, -1000) for m in legal_moves_uci])
            probs = F.softmax(scores / temperature, dim=0)

            return {move: probs[i].item() for i, move in enumerate(legal_moves_uci)}

        except Exception as e:
            print(f"Error getting Stockfish distribution: {e}")
            # Return uniform distribution on error
            uniform_prob = 1.0 / len(legal_moves_uci)
            return {move: uniform_prob for move in legal_moves_uci}

    def compute_ce_loss(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            predicted: Model predictions (N, n_predictions, vocab_size)
            targets: Ground truth moves (N, n_predictions)
            lengths: Sequence lengths (N, 1)

        Returns:
            Scalar loss value
        """
        # Remove pad-positions and flatten
        predicted = predicted[self.indices < lengths]  # (sum(lengths), vocab_size)
        targets = targets[self.indices < lengths]  # (sum(lengths))

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = (
            torch.zeros_like(predicted)
            .scatter(dim=1, index=targets.unsqueeze(1), value=1.0)
            .to(self.device)
        )  # (sum(lengths), vocab_size), one-hot

        target_vector = target_vector * (1.0 - self.eps) + self.eps / target_vector.size(1)

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(dim=1)

        return torch.mean(loss)

    def compute_kl_loss(
        self,
        predicted: torch.Tensor,
        board_states: list,  # List of FEN strings
        legal_moves: list,   # List of lists of legal moves (indices)
        move_vocab: list     # Vocabulary mapping indices to UCI moves
    ) -> torch.Tensor:
        """
        Compute KL-divergence between model predictions and Stockfish distribution.

        KL(Stockfish || Model) = Σ p_sf(a) log(p_sf(a) / p_model(a))

        Args:
            predicted: Model predictions (N, vocab_size)
            board_states: List of board positions in FEN notation
            legal_moves: List of legal move indices for each position
            move_vocab: Vocabulary mapping indices to UCI notation

        Returns:
            Scalar KL-divergence loss
        """
        if self.engine is None or self.gt_weight == 0:
            return torch.tensor(0.0, device=self.device)

        batch_size = predicted.shape[0]
        kl_losses = []

        for i in range(batch_size):
            try:
                # Get legal moves for this position
                legal_move_indices = legal_moves[i]
                legal_moves_uci = tuple([move_vocab[idx] for idx in legal_move_indices])

                # Get Stockfish distribution
                sf_dist = self._get_stockfish_distribution(
                    board_states[i],
                    legal_moves_uci
                )

                # Create target distribution tensor
                sf_probs = torch.zeros(predicted.shape[1], device=self.device)
                for idx, uci_move in zip(legal_move_indices, legal_moves_uci):
                    sf_probs[idx] = sf_dist[uci_move]

                # Get model distribution
                model_log_probs = F.log_softmax(predicted[i], dim=0)

                # Compute KL divergence: KL(p||q) = Σ p(x) log(p(x)/q(x))
                #                                  = Σ p(x) log p(x) - Σ p(x) log q(x)
                kl = (sf_probs * (torch.log(sf_probs + 1e-10) - model_log_probs)).sum()
                kl_losses.append(kl)

            except Exception as e:
                print(f"Error computing KL loss for position {i}: {e}")
                kl_losses.append(torch.tensor(0.0, device=self.device))

        return torch.stack(kl_losses).mean()

    def forward(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        board_states: Optional[list] = None,
        legal_moves: Optional[list] = None,
        move_vocab: Optional[list] = None
    ) -> tuple:
        """
        Compute combined loss: L = L_CE + λ * L_KL

        Args:
            predicted: Model predictions (N, n_predictions, vocab_size)
            targets: Ground truth moves (N, n_predictions)
            lengths: Sequence lengths (N, 1)
            board_states: Optional list of FEN strings for GT regularization
            legal_moves: Optional list of legal move indices for GT regularization
            move_vocab: Optional move vocabulary for GT regularization

        Returns:
            Tuple of (total_loss, ce_loss, kl_loss) for logging
        """
        # Compute cross-entropy loss
        ce_loss = self.compute_ce_loss(predicted, targets, lengths)

        # Compute KL-divergence loss if game-theoretic regularization enabled
        kl_loss = torch.tensor(0.0, device=self.device)
        if (self.gt_weight > 0 and
            board_states is not None and
            legal_moves is not None and
            move_vocab is not None):

            # Use only first prediction for KL loss (next move)
            first_pred = predicted[:, 0, :]  # (N, vocab_size)
            kl_loss = self.compute_kl_loss(
                first_pred,
                board_states,
                legal_moves,
                move_vocab
            )

        # Combined loss
        total_loss = ce_loss + self.gt_weight * kl_loss

        return total_loss, ce_loss, kl_loss

    def __del__(self):
        """Clean up Stockfish engine on deletion."""
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass


class LabelSmoothedCE(torch.nn.Module):
    """
    Standard cross-entropy loss with label smoothing (for baseline experiments).

    This is the same as the loss in chess-transformers/transformers/criteria.py
    """

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
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Returns scalar loss for compatibility with training loop.
        """
        # Remove pad-positions and flatten
        predicted = predicted[self.indices < lengths]
        targets = targets[self.indices < lengths]

        device = predicted.device

        # "Smoothed" one-hot vectors
        target_vector = (
            torch.zeros_like(predicted)
            .scatter(dim=1, index=targets.unsqueeze(1), value=1.0)
            .to(device)
        )
        target_vector = target_vector * (1.0 - self.eps) + self.eps / target_vector.size(1)

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(dim=1)

        return torch.mean(loss)
