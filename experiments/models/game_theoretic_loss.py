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
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        disable_stockfish: bool = False,
        precomputed_cache_path: Optional[str] = None  # NEW: path to precomputed distributions
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
            disable_stockfish: If True, skip real-time Stockfish calls (for stability)
            precomputed_cache_path: Path to precomputed Stockfish distributions (.pkl file)
        """
        super(GameTheoreticLoss, self).__init__()

        self.eps = eps
        self.gt_weight = gt_weight
        self.stockfish_path = stockfish_path
        self.stockfish_depth = stockfish_depth
        self.stockfish_time_limit = stockfish_time_limit
        self.use_cache = use_cache
        self.device = device
        self.disable_stockfish = disable_stockfish
        self.precomputed_cache_path = precomputed_cache_path

        # Load precomputed distributions if provided
        self.precomputed_distributions = None
        if precomputed_cache_path:
            try:
                import pickle
                with open(precomputed_cache_path, 'rb') as f:
                    self.precomputed_distributions = pickle.load(f)
                print(f"✓ Loaded {len(self.precomputed_distributions)} precomputed Stockfish distributions")
                print(f"  Cache file: {precomputed_cache_path}")
            except Exception as e:
                print(f"Warning: Could not load precomputed cache: {e}")
                self.precomputed_distributions = None

        # Initialize Stockfish engine if path provided and not disabled
        self.engine = None
        if stockfish_path and gt_weight > 0 and not disable_stockfish and not precomputed_cache_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"✓ Stockfish initialized successfully at {stockfish_path}")
            except Exception as e:
                print(f"Warning: Could not initialize Stockfish: {e}")
                print("Game-theoretic regularization will be disabled")
        elif disable_stockfish:
            print("Stockfish calls disabled - GT regularization will return uniform distributions")
        elif precomputed_cache_path:
            print("Using precomputed Stockfish distributions - no real-time engine needed")

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

        # Move vocabulary will be set by set_move_vocab() after initialization
        self.uci_to_idx = None
        self.idx_to_uci = None
        
    def set_move_vocab(self, move_to_index: dict, index_to_move: dict):
        """
        Set the move vocabulary from the dataset.
        
        Args:
            move_to_index: Dictionary mapping UCI moves to indices
            index_to_move: Dictionary mapping indices to UCI moves
        """
        self.uci_to_idx = move_to_index
        self.idx_to_uci = index_to_move
        print(f"Move vocabulary set with {len(self.uci_to_idx)} moves")

    def _board_tensor_to_fen(self, board_tensor: torch.Tensor) -> str:
        """
        Convert board position tensor to FEN string.
        
        Args:
            board_tensor: Tensor of shape (64,) with piece encodings
                         Encoding: 0=empty, 1=P, 2=N, 3=B, 4=R, 5=Q, 6=K,
                                  7=p, 8=n, 9=b, 10=r, 11=q, 12=k
        
        Returns:
            FEN string representing the position
        """
        # Piece type mapping
        piece_map = {
            0: '.',   # empty
            1: 'P',   # white pawn
            2: 'N',   # white knight
            3: 'B',   # white bishop
            4: 'R',   # white rook
            5: 'Q',   # white queen
            6: 'K',   # white king
            7: 'p',   # black pawn
            8: 'n',   # black knight
            9: 'b',   # black bishop
            10: 'r',  # black rook
            11: 'q',  # black queen
            12: 'k',  # black king
        }

        # Convert tensor to CPU and numpy for easier processing
        board_array = board_tensor.cpu().numpy()

        # Build FEN string rank by rank (from rank 8 to rank 1)
        fen_ranks = []
        for rank in range(7, -1, -1):  # Ranks 8 to 1
            fen_rank = ""
            empty_count = 0
            
            for file in range(8):  # Files a to h
                square_idx = rank * 8 + file
                piece_code = int(board_array[square_idx])
                piece_char = piece_map.get(piece_code, '.')
                
                if piece_char == '.':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_rank += str(empty_count)
                        empty_count = 0
                    fen_rank += piece_char
            
            if empty_count > 0:
                fen_rank += str(empty_count)
            
            fen_ranks.append(fen_rank)

        # Join ranks with '/'
        board_fen = '/'.join(fen_ranks)

        # For simplicity, assume white to move, no castling rights, no en passant
        # In a full implementation, these would come from additional batch fields
        full_fen = f"{board_fen} w KQkq - 0 1"

        return full_fen

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
            # Check if engine is still alive, restart if needed
            if not hasattr(self.engine, 'process') or self.engine.process.poll() is not None:
                print("Stockfish engine died, restarting...")
                try:
                    self.engine.quit()
                except:
                    pass
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                
            board = chess.Board(fen)

            # Analyze with multi-PV to get top moves
            num_legal = len(list(board.legal_moves))
            info = self.engine.analyse(
                board,
                chess.engine.Limit(
                    depth=self.stockfish_depth,
                    time=self.stockfish_time_limit
                ),
                multipv=min(num_legal, 5)  # Top 5 moves
            )

            # Extract centipawn scores and moves
            move_scores = {}
            for result in (info if isinstance(info, list) else [info]):
                if "pv" in result and len(result["pv"]) > 0:
                    move = result["pv"][0].uci()
                    score = result.get("score", chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE))

                    # Convert score to centipawns (from perspective of side to move)
                    pov_score = score.white() if board.turn == chess.WHITE else score.black()
                    
                    if pov_score.is_mate():
                        # Mate scores: very high for winning, very low for losing
                        mate_in = pov_score.mate()
                        cp = 10000 if mate_in > 0 else -10000
                    else:
                        cp = pov_score.score()

                    move_scores[move] = cp

            # If no moves were evaluated, return uniform
            if not move_scores:
                uniform_prob = 1.0 / len(legal_moves_uci)
                return {move: uniform_prob for move in legal_moves_uci}

            # Convert centipawn scores to probabilities via softmax
            # Higher centipawn score = better move = higher probability
            temperature = 100.0  # Temperature for softmax (tune this)
            
            # Assign very low score to moves not in top-k
            default_score = min(move_scores.values()) - 500 if move_scores else -1000
            scores = torch.tensor([move_scores.get(m, default_score) for m in legal_moves_uci])
            probs = F.softmax(scores / temperature, dim=0)

            return {move: probs[i].item() for i, move in enumerate(legal_moves_uci)}

        except Exception as e:
            # Silently return uniform on error to avoid spam
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
        board_states: torch.Tensor,
        subsample_ratio: float = 0.05  # Only evaluate 5% of positions
    ) -> torch.Tensor:
        """
        Compute KL-divergence between model predictions and Stockfish distribution.
        
        If precomputed distributions are available, uses those. Otherwise, falls back
        to real-time Stockfish evaluation with subsampling.

        KL(Stockfish || Model) = Σ p_sf(a) log(p_sf(a) / p_model(a))

        Args:
            predicted: Model predictions (N, vocab_size)
            board_states: Board position tensors (N, 64)
            subsample_ratio: Fraction of batch to evaluate (only used if calling Stockfish live)

        Returns:
            Scalar KL-divergence loss
        """
        if self.gt_weight == 0:
            return torch.tensor(0.0, device=self.device)
        
        if self.uci_to_idx is None:
            return torch.tensor(0.0, device=self.device)

        batch_size = predicted.shape[0]
        
        # If using precomputed distributions, we can evaluate ALL positions (no subsampling needed)
        if self.precomputed_distributions is not None:
            sample_indices = list(range(batch_size))
            scaling_factor = 1.0  # No scaling needed
        else:
            # Real-time Stockfish: subsample to keep training fast
            if self.engine is None:
                return torch.tensor(0.0, device=self.device)
            
            num_samples = max(1, int(batch_size * subsample_ratio))
            sample_indices = torch.randperm(batch_size, device='cpu')[:num_samples].tolist()
            scaling_factor = 1.0 / subsample_ratio  # Scale up to account for subsampling
        
        kl_losses = []

        for idx in sample_indices:
            try:
                # Convert board tensor to FEN
                fen = self._board_tensor_to_fen(board_states[idx])
                
                # Get Stockfish distribution (from cache or live)
                if self.precomputed_distributions is not None:
                    # Use precomputed distribution
                    if fen not in self.precomputed_distributions:
                        continue  # Skip if position not in cache
                    sf_dist = self.precomputed_distributions[fen]
                else:
                    # Call Stockfish live
                    board = chess.Board(fen)
                    legal_moves_uci = tuple([move.uci() for move in board.legal_moves])
                    
                    if len(legal_moves_uci) == 0:
                        continue
                    
                    sf_dist = self._get_stockfish_distribution(fen, legal_moves_uci)

                if not sf_dist:  # Empty distribution
                    continue

                # Create target distribution tensor (only over legal moves)
                sf_probs = torch.zeros(predicted.shape[1], device=self.device)
                model_mask = torch.zeros(predicted.shape[1], dtype=torch.bool, device=self.device)
                
                for uci_move, prob in sf_dist.items():
                    if uci_move in self.uci_to_idx:
                        move_idx = self.uci_to_idx[uci_move]
                        if move_idx < predicted.shape[1]:  # Safety check
                            sf_probs[move_idx] = prob
                            model_mask[move_idx] = True

                # Check if we found any valid moves
                if not model_mask.any():
                    continue

                # Get model distribution (masked to legal moves only)
                model_logits_masked = predicted[idx].clone()
                model_logits_masked[~model_mask] = float('-inf')
                model_log_probs = F.log_softmax(model_logits_masked, dim=0)

                # Compute KL divergence
                kl = (sf_probs[model_mask] * (
                    torch.log(sf_probs[model_mask] + 1e-10) - model_log_probs[model_mask]
                )).sum()
                
                kl_losses.append(kl)

            except Exception as e:
                # Silently skip errors
                continue

        if len(kl_losses) == 0:
            return torch.tensor(0.0, device=self.device)

        # Scale by subsample ratio if using live Stockfish
        return torch.stack(kl_losses).mean() * scaling_factor

    def forward(
        self,
        predicted: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        board_states: Optional[torch.Tensor] = None,
        **kwargs  # Accept and ignore other arguments for compatibility
    ) -> tuple:
        """
        Compute combined loss: L = L_CE + λ * L_KL

        Args:
            predicted: Model predictions (N, n_predictions, vocab_size)
            targets: Ground truth moves (N, n_predictions)
            lengths: Sequence lengths (N, 1)
            board_states: Optional board position tensors (N, 64) for GT regularization
            **kwargs: Other arguments (ignored, for backward compatibility)

        Returns:
            Tuple of (total_loss, ce_loss, kl_loss) for logging
        """
        # Compute cross-entropy loss
        ce_loss = self.compute_ce_loss(predicted, targets, lengths)

        # Compute KL-divergence loss if game-theoretic regularization enabled
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.gt_weight > 0 and board_states is not None:
            # Use only first prediction for KL loss (next move)
            first_pred = predicted[:, 0, :]  # (N, vocab_size)
            kl_loss = self.compute_kl_loss(first_pred, board_states)

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
        lengths: torch.Tensor,
        **kwargs  # Accept and ignore extra arguments for compatibility
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
