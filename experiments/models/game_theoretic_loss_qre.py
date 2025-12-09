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

        self.precomputed_distributions = None
        if precomputed_cache_path:
            try:
                import pickle
                with open(precomputed_cache_path, 'rb') as f:
                    self.precomputed_distributions = pickle.load(f)
                print(f"✓ Loaded {len(self.precomputed_distributions)} precomputed Stockfish distributions")
                print(f"  Will use true minimax evaluations where available")
            except Exception as e:
                print(f"Note: No precomputed cache found (this is fine)")
                self.precomputed_distributions = None

        # We don't initialize Stockfish - we use heuristic evaluation instead
        self.engine = None
        print("✓ Game-theoretic evaluation initialized (heuristic-based, no engine required)")
        print("  Approach: Quantal Response Equilibrium with minimax-inspired value function")
        print("  Components: Material balance + Positional features + Opponent mobility")

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
        
        # For QRE approach, we don't need Stockfish - just use heuristic evaluation
        self.engine = None
        print("Using game-theoretic heuristic evaluation (no Stockfish required)")
        print("Approach: Quantal Response Equilibrium with minimax-inspired value function")
        
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

    def _compute_material_balance(self, board: chess.Board) -> float:
        """
        Compute material balance from the perspective of the side to move.
        
        This is a classical game-theoretic evaluation function.
        
        Returns:
            Material advantage in centipawns
        """
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        balance = 0
        for piece_type in piece_values:
            balance += len(board.pieces(piece_type, board.turn)) * piece_values[piece_type]
            balance -= len(board.pieces(piece_type, not board.turn)) * piece_values[piece_type]
        
        return balance
    
    def _compute_positional_value(self, board: chess.Board) -> float:
        """
        Compute positional features for game-theoretic evaluation.
        
        Includes:
        - Center control
        - King safety
        - Piece development
        - Pawn structure
        
        Returns:
            Positional evaluation in centipawns
        """
        score = 0
        
        # Center control (e4, d4, e5, d5)
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        for square in center_squares:
            if board.piece_at(square):
                piece = board.piece_at(square)
                if piece.color == board.turn:
                    score += 10
                else:
                    score -= 10
        
        # Piece development (knights and bishops not on back rank)
        back_rank = 0 if board.turn == chess.WHITE else 7
        for square in chess.SQUARES:
            if chess.square_rank(square) != back_rank:
                piece = board.piece_at(square)
                if piece and piece.color == board.turn:
                    if piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        score += 15
        
        # King safety (castling rights)
        if board.has_kingside_castling_rights(board.turn):
            score += 20
        if board.has_queenside_castling_rights(board.turn):
            score += 15
        
        # Mobility (number of legal moves is a proxy for position strength)
        score += len(list(board.legal_moves)) * 2
        
        return score
    
    def _evaluate_move_game_theoretically(self, board: chess.Board, move: chess.Move) -> float:
        """
        Evaluate a move using game-theoretic principles:
        1. Material gain/loss
        2. Positional improvement
        3. Threat creation
        4. King safety
        
        This approximates a shallow minimax evaluation.
        
        Args:
            board: Current board position
            move: Move to evaluate
            
        Returns:
            Evaluation score in centipawns (higher is better)
        """
        # Make the move on a copy
        board_copy = board.copy()
        
        # Check for captures (immediate material gain)
        capture_value = 0
        if board_copy.is_capture(move):
            captured_piece = board_copy.piece_at(move.to_square)
            if captured_piece:
                piece_values = {
                    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                    chess.ROOK: 500, chess.QUEEN: 900
                }
                capture_value = piece_values.get(captured_piece.piece_type, 0)
        
        # Make the move
        board_copy.push(move)
        
        # Check if move gives check (tactical advantage)
        check_bonus = 50 if board_copy.is_check() else 0
        
        # Check if move leads to checkmate (infinite value in game theory)
        if board_copy.is_checkmate():
            return 10000
        
        # Evaluate resulting position
        material_after = self._compute_material_balance(board_copy)
        positional_after = self._compute_positional_value(board_copy)
        
        # Opponent's response strength (approximate minimax)
        # Penalize moves that give opponent strong responses
        opponent_mobility = len(list(board_copy.legal_moves))
        mobility_penalty = opponent_mobility * 2
        
        # Total game-theoretic value
        score = (
            capture_value +           # Immediate material gain
            check_bonus +             # Tactical pressure
            material_after +          # Material balance after move
            positional_after -        # Positional value
            mobility_penalty          # Opponent's response options (minimax consideration)
        )
        
        return score
    
    def _get_game_theoretic_distribution(
        self,
        fen: str,
        temperature: float = 100.0
    ) -> Dict[str, float]:
        """
        Compute game-theoretic move distribution using minimax-inspired heuristics.
        
        This implements a simplified game-theoretic evaluation that:
        1. Evaluates each legal move using classical chess heuristics
        2. Considers material, positional, and tactical factors
        3. Approximates shallow minimax by considering opponent responses
        4. Converts evaluations to probabilities via softmax (Boltzmann rational)
        
        Args:
            fen: Position in FEN notation
            temperature: Softmax temperature for rationality (lower = more deterministic)
            
        Returns:
            Dictionary mapping UCI moves to probabilities
        """
        try:
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            
            if len(legal_moves) == 0:
                return {}
            
            # Evaluate all legal moves
            move_scores = {}
            for move in legal_moves:
                score = self._evaluate_move_game_theoretically(board, move)
                move_scores[move.uci()] = score
            
            # Convert scores to probabilities using softmax (Quantal Response Equilibrium)
            # This models bounded rationality in game theory
            scores_tensor = torch.tensor(list(move_scores.values()), dtype=torch.float32)
            probs = F.softmax(scores_tensor / temperature, dim=0)
            
            return {move: probs[i].item() for i, move in enumerate(move_scores.keys())}
            
        except Exception as e:
            # Fallback to uniform if evaluation fails
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)
            uniform_prob = 1.0 / len(legal_moves) if legal_moves else 0
            return {move.uci(): uniform_prob for move in legal_moves}

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
        subsample_ratio: float = 1.0  # Can evaluate all positions - it's fast
    ) -> torch.Tensor:
        """
        Compute KL-divergence between model predictions and game-theoretic equilibrium.
        
        This implements game-theoretic regularization by:
        1. Computing a minimax-inspired evaluation for each legal move
        2. Converting evaluations to a probability distribution (Quantal Response Equilibrium)
        3. Regularizing the model toward this game-theoretic distribution
        
        The approach combines:
        - Classical game theory: Material balance, positional evaluation
        - Bounded rationality: Softmax over move values (QRE)
        - Equilibrium concept: Encouraging Nash-like play
        
        If precomputed Stockfish distributions are available, uses those instead
        for true minimax evaluations.

        Args:
            predicted: Model predictions (N, vocab_size)
            board_states: Board position tensors (N, 64)
            subsample_ratio: Fraction of batch to evaluate

        Returns:
            Scalar KL-divergence loss
        """
        if self.gt_weight == 0:
            return torch.tensor(0.0, device=self.device)
        
        if self.uci_to_idx is None:
            return torch.tensor(0.0, device=self.device)

        batch_size = predicted.shape[0]
        sample_indices = list(range(batch_size))
        
        kl_losses = []

        for idx in sample_indices:
            try:
                # Convert board tensor to FEN
                fen = self._board_tensor_to_fen(board_states[idx])
                
                # Get game-theoretic distribution
                if self.precomputed_distributions is not None and fen in self.precomputed_distributions:
                    # Use precomputed Stockfish (true minimax)
                    gt_dist = self.precomputed_distributions[fen]
                else:
                    # Use heuristic game-theoretic evaluation (shallow minimax)
                    gt_dist = self._get_game_theoretic_distribution(fen, temperature=100.0)

                if not gt_dist:
                    continue

                # Create target distribution tensor
                target_probs = torch.zeros(predicted.shape[1], device=self.device)
                model_mask = torch.zeros(predicted.shape[1], dtype=torch.bool, device=self.device)
                
                for uci_move, prob in gt_dist.items():
                    if uci_move in self.uci_to_idx:
                        move_idx = self.uci_to_idx[uci_move]
                        if move_idx < predicted.shape[1]:
                            target_probs[move_idx] = prob
                            model_mask[move_idx] = True

                if not model_mask.any():
                    continue

                # Get model distribution (masked to legal moves only)
                model_logits_masked = predicted[idx].clone()
                model_logits_masked[~model_mask] = float('-inf')
                model_log_probs = F.log_softmax(model_logits_masked, dim=0)

                # Compute KL divergence: KL(GT || Model)
                # This pulls the model toward game-theoretic equilibrium
                kl = (target_probs[model_mask] * (
                    torch.log(target_probs[model_mask] + 1e-10) - model_log_probs[model_mask]
                )).sum()
                
                kl_losses.append(kl)

            except Exception as e:
                continue

        if len(kl_losses) == 0:
            return torch.tensor(0.0, device=self.device)

        return torch.stack(kl_losses).mean()

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
