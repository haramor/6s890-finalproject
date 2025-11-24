import argparse
import json
import subprocess
import sys
from pathlib import Path
import shutil

import chess.pgn

# --- Paths relative to this script -----------------------------------------

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent  # /workspace/6s890-finalproject
CHESS_REPO = ROOT / "chess-transformers"

# Make chess_transformers importable
sys.path.insert(0, str(CHESS_REPO))

from chess_transformers.configs import import_config  # type: ignore
from chess_transformers.data.prep import prepare_data  # type: ignore


def load_progress(progress_path: Path) -> dict:
    if progress_path.exists():
        return json.loads(progress_path.read_text())
    # fresh progress
    return {
        "games_processed": 0,
        "next_chunk_index": 1,
        "byte_offset": 0,  # new: track file byte offset
    }

def save_progress(progress_path: Path, progress: dict) -> None:
    progress_path.write_text(json.dumps(progress, indent=2))


def stream_games(pgn_path: Path, start_offset: int = 0, skip_n: int = 0):
    """
    Generator over games in a PGN file.

    If start_offset > 0, seek to that byte position in the file and start
    reading games from there.

    Otherwise, skip the first skip_n games by parsing them.
    Yields (game, offset_after_game) tuples, where offset_after_game is the
    file position immediately after reading that game.
    """
    with pgn_path.open("r", encoding="utf-8", errors="ignore") as f:
        if start_offset and start_offset > 0:
            # Fast resume: jump straight to the stored byte offset
            f.seek(start_offset)
        else:
            # Slow path: skip already-processed games by parsing them
            for _ in range(skip_n):
                g = chess.pgn.read_game(f)
                if g is None:
                    return  # EOF reached early

        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            # We record the file position *after* reading this game
            yield game, f.tell()


def write_chunk_pgn(games, out_path: Path, max_games: int) -> tuple[int, int]:
    """
    Write up to max_games from iterator `games` into out_path.
    `games` should yield (game, offset_after_game) tuples.

    Returns:
        (num_games_written, last_offset_after_game)
    """
    count = 0
    last_offset = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for game, offset in games:
            if count >= max_games:
                break
            f_out.write(str(game))
            f_out.write("\n\n")
            count += 1
            last_offset = offset
    return count, last_offset


def run_build_dataset(chunk_dir: Path) -> None:
    """
    Run build_dataset.sh inside chunk_dir.
    Assumes build_dataset.sh lives in ROOT/scripts.
    """
    # Copy the user's tags.txt into the chunk if present
    user_tags = ROOT / "scripts" / "tags.txt"    # or wherever your tags live
    if user_tags.exists():
        shutil.copy(user_tags, chunk_dir / "tags.txt")
    build_script = ROOT / "scripts" / "build_dataset.sh"
    if not build_script.exists():
        raise FileNotFoundError(f"build_dataset.sh not found at {build_script}")

    subprocess.run(
        ["bash", str(build_script)],
        cwd=str(chunk_dir),
        check=True,
    )


def run_prepare_data(
    chunk_dir: Path,
    h5_name: str,
    config_name: str,
) -> None:
    """
    Call chess_transformers.data.prep.prepare_data() directly for this chunk.
    """
    CONFIG = import_config(config_name)

    # Use hyperparameters from config, but override data_folder & h5_file
    data_folder = str(chunk_dir)
    h5_file = h5_name

    expected_rows = getattr(CONFIG, "EXPECTED_ROWS", 500_000)
    max_seq_len = getattr(CONFIG, "MAX_MOVE_SEQUENCE_LENGTH", 80)
    val_split_fraction = getattr(CONFIG, "VAL_SPLIT_FRACTION", None)

    print(
        f"[prepare_data] chunk_dir={data_folder}, h5_file={h5_file}, "
        f"max_seq_len={max_seq_len}, expected_rows={expected_rows}, "
        f"val_split_fraction={val_split_fraction}"
    )

    prepare_data(
        data_folder=data_folder,
        h5_file=h5_file,
        max_move_sequence_length=max_seq_len,
        expected_rows=expected_rows,
        val_split_fraction=val_split_fraction,
    )


def run_analyze(chunk_dir: Path) -> None:
    """
    Run scripts/analyze_games.py on this chunk's filtered_games.pgn if present.
    """
    filtered = chunk_dir / "filtered_games.pgn"
    if not filtered.exists():
        print(f"[analyze] No filtered_games.pgn in {chunk_dir}, skipping analysis.")
        return

    analyze_script = ROOT / "scripts" / "analyze_games.py"
    if not analyze_script.exists():
        print("[analyze] analyze_games.py not found, skipping analysis.")
        return
    out_json = chunk_dir / "stats.json"
    print(f"[analyze] Running analysis -> {out_json}")
    subprocess.run(
        [
            sys.executable,
            str(analyze_script),
            str(filtered),
            "--save-json",
            str(out_json),
        ],
        check=True,
    )

    #print(f"[analyze] Analyzing {filtered}")
    #subprocess.run(
    #    [sys.executable, str(analyze_script), str(filtered)],
    #    cwd=str(ROOT),
    #    check=True,
    #)


def process_chunks(
    huge_pgn: Path,
    chunks_dir: Path,
    chunk_size: int,
    config_name: str,
    max_chunks: int | None = None,
) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    progress_path = chunks_dir / "progress.json"
    progress = load_progress(progress_path)

    games_processed = int(progress.get("games_processed", 0))
    next_chunk_index = int(progress.get("next_chunk_index", 1))
    byte_offset = int(progress.get("byte_offset", 0))

    print(
        f"[info] Starting from games_processed={games_processed}, "
        f"next_chunk_index={next_chunk_index}, byte_offset={byte_offset}"
    )

    # If we have a byte_offset, use it (fast resume).
    # Otherwise, fall back to skipping games by parsing them once.
    if byte_offset > 0:
        game_iter = stream_games(huge_pgn, start_offset=byte_offset, skip_n=0)
    else:
        game_iter = stream_games(huge_pgn, start_offset=0, skip_n=games_processed)

    chunks_done = 0

    while True:
        if max_chunks is not None and chunks_done >= max_chunks:
            print(f"[info] Reached max_chunks={max_chunks}, stopping.")
            break

        chunk_id = next_chunk_index
        chunk_dir = chunks_dir / f"chunk_{chunk_id:05d}"
        raw_pgn = chunk_dir / "raw_games.pgn"
        h5_name = f"chunk_{chunk_id:05d}.h5"
        h5_path = chunk_dir / h5_name

        if h5_path.exists():
            # Already processed this chunk; skip but still account games_processed
            print(f"[info] {h5_path} already exists; skipping chunk {chunk_id}.")
            # NOTE: For simplicity, we assume progress.json is correct and aligned
            next_chunk_index += 1
            chunks_done += 1
            continue

        print(f"[chunk {chunk_id}] Writing up to {chunk_size} games to {raw_pgn}...")
        n_games_chunk, last_offset = write_chunk_pgn(
            game_iter, raw_pgn, max_games=chunk_size
        )

        if n_games_chunk == 0:
            print("[info] No more games to process. Done.")
            break

        print(
            f"[chunk {chunk_id}] Wrote {n_games_chunk} games. "
            f"Last file offset={last_offset}"
        )
        # Run build_dataset.sh inside this chunk directory
        print(f"[chunk {chunk_id}] Running build_dataset.sh...")
        run_build_dataset(chunk_dir)

        # Run prepare_data to create H5 for this chunk
        print(f"[chunk {chunk_id}] Running prepare_data -> {h5_name}...")
        run_prepare_data(chunk_dir, h5_name=h5_name, config_name=config_name)

        # Optional: run analysis on this chunk
        print(f"[chunk {chunk_id}] Running analyze_games.py...")
        run_analyze(chunk_dir)
       
        # Only now update progress (so a crash mid-chunk doesn’t “skip” games)
        games_processed += n_games_chunk
        next_chunk_index += 1
        chunks_done += 1

        progress["games_processed"] = games_processed
        progress["next_chunk_index"] = next_chunk_index
        progress["byte_offset"] = last_offset
        save_progress(progress_path, progress)

        print(
            f"[chunk {chunk_id}] Done. Total games_processed={games_processed}. "
            f"Progress saved to {progress_path}."
        )

    print("[info] All done / no more games.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--huge-pgn",
        type=str,
        default=str(ROOT / "data" / "huge.pgn"),
        help="Path to the big PGN file.",
    )
    parser.add_argument(
        "--chunks-dir",
        type=str,
        default=str(ROOT / "data" / "chunks"),
        help="Directory to store per-chunk data.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of games per chunk.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="custom_config",
        help="Name of chess_transformers config (e.g., custom_config).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional max number of chunks to process this run.",
    )
    args = parser.parse_args()

    process_chunks(
        huge_pgn=Path(args.huge_pgn),
        chunks_dir=Path(args.chunks_dir),
        chunk_size=args.chunk_size,
        config_name=args.config_name,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
