# scripts/process_chunk.py
# # One-time: make scripts a package
# touch scripts/__init__.py

# # Make build_dataset.sh executable if it isn't
# chmod +x scripts/build_dataset.sh
# cd final_project

# python -m scripts.process_chunk \
#   --big-pgn data/huge.pgn \
#   --chunk-games 50000 \
#   --config-name custom_config
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence, Optional


def run(cmd: Sequence[str], cwd: Optional[Path] = None, quiet: bool = True):
    """Run a shell command, optionally suppressing stdout."""
    pretty = " ".join(cmd)
    print(f"\n>> {pretty}")
    try:
        subprocess.run(
            cmd,
            cwd=str(cwd) if cwd is not None else None,
            check=True,
            stdout=subprocess.DEVNULL if quiet else None,
            stderr=None,
        )
    except subprocess.CalledProcessError as e:
        print(f"Command failed ({e.returncode}): {pretty}", file=sys.stderr)
        sys.exit(e.returncode)


def load_state(state_path: Path) -> int:
    if not state_path.exists():
        return 1
    with state_path.open("r") as f:
        data = json.load(f)
    return int(data.get("next_firstgame", 1))


def save_state(state_path: Path, next_firstgame: int):
    state_path.write_text(json.dumps({"next_firstgame": next_firstgame}, indent=2))


def patch_build_script_keep_filtered(build_script_path: Path):
    """
    In the copied build_dataset.sh, change:
        rm -f filtered_games.pgn shuffled_filtered_games.pgn
    to:
        rm -f shuffled_filtered_games.pgn
    so filtered_games.pgn survives for analysis.
    """
    text = build_script_path.read_text()
    old = "rm -f filtered_games.pgn shuffled_filtered_games.pgn"
    new = "rm -f shuffled_filtered_games.pgn"
    if old in text:
        text = text.replace(old, new)
        build_script_path.write_text(text)
        print("Patched build_dataset.sh to keep filtered_games.pgn.")
    else:
        print("Warning: could not patch build_dataset.sh (rm line not found).")


def main():
    parser = argparse.ArgumentParser(
        description="Process huge PGN in chunks: build_dataset -> H5 -> stats."
    )
    parser.add_argument(
        "--big-pgn",
        required=True,
        help="Path to the huge PGN file (e.g., data/huge.pgn).",
    )
    parser.add_argument(
        "--chunk-games",
        type=int,
        default=100_000,
        help="Number of games to extract for each chunk.",
    )
    parser.add_argument(
        "--firstgame",
        type=int,
        default=None,
        help="Optional override for starting game index (1-based).",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="chunk_state.json",
        help="Path to state JSON file for resuming.",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="custom_config",
        help="Config name to pass to scripts.prepare_data.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    scripts_dir = repo_root / "scripts"
    data_dir = repo_root / "data"
    chunks_dir = data_dir / "chunks"
    h5_dir = data_dir / "h5"
    stats_dir = data_dir / "stats"

    chunks_dir.mkdir(parents=True, exist_ok=True)
    h5_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    big_pgn = Path(args.big_pgn).resolve()
    if not big_pgn.exists():
        print(f"Huge PGN not found: {big_pgn}", file=sys.stderr)
        sys.exit(1)

    build_script_src = scripts_dir / "build_dataset.sh"
    if not build_script_src.exists():
        print(f"build_dataset.sh not found at {build_script_src}", file=sys.stderr)
        sys.exit(1)

    state_path = repo_root / args.state_file
    if args.firstgame is not None:
        firstgame = args.firstgame
    else:
        firstgame = load_state(state_path)

    chunk_games = args.chunk_games
    lastgame = firstgame + chunk_games - 1

    print(f"=== Processing games {firstgame} .. {lastgame} from {big_pgn.name} ===")

    # Chunk directory under data/chunks
    chunk_name = f"chunk_{firstgame}_{lastgame}"
    chunk_dir = chunks_dir / chunk_name
    if chunk_dir.exists():
        print(f"Chunk dir {chunk_dir} already exists; refusing to overwrite.", file=sys.stderr)
        sys.exit(1)
    chunk_dir.mkdir()

    # Copy and patch build_dataset.sh for this chunk
    chunk_build_script = chunk_dir / "build_dataset.sh"
    shutil.copy2(build_script_src, chunk_build_script)
    chunk_build_script.chmod(chunk_build_script.stat().st_mode | 0o111)
    patch_build_script_keep_filtered(chunk_build_script)

    # 1) Extract raw slice from the huge PGN
    chunk_pgn = chunk_dir / "chunk_raw.pgn"
    extract_cmd = [
        "pgn-extract",
        "-s",  # silent
        "--firstgame",
        str(firstgame),
        "--stopafter",
        str(chunk_games),
        "-o",
        str(chunk_pgn),
        str(big_pgn),
    ]
    run(extract_cmd)

    if not chunk_pgn.exists() or chunk_pgn.stat().st_size == 0:
        print("No games extracted (chunk_raw.pgn is empty). Probably at end of file.")
        save_state(state_path, firstgame + chunk_games)
        return

    print(f"Extracted chunk PGN at: {chunk_pgn}")

    # 2) Run build_dataset.sh in the chunk dir
    run(["bash", "build_dataset.sh"], cwd=chunk_dir, quiet=True)

    # 3) Run prepare_data.py to build H5 into data/h5/
    h5_file_name = f"{chunk_name}.h5"
    h5_target_path = h5_dir / h5_file_name

    print("\n[prepare_data] Building H5 for this chunk...")
    run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_data",
            args.config_name,
            "--data-folder",
            str(chunk_dir),
            "--h5-file",
            h5_file_name,
        ],
        cwd=repo_root,
        quiet=False,
    )

    # Move H5 file from chunk_dir to data/h5/ (prepare_data writes inside data_folder)
    chunk_h5_path = chunk_dir / h5_file_name
    if not chunk_h5_path.exists():
        print(f"ERROR: Expected H5 at {chunk_h5_path}, but it does not exist.", file=sys.stderr)
        sys.exit(1)
    shutil.move(str(chunk_h5_path), str(h5_target_path))
    print(f"[prepare_data] H5 stored at: {h5_target_path}")

    # 4) Analyze filtered_games.pgn (if present) and store stats into data/stats/
    filtered_pgn = chunk_dir / "filtered_games.pgn"
    if filtered_pgn.exists():
        out_prefix = stats_dir / chunk_name
        print(f"\n[analyze_games] Analyzing {filtered_pgn} ...")
        run(
            [
                sys.executable,
                "-m",
                "scripts.analyze_games",
                "--pgn",
                str(filtered_pgn),
                "--out-prefix",
                str(out_prefix),
            ],
            cwd=repo_root,
            quiet=False,
        )

        # Append this chunk's filtered games to global data/filtered.pgn
        global_filtered = data_dir / "filtered.pgn"
        with filtered_pgn.open("rb") as src, global_filtered.open("ab") as dst:
            shutil.copyfileobj(src, dst)
        print(f"Appended filtered games to {global_filtered}")
    else:
        print("Warning: filtered_games.pgn not found; skipping analysis and global append.")

    # 5) Update state for next run
    next_firstgame = firstgame + chunk_games
    save_state(state_path, next_firstgame)
    print(
        f"\n=== Done with chunk {firstgame}..{lastgame}. "
        f"Next run will start at game {next_firstgame}. ==="
    )


if __name__ == "__main__":
    main()
