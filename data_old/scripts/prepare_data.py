# scripts/prepare_data.py

import os
import argparse
import tables as tb
from tqdm import tqdm

from chess_transformers.configs import import_config
from chess_transformers.data.utils import encode, parse_fen
from chess_transformers.data.levels import TURN, PIECES, SQUARES, UCI_MOVES, BOOL


def prepare_data(
    data_folder,
    h5_file,
    max_move_sequence_length,
    expected_rows,
    val_split_fraction=None,
):
    """
    Transform raw data (FENs and moves) into a form that can be consumed
    by a neural network.

    Stored in two tables in an H5 file (human-readable and encoded).
    """

    # Get names of files/chunks containing moves and FENs
    moves_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".moves")])
    fens_files = sorted([f for f in os.listdir(data_folder) if f.endswith(".fens")])
    assert len(moves_files) == len(fens_files)
    print("\nMoves and FENs are stored in %d chunk(s).\n" % len(moves_files))

    # Create table description for H5 file
    class ChessTable(tb.IsDescription):
        board_position = tb.StringCol(64)
        turn = tb.StringCol(1)
        white_kingside_castling_rights = tb.BoolCol()
        white_queenside_castling_rights = tb.BoolCol()
        black_kingside_castling_rights = tb.BoolCol()
        black_queenside_castling_rights = tb.BoolCol()
        moves = tb.StringCol(
            shape=(max_move_sequence_length + 1), itemsize=8, dflt="<pad>"
        )
        length = tb.Int8Col()
        from_square = tb.StringCol(2)
        to_square = tb.StringCol(2)

    class EncodedChessTable(tb.IsDescription):
        board_position = tb.Int8Col(shape=(64))
        turn = tb.Int8Col()
        white_kingside_castling_rights = tb.Int8Col()
        white_queenside_castling_rights = tb.Int8Col()
        black_kingside_castling_rights = tb.Int8Col()
        black_queenside_castling_rights = tb.Int8Col()
        moves = tb.Int16Col(shape=(max_move_sequence_length + 1))
        length = tb.Int8Col()
        from_square = tb.Int8Col()
        to_square = tb.Int8Col()

    # Delete H5 file if it already exists; start anew
    h5_path = os.path.join(data_folder, h5_file)
    if os.path.exists(h5_path):
        os.remove(h5_path)

    # Create new H5 file
    h5_file_handle = tb.open_file(h5_path, mode="w", title="data file")

    # Create table in H5 file
    table = h5_file_handle.create_table("/", "data", ChessTable, expectedrows=expected_rows)

    # Create encoded table in H5 file
    encoded_table = h5_file_handle.create_table(
        "/", "encoded_data", EncodedChessTable, expectedrows=table.nrows
    )

    # Create pointer to next row in these tables
    row = table.row
    encoded_row = encoded_table.row

    # Keep track of row numbers where new games begin
    new_game_index = 0
    new_game_indices = []

    # Keep track of errors
    n_wrong_results = 0
    n_move_fen_mismatches = 0

    # Iterate through chunks
    for i in range(len(moves_files)):
        print("Now reading %s and %s...\n" % (moves_files[i], fens_files[i]))

        all_moves = open(os.path.join(data_folder, moves_files[i]), "r").read()
        all_fens = open(os.path.join(data_folder, fens_files[i]), "r").read()
        all_moves = all_moves.split("\n\n")[:-1]
        all_fens = all_fens.split("\n\n")[:-1]
        assert len(all_moves) == len(all_fens)
        print("There are %d games.\n" % len(all_moves))

        for j in tqdm(range(len(all_moves)), desc="Adding rows to table"):
            moves = all_moves[j].split("\n")
            result = moves.pop(-1)
            moves = [move.lower() for move in moves]
            moves.append("<loss>")  # EOS-like token
            fens = all_fens[j].split("\n")

            if len(moves) != len(fens):
                n_move_fen_mismatches += 1
                continue

            start_index = 0 if result == "1-0" else 1

            if len(moves) % 2 != start_index:
                n_wrong_results += 1
                continue

            for k in range(start_index, len(moves), 2):
                t, b, wk, wq, bk, bq = parse_fen(fens[k])
                ms = (
                    ["<move>"]
                    + moves[k : k + max_move_sequence_length]
                    + ["<pad>"] * ((k + max_move_sequence_length) - len(moves))
                )
                msl = len([m for m in ms if m != "<pad>"]) - 1

                row["board_position"] = b
                encoded_row["board_position"] = encode(b, PIECES)

                row["turn"] = t
                encoded_row["turn"] = encode(t, TURN)

                row["white_kingside_castling_rights"] = wk
                row["white_queenside_castling_rights"] = wq
                row["black_kingside_castling_rights"] = bk
                row["black_queenside_castling_rights"] = bq
                encoded_row["white_kingside_castling_rights"] = encode(wk, BOOL)
                encoded_row["white_queenside_castling_rights"] = encode(wq, BOOL)
                encoded_row["black_kingside_castling_rights"] = encode(bk, BOOL)
                encoded_row["black_queenside_castling_rights"] = encode(bq, BOOL)

                row["moves"] = ms
                encoded_row["moves"] = encode(ms, UCI_MOVES)

                row["length"] = msl
                encoded_row["length"] = msl

                row["from_square"] = ms[1][:2]
                encoded_row["from_square"] = encode(ms[1][:2], SQUARES)
                row["to_square"] = ms[1][2:4]
                encoded_row["to_square"] = encode(ms[1][2:4], SQUARES)

                row.append()
                encoded_row.append()

            new_game_index += k + 1
            new_game_indices.append(new_game_index)

        table.flush()
        print("\nA total of %d datapoints have been saved to disk.\n" % table.nrows)

    print("...done.\n")

    if n_move_fen_mismatches > 0:
        print(
            "NOTE: %d game(s) excluded because number of moves and FENs did not match.\n"
            % n_move_fen_mismatches
        )
    if n_wrong_results > 0:
        print(
            "NOTE: %d game(s) (%.2f percent) excluded that had the wrong result recorded.\n"
            % (
                n_wrong_results,
                n_wrong_results
                * 100.0
                / (len(new_game_indices) + n_wrong_results + n_move_fen_mismatches),
            )
        )

    if val_split_fraction is not None:
        val_split_index = None
        for i in new_game_indices:
            if val_split_index is None:
                if i / table.nrows >= val_split_fraction:
                    val_split_index = i
            else:
                break
        print(
            "The training set will start at index 0, the validation set at index %d (%2.6f%%).\n"
            % (val_split_index, 100.0 * val_split_index / table.nrows)
        )
        encoded_table.attrs.val_split_index = val_split_index

    h5_file_handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    parser.add_argument(
        "--data-folder",
        type=str,
        default=None,
        help="Override DATA_FOLDER from config (e.g. a chunk dir).",
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        default=None,
        help="Override H5_FILE from config (file name only, stored inside data_folder).",
    )
    args = parser.parse_args()

    CONFIG = import_config(args.config_name)

    data_folder = args.data_folder or CONFIG.DATA_FOLDER
    h5_file = args.h5_file or CONFIG.H5_FILE

    prepare_data(
        data_folder=data_folder,
        h5_file=h5_file,
        max_move_sequence_length=CONFIG.MAX_MOVE_SEQUENCE_LENGTH,
        expected_rows=CONFIG.EXPECTED_ROWS,
        val_split_fraction=CONFIG.VAL_SPLIT_FRACTION,
    )


if __name__ == "__main__":
    main()
