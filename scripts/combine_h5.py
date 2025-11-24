import argparse
from pathlib import Path
import tables as tb


def table_nrows(table):
    """Safely get the number of rows in a PyTables table."""
    # new versions
    if hasattr(table, "nrows"):
        return table.nrows
    # fallback
    try:
        return table.shape[0]
    except Exception:
        return len(table)


def combine_h5(chunks_dir: Path, output_path: Path, h5_name_pattern: str = "chunk_*.h5"):
    h5_files = sorted(chunks_dir.glob(f"chunk_*/{h5_name_pattern}"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files in {chunks_dir}/chunk_*/")

    print(f"[combine] Found {len(h5_files)} H5 files:")
    for f in h5_files:
        print(f"  - {f}")

    if output_path.exists():
        print(f"[combine] Output {output_path} exists; removing.")
        output_path.unlink()

    h5_out = None
    out_data = None
    out_encoded = None

    try:
        for idx, h5_file in enumerate(h5_files):
            print(f"[combine] Reading {h5_file}...")
            h5_in = tb.open_file(h5_file, mode="r")
            data_in = h5_in.root.data
            encoded_in = h5_in.root.encoded_data

            n_data_rows = table_nrows(data_in)
            n_encoded_rows = table_nrows(encoded_in)

            if h5_out is None:
                # first file -> initialize output
                h5_out = tb.open_file(output_path, mode="w", title="combined data")

                out_data = h5_out.create_table(
                    "/", "data", data_in.description, expectedrows=n_data_rows * len(h5_files)
                )
                out_encoded = h5_out.create_table(
                    "/", "encoded_data", encoded_in.description, expectedrows=n_encoded_rows * len(h5_files)
                )

            # append rows
            out_data.append(data_in[:])
            out_encoded.append(encoded_in[:])

            print(f"[combine] Appended {n_data_rows} rows from {h5_file}.")

            h5_in.close()

        out_data.flush()
        out_encoded.flush()

        print(f"[combine] Done. Combined H5 saved at {output_path}")

    finally:
        if h5_out is not None:
            h5_out.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    combine_h5(
        chunks_dir=Path(args.chunks_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
