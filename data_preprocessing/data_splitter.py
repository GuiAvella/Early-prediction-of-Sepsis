"""
data_splitter.py
────────────────
Import this module into your own Python code to split and concatenate
large CSV / Excel files without hitting GitHub's 100 MB file size limit.

Public API
──────────
    split_dataframe(df, file_path, max_mb, output_dir)  →  list[str]
    split_file(file_path, max_mb, output_dir)           →  list[str]
    concat_chunks(chunk_paths)                           →  pd.DataFrame
    concat_from_manifest(manifest_path)                  →  pd.DataFrame
    concat_from_dir(directory, pattern)                  →  pd.DataFrame

Example usage
─────────────
    from data_splitter import split_dataframe, concat_chunks

    # ── split ──────────────────────────────────────────
    import pandas as pd
    df = pd.read_csv("my_large_dataset.csv")

    chunk_paths = split_dataframe(df, "my_large_dataset.csv", max_mb=90)
    # → ['chunks/my_large_dataset_part001.csv',
    #    'chunks/my_large_dataset_part002.csv', ...]

    # ── concatenate back into one DataFrame ────────────
    full_df = concat_chunks(chunk_paths)

    # Or load straight from the manifest that was written during splitting:
    full_df = concat_from_manifest("chunks/my_large_dataset_manifest.txt")

    # Or scan a directory automatically:
    full_df = concat_from_dir("chunks")
"""

from __future__ import annotations

import math
import os
import glob
from typing import Optional

import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_format(path: str) -> str:
    """Return 'csv' or 'excel' based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return "csv"
    if ext in (".xlsx", ".xls", ".xlsm"):
        return "excel"
    raise ValueError(
        f"Unsupported file type '{ext}'. "
        "Only .csv and .xlsx / .xls / .xlsm are supported."
    )


def _human_size(byte_count: int) -> str:
    """Return a human-readable byte size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if byte_count < 1024:
            return f"{byte_count:.1f} {unit}"
        byte_count /= 1024
    return f"{byte_count:.1f} TB"


def _read_file(path: str, fmt: str) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame."""
    if fmt == "csv":
        return pd.read_csv(path)
    return pd.read_excel(path, engine="openpyxl")


def _write_file(df: pd.DataFrame, path: str, fmt: str) -> None:
    """Save a DataFrame as CSV or Excel."""
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_excel(path, index=False, engine="openpyxl")


def _write_manifest(
    output_dir: str,
    base_name: str,
    chunk_paths: list[str],
    total_rows: int,
    fmt: str,
) -> str:
    """Write a manifest file and return its path."""
    manifest_path = os.path.join(output_dir, f"{base_name}_manifest.txt")
    with open(manifest_path, "w") as f:
        f.write(f"# Manifest for {base_name}\n")
        f.write(f"# Total rows : {total_rows}\n")
        f.write(f"# Format     : {fmt}\n")
        f.write(f"# Chunks     : {len(chunk_paths)}\n\n")
        for p in chunk_paths:
            f.write(os.path.basename(p) + "\n")
    return manifest_path


# ═══════════════════════════════════════════════════════════════════════════════
# Public — Split
# ═══════════════════════════════════════════════════════════════════════════════

def split_dataframe(
    df: pd.DataFrame,
    file_path: str,
    max_mb: float = 90.0,
    output_dir: str = "chunks",
    verbose: bool = True,
) -> list[str]:
    """
    Split a DataFrame into chunk files that are each at most *max_mb* MB.

    Parameters
    ----------
    df          : The DataFrame to split.
    file_path   : Original file path — used only to derive the base name and
                  output format (.csv or .xlsx).
    max_mb      : Maximum size per chunk in megabytes (default 90).
    output_dir  : Directory where chunk files will be written (created if needed).
    verbose     : Print progress messages (default True).

    Returns
    -------
    list[str]   : Sorted list of chunk file paths that were written.

    Example
    -------
    >>> chunk_paths = split_dataframe(df, "sales_data.csv", max_mb=90)
    >>> print(chunk_paths)
    ['chunks/sales_data_part001.csv', 'chunks/sales_data_part002.csv']
    """
    fmt = _detect_format(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ext = ".csv" if fmt == "csv" else ".xlsx"
    os.makedirs(output_dir, exist_ok=True)

    # Estimate bytes per row from a temp file of the first 1 000 rows
    sample_path = os.path.join(output_dir, f"_sample_tmp{ext}")
    _write_file(df.head(1_000), sample_path, fmt)
    sample_bytes = os.path.getsize(sample_path)
    os.remove(sample_path)

    sample_rows = min(1_000, len(df))
    bytes_per_row = sample_bytes / sample_rows if sample_rows else 1
    max_bytes = max_mb * 1024 * 1024
    rows_per_chunk = max(1, int(max_bytes / bytes_per_row))
    total_rows = len(df)
    num_chunks = math.ceil(total_rows / rows_per_chunk)

    if verbose:
        print(f"📊  Rows          : {total_rows:,}")
        print(f"📏  Est. row size : {_human_size(int(bytes_per_row))}")
        print(f"🎯  Chunk limit   : {max_mb} MB  →  ~{rows_per_chunk:,} rows/chunk")
        print(f"📦  Chunks needed : {num_chunks}")

    chunk_paths: list[str] = []
    for i in range(num_chunks):
        start = i * rows_per_chunk
        end = min(start + rows_per_chunk, total_rows)
        chunk_df = df.iloc[start:end].reset_index(drop=True)

        chunk_name = f"{base_name}_part{i + 1:03d}{ext}"
        chunk_path = os.path.join(output_dir, chunk_name)
        _write_file(chunk_df, chunk_path, fmt)

        if verbose:
            size = _human_size(os.path.getsize(chunk_path))
            print(f"  ✔  {chunk_name}  ({end - start:,} rows, {size})")

        chunk_paths.append(chunk_path)

    manifest_path = _write_manifest(output_dir, base_name, chunk_paths, total_rows, fmt)

    if verbose:
        print(f"📋  Manifest      : {manifest_path}")
        print(f"\n🎉  Done! {num_chunks} chunk(s) saved to '{output_dir}'")

    return chunk_paths


def split_file(
    file_path: str,
    max_mb: float = 90.0,
    output_dir: str = "chunks",
    verbose: bool = True,
) -> list[str]:
    """
    Load a CSV / Excel file from disk and split it into chunks.

    Parameters
    ----------
    file_path   : Path to the source CSV or Excel file.
    max_mb      : Maximum size per chunk in megabytes (default 90).
    output_dir  : Directory where chunk files will be written.
    verbose     : Print progress messages (default True).

    Returns
    -------
    list[str]   : Sorted list of chunk file paths that were written.

    Example
    -------
    >>> chunk_paths = split_file("data/my_dataset.csv", max_mb=90)
    """
    if verbose:
        size = _human_size(os.path.getsize(file_path))
        print(f"📂  Loading {file_path}  ({size}) …")

    fmt = _detect_format(file_path)
    df = _read_file(file_path, fmt)
    return split_dataframe(df, file_path, max_mb=max_mb, output_dir=output_dir, verbose=verbose)


# ═══════════════════════════════════════════════════════════════════════════════
# Public — Concatenate
# ═══════════════════════════════════════════════════════════════════════════════

def concat_chunks(
    chunk_paths: list[str],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Concatenate a list of chunk file paths into a single DataFrame.

    Parameters
    ----------
    chunk_paths : Ordered list of file paths (as returned by split_dataframe /
                  split_file, or built manually).
    verbose     : Print progress messages (default True).

    Returns
    -------
    pd.DataFrame : The combined DataFrame with a reset index.

    Example
    -------
    >>> df = concat_chunks(["chunks/data_part001.csv", "chunks/data_part002.csv"])
    >>> print(df.shape)
    (500000, 12)
    """
    if not chunk_paths:
        raise ValueError("chunk_paths is empty — nothing to concatenate.")

    if verbose:
        print(f"🔄  Concatenating {len(chunk_paths)} chunk(s) …")

    dfs: list[pd.DataFrame] = []
    for path in chunk_paths:
        fmt = _detect_format(path)
        chunk = _read_file(path, fmt)
        dfs.append(chunk)
        if verbose:
            print(f"  ✔  {os.path.basename(path)}  ({len(chunk):,} rows)")

    combined = pd.concat(dfs, ignore_index=True)

    if verbose:
        print(f"\n✅  Combined DataFrame: {len(combined):,} rows × {len(combined.columns)} columns")

    return combined


def concat_from_manifest(
    manifest_path: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read a manifest file produced by split_dataframe / split_file and
    concatenate all listed chunks into a single DataFrame.

    Parameters
    ----------
    manifest_path : Path to the *_manifest.txt file.
    verbose       : Print progress messages (default True).

    Returns
    -------
    pd.DataFrame  : The fully reconstructed DataFrame.

    Example
    -------
    >>> df = concat_from_manifest("chunks/my_dataset_manifest.txt")
    """
    dir_path = os.path.dirname(os.path.abspath(manifest_path))
    chunk_paths: list[str] = []

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                chunk_paths.append(os.path.join(dir_path, line))

    if not chunk_paths:
        raise FileNotFoundError(f"No chunk entries found in manifest: {manifest_path}")

    if verbose:
        print(f"📋  Manifest loaded: {manifest_path}")

    return concat_chunks(chunk_paths, verbose=verbose)


def concat_from_dir(
    directory: str,
    pattern: str = "*_part*.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Scan a directory for chunk files matching *pattern* and concatenate them.

    Files are sorted alphabetically, so the _part001, _part002 … naming
    convention produced by the splitter gives the correct order automatically.

    Parameters
    ----------
    directory   : Folder to scan.
    pattern     : Glob pattern (default ``*_part*.csv``).
                  Use ``*_part*.xlsx`` for Excel chunks.
    verbose     : Print progress messages (default True).

    Returns
    -------
    pd.DataFrame : The combined DataFrame.

    Example
    -------
    >>> df = concat_from_dir("chunks")
    >>> df = concat_from_dir("chunks", pattern="*_part*.xlsx")
    """
    search = os.path.join(directory, pattern)
    chunk_paths = sorted(glob.glob(search))

    if not chunk_paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{directory}'."
        )

    if verbose:
        print(f"📁  Found {len(chunk_paths)} file(s) in '{directory}' matching '{pattern}'")

    return concat_chunks(chunk_paths, verbose=verbose)
