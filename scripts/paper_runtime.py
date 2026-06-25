#!/usr/bin/env python3
"""Shared setup for publication experiment and analysis scripts."""

from __future__ import annotations

import os
import re
import shutil
import sys
import types
from pathlib import Path


def _force_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target, target_is_directory=target.is_dir())


def _remove_existing_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _symlink_entry(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        _remove_existing_path(link)
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def prepare_source_data_compat_dir(source_dir: Path, compat_dir: Path) -> None:
    """Create notebook-compatible source-data links.

    The Zenodo-facing source bundle is allowed to flatten small legacy wrapper
    directories. Notebook-derived code still expects the original paths for
    KRAS and ProteinGym caches, so this creates a local compatibility view.
    """

    source_dir = Path(source_dir).resolve()
    compat_dir = Path(compat_dir)
    if compat_dir.is_symlink() or compat_dir.exists():
        _remove_existing_path(compat_dir)
    compat_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(source_dir.iterdir()):
        if child.name == "kras_genetic_arch" and not (child / "RESULTS").exists():
            kras_dir = compat_dir / "kras_genetic_arch"
            results_dir = kras_dir / "RESULTS"
            results_dir.mkdir(parents=True, exist_ok=True)
            for item in sorted(child.iterdir()):
                _symlink_entry(item, kras_dir / item.name)
                if item.is_file():
                    _symlink_entry(item, results_dir / item.name)
            continue

        if child.name == "protein_gym" and not (child / "DMS_assays_substitutions").exists():
            protein_gym_dir = compat_dir / "protein_gym"
            dms_dir = protein_gym_dir / "DMS_assays_substitutions"
            dms_dir.mkdir(parents=True, exist_ok=True)
            for item in sorted(child.iterdir()):
                _symlink_entry(item, protein_gym_dir / item.name)
                if item.is_file():
                    _symlink_entry(item, dms_dir / item.name)
            continue

        _symlink_entry(child, compat_dir / child.name)


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _looks_like_source_datasets(path: Path) -> bool:
    expected_children = [
        "megascale_folding",
        "combinatorial_core",
        "kras_genetic_arch",
        "lycov_combination_antibodies",
        "protein_stability_arch",
    ]
    return path.is_dir() and any((path / child).exists() for child in expected_children)


ALISIM_FASTA_SUFFIXES = {".fa", ".fasta", ".fas", ".faa"}


def _is_aligned_alisim_fasta(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.is_file()
        and path.suffix.lower() in ALISIM_FASTA_SUFFIXES
        and "unaligned" not in name
    )


def _looks_like_alisim_results(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(_is_aligned_alisim_fasta(candidate) for candidate in path.rglob("*"))


def _discover_aligned_alisim_fastas(path: Path) -> list[Path]:
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(candidate for candidate in path.rglob("*") if _is_aligned_alisim_fasta(candidate))


def _replicate_number_from_fasta(path: Path, fallback_index: int) -> int:
    stem = Path(path).stem
    match = re.search(r"(?:replicate|rep|sim)[_-]?0*([0-9]+)", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"([0-9]+)", stem)
    if match:
        return int(match.group(1))
    return fallback_index


def _prepare_alisim_compat_dir(source_dir: Path, compat_dir: Path) -> None:
    """Create canonical replicate_XXX.fa links for notebook-derived code."""

    files = _discover_aligned_alisim_fastas(source_dir)
    if not files:
        raise FileNotFoundError(f"No aligned AliSim FASTA files found in {source_dir}")

    if compat_dir.is_symlink() or compat_dir.exists():
        if compat_dir.resolve() == source_dir.resolve() and all(path.name.startswith("replicate_") for path in files):
            return
        if compat_dir.is_symlink() or compat_dir.is_file():
            compat_dir.unlink()
        else:
            for child in compat_dir.iterdir():
                if child.is_dir() and not child.is_symlink():
                    import shutil

                    shutil.rmtree(child)
                else:
                    child.unlink()
    compat_dir.mkdir(parents=True, exist_ok=True)

    used_names = set()
    for index, fasta in enumerate(files, start=1):
        rep_num = _replicate_number_from_fasta(fasta, index)
        name = f"replicate_{rep_num:03d}.fa"
        if name in used_names:
            name = f"replicate_{index:03d}.fa"
        used_names.add(name)
        link = compat_dir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(fasta.resolve())


def _resolve_data_dir(
    *,
    label: str,
    env_var: str | None,
    candidates: list[Path],
    predicate,
) -> Path:
    """Resolve a data directory across publication and project-record layouts."""

    ordered_candidates = []
    if env_var and os.environ.get(env_var):
        ordered_candidates.append(Path(os.environ[env_var]))
    ordered_candidates.extend(candidates)

    checked = []
    for path in _unique_paths(ordered_candidates):
        checked.append(path)
        if predicate(path):
            return path

    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        f"Could not locate {label}. Checked:\n{checked_text}\n\n"
        "Unpack the publication data bundle at the repository root, or set the "
        f"{env_var} environment variable to the correct directory."
    )


def resolve_publication_data_dirs(project_root: Path) -> dict[str, Path]:
    """Resolve source-data and AliSim directories accepted by this repo."""

    data_files = _resolve_data_dir(
        label="source datasets",
        env_var="PAPER_DATA_FILES",
        candidates=[
            project_root / "data" / "source_datasets",
            project_root / "data" / "external" / "graph_ruggedness_de" / "data_files",
            project_root / "data_files",
        ],
        predicate=_looks_like_source_datasets,
    )
    alisim_results = _resolve_data_dir(
        label="AliSim replicate results",
        env_var="PAPER_ALISIM_RESULTS",
        candidates=[
            project_root / "data" / "alisim_results",
            project_root / "data" / "raw" / "alisim_results",
            project_root / "alisim_results",
        ],
        predicate=_looks_like_alisim_results,
    )
    return {
        "data_files": data_files,
        "alisim_results": alisim_results,
    }


def _mkdir_parent_for_output(path_like: object) -> None:
    if isinstance(path_like, (str, os.PathLike)):
        parent = Path(path_like).expanduser().parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)


def _patch_matplotlib_savefig() -> None:
    try:
        import matplotlib.figure as mpl_figure
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not getattr(mpl_figure.Figure.savefig, "_paper_mkdir_parent", False):
        original_figure_savefig = mpl_figure.Figure.savefig

        def figure_savefig(self, fname, *args, **kwargs):
            _mkdir_parent_for_output(fname)
            return original_figure_savefig(self, fname, *args, **kwargs)

        figure_savefig._paper_mkdir_parent = True
        mpl_figure.Figure.savefig = figure_savefig

    if not getattr(plt.savefig, "_paper_mkdir_parent", False):
        original_pyplot_savefig = plt.savefig

        def pyplot_savefig(*args, **kwargs):
            if args:
                _mkdir_parent_for_output(args[0])
            elif "fname" in kwargs:
                _mkdir_parent_for_output(kwargs["fname"])
            return original_pyplot_savefig(*args, **kwargs)

        pyplot_savefig._paper_mkdir_parent = True
        plt.savefig = pyplot_savefig


def find_project_root(start: str | os.PathLike[str]) -> Path:
    """Find the repository root from a script or directory path."""

    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    for path in [current, *current.parents]:
        if (path / "README.md").is_file() and (path / "experiments").is_dir() and (path / "analyses").is_dir():
            return path
    raise FileNotFoundError(f"Could not locate repository root from {start!s}")


def prepare_native_experiment(script_file: str | os.PathLike[str]) -> dict[str, Path]:
    """Create the path/runtime context used by publication experiment scripts."""

    script_dir = Path(script_file).resolve().parent
    project_root = find_project_root(script_dir)
    output_dir = Path(os.environ.get("PAPER_OUTPUT_DIR", script_dir / "outputs")).resolve()
    work_dir = Path(os.environ.get("PAPER_WORKDIR", script_dir / "work")).resolve()
    source_root = script_dir / "source"
    notebook_dir = source_root / "figure_notebooks_rev"
    processed_dir = Path(
        os.environ.get("PAPER_PROCESSED_DIR", project_root / "data" / "processed")
    ).resolve()
    data_dirs = resolve_publication_data_dirs(project_root)
    data_files = data_dirs["data_files"]
    runtime_data_files = data_files
    alisim_results = data_dirs["alisim_results"]

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "si_figures").mkdir(parents=True, exist_ok=True)

    if os.environ.get("PAPER_RUN_MODE") != "node_label_permutation_null":
        runtime_data_files = source_root / "data_files"
        prepare_source_data_compat_dir(data_files, runtime_data_files)
        if script_dir.name == "figure-1_sparse-phylogenetic-diffusion-scale-validation":
            _prepare_alisim_compat_dir(alisim_results, source_root / "alisim_results")
        else:
            _force_symlink(alisim_results, source_root / "alisim_results")
        _force_symlink(output_dir / "figures", source_root / "figures")
        _force_symlink(output_dir / "si_figures", source_root / "si_figures")
    _patch_matplotlib_savefig()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["PAPER_PROJECT_ROOT"] = str(project_root)
    os.environ["PAPER_OUTPUT_DIR"] = str(output_dir)
    os.environ["PAPER_PROCESSED_DIR"] = str(processed_dir)
    os.environ["PAPER_DATA_FILES"] = str(runtime_data_files)
    os.environ["PAPER_ALISIM_RESULTS"] = str(alisim_results)

    for path in [notebook_dir, project_root / "scripts"]:
        path_s = str(path)
        if path_s not in sys.path:
            sys.path.insert(0, path_s)

    sys.modules.setdefault("py3Dmol", types.ModuleType("py3Dmol"))

    return {
        "script_dir": script_dir,
        "project_root": project_root,
        "output_dir": output_dir,
        "work_dir": work_dir,
        "source_root": source_root,
        "notebook_dir": notebook_dir,
        "processed_dir": processed_dir,
        "data_files": runtime_data_files,
        "alisim_results": alisim_results,
    }


def run_postprocess(postprocess_path: Path, namespace: dict[str, object]) -> None:
    """Run a postprocess script in the experiment namespace."""

    if not postprocess_path.exists():
        return
    print(f"[paper-exp] Running postprocess {postprocess_path}", flush=True)
    exec(compile(postprocess_path.read_text(encoding="utf-8"), str(postprocess_path), "exec"), namespace)
