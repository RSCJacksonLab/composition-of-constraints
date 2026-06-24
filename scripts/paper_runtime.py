#!/usr/bin/env python3
"""Shared setup for publication experiment and analysis scripts."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def _force_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target, target_is_directory=target.is_dir())


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
    data_files = Path(
        os.environ.get(
            "PAPER_DATA_FILES",
            project_root / "data" / "source_datasets",
        )
    ).resolve()
    alisim_results = project_root / "data" / "alisim_results"

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "si_figures").mkdir(parents=True, exist_ok=True)

    if os.environ.get("PAPER_RUN_MODE") != "node_label_permutation_null":
        _force_symlink(data_files, source_root / "data_files")
        _force_symlink(alisim_results, source_root / "alisim_results")
        _force_symlink(output_dir / "figures", source_root / "figures")
        _force_symlink(output_dir / "si_figures", source_root / "si_figures")
    _patch_matplotlib_savefig()

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["PAPER_PROJECT_ROOT"] = str(project_root)
    os.environ["PAPER_OUTPUT_DIR"] = str(output_dir)
    os.environ["PAPER_PROCESSED_DIR"] = str(processed_dir)
    os.environ["PAPER_DATA_FILES"] = str(data_files)

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
        "data_files": data_files,
    }


def run_postprocess(postprocess_path: Path, namespace: dict[str, object]) -> None:
    """Run a postprocess script in the experiment namespace."""

    if not postprocess_path.exists():
        return
    print(f"[paper-exp] Running postprocess {postprocess_path}", flush=True)
    exec(compile(postprocess_path.read_text(encoding="utf-8"), str(postprocess_path), "exec"), namespace)
