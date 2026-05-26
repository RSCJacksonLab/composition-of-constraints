from __future__ import annotations

import json
import textwrap
import uuid
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "figure_notebooks_rev"
SUMMARY_NOTEBOOK_PATH = NOTEBOOK_DIR / "Figure_4_boundary_enrichment_summary.ipynb"

SEARCH_TERM = "#SEARCH TERM: interface energy share"
EXPORT_MARKER = "# Boundary enrichment export"

SYSTEM_SPECS = [
    {
        "notebook": "Figure_4_and_SI_KRAS_DARPin_K27.ipynb",
        "slug": "kras_darpin_k27",
        "system_name": "KRAS DARPin K27",
        "plot_label": "KRAS\nDARPin K27",
    },
    {
        "notebook": "Figure_4_and_SI_KRAS_DARPin_K55.ipynb",
        "slug": "kras_darpin_k55",
        "system_name": "KRAS DARPin K55",
        "plot_label": "KRAS\nDARPin K55",
    },
    {
        "notebook": "Figure_4_and_SI_KRAS_PIK3CG.ipynb",
        "slug": "kras_pik3cg",
        "system_name": "KRAS PIK3CG",
        "plot_label": "KRAS\nPIK3CG",
    },
    {
        "notebook": "Figure_4_and_SI_KRAS_RAF1.ipynb",
        "slug": "kras_raf1",
        "system_name": "KRAS RAF1",
        "plot_label": "KRAS\nRAF1",
    },
    {
        "notebook": "Figure_4_and_SI_KRAS_RALGDS.ipynb",
        "slug": "kras_ralgds",
        "system_name": "KRAS RALGDS",
        "plot_label": "KRAS\nRALGDS",
    },
    {
        "notebook": "Figure_4_and_SI_KRAS_SOS1.ipynb",
        "slug": "kras_sos1",
        "system_name": "KRAS SOS1",
        "plot_label": "KRAS\nSOS1",
    },
    {
        "notebook": "Figure_4_and_SI_LYCoV.ipynb",
        "slug": "lycov",
        "system_name": "LYCoV",
        "plot_label": "LYCoV",
    },
]


def to_source(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def make_id() -> str:
    return uuid.uuid4().hex[:8]


def build_export_block(spec: dict[str, str]) -> str:
    return textwrap.dedent(
        f"""
        {EXPORT_MARKER}
        from pathlib import Path
        import json

        system_name = {spec["system_name"]!r}
        system_slug = {spec["slug"]!r}
        source_notebook = {spec["notebook"]!r}

        def resolve_boundary_enrichment_output_dir():
            cwd = Path.cwd().resolve()
            for candidate in (cwd, *cwd.parents):
                if (candidate / "figure_notebooks_rev").is_dir():
                    return candidate / "figure_notebooks_rev" / "boundary_enrichment_results"
            return cwd / "boundary_enrichment_results"

        result_dir = resolve_boundary_enrichment_output_dir()
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{{system_slug}}_boundary_enrichment.json"

        result_payload = {{
            "system_name": system_name,
            "system_slug": system_slug,
            "source_notebook": source_notebook,
            "summary_null_model": "same_size_random_edge_subsets",
            "observed": {{
                "n_edges_all": int(n_edges_all),
                "n_edges_defined": int(n_edges_defined),
                "n_boundary_edges": int(n_boundary),
                "boundary_edge_share": float(obs_boundary_edge_share),
                "boundary_energy_share": float(obs_boundary_energy_share),
                "energy_share_excess": float(obs_energy_share_excess),
                "energy_share_enrichment": float(obs_energy_share_enrichment),
            }},
            "null_same_size_random_edge_subsets": {{
                "n_perm": int(n_perm_energy_share),
                "p_energy_share": float(p_energy_share_fixed),
                "p_excess": float(p_excess_fixed),
                "p_enrichment": float(p_enrichment_fixed),
                "energy_share": null_energy_share_fixed.tolist(),
                "energy_excess": null_energy_excess_fixed.tolist(),
                "energy_enrichment": null_energy_enrichment_fixed.tolist(),
            }},
            "null_shuffled_limiter_labels": {{
                "n_perm": int(n_perm_energy_share),
                "p_energy_share": float(p_energy_share_label),
                "p_excess": float(p_excess_label),
                "p_enrichment": float(p_enrichment_label),
                "boundary_edge_share": null_edge_share_label.tolist(),
                "energy_share": null_energy_share_label.tolist(),
                "energy_excess": null_energy_excess_label.tolist(),
                "energy_enrichment": null_energy_enrichment_label.tolist(),
            }},
        }}

        result_path.write_text(json.dumps(result_payload, indent=2) + "\\n")
        print(f"\\nWrote boundary enrichment summary to {{result_path}}")
        """
    ).lstrip()


def patch_notebook(spec: dict[str, str]) -> None:
    path = NOTEBOOK_DIR / spec["notebook"]
    data = json.loads(path.read_text())

    target_cell = None
    for cell in data["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if SEARCH_TERM in source:
            target_cell = cell
            break

    if target_cell is None:
        raise RuntimeError(f"Could not find enrichment cell in {path}")

    source = "".join(target_cell["source"])
    if EXPORT_MARKER in source:
        source = source.split(EXPORT_MARKER, 1)[0].rstrip() + "\n\n"
    else:
        source = source.rstrip() + "\n\n"

    target_cell["source"] = to_source(source + build_export_block(spec))
    path.write_text(json.dumps(data, indent=1) + "\n")


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": make_id(),
        "metadata": {},
        "outputs": [],
        "source": to_source(source),
    }


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": make_id(),
        "metadata": {},
        "source": to_source(source),
    }


def build_summary_notebook() -> dict:
    metadata_source = NOTEBOOK_DIR / SYSTEM_SPECS[0]["notebook"]
    base_metadata = json.loads(metadata_source.read_text()).get("metadata", {})

    specs_literal = json.dumps(
        [
            {
                "slug": spec["slug"],
                "system_name": spec["system_name"],
                "plot_label": spec["plot_label"],
            }
            for spec in SYSTEM_SPECS
        ],
        indent=4,
    )

    load_cell = (
        textwrap.dedent(
            """
            import json
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            RESULT_SPECS = """
        ).strip()
        + specs_literal
        + "\n\n"
        + textwrap.dedent(
            """
            def resolve_boundary_enrichment_output_dir():
                cwd = Path.cwd().resolve()
                for candidate in (cwd, *cwd.parents):
                    if (candidate / "figure_notebooks_rev").is_dir():
                        return candidate / "figure_notebooks_rev" / "boundary_enrichment_results"
                return cwd / "boundary_enrichment_results"

            result_dir = resolve_boundary_enrichment_output_dir()
            missing = [
                result_dir / f"{spec['slug']}_boundary_enrichment.json"
                for spec in RESULT_SPECS
                if not (result_dir / f"{spec['slug']}_boundary_enrichment.json").exists()
            ]
            if missing:
                missing_list = "\\n".join(str(path) for path in missing)
                raise FileNotFoundError(
                    "Missing boundary enrichment result files. Re-run the corresponding Figure 4 notebooks first:\\n"
                    + missing_list
                )

            records = []
            for order, spec in enumerate(RESULT_SPECS):
                payload = json.loads((result_dir / f"{spec['slug']}_boundary_enrichment.json").read_text())
                records.append(
                    {
                        "sort_order": order,
                        "system_slug": spec["slug"],
                        "system_name": spec["system_name"],
                        "plot_label": spec["plot_label"],
                        "observed_enrichment": float(payload["observed"]["energy_share_enrichment"]),
                        "null_p_enrichment": float(payload["null_same_size_random_edge_subsets"]["p_enrichment"]),
                        "n_perm": int(payload["null_same_size_random_edge_subsets"]["n_perm"]),
                    }
                )

            summary_df = pd.DataFrame.from_records(records).sort_values("sort_order").reset_index(drop=True)
            summary_df
            """
        ).strip()
    )

    plot_cell = textwrap.dedent(
        """
        GREY = "#d3d3d3"

        def format_p_value(p):
            if p < 0.001:
                return "p<0.001"
            return f"p={p:.3f}"

        x = np.arange(len(summary_df))
        y_max = float(summary_df["observed_enrichment"].max()) * 1.18

        fig, ax = plt.subplots(figsize=(6.4, 2.8))

        ax.bar(
            x,
            summary_df["observed_enrichment"],
            width=0.65,
            color="black",
            edgecolor="black",
            linewidth=0.8,
        )

        for xi, y, p in zip(x, summary_df["observed_enrichment"], summary_df["null_p_enrichment"]):
            ax.text(
                xi,
                y + 0.03 * y_max,
                format_p_value(float(p)),
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
        ax.set_xticks(x, summary_df["plot_label"])
        ax.set_ylabel("Boundary energy-share enrichment (x)")
        ax.set_xlabel("")
        ax.set_title("")
        ax.set_ylim(0, y_max)
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", color=GREY)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()

        png_path = result_dir / "figure_4_boundary_enrichment_summary.png"
        pdf_path = result_dir / "figure_4_boundary_enrichment_summary.pdf"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")

        print(f"Saved summary plot to {png_path}")
        print(f"Saved summary plot to {pdf_path}")
        plt.show()
        """
    ).strip()

    notebook = {
        "cells": [
            markdown_cell(
                textwrap.dedent(
                    """
                    # Figure 4 Boundary Enrichment Summary

                    This notebook reads the boundary-edge enrichment JSON files written by the non-SH3 Figure 4 system notebooks and plots the observed enrichment for each system, with permutation p-values from the matched-size random-edge null shown above each bar.
                    """
                ).strip()
            ),
            code_cell(load_cell),
            code_cell(plot_cell),
        ],
        "metadata": base_metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def main() -> None:
    for spec in SYSTEM_SPECS:
        patch_notebook(spec)

    SUMMARY_NOTEBOOK_PATH.write_text(json.dumps(build_summary_notebook(), indent=1) + "\n")


if __name__ == "__main__":
    main()
