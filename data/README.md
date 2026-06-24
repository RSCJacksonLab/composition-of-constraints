# Data Layout

Large data and generated result files are distributed outside Git as the
publication data/results zip.

Unpack that zip at the repository root so the following paths exist:

```text
data/source_datasets/
data/alisim_results/
data/precomputed/
data/processed/
```

Expected contents:

- `data/source_datasets/`: source datasets consolidated from the legacy project, including DMS tables, KRAS ddPCA decomposition tables, SH3 combinatorial-core tables, LyCoV antibody data, PDB cache files, and auxiliary source files.
- `data/alisim_results/`: AliSim replicate FASTA files used for sparse phylogenetic validation.
- `data/precomputed/`: legacy precomputed caches, where supplied.
- `data/processed/`: regenerated or bundled experiment outputs consumed by the analysis notebooks.

The experiment `run.sh` scripts write into `data/processed/` by default.
