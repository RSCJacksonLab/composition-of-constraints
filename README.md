# The composition of biophysical constraints generates complex, rugged regions of protein fitness landscapes
This repository accompanies the manuscript and contains the code needed to
reproduce the paper figures from the associated data bundle.

Large source datasets and generated result files are not tracked in Git. After
downloading the publication data bundle (), unpack it at the repository root to
create:

```text
data/source_datasets/
data/alisim_results/
data/precomputed/
data/processed/
```

## Setup

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The wrapper scripts automatically use `.venv/bin/python` when that environment
exists. If you use a different environment, pass it explicitly with `PYTHON`.

## Reproduction

Run the experiment scripts:

```bash
bash scripts/run_all_experiments.sh
```

Then run the figure/SI analyses:

```bash
bash scripts/run_all_analyses.sh
```

The Figure 2 node-label permutation nulls are optional to regenerate because
they are expensive and can be supplied in `data/processed/`. To regenerate the
full public null chunk:

```bash
PAPER_RUN_MODE=node_label_permutation_null \
PAPER_NULL_PERM_START=0 \
PAPER_NULL_PERM_STOP=100 \
bash experiments/figure-2_stability-dms-ruggedness-and-edge-cutsets/run.sh
```

## Layout

- `experiments/`: figure-scoped scripts that regenerate processed outputs.
- `analyses/`: notebooks and paired Python exports that regenerate figures and
  tables.
- `scripts/`: shared runtime helpers and wrapper commands.
- `data/`: local location for the external data bundle and regenerated outputs.

The folder names indicate which figure each experiment or analysis supports.
Run a lightweight structural check with:

```bash
bash scripts/check_reproduction_tree.sh
```
