#!/usr/bin/env bash
set -euo pipefail

# Config
OUTDIR="alisim_results"
ALIGNMENT="alignment_bd"
MODEL="LG+G4"
TREE='RANDOM{bd{0.1/0.05}/1000}'
INDEL="0.0005,0.005"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <num_replicates>"
    exit 1
fi

# Check IQTREE3.0 installation
if ! command -v iqtree3 >/dev/null 2>&1; then
    echo "Error: iqtree3 not found in PATH"
    exit 1
fi

NREPS="$1"
mkdir -p "$OUTDIR"

# Width for zero-padding (e.g. 0001, 0002, ...)
PAD_WIDTH=${#NREPS}

# run replicates
for ((i=1; i<=NREPS; i++)); do
    IDX=$(printf "%0*d" "$PAD_WIDTH" "$i")
    PREFIX="${OUTDIR}/replicate_${IDX}"

    echo "Running replicate ${IDX}/${NREPS}"

    iqtree3 \
        --alisim "$PREFIX" "$ALIGNMENT" \
        -t "$TREE" \
        -m "$MODEL" \
        --indel "$INDEL" \
        --out-format fasta \
        -redo
done

