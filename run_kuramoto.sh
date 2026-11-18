#!/bin/bash

#SBATCH --job-name=kuramoto
#SBATCH --partition=cpu
#SBATCH --array=0-49              
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00           
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --mail-type=END --mail-user=UO288280@uniovi.es

mkdir -p logs

N=${1:?"Falta N: sbatch run_kuramoto.sh <N> [args de simulate.py]"}
shift

SIF=${SIF:-$PWD/kuramoto.sif}
[ -f "$SIF" ] || { echo "No existe la imagen: $SIF" >&2; exit 1; }

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

apptainer run \
    --bind "$PWD":/work --pwd /work \
    "$SIF" \
    --N "$N" --sim-index "${SLURM_ARRAY_TASK_ID:-0}" "$@"