#!/bin/bash
#SBATCH --job-name=PDB_Resume_Energy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --array=0
#SBATCH --mail-type=END,FAIL
#SBATCH --exclude=gpu029
#SBATCH --mail-user=amb755@scarletmail.rutgers.edu

# ========================================
# 1. DEFINE HYPERPARAMETERS
# ========================================
alphas=(1)
examples=(100)          # <-- now training on 100 examples
learning_rates=(100)

NUM_ALPHA=${#alphas[@]}
NUM_EXAMPLES=${#examples[@]}
NUM_LR=${#learning_rates[@]}

# ========================================
# 2. MAP ARRAY INDEX TO PARAMETER GRID
# ========================================
TASK=$SLURM_ARRAY_TASK_ID

a_idx=$(( TASK / (NUM_EXAMPLES * NUM_LR) ))
rem=$(( TASK % (NUM_EXAMPLES * NUM_LR) ))

e_idx=$(( rem / NUM_LR ))
lr_idx=$(( rem % NUM_LR ))

CURRENT_ALPHA=${alphas[$a_idx]}
CURRENT_EXAMPLES=${examples[$e_idx]}
CURRENT_LR=${learning_rates[$lr_idx]}

# ========================================
# 3. DEFINE OUTPUT FOLDER (pinned to the original E1 run)
# ========================================
FOLDER_NAME="PDB_Ensemble_A1_E1_LR100p0000"
mkdir -p "$FOLDER_NAME"

CHECKPOINT_PATH="./${FOLDER_NAME}/${FOLDER_NAME}.pt"

# ========================================
# 4. DEFINE LOG FILES
# ========================================
OUT_FILE="${FOLDER_NAME}/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}.out"
ERR_FILE="${FOLDER_NAME}/job_${SLURM_ARRAY_JOB_ID}_task_${SLURM_ARRAY_TASK_ID}.err"

exec > "$OUT_FILE" 2> "$ERR_FILE"

# ========================================
# 5. DEBUG INFORMATION
# ========================================
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Alpha: $CURRENT_ALPHA"
echo "Examples: $CURRENT_EXAMPLES"
echo "Learning Rate: $CURRENT_LR"
echo "Folder: $FOLDER_NAME"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Started at: $(date)"
echo "========================================"

# ========================================
# 6. RUN TRAINING
# ========================================
srun -N1 -n1 python -u Training_2opt.py \
    --data_folders "/projects/ccib/lamoureux/amb755/PDB_Ensambles_Centered_Interactions_Pickles_Split" \
    --examples_to_run "$CURRENT_EXAMPLES" \
    --alpha "$CURRENT_ALPHA" \
    --learning_rate "$CURRENT_LR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --cluster_file "./cleaned_results.csv" \
    --num_epochs 10000

EXIT_CODE=$?

echo "========================================"
echo "Finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE