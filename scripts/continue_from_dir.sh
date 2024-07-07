#!/bin/bash
#SBATCH --job-name=ocp-run
#SBATCH --cpus-per-task=4
#SBATCH --mem=80GB
#SBATCH --gres=gpu:a100l:1
#SBATCH --output="/network/scratch/a/ali.ramlaoui/ocp/runs/output-%j.txt"

module load anaconda/3
conda activate ocp
# source venv/bin/activate

echo "Continue from model with job_id: $1"
echo "Mode: $2"

# if --config is provided to the script, use it, otherwise use the default config faenet-is2re-all
if [ -z "$3" ]
then
    echo "Using default config"
    python main.py --cp_data_to_tmp_dir=True --continue_from_dir=$SCRATCH/ocp/runs/$1  --mode=$2 --config=faenet-is2re-all $4 
else
    echo "Using config: $3"
    python main.py --cp_data_to_tmp_dir=True --continue_from_dir=$SCRATCH/ocp/runs/$1  --mode=$2 --config=$3 --reload_config --optim.eval_batch_size=1
fi
