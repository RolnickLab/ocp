#!/bin/bash
#SBATCH --job-name=test_lcf
#SBATCH --ntasks=1
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --output="/network/scratch/t/theo.saulus/ocp/runs/output-%j.txt"  # replace: location where you want to store the output of the job

module load anaconda/3 # replace: load anaconda module
module load cuda/11.8 
conda activate ocp  # replace: conda env name
cd /home/mila/t/theo.saulus/code/ocp # replace: location of the code

# python scripts/eval_model.py job_id=4757143
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="SInvE3_SFA" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4903930"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="SInvE3_SFA" --cano_args.equivariance_module=untrained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4897787"
