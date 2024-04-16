#!/bin/bash
#SBATCH --job-name=test_lcf
#SBATCH --ntasks=1
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output="/network/scratch/t/theo.saulus/ocp/runs/output-%j.txt"  # replace: location where you want to store the output of the job

module load anaconda/3 # replace: load anaconda module
module load cuda/11.8 
conda activate ocp  # replace: conda env name
cd /home/mila/t/theo.saulus/code/ocp # replace: location of the code
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="FAENet with LCF" --fa_method=all --frame_averaging=3D
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="FAENet with LCF" --fa_method=all --frame_averaging=3D

# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=untrained_cano-is2re-all --note="Untrained Canonicalisation" --canonicalisation=3D

# python main.py --test_ri=True --cp_data_to_tmp_dir=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --optim.force_coefficient=50 --config=faenet-is2re-all --note="FAENet with LCF" --fa_method=all --frame_averaging=3D

# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test Trained Canonicalisation" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --optim.max_epochs=1 --model.num_interactions=1 --inference_time_loops=0
python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=dgcnn --inference_time_loops=0 --model.num_interactions=3 --model.hidden_channels=64
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-10k --note="Test Trained Canonicalisation" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --optim.max_epochs=1 --inference_time_loops=0 --model.num_interactions=1
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=fa --cano_args.cano_type=3D --cano_args.fa_method=all --optim.batch_size=192 --optim.eval_batch_size=192
