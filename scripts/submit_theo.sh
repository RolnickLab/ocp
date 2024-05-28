#!/bin/bash
#SBATCH --job-name=test_lcf
#SBATCH --ntasks=1
#SBATCH --mem=48GB
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
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
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test Trained Canonicalisation" --cano_args.equivariance_module=fa --cano_args.cano_type=3D --frame_averaging=3D --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=fa --cano_args.cano_type=3D --cano_args.fa_method=all --optim.batch_size=192 --optim.eval_batch_size=192

# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=fa --cano_args.cano_type=DA



# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=simple --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783372"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=pointnet --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783395"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=dgcnn --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783408"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=fa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783394"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=untrained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783403"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=sign_equiv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.batch_size=128 --optim.eval_batch_size=16

# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=simple --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783406"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=pointnet --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783407"
# # python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=dgcnn --inference_time_loops=0 --optim.max_steps=1
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test S2EF_direct_with_grad" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4783405"



python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=simple --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4757142"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=pointnet --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4757143"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=dgcnn --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4757145"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=fa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4758869"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=untrained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4771801"

# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=simple --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4757151"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=pointnet --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4757152"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=dgcnn --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4757153"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation QM7x" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_steps=1 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4771798"


# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation QM9" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=simple --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation QM9" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=pointnet --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation QM9" --cano_args.equivariance_module=untrained_cano --cano_args.cano_type=3D --cano_args.cano_method=dgcnn --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation QM9" --cano_args.equivariance_module=fa --cano_args.cano_type=3D --inference_time_loops=0

# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation QM9" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=simple --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation QM9" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=pointnet --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation QM9" --cano_args.equivariance_module=trained_cano --cano_args.cano_type=3D --cano_args.cano_method=dgcnn --inference_time_loops=0


# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=sign_equiv_sfa --cano_args.cano_type=3D --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=sign_equiv_sfa --cano_args.cano_type=2D --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test Trained Canonicalisation" --cano_args.equivariance_module=sign_equiv_sfa --cano_args.cano_type=3D --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test Trained Canonicalisation" --cano_args.equivariance_module=sign_equiv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_epochs=1 --optim.batch_size=16 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4770661"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=sign_equiv_sfa --cano_args.cano_type=3D --inference_time_loops=0
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=sign_equiv_sfa --cano_args.cano_type=3D --inference_time_loops=0


# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=untrained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test Trained Canonicalisation" --cano_args.equivariance_module=untrained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_epochs=1 --optim.batch_size=128 
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=untrained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4771801"
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=untrained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 

# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-is2re-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-s2ef-2M --note="Test Trained Canonicalisation" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --optim.max_epochs=1 --optim.batch_size=128 
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm7x-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 
# python main.py --test_ri=True --mode=train --wandb_tags=faenet++ --wandb_project=faenet++ --config=faenet-qm9-all --note="Test Trained Canonicalisation" --cano_args.equivariance_module=trained_sign_inv_sfa --cano_args.cano_type=3D --inference_time_loops=0 --continue_from_dir="/network/scratch/t/theo.saulus/ocp/runs/4771799"

# python scripts/eval_model.py job_id=4757143
# python scripts/eval_model.py job_id=4757145
# python scripts/eval_model.py job_id=4757151
# python scripts/eval_model.py job_id=4757152
# python scripts/eval_model.py job_id=4757153
# python scripts/eval_model.py job_id=4758869
# python scripts/eval_model.py job_id=4767093
# python scripts/eval_model.py job_id=4771798
# python scripts/eval_model.py job_id=4771801
# python scripts/eval_model.py job_id=4779142

