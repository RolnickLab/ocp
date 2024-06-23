# ----Inspired from Scripts/submit_example.sh-------
# module load python/3.8 
# cd /home/mila/b/basile.terver/ocp/ocp # If did not run load alias from .bashrc
# source venv/bin/activate

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 py_args="--test_ri=True --cp_data_to_tmp_dir=True --mode='train' --wandb_tags='faenet++' --optim.force_coefficient=50 --config='faenet-is2re-all' --note='FAENet with ewald different downprojlayer'"
#exp_name=jmlr/last-runs 
#--model.regress_forces='direct'
# job_name=jmlr 

# 6th February
# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re_aux-10k --model.num_interactions=10 --mode=train --note='is2re_aux with 10 interactions'"

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re-all --model.num_interactions=5 --mode=train --note='is2re-all with 5 interactions'"

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re_aux-all --model.num_interactions=5 --mode=train --note='is2re_aux-all with 5 interactions'"

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re_aux-all --model.num_interactions=10 --mode=train --note='is2re_aux-all with 10 interactions'"

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re_aux-all --model.num_interactions=10 --mode=train --note='noising in dataloader: is2re_aux-all with 10 interactions'"

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re_aux-all --model.num_interactions=10 --dataset.train.noisy_nodes.type=constant --mode=train --note='noising in dataloader, constant noise: is2re_aux-all with 10 interactions'"

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re_aux-10k --model.num_interactions=5 --optim.max_epochs=50 --dataset.train.noisy_nodes.type=constant --mode=train --note='constant noise, more epochs'"
# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long time=16:00:00 py_args="--config=faenet-is2re_aux-10k --model.num_interactions=5 --optim.max_epochs=50 --dataset.train.noisy_nodes.type=constant --mode=train --note='constant noise, more epochs'"
# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long time=16:00:00 py_args="--config=faenet-is2re_aux-10k --model.num_interactions=5 --dataset.train.noisy_nodes.type=constant --mode=train --note='constant noise, back normal nb of epochs for debugging'"
python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 partition=long py_args="--config=faenet-is2re_aux-10k --model.num_interactions=5 --dataset.train.noisy_nodes.type=constant --mode=train --note='constant noise, back normal nb of epochs for debugging'"