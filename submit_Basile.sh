# ----Inspired from Scripts/submit_example.sh-------
# module load python/3.8 
# cd /home/mila/b/basile.terver/ocp/ocp # If did not run load alias from .bashrc
# source venv/bin/activate

# python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 py_args="--test_ri=True --cp_data_to_tmp_dir=True --mode='train' --wandb_tags='faenet++' --optim.force_coefficient=50 --config='faenet-is2re-all' --note='FAENet with ewald different downprojlayer'"
#exp_name=jmlr/last-runs 
#--model.regress_forces='direct'
# job_name=jmlr 

python mila/sbatch.py mem=32GB cpus=4 gres=gpu:1 py_args="--cp_data_to_tmp_dir=True --optim.force_coefficient=50 --config=faenet-is2re-10k --mode=train --is_debug --logger=dummy --note='Debugging Noisy nodes experiment' "