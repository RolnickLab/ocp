# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='2D' --fa_frames='all' --test_ri=True --note='2D all pbc fa_cell corrected version'" env=ocp


# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=True --test_ri=True --note='Baseline with fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='3D' --fa_frames='all' --test_ri=True --note='3D all with pbc and fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='2D' --fa_frames='all' --test_ri=True --note='2D all pbc fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='2D' --fa_frames='det' --test_ri=True --note='2D det pbc fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='2D' --fa_frames='se3-det' --test_ri=True --note='se3 det pbc fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='2D' --fa_frames='random' --test_ri=True --note='2D random pbc fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='3D' --test_ri=True --note='3D rando pbc fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config sfarinet-is2re-10k --optim.batch_size=64 --optim.eval_batch_size=64 --optim.lr_initial=0.006 --optim.lr_gamma=0.028 --optim.warmup_steps=500 --optim.warmup_factor=0.3 --optim.max_epochs=20 --model.hidden_channels=180 --model.num_interactions=2 --model.num_gaussians=50 --model.cutoff=4.0 --model.tag_hidden_channels=64 --model.pg_hidden_channels=0 --model.phys_embeds=True --model.phys_hidden_channels=0 --graph_rewiring='remove-tag-0' --model.energy_head='weighted-av-initial-embebds' --model.use_pbc=False --frame_averaging='da' --test_ri=True --note='DA pbc fa_cell corrected version'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config schnet-is2re-10k --graph_rewiring='remove-tag-0' --test_ri=True --note='Baseline no pbc'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config forcenet-is2re-10k --graph_rewiring='remove-tag-0' --test_ri=True --note='Baseline no pbc'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config forcenet-is2re-10k --graph_rewiring='remove-tag-0' --test_ri=True --frame_averaging='da' --note='DA Forcenet no pbc'" env=ocp

# python sbatch.py gres='gpu:rtx8000:1' partition=long time='16:00:00' cpus=4 mem=32GB py_args="--mode train --config dimenet_plus_plus-is2re-10k --graph_rewiring='remove-tag-0' --test_ri=True --note='Baseline no pbc'" env=ocp

