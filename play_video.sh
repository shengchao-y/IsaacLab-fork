# task=Cartwheel
# task_folder=cartwheel_my
# task=Antonball #Cartwheel
# task_folder=antonball #cartwheel
task=Humandribble-Direct
task_folder=dribbling
# task=Go2beam
# task_folder=go2beam
# task=Humanrope
# task_folder=humanrope-F

# load_run=test
# checkpoint=model_15800.pt
# python source/standalone/workflows/rsl_rl/play.py --task Isaac-$task-v0 --num_envs 2 --load_run $load_run --checkpoint $checkpoint


python source/standalone/workflows/rsl_rl/play_n.py --task Isaac-$task-v0 --model_path_dir ~/Documents/papers/2025TMLR/experiment/models/$task_folder --video_length 800  --video