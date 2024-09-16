for seed in 26202127 26192416 1484620 72346654 32225970 31415816 68630553 42161619 14201156 30132438
do
    python source/standalone/workflows/rsl_rl/train.py --task Isaac-??-v0 --num_envs 1024 --headless agent.seed=$seed
done