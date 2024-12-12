for seed in 26202127 26192416 1484620 72346654 32225970 31415816 68630553 42161619 14201156 30132438
do
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Go2beam-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=0.75 agent.rewards_expect.rew_progress=6.0
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Go2beam-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=0.75 agent.rewards_expect.rew_progress=20.0
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Go2beam-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=0 agent.ent_schedule_end=0.75
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Go2beam-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=0 agent.ent_schedule_end=0.25
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Go2beam-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=800
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Go2beam-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=8000
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humandribble-Direct-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=0 agent.ent_schedule_end=0.75
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humandribble-Direct-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=0 agent.ent_schedule_end=0.5
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humandribble-Direct-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=0 agent.ent_schedule_end=0.25
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humandribble-Direct-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=1000
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humandribble-Direct-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=5000
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humandribble-Direct-v0 --num_envs 1024 --headless agent.seed=$seed agent.ent_schedule_iterations=10000
    # python source/standalone/workflows/rsl_rl/train_rnd.py --task Isaac-Antonball-v0 --num_envs 1024 --headless agent.seed=$seed
    # python source/standalone/workflows/rsl_rl/train_rnd.py --task Isaac-Humandribble-Direct-v0 --num_envs 1024 --headless agent.seed=$seed
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Poleonhuman-v0 --num_envs 1024 --headless agent.seed=$seed
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humanrope-v0 --num_envs 1024 --headless agent.seed=$seed
    python source/standalone/workflows/rsl_rl/train_rnd.py --task Isaac-Go2beam-v0 --num_envs 1024 --headless agent.seed=$seed
    # # python source/standalone/workflows/rsl_rl/train.py --task Isaac-G1run-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=0.75 agent.rewards_expect.rew_progress=1.8
    # # python source/standalone/workflows/rsl_rl/train.py --task Isaac-G1run-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=0.75 agent.rewards_expect.rew_progress=3.0
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humanoid-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=1.0 agent.rewards_expect.rew_progress=40.0
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humanoid-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=1.0 agent.rewards_expect.rew_progress=
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humanoid-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=1.0 agent.rewards_expect.rew_progress=2.8
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humanoid-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=1.0 agent.rewards_expect.rew_progress=
    # python source/standalone/workflows/rsl_rl/train.py --task Isaac-Humanoid-v0 --num_envs 1024 --headless agent.seed=$seed agent.gage_init_std=0.0
done