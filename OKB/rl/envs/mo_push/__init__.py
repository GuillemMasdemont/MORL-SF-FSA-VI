from gymnasium.envs.registration import register


register(
    id="mo-fetch-push-d4-v2",
    entry_point="envs.mo_push.mo_push:MujocoMOFetchPushEnv",
    kwargs={"reward_type": "dense", "d": 4},
    max_episode_steps=25,
)

register(
    id="mo-fetch-push-d6-v2",
    entry_point="envs.mo_push.mo_push:MujocoMOFetchPushEnv",
    kwargs={"reward_type": "dense", "d": 6},
    max_episode_steps=25,
)

register(
    id="mo-fetch-push-d8-v2",
    entry_point="rl.envs.mo_push.mo_push:MujocoMOFetchPushEnv",
    kwargs={"reward_type": "dense", "d": 8},
    max_episode_steps=25,
)


register(
    id="mo-pickandplace-d2-v2",
    entry_point="rl.envs.mo_push.mo_pickplace:MujocoMOFetchPickAndPlaceEnv",
    kwargs={"reward_type": "dense", "d": 2},
    max_episode_steps=25,
)

register(
    id="mo-pickandplace-d4-v2",
    entry_point="rl.envs.mo_push.mo_pickplace:MujocoMOFetchPickAndPlaceEnv",
    kwargs={"reward_type": "dense", "d": 4},
    max_episode_steps=25,
)

register(
    id="mo-pickandplace-d6-v2",
    entry_point="rl.envs.mo_push.mo_pickplace:MujocoMOFetchPickAndPlaceEnv",
    kwargs={"reward_type": "dense", "d": 6},
    max_episode_steps=25,
)

register(
    id="mo-pickandplace-d8-v2",
    entry_point="rl.envs.mo_push.mo_pickplace:MujocoMOFetchPickAndPlaceEnv",
    kwargs={"reward_type": "dense", "d": 8},
    max_episode_steps=25,
)

register(
    id="mo-pickandplace-d10-v2",
    entry_point="rl.envs.mo_push.mo_pickplace:MujocoMOFetchPickAndPlaceEnv",
    kwargs={"reward_type": "dense", "d": 10},
    max_episode_steps=25,
)


register(
    id="mo-reach-d4-v2",
    entry_point="rl.envs.mo_push.mo_reach:MujocoMOFetchReachEnv",
    kwargs={"reward_type": "dense", "d": 4},
    max_episode_steps=25,
)

register(
    id="mo-reach-d6-v2",
    entry_point="rl.envs.mo_push.mo_reach:MujocoMOFetchReachEnv",
    kwargs={"reward_type": "dense", "d": 6},
    max_episode_steps=25,
)

register(
    id="mo-reach-d8-v2",
    entry_point="rl.envs.mo_push.mo_reach:MujocoMOFetchReachEnv",
    kwargs={"reward_type": "dense", "d": 8},
    max_episode_steps=25,
)