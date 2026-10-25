from gymnasium.envs.registration import register


register(id="sf-highway-v1", 
         entry_point="rl.envs.highway.highway:MOHighwayEnv",
         nondeterministic=True)

register(id="sf-highway-fast-v1", 
         entry_point="rl.envs.highway.highway:MOHighwayEnvFast",
         nondeterministic=True)