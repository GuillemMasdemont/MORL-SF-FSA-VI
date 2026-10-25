from gymnasium.envs.registration import register


register(
        id="item-collector-v0",
        entry_point="rl.envs.item_collector.item_collector:ItemCollector",
        max_episode_steps=50,
)

register(
        id="item-collector-sequential-v0",
        entry_point="rl.envs.item_collector.item_collector:ItemCollector",
        max_episode_steps=50,
        kwargs={"single_task": "sequential"}
)

register(
        id="item-collector-balanced-v0",
        entry_point="rl.envs.item_collector.item_collector:ItemCollector",
        max_episode_steps=50,
        kwargs={"single_task": "balanced"}
)