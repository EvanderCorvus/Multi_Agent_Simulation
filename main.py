from utils import *
from environemnt import CustomEnv
from ray import tune
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.dqn import DQNTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.tune.registry import register_env


hyperparams = hyperparams_dict("Hyperparameters")
CustomEnv.configure_spaces(hyperparams["grid_size"], hyperparams["n_agents"], hyperparams["seed"])

def env_creator(env_config):
    return CustomEnv(env_config)  # return an env instance

register_env("custom_multi_agent_env", env_creator)


# Define the multi-agent configuration
def policy_mapping_fn(agent_id):
    return "dqn_policy"  # Assuming you use the same policy for all agents

multiagent_config = {
    "multiagent": {
        "policies": {
            "dqn_policy": (DQNTorchPolicy, CustomEnv.observation_space, CustomEnv.action_space, {}),
        },
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": ["dqn_policy"],
    },
    "env": "custom_multi_agent_env",
    "env_config": hyperparams,
}

# Merge with the default DQN config
config = DQNConfig()
config.update_from_dict(multiagent_config)


agent = DQN(config=config, env="custom_multi_agent_env")

for _ in range(hyperparams["max_epochs"]):
    result = DQN.train()
    print(result)