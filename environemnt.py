import numpy as np
import gymnasium as gym
from ray.rllib.env import MultiAgentEnv
from gymnasium import spaces
import random as rand

class FoodGrid():
    def __init__(self, config):
        self.size = config["grid_size"]
        self.growth_rate = config["growth_rate"]
        self.eat_rate = config["eat_rate"]
        #carrying capacity measures the population size at which
        #the population produces exactly enough offspring to just replace itself.
        self.carry_capacity = config["carry_capacity"]
        self.grid = np.ones((self.size, self.size), dtype=float)

    def refresh_food_grid(self):
        self.grid = np.ones((self.size, self.size), dtype=float)
        
    def time_step(self):
        self.grid += self.growth_rate*self.grid(1-self.grid/self.carry_capacity)

    # returns the amount of food eaten at each location
    def food_eat(self, positions):
        food = self.grid.copy()

        for position in positions:
            self.grid[position[0], position[1]] *= (1-self.eat_rate)
        
        
        food_eaten = np.abs(food - self.grid)
        return food_eaten

class CustomEnv(gym.Env):
    action_space = None
    observation_space = None
    seed_val = None
    PYTHONWARNINGS="ignore::DeprecationWarning"

    def __init__(self, config):
        super(CustomEnv, self).__init__()
        self.size = config["grid_size"]
        self.n_agents = config["n_agents"]
        self.max_episode_steps = config["max_episode_steps"]
        self._agent_ids = [f"agent_{i}" for i in range(self.n_agents)]
        self.food_grid = FoodGrid(config)

        self.terminateds = {**{f"agent_{i}": False for i in range(self.n_agents)}, "__all__": False}
        self.truncateds = {**{f"agent_{i}": False for i in range(self.n_agents)}, "__all__": False}

        self.seed(config["seed"])
        CustomEnv.configure_spaces(self.size, self.n_agents, self.seed_val)

        self.reset()

    def step(self, action: dict): # returns: (next_state, reward, done, info)
        boundary_condition = 'hard'
        # Update position based on action
        # shape of position is (n_agents, 2)
        position = [[x, y] for x, y, _, _ in self.agent_states.values()]
        step_direction = [self._action_conversion(a) for a in action.values()]
        new_position = np.add(position, step_direction)

        geometric_center = [np.mean(new_position[:, 0]),
                            np.mean(new_position[:, 1])]

        if boundary_condition == 'hard':
            new_position = np.clip(new_position, 0, self.size-1)
        elif boundary_condition == 'periodic':
            new_position = new_position%self.size

        reward = self._get_reward(new_position)
        next_state = {f"agent_{i}": (new_position[i][0],
                                     new_position[i][1],
                                    float(self.food_grid[new_position[i][0], new_position[i][1]]),
                                    list(geometric_center)) for i in range(self.n_agents)}

        self.food_grid.time_step()

        info = {f"agent_{i}": {} for i in range(self.n_agents)}
        return next_state, reward, self.terminateds, self.truncateds, info
    
    # Random initialization of position of agents
    def reset(self, seed=None, options=None):
        self.food_grid.refresh_food_grid()
        init_positions = np.random.randint(0, self.size, (self.n_agents, 2))
        geometric_center = [np.mean(init_positions[:, 0]),
                            np.mean(init_positions[:, 1])]
        
        agent_states = {f"agent_{i}": (init_positions[i][0],
                                            init_positions[i][1],
                                            1.,
                                            geometric_center) for i in range(self.n_agents)}
        
        self.agent_states = agent_states
        info = {f"agent_{i}": {} for i in range(self.n_agents)}
        
        return self.agent_states, info

    @classmethod
    def configure_spaces(cls, grid_size, n_agents, seed):
        cls.action_space = {f"agent_{i}": spaces.Discrete(9) for i in range(n_agents)}
        cls.observation_space = {f"agent_{i}": spaces.Tuple(
                                                    (spaces.Discrete(grid_size),
                                                     spaces.Discrete(grid_size),
                                                     spaces.Box(0, 1, shape=(1,), dtype=float),
                                                     spaces.Box(-grid_size,
                                                                grid_size,
                                                                shape=(2,), dtype=float)
                                                     ),
                                                     seed=seed) for i in range(n_agents)}

    def render(self):
        return None
    
    def seed(self, seed=None):
        self.seed_val = seed
        rand.seed(seed)
        np.random.seed(seed)
        
    def _action_conversion(self, action):
        # Map action to movement
        action_map = {
            0: [0, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1],
            4: [-1, 0],
            5: [1, 1],
            6: [-1, -1],
            7: [-1, 1],
            8: [1, -1]
        }
        return action_map.get(action, "Invalid action")  # Default case if action is not in the map

    def _element_count(self, array):
        element_to_count = {}
        for element in array:
            if element in element_to_count:
                element_to_count[element] += 1
            else:
                element_to_count[element] = 1
    
    def _get_reward(self, new_position, geometric_center):
        reward_array = np.zeros(self.n_agents, dtype=float)
        count = self._element_count(new_position)
        food_eaten = self.food_grid.food_eat(new_position)

        for i in range(len(new_position)):
            # Food Reward
            reward_array[i] = food_eaten[new_position[i][0], new_position[i][1]]/count[i]
            # Predation Reward
            normalized_distance = np.linalg.norm(new_position[i] - geometric_center)/self.size
            if rand.gauss(0.5, 0.1) < normalized_distance:
                reward_array[new_position[i][0], new_position[i][1]] -= normalized_distance
        reward = {f"agent_{i}": reward_array[i] for i in range(self.n_agents)}
        return reward
    

'''
ValueError: Your environment (<MultiAgentEnvCompatibility instance>) does not abide to the new gymnasium-style API!     
From Ray 2.3 on, RLlib only supports the new (gym>=0.26 or gymnasium) Env APIs.
In particular, the `reset()` method seems to be faulty.
Learn more about the most important changes here:
https://github.com/openai/gym and here: https://github.com/Farama-Foundation/Gymnasium

In order to fix this problem, do the following:

1) Run `pip install gymnasium` on your command line.
2) Change all your import statements in your code from
   `import gym` -> `import gymnasium as gym` OR
   `from gym.space import Discrete` -> `from gymnasium.spaces import Discrete`

For your custom (single agent) gym.Env classes:
3.1) Either wrap your old Env class via the provided `from gymnasium.wrappers import
     EnvCompatibility` wrapper class.
3.2) Alternatively to 3.1:
 - Change your `reset()` method to have the call signature 'def reset(self, *,
   seed=None, options=None)'
 - Return an additional info dict (empty dict should be fine) from your `reset()`
   method.
 - Return an additional `truncated` flag from your `step()` method (between `done` and
   `info`). This flag should indicate, whether the episode was terminated prematurely
   due to some time constraint or other kind of horizon setting.

For your custom RLlib `MultiAgentEnv` classes:
4.1) Either wrap your old MultiAgentEnv via the provided
     `from ray.rllib.env.wrappers.multi_agent_env_compatibility import
     MultiAgentEnvCompatibility` wrapper class.
4.2) Alternatively to 4.1:
 - Change your `reset()` method to have the call signature
   'def reset(self, *, seed=None, options=None)'
 - Return an additional per-agent info dict (empty dict should be fine) from your
   `reset()` method.
 - Rename `dones` into `terminateds` and only set this to True, if the episode is really
   done (as opposed to has been terminated prematurely due to some horizon/time-limit
   setting).
 - Return an additional `truncateds` per-agent dictionary flag from your `step()`
   method, including the `__all__` key (100% analogous to your `dones/terminateds`
   per-agent dict).
   Return this new `truncateds` dict between `dones/terminateds` and `infos`. This
   flag should indicate, whether the episode (for some agent or all agents) was
   terminated prematurely due to some time constraint or other kind of horizon setting.
'''