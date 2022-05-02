import functools
import numpy as np
import gym
from gym.spaces import Box, Discrete
from pettingzoo.utils.wrappers import OrderEnforcingWrapper as PettingzooWrap
from pettingzoo.utils.wrappers import BaseParallelWraper

class BaseModifier:
    def __init__(self):
        pass

    def reset(self, seed=None):
        pass

    def modify_obs(self, obs):
        self.cur_obs = obs
        return obs

    def get_last_obs(self):
        return self.cur_obs

    def modify_obs_space(self, obs_space):
        self.observation_space = obs_space
        return obs_space

    def modify_action(self, act):
        return act

    def modify_action_space(self, act_space):
        self.action_space = act_space
        return act_space

class shared_wrapper_aec(PettingzooWrap):
    def __init__(self, env, modifier_class):
        super().__init__(env)

        self.modifier_class = modifier_class
        self.modifiers = {}
        self._cur_seed = None

        if hasattr(self.env, "possible_agents"):
            self.add_modifiers(self.env.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.modifiers[agent].modify_obs_space(self.env.observation_space(agent))

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.modifiers[agent].modify_action_space(self.env.action_space(agent))

    def add_modifiers(self, agents_list):
        for agent in agents_list:
            if agent not in self.modifiers:
                # populate modifier spaces
                self.modifiers[agent] = self.modifier_class()
                self.observation_space(agent)
                self.action_space(agent)
                self.modifiers[agent].reset(seed=self._cur_seed)

                # modifiers for each agent has a different seed
                if self._cur_seed is not None:
                    self._cur_seed += 1

    def reset(self, seed=None):
        self._cur_seed = seed
        for mod in self.modifiers.values():
            mod.reset(seed=seed)
        super().reset(seed=seed)
        self.add_modifiers(self.agents)
        self.modifiers[self.agent_selection].modify_obs(
            super().observe(self.agent_selection)
        )

    def step(self, action):
        mod = self.modifiers[self.agent_selection]
        action = mod.modify_action(action)
        if self.dones[self.agent_selection]:
            action = None
        super().step(action)
        self.add_modifiers(self.agents)
        self.modifiers[self.agent_selection].modify_obs(
            super().observe(self.agent_selection)
        )

    def observe(self, agent):
        return self.modifiers[agent].get_last_obs()

def frame_stack_v1(env, stack_size):
    class FrameStackModifier(BaseModifier):
        def modify_obs_space(self, obs_space):
            if isinstance(obs_space, Box):
                assert (
                    1 <= len(obs_space.shape) <= 3
                ), "frame_stack only works for 1, 2 or 3 dimensional observations"
            elif isinstance(obs_space, Discrete):
                pass
            else:
                assert (
                    False
                ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                    obs_space
                )

            self.old_obs_space = obs_space
            self.observation_space = stack_obs_space(obs_space, stack_size)
            return self.observation_space

        def reset(self, seed=None):
            self.stack = stack_init(self.old_obs_space, stack_size)

        def modify_obs(self, obs):
            self.stack = stack_obs(
                self.stack,
                obs,
                self.old_obs_space,
                stack_size,
            )
            return self.stack

        def get_last_obs(self):
            return self.stack
        def stack_obs_space(obs_space, stack_size):
            """
            obs_space_dict: Dictionary of observations spaces of agents
            stack_size: Number of frames in the observation stack
            Returns:
                New obs_space_dict
            """
            if isinstance(obs_space, Box):
                dtype = obs_space.dtype
                # stack 1-D frames and 3-D frames
                tile_shape, new_shape = get_tile_shape(obs_space.low.shape, stack_size)

                low = np.tile(obs_space.low.reshape(new_shape), tile_shape)
                high = np.tile(obs_space.high.reshape(new_shape), tile_shape)
                new_obs_space = Box(low=low, high=high, dtype=dtype)
                return new_obs_space
            elif isinstance(obs_space, Discrete):
                return Discrete(obs_space.n ** stack_size)
            else:
                assert (
                    False
                ), "Stacking is currently only allowed for Box and Discrete observation spaces. The given observation space is {}".format(
                    obs_space
                )
        def stack_init(obs_space, stack_size):
            if isinstance(obs_space, Box):
                tile_shape, new_shape = get_tile_shape(obs_space.low.shape, stack_size)
                return np.tile(np.zeros(new_shape, dtype=obs_space.dtype), tile_shape)
            else:
                return 0


        def stack_obs(frame_stack, obs, obs_space, stack_size):
            """
            Parameters
            ----------
            frame_stack : if not None, it is the stack of frames
            obs : new observation
                Rearranges frame_stack. Appends the new observation at the end.
                Throws away the oldest observation.
            stack_size : needed for stacking reset observations
            """
            if isinstance(obs_space, Box):
                obs_shape = obs.shape
                agent_fs = frame_stack

                if len(obs_shape) == 1:
                    size = obs_shape[0]
                    agent_fs[:-size] = agent_fs[size:]
                    agent_fs[-size:] = obs
                elif len(obs_shape) == 2:
                    agent_fs[:, :, :-1] = agent_fs[:, :, 1:]
                    agent_fs[:, :, -1] = obs
                elif len(obs_shape) == 3:
                    nchannels = obs_shape[-1]
                    agent_fs[:, :, :-nchannels] = agent_fs[:, :, nchannels:]
                    agent_fs[:, :, -nchannels:] = obs
                return agent_fs
            elif isinstance(obs_space, Discrete):
                return (frame_stack * obs_space.n + obs) % (obs_space.n ** stack_size)

        def get_tile_shape(shape, stack_size):
            obs_dim = len(shape)

            if obs_dim == 1:
                tile_shape = (stack_size,)
                new_shape = shape
            elif obs_dim == 3:
                tile_shape = (1, 1, stack_size)
                new_shape = shape
            # stack 2-D frames
            elif obs_dim == 2:
                tile_shape = (1, 1, stack_size)
                new_shape = shape + (1,)
            else:
                assert False, "Stacking is only avaliable for 1,2 or 3 dimentional arrays"

            return tile_shape, new_shape
    return shared_wrapper_aec(env, FrameStackModifier)