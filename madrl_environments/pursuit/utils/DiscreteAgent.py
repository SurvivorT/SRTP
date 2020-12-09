import numpy as np

from gym import spaces
from madrl_environments import Agent


#################################################################
# Implements the Single 2D Agent Dynamics
#################################################################

class DiscreteAgent(Agent):

    # constructor
    def __init__(self,
                 xs,
                 ys,
                 zs,
                 map_matrix, # the map of the environemnt (-1 are buildings)
                 obs_range=3,
                 n_channels=3, # number of observation channels
                 seed=1,
                 flatten=False):

        self.random_state = np.random.RandomState(seed)

        self.xs = xs
        self.ys = ys
        self.zs = zs
        self.eactions = [0, # move left
                         1, # move right
                         2, # move up
                         3, # move down
                         4, # stay
                         5, # move upstairs
                         6] # move downstairs

        self.motion_range = [[-1, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, -1, 0],
                             [0, 0, 0],
                             [0, 0, 1],
                             [0, 0, -1]]

        self.current_pos = np.zeros(3, dtype=np.int32) # x and y position
        self.last_pos = np.zeros(3, dtype=np.int32)
        self.temp_pos = np.zeros(3, dtype=np.int32)

        self.map_matrix = map_matrix

        self.terminal = False

        self._obs_range = obs_range
	#推倒？
        if flatten:
            self._obs_shape = (n_channels * obs_range**2 + 1,)
        else:
            self._obs_shape = (obs_range, obs_range, 4)
            #self._obs_shape = (4, obs_range, obs_range)


    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=self._obs_shape)

    @property
    def action_space(self):
        return spaces.Discrete(7)


    ################################################################# 
    # Dynamics Functions
    ################################################################# 
    def step(self, a):
        cpos = self.current_pos
        lpos = self.last_pos
        # print("current_pos" + str(cpos))
        # print("last_pos" + str(lpos))
        # if dead or reached goal dont move
        if self.terminal:
            return cpos
        # if in building, dead, and stay there
        if self.inbuilding(cpos[0], cpos[1], cpos[2]):
            self.terminal = True
            return cpos
        tpos = self.temp_pos
        tpos[0] = cpos[0]
        tpos[1] = cpos[1]
        tpos[2] = cpos[2]
        # transition is deterministic 
        tpos += self.motion_range[a] 
        x = tpos[0]
        y = tpos[1]
        z = tpos[2]
        # check bounds

        if not self.inbounds(x, y, z):
            return cpos
        if (self.motion_range[a][2] != 0) and (not self.hasStairs(x, y, z)):
            return cpos
        # if bumped into building, then stay
        if self.inbuilding(x, y, z):
            return cpos
        else:
            lpos[0] = cpos[0]
            lpos[1] = cpos[1]
            lpos[2] = cpos[2]
            cpos[0] = x
            cpos[1] = y
            cpos[2] = z
            return cpos

    def get_state(self):
        return self.current_pos

    ################################################################# 
    # Helper Functions
    #################################################################
    def hasStairs(self, x, y, z):
        if (self.map_matrix[z][x, y] == 2):
            return True
        return False
    def inbounds(self, x, y, z):
        if 0 <= x < self.xs and 0 <= y < self.ys and 0 <= z < self.zs:
            return True
        return False

    def inbuilding(self, x, y, z):
        if self.map_matrix[z][x,y] == -1:
            return True
        return False

    def nactions(self):
        return len(self.eactions)

    def set_position(self, xs, ys, zs):
        self.current_pos[0] = xs
        self.current_pos[1] = ys
        self.current_pos[2] = zs


    def current_position(self):
        return self.current_pos

    def last_position(self):
        return self.last_pos

