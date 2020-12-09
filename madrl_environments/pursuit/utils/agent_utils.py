import numpy as np

from six.moves import xrange

from .DiscreteAgent import DiscreteAgent

#################################################################
# Implements utility functions for multi-agent DRL
#################################################################


def create_agents(nagents, map_matrix, obs_range, flatten=False, randinit=False, constraints=None):
    """
    Initializes the agents on a map (map_matrix)
    -nagents: the number of agents to put on the map
    -randinit: if True will place agents in random, feasible locations
               if False will place all agents at 0
    """
    zs, xs, ys = map_matrix.shape
    agents = []
    for i in xrange(nagents):
        xinit, yinit, zinit = (0, 0, 0)
        if randinit:
            xinit, yinit, zinit = feasible_position(map_matrix, constraints=constraints)
        agent = DiscreteAgent(xs, ys, zs,  map_matrix, obs_range=obs_range, flatten=flatten)
        agent.set_position(xinit, yinit, zinit)
        agents.append(agent)
    return agents


def feasible_position(map_matrix, constraints=None):
    """
    Returns a feasible position on map (map_matrix)
    """
    xs, ys, zs = map_matrix.shape
    loop_count = 0
    while True:
        if constraints is None:
            x = np.random.randint(xs)
            y = np.random.randint(ys)
            z = np.random.randint(zs)
        else:
            xl, xu = constraints[0]
            yl, yu = constraints[1]
            zl, zu = constraints[2]
            x = np.random.randint(xl, xu)
            y = np.random.randint(yl, yu)
            z = np.random.randint(zl, zu)
        if map_matrix[z][x, y] != -1:
            return (x, y, z)


def set_agents(agent_matrix, map_matrix):
    # check input sizes
    if agent_matrix.shape != map_matrix.shape:
        raise ValueError("Agent configuration and map matrix have mis-matched sizes")

    agents = []
    zs, xs, ys = agent_matrix.shape
    for k in xrange(zs):
        for i in xrange(xs):
            for j in xrange(ys):
                n_agents = agent_matrix[k][i, j]
                if n_agents > 0:
                    if map_matrix[k][i, j] == -1:
                        raise ValueError(
                            "Trying to place an agent into a building: check map matrix and agent configuration")
                    agent = DiscreteAgent(xs, ys, zs, map_matrix)
                    agent.set_position(i, j, k)
                    agents.append(agent)
    return agents
