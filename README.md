# bomberman_rl
Code of the team consisting of Christian Lang and Joshua Bossert for the Bomberman competition. (See https://github.com/ukoethe/bomberman_rl)

The submitted agent was the linear agent.

Training code for the linear agent can be found in agent_code\linear_agent. A trained linear agent, without the agent code, can be found in agent_code\best_linear_agent.

Both the DQN-agent and CDQN-agent share the same code, they only differ in the parameters since the network architecture is described in a dictionary.
The required training code for both agents is located in agent_code\DQN_agent.

agent_code\best_DQN_agent and agent_code\best_CDQN_agent contain trained agents, without the training code, that could be used for the competition, but perform worse than the linear agent.