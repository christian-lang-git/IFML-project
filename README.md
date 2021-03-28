# bomberman_rl
Code of the team consisting of Christian Lang and Joshua Bossert for the Bomberman competition. (See https://github.com/ukoethe/bomberman_rl)

The submitted agent was the linear agent found in agent_code\best_linear_agent.

Training of the linear agent was done in agent_code\linear_agent, where results are stored in saved_agents. 

Both the DQN-agent and CDQN-agent share the same code, they only differ in the parameters since the network architecture is described in a dictionary.
They were trained in agent_code\DQN_agent, where the training results are stored in saved_agents.

agent_code\best_DQN_agent and agent_code\best_CDQN_agent contain trained agents, that could be used for the competition, but perform worse than the linear agent.