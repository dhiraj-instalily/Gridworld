from cls.gridworld import GridWorld
from cls.q_learning import QAgent
from test_policies import test_policy

def train_agents(env, red_agent, blue_agent, episodes=1000):
    actions = ['up', 'down', 'left', 'right']
    
    for episode in range(episodes):
        state = env.reset()  # This will now be random if env.random_init is True
        done = False
        
        while not done:
            # Agents choose actions
            action_red = red_agent.choose_action(state, actions)
            action_blue = blue_agent.choose_action(state, actions)
            
            # Take step
            next_state, (reward_red, reward_blue), done = env.step(action_red, action_blue)
            
            # Update Q-tables
            red_agent.update_q(state, action_red, reward_red, next_state, actions)
            blue_agent.update_q(state, action_blue, reward_blue, next_state, actions)
            
            state = next_state
        
        # Decay exploration rate
        red_agent.epsilon = max(0.01, red_agent.epsilon * 0.995)
        blue_agent.epsilon = max(0.01, blue_agent.epsilon * 0.995)
        
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}, ε={red_agent.epsilon:.2f}")

def train_and_test():
    from cls import q_learning

    # Train environment: random init, random obstacles, random treasure
    train_env = GridWorld(
        size=3,
        random_init=True,
        random_obstacles=True,
        random_treasure=True,
        num_obstacles=1,
        num_treasures=1
    )

    red_agent = q_learning.QAgent()
    blue_agent = q_learning.QAgent()

    print("=== Training with random obstacles, treasure, and start positions ===")
    train_agents(train_env, red_agent, blue_agent, episodes=10000)

    # Then test on a few *fixed* scenarios to see how well it generalizes:
    print("\n=== Test: 3x3 Grid (No obstacles, fixed treasure in center) ===")
    test_env1 = GridWorld(size=3, treasures=[(1,1)], obstacles=[], random_init=False, random_obstacles=False, random_treasure=False)
    test_policy(test_env1, red_agent, blue_agent)

    print("\n=== Test: 3x3 Grid (Corner treasure, 2 obstacles) ===")
    test_env2 = GridWorld(size=3, treasures=[(0,2)], obstacles=[(1,0), (1,1)], random_init=False, random_obstacles=False, random_treasure=False)
    test_policy(test_env2, red_agent, blue_agent)


if __name__ == "__main__":
    train_and_test()
