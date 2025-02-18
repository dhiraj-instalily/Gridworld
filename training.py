from cls import gridworld
from cls import q_learning
from test_policies import test_policy, test_policy_previous

def train_agents(env, red_agent, blue_agent, episodes=1000):
    actions = ['up', 'down', 'left', 'right']
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Agents choose actions
            action_red = red_agent.choose_action(state, actions)
            action_blue = blue_agent.choose_action(state, actions)
            
            # Take step in environment
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

# Initialize
env = gridworld.GridWorld(size=3, random_treasures=True)
red_agent = q_learning.QAgent()
blue_agent = q_learning.QAgent()


print("Before training ...")
# test_policy(env, red_agent, blue_agent)

train_agents(env, red_agent, blue_agent, episodes=10000)

print("Before training ...")

new = True
if new:
    test_policy(env, red_agent, blue_agent)
elif not new:
    test_policy_previous(env, red_agent, blue_agent)