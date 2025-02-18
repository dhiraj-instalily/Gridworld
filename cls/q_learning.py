import numpy as np

class QAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=1.0, beta=0.01):
        self.q_table = {}  # Format: {state: {action: q_value}}
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.beta = beta  # Entropy regularization strength

    def choose_action(self, state, possible_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(possible_actions)  # Explore
        else:
            return self._best_action(state, possible_actions)  # Exploit

    def _best_action(self, state, possible_actions):
        state_actions = self.q_table.get(state, {})
        q_values = [state_actions.get(a, 0) for a in possible_actions]
        return possible_actions[np.argmax(q_values)]

    def update_q(self, state, action, reward, next_state, possible_actions):
        # Initialize state if not in Q-table
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in possible_actions}

        # Calculate entropy of current policy
        action_probs = self._get_action_probs(state, possible_actions)
        entropy = -np.sum([p * np.log(p) for p in action_probs.values() if p > 0])

        # Q-update with entropy regularization
        old_q = self.q_table[state].get(action, 0)
        max_future_q = max(self.q_table.get(next_state, {}).values(), default=0)
        new_q = old_q + self.alpha * (
            reward + self.gamma * max_future_q - old_q + self.beta * entropy
        )
        self.q_table[state][action] = new_q

    def _get_action_probs(self, state, possible_actions):
        # Get action probabilities for entropy calculation
        state_actions = self.q_table.get(state, {})
        q_values = np.array([state_actions.get(a, 0) for a in possible_actions])
        exp_q = np.exp(q_values - np.max(q_values))  # Numerically stable softmax
        probs = exp_q / exp_q.sum()
        return {a: p for a, p in zip(possible_actions, probs)}