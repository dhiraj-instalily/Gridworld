import numpy as np


def test_policy(env, red_agent, blue_agent, max_steps=10):
    state = env.reset()
    done = False
    step = 0
    total_red_reward = 0
    total_blue_reward = 0
    
    print(f"Initial grid ({env.size}x{env.size}):")
    _visualize(state, env.size, env.obstacles)  # Pass obstacles
    
    while not done and step < max_steps:
        # Store previous state for comparison
        prev_red_pos = state[0]
        prev_blue_pos = state[1]
        prev_treasures = state[2]
        
        # Get actions
        action_red = red_agent._best_action(state, ['up', 'down', 'left', 'right'])
        action_blue = blue_agent._best_action(state, ['up', 'down', 'left', 'right'])
        
        # Take step
        next_state, rewards, done = env.step(action_red, action_blue)
        step += 1
        
        # Calculate reward components
        collision = (next_state[0] == next_state[1])
        red_treasure = prev_red_pos not in prev_treasures and state[0] in prev_treasures
        blue_treasure = prev_blue_pos not in prev_treasures and state[1] in prev_treasures
        
        # Print detailed information
        print(f"\nStep {step}:")
        print(f"🟥 chose {action_red} (from {prev_red_pos} → {next_state[0]})")
        print(f"🟦 chose {action_blue} (from {prev_blue_pos} → {next_state[1]})")
        print(f"Collision occurred: {'YES' if collision else 'NO'}")
        print(f"Treasures remaining: {len(next_state[2])}")
        
        # Breakdown rewards
        print("\nReward Calculation:")
        print(f"🟥: {'Treasure (+10)' if red_treasure else 'No treasure'} | " 
              f"{'Collision (-20)' if collision else 'No collision'} | Step penalty (-1)")
        print(f"🟦: {'Treasure (+10)' if blue_treasure else 'No treasure'} | "
              f"{'Collision (-20)' if collision else 'No collision'} | Step penalty (-1)")
        
        # Track cumulative rewards
        total_red_reward += rewards[0]
        total_blue_reward += rewards[1]
        print(f"\nStep Rewards: 🟥={rewards[0]}, 🟦={rewards[1]}")
        print(f"Total Rewards: 🟥={total_red_reward}, 🟦={total_blue_reward}")
        
        _visualize(next_state, env.size, env.obstacles)  # Pass obstacles here
        state = next_state
    
    print("\nFinal rewards:")
    print(f"🟥 Total: {total_red_reward}")
    print(f"🟦 Total: {total_blue_reward}")

def _visualize(state, grid_size, obstacles):
    # Create grid with uniform-width characters
    grid = np.full((grid_size, grid_size), '・', dtype='<U2')
    red_pos, blue_pos, treasures = state
    
    # Mark obstacles first (using ■ for better visibility)
    for (x, y) in obstacles:
        grid[x][y] = '■'
    
    # Mark treasures (overwrite obstacles if needed, though they shouldn't overlap)
    for (x, y) in treasures:
        grid[x][y] = '★'
    
    # Check for collision and mark agents
    collision = (red_pos == blue_pos)
    if collision:
        x, y = red_pos
        grid[x][y] = '💥'
    else:
        # Mark red agent if not on obstacle
        if red_pos not in obstacles:
            grid[red_pos[0]][red_pos[1]] = '🔴'
        # Mark blue agent if not on obstacle
        if blue_pos not in obstacles:
            grid[blue_pos[0]][blue_pos[1]] = '🔵'
    
    # Print with fixed-width formatting
    for row in grid:
        # Use | as separators for better visual structure
        formatted_row = [f"{cell:2}" for cell in row]
        print('|' + '|'.join(formatted_row) + '|')