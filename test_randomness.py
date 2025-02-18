from cls.gridworld import GridWorld

def test_random_init_positions():
    """
    Quick test to confirm that random_init=True indeed randomizes the agent start positions.
    We'll reset multiple times and print the initial agent states.
    """
    test_env = GridWorld(size=3, random_init=True)
    
    positions_seen = set()
    for i in range(10):
        state = test_env.reset()
        red_start, blue_start, treasures = state
        positions_seen.add((red_start, blue_start))
        print(f"Episode {i+1} -> Red: {red_start}, Blue: {blue_start}")
    
    print(f"\nDistinct (red, blue) start pairs over 10 resets: {len(positions_seen)}")
    # Typically, you'd expect more than 1 (unless your grid is extremely constrained).
    
if __name__ == "__main__":
    test_random_init_positions()
