import random

class GridWorld:
    def __init__(
        self, 
        size=3, 
        treasures=None, 
        obstacles=None, 
        random_init=False,
        random_obstacles=False,
        random_treasure=False,
        num_obstacles=1,
        num_treasures=1
    ):
        """
        - size: grid dimension
        - treasures: if given, use these exact treasure positions (unless random_treasure=True)
        - obstacles: if given, use these exact obstacle positions (unless random_obstacles=True)
        - random_init: randomize agent start positions
        - random_obstacles: randomize obstacle placement
        - random_treasure: randomize treasure placement
        - num_obstacles: how many obstacles to place randomly
        - num_treasures: how many treasures to place randomly
        """
        self.size = size
        self._default_treasures = treasures if treasures else [(1,1)]
        self._default_obstacles = obstacles if obstacles else []
        self.random_init = random_init
        self.random_obstacles = random_obstacles
        self.random_treasure = random_treasure
        self.num_obstacles = num_obstacles
        self.num_treasures = num_treasures

        # Internal placeholders that get set each reset
        self.obstacles = []
        self.treasures = []

        self.reset()

    def reset(self):
        # 1. Randomize obstacles if requested
        if self.random_obstacles:
            all_positions = [(x, y) for x in range(self.size) for y in range(self.size)]
            # We'll choose random obstacle positions
            self.obstacles = random.sample(all_positions, self.num_obstacles)
        else:
            # Use fixed obstacles
            self.obstacles = self._default_obstacles.copy()

        # 2. Randomize treasure if requested
        if self.random_treasure:
            valid_positions = [(x, y) for x in range(self.size)
                                        for y in range(self.size)
                                        if (x, y) not in self.obstacles]
            self.treasures = random.sample(valid_positions, self.num_treasures)
        else:
            self.treasures = self._default_treasures.copy()

        # 3. Randomize agents if requested
        if self.random_init:
            valid_positions = [(x, y) for x in range(self.size)
                                        for y in range(self.size)
                                        if (x, y) not in self.obstacles]
            # sample 2 distinct positions for red & blue
            if len(valid_positions) < 2:
                raise ValueError("Not enough valid cells to place both agents.")
            self.red_pos, self.blue_pos = random.sample(valid_positions, 2)
        else:
            # Default corners
            self.red_pos = (0, 0)
            self.blue_pos = (self.size - 1, self.size - 1)

        self.remaining_treasures = self.treasures.copy()
        self.done = False
        return self._get_state()

    def _get_state(self):
        return (self.red_pos, self.blue_pos, tuple(self.remaining_treasures))

    def step(self, action_red, action_blue):
        new_red = self._move(self.red_pos, action_red)
        new_blue = self._move(self.blue_pos, action_blue)
        collision = (new_red == new_blue)

        if collision:
            rewards = (-20, -20)
        else:
            self.red_pos = new_red
            self.blue_pos = new_blue
            rewards = self._calculate_rewards()

        for agent_pos in [self.red_pos, self.blue_pos]:
            if agent_pos in self.remaining_treasures:
                self.remaining_treasures.remove(agent_pos)

        if not self.remaining_treasures:
            self.done = True

        # step penalty
        rewards = (rewards[0] - 1, rewards[1] - 1)
        return self._get_state(), rewards, self.done

    def _move(self, pos, action):
        x, y = pos
        actions = {
            'up':    (x-1, y),
            'down':  (x+1, y),
            'left':  (x, y-1),
            'right': (x, y+1)
        }
        next_pos = actions.get(action, pos)

        # boundaries & obstacles
        if (0 <= next_pos[0] < self.size 
            and 0 <= next_pos[1] < self.size 
            and next_pos not in self.obstacles):
            return next_pos
        return pos

    def _calculate_rewards(self):
        red_reward = 10 if self.red_pos in self.remaining_treasures else 0
        blue_reward = 10 if self.blue_pos in self.remaining_treasures else 0
        return (red_reward, blue_reward)
