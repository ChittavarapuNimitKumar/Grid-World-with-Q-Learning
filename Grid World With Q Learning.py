import random

# Grid World Environment
grid = [
    [0, 0, 0, 1],    # 0: empty, 1: goal
    [0, -1, 0, -1],  # -1: obstacle
    [0, 0, 0, 0]
]
rows, cols = len(grid), len(grid[0])
actions = ['up', 'down', 'left', 'right']

# Q-table initialization
Q = {}
for r in range(rows):
    for c in range(cols):
        Q[(r, c)] = {a: 0.0 for a in actions}

def is_valid(r, c):
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] != -1

def get_next_state(r, c, action):
    if action == 'up':
        new_r, new_c = r - 1, c
    elif action == 'down':
        new_r, new_c = r + 1, c
    elif action == 'left':
        new_r, new_c = r, c - 1
    elif action == 'right':
        new_r, new_c = r, c + 1
    else:
        new_r, new_c = r, c
    if is_valid(new_r, new_c):
        return new_r, new_c
    else:
        return r, c  # Stay if invalid

def get_reward(r, c):
    if grid[r][c] == 1:
        return 1   # Goal
    elif grid[r][c] == -1:
        return -1  # Obstacle
    else:
        return 0   # Empty

# Q-learning parameters
alpha = 0.5
gamma = 0.9
epsilon = 0.2
episodes = 200

# Training
for ep in range(episodes):
    r, c = 0, 0  # Start position
    while grid[r][c] != 1:
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            action = max(Q[(r, c)], key=Q[(r, c)].get)
        new_r, new_c = get_next_state(r, c, action)
        reward = get_reward(new_r, new_c)
        old_value = Q[(r, c)][action]
        next_max = max(Q[(new_r, new_c)].values())
        Q[(r, c)][action] = old_value + alpha * (reward + gamma * next_max - old_value)
        r, c = new_r, new_c

# Demo: Find the optimal path from start to goal
r, c = 0, 0
path = [(r, c)]
while grid[r][c] != 1 and len(path) < 20:
    action = max(Q[(r, c)], key=Q[(r, c)].get)
    r, c = get_next_state(r, c, action)
    path.append((r, c))

print("Learned path to goal:", path)
