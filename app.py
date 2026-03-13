# ==========================================
# Animated Q-Learning Gridworld App (Smaller Figure)
# ==========================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random
import time

st.set_page_config(
    page_title="Animated Q-Learning Gridworld",
    page_icon="🤖",
    layout="wide"
)

st.markdown('<h1 style="text-align:center;">🤖 Animated Q-Learning Gridworld</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;">Watch the agent learn and move step by step!</p>', unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# Sidebar: Environment Settings
# -----------------------------
st.sidebar.header("🏗 Environment Settings")
grid_size = st.sidebar.slider("Grid Size", 4, 8, 5)

start_x = st.sidebar.number_input("Start X", 0, grid_size-1, 0)
start_y = st.sidebar.number_input("Start Y", 0, grid_size-1, 0)
goal_x = st.sidebar.number_input("Goal X", 0, grid_size-1, grid_size-1)
goal_y = st.sidebar.number_input("Goal Y", 0, grid_size-1, grid_size-1)

num_obstacles = st.sidebar.slider("Number of Obstacles", 0, grid_size*grid_size//2, 4)
obstacles = []
for i in range(num_obstacles):
    obs_x = st.sidebar.number_input(f"Obstacle {i+1} X", 0, grid_size-1, random.randint(0, grid_size-1))
    obs_y = st.sidebar.number_input(f"Obstacle {i+1} Y", 0, grid_size-1, random.randint(0, grid_size-1))
    obstacles.append((obs_x, obs_y))

start = (start_x, start_y)
goal = (goal_x, goal_y)

# -----------------------------
# Sidebar: Q-Learning Hyperparameters
# -----------------------------
st.sidebar.header("⚙️ Q-Learning Hyperparameters")
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.99, 0.9)
epsilon = st.sidebar.slider("Exploration Rate (ε)", 0.0, 1.0, 0.2)
episodes = st.sidebar.slider("Episodes", 100, 3000, 1000)
delay = st.sidebar.slider("Animation Delay (s)", 0.05, 1.0, 0.3)

# -----------------------------
# Q-Learning Helper Functions
# -----------------------------
def get_state(pos):
    return pos[0]*grid_size + pos[1]

def step(position, action):
    x, y = position
    if action == 0: x -= 1  # up
    elif action == 1: x += 1  # down
    elif action == 2: y -= 1  # left
    elif action == 3: y += 1  # right

    if x < 0 or x >= grid_size or y < 0 or y >= grid_size:
        return position, -5

    new_pos = (x, y)
    if new_pos in obstacles:
        return new_pos, -10
    if new_pos == goal:
        return new_pos, 10
    return new_pos, -1

# -----------------------------
# Train and Animate
# -----------------------------
if st.button("🏃 Train & Animate Agent"):

    actions = 4
    total_states = grid_size * grid_size
    q_table = np.zeros((total_states, actions))

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Q-Learning Training
    for episode in range(episodes):
        position = start
        while position != goal:
            state = get_state(position)
            if random.uniform(0,1) < epsilon:
                action = random.randint(0,3)
            else:
                action = np.argmax(q_table[state])
            new_position, reward = step(position, action)
            new_state = get_state(new_position)
            old_value = q_table[state, action]
            next_max = np.max(q_table[new_state])
            new_value = old_value + alpha * (reward + gamma * next_max - old_value)
            q_table[state, action] = new_value
            position = new_position
        if episode % max(1, episodes//100) == 0:
            progress_bar.progress(episode / episodes)
            status_text.text(f"Training Episode: {episode}/{episodes}")

    st.success("✅ Training Completed")
    st.balloons()

    # -----------------------------
    # Compute Learned Path
    # -----------------------------
    position = start
    path = [position]
    while position != goal:
        state = get_state(position)
        action = np.argmax(q_table[state])
        position, _ = step(position, action)
        path.append(position)

    st.subheader("📍 Learned Path")
    #st.write(path)

    # -----------------------------
    # Animate Path (Smaller Figure)
    # -----------------------------
    st.subheader("🎬 Agent Animation")
    placeholder = st.empty()

    for i in range(len(path)):
        # Smaller figure
        fig, ax = plt.subplots(figsize=(4,4))  # smaller overall figure

        # Draw Path So Far
        x = [p[1] for p in path[:i+1]]
        y = [p[0] for p in path[:i+1]]
        ax.plot(x, y, marker='o', linestyle='-', color='red', markersize=6, label='Agent Path')

        # Start & Goal
        ax.scatter(start[1], start[0], s=100, c='blue', label='Start')
        ax.scatter(goal[1], goal[0], s=100, c='gold', label='Goal')

        # Obstacles
        ox = [o[1] for o in obstacles]
        oy = [o[0] for o in obstacles]
        ax.scatter(ox, oy, marker='*', s=80, c='green', label='Obstacles')

        ax.set_title("Q-Learning Agent Path Animation", fontsize=10)
        ax.set_xlabel("X Position", fontsize=8)
        ax.set_ylabel("Y Position", fontsize=8)
        ax.set_xlim(-0.5, grid_size-0.5)
        ax.set_ylim(-0.5, grid_size-0.5)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.invert_yaxis()

        placeholder.pyplot(fig)
        time.sleep(delay)