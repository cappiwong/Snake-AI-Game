import torch
import random
import numpy as np
from collections import deque
from model import Dueling_QNet
from replay_buffer import PrioritizedReplayBuffer
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100_000
BATCH_SIZE = 64
LR = 0.001

class RainbowAgent:
    def __init__(self, input_size, hidden_size, output_size, n_steps=3):
        self.n_games = 0
        self.gamma = 0.99
        self.n_steps = n_steps  # Multi-step learning
        self.n_step_buffer = deque(maxlen=n_steps)
        self.memory = PrioritizedReplayBuffer(MAX_MEMORY)
        self.model = Dueling_QNet(input_size, hidden_size, output_size)
        self.target_model = Dueling_QNet(input_size, hidden_size, output_size)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = torch.nn.MSELoss()
        self.epsilon = 1.0  # Initial exploration rate

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Add distances to food
        food_dx = (game.food.x - head.x) / game.w
        food_dy = (game.food.y - head.y) / game.h

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            # food_dx,
            # food_dy,
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done):
        # Add to the n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) == self.n_steps:
            # Compute multi-step reward
            multi_step_reward = sum([self.n_step_buffer[i][2] * (self.gamma ** i) for i in range(self.n_steps)])
            state, action, _, next_state, done = self.n_step_buffer[0]
            self.memory.add(state, action, multi_step_reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        # Convert inputs to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float)
        done = torch.tensor([done], dtype=torch.float)

        # Compute Q-values and target Q-values
        q_values = self.model(state).gather(1, action.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_state).max(1)[0]
        target_q_values = reward + self.gamma * next_q_values * (1 - done)

        # Compute the loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_long_memory(self):
        if len(self.memory.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(BATCH_SIZE)
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(next_states, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.float)
        weights = torch.tensor(weights, dtype=torch.float)

        q_values = self.model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.target_model(next_states).max(1)[0]
        target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        td_errors = q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.abs().detach().numpy())

    def get_action(self, state):
        # Epsilon-decay for exploration
        self.epsilon = max(0.01, 1.0 - self.n_games * 0.001)  # Decay over games

        if random.random() < self.epsilon:  # Explore
            action = random.randint(0, 2)
        else:  # Exploit
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state)
            action = q_values.argmax().item()
        return action


    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

