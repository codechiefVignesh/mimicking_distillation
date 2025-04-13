import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import matplotlib.pyplot as plt
import time

# Create save directory for models
save_dir = "saved_models"
os.makedirs(save_dir, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculate the size of the output from the last convolutional layer
        conv_out_size = self._get_conv_output(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape  # Should be (C, H, W) format for PyTorch
        self.action_size = action_size
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    # discount factor
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.update_target_frequency = 1000
        self.batch_size = 32
        self.train_start = 1000
        self.step_counter = 0
        
        # Initialize main and target networks
        self.policy_net = DQN(state_shape, action_size).to(device)
        self.target_net = DQN(state_shape, action_size).to(device)
        self.update_target_network()
        
        # Ensure target network has same weights as policy network but doesn't require gradients
        for param in self.target_net.parameters():
            param.requires_grad = False
            
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        print("Target network updated")
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Convert state to PyTorch tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) / 255.0
        
        # Get action values from policy network
        with torch.no_grad():
            act_values = self.policy_net(state_tensor)
        
        return torch.argmax(act_values).item()
    
    def replay(self):
        if len(self.memory) < self.train_start:
            return
        
        # Sample a batch of experiences
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Convert to PyTorch tensors
        states = torch.FloatTensor(np.array([transition[0] for transition in minibatch])).to(device) / 255.0
        actions = torch.LongTensor(np.array([transition[1] for transition in minibatch])).to(device)
        rewards = torch.FloatTensor(np.array([transition[2] for transition in minibatch])).to(device)
        next_states = torch.FloatTensor(np.array([transition[3] for transition in minibatch])).to(device) / 255.0
        dones = torch.FloatTensor(np.array([transition[4] for transition in minibatch])).to(device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        # Compute target Q values
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update policy network
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.step_counter += 1
        if self.step_counter % self.update_target_frequency == 0:
            self.update_target_network()
    
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
    
    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

def preprocess_env(env_name):
    """Create and preprocess the Atari environment."""
    # Create the environment with frame skip set to 1 (no frame skip)
    env = gym.make(env_name, frameskip=1)
    
    # Apply standard Atari preprocessing: grayscale, frame skipping, etc.
    env = AtariPreprocessing(
        env, 
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,
        terminal_on_life_loss=True
    )
    
    # Stack 4 frames to capture motion
    env = FrameStack(env, 4)
    return env

def process_state(state):
    """Convert state from LazyFrames to numpy array with channel-first format for PyTorch."""
    if not isinstance(state, np.ndarray):
        state = np.array(state)
    # Channel first format for PyTorch (frames, height, width)
    return state

def train_agent(env_name, episodes=1000, render=False):
    """Train the DQN agent on the specified Atari game."""
    env = preprocess_env(env_name)
    
    # Get state shape and action size
    # PyTorch uses channel-first format (C, H, W)
    state_shape = (4, 84, 84)  # 4 stacked frames of 84x84 grayscale images
    action_size = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_shape, action_size)
    
    # Training metrics
    scores = []
    average_scores = []
    epsilon_values = []
    
    for e in range(1, episodes+1):
        state, _ = env.reset()
        # Convert state and prepare for PyTorch
        state = process_state(state)
        
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            step += 1
            
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = process_state(next_state)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Learn from experience
            agent.replay()
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Save scores and epsilon
        scores.append(total_reward)
        epsilon_values.append(agent.epsilon)
        
        # Calculate average score of last 100 episodes
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        average_scores.append(avg_score)
        
        # Print episode stats
        print(f"Episode: {e}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.4f}, Avg Score: {avg_score:.2f}")
        
        # Save model every 100 episodes
        if e % 100 == 0:
            agent.save(f"{save_dir}/boxing_dqn_ep{e}.pt")
            plot_training_results(e, scores, average_scores, epsilon_values)
    
    # Save final model
    agent.save(f"{save_dir}/boxing_dqn_final.pt")
    
    # Plot training results
    plot_training_results(episodes, scores, average_scores, epsilon_values)
    
    env.close()
    return agent, scores, average_scores

def plot_training_results(episodes, scores, avg_scores, epsilons):
    """Plot the training results."""
    plt.figure(figsize=(15, 5))
    
    # Plot scores
    plt.subplot(1, 2, 1)
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Average Score (last 100)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Scores')
    plt.legend()
    
    # Plot epsilon decay
    plt.subplot(1, 2, 2)
    plt.plot(epsilons)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/boxing_training_results_ep{episodes}.png")
    plt.close()

def test_agent(env_name, model_path, episodes=10, render=True):
    """Test the trained agent."""
    env = preprocess_env(env_name)
    
    # Get state shape and action size
    state_shape = (4, 84, 84)  # PyTorch format (C, H, W)
    action_size = env.action_space.n
    
    # Initialize agent
    agent = DQNAgent(state_shape, action_size)
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during testing
    
    scores = []
    
    for e in range(1, episodes+1):
        state, _ = env.reset()
        state = process_state(state)
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Select action
            action = agent.act(state, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Process next state
            next_state = process_state(next_state)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        scores.append(total_reward)
        print(f"Episode: {e}/{episodes}, Score: {total_reward}, Steps: {steps}")
    
    avg_score = np.mean(scores)
    print(f"Average Score over {episodes} episodes: {avg_score:.2f}")
    
    env.close()
    return scores

if __name__ == "__main__":
    # Environment name
    env_name = "ALE/Boxing-v5"
    # env_name = "Boxing-v5"
    # env_name = "ALE/BoxingNoFrameskip-v4"
    
    # Training parameters
    training_episodes = 1000
    
    # Train agent
    print("Starting training...")
    agent, scores, avg_scores = train_agent(env_name, episodes=training_episodes)
    print("Training completed!")
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    model_path = f"{save_dir}/boxing_dqn_final.pt"
    test_scores = test_agent(env_name, model_path, episodes=5)
    print("Testing completed!")