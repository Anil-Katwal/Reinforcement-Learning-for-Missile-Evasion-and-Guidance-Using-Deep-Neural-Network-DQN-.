import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import gym
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Custom Environment with Rendering
class MissileEnv(gym.Env):
    def __init__(self):
        super(MissileEnv, self).__init__()
        self.state_space = 8  # [dm, θm, d_m, θ_m, di, θi, d_i, θ_i]
        self.action_space = 2  # Guidance (0) or Evasion (1)
        self.max_steps = 200
        self.reset()
        self.target_position = np.array([0.0, 0.0])  # Fixed target position

    def reset(self):
        # Initialize state: [dm, θm, d_m, θ_m, di, θi, d_i, θ_i]
        self.state = np.random.uniform(-1, 1, self.state_space)
        self.missile_position = np.array([1.0, 1.0])  # Initial missile position
        self.steps = 0
        return self.state

    def step(self, action):
        # Simulate environment dynamics
        reward = self.calculate_reward(action)
        done = self.check_termination()
        self.steps += 1

        # Update missile position (for visualization)
        self.missile_position += np.random.uniform(-0.05, 0.05, size=2)

        next_state = self.state + np.random.normal(0, 0.01, size=self.state_space)  # Add noise
        return next_state, reward, done, {}

    def calculate_reward(self, action):
        dm, di = self.state[0], self.state[4]  # missile-target and missile-interceptor distances
        if dm <= 0.01:  # Hit target
            return 10
        if di <= 0.01:  # Intercepted
            return -10
        return -0.01 * np.linalg.norm(action)  # Penalize energy usage

    def check_termination(self):
        dm, di = self.state[0], self.state[4]
        return dm <= 0.01 or di <= 0.01 or self.steps >= self.max_steps

    def render(self, ax):
        # Clear the axes
        ax.clear()

        # Draw the target and missile
        ax.scatter(*self.target_position, color='red', s=100, label="Target")
        ax.scatter(*self.missile_position, color='blue', s=100, label="Missile")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title("Missile Guidance")
        ax.legend()
        ax.grid()


# Neural Network for Policy and Value
class ActorCritic(tf.keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu')
        ])
        self.actor = tf.keras.layers.Dense(action_dim, activation='softmax')
        self.critic = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.shared(state)
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value


# PPO Algorithm
class PPOAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, clip_ratio=0.2, learning_rate=3e-4):
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def select_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action_probs, _ = self.actor_critic(state)
        action_dist = tfp.distributions.Categorical(probs=action_probs)
        action = action_dist.sample()
        return action.numpy()[0], action_dist.log_prob(action)

    def compute_advantages(self, rewards, values, next_values, dones):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        running_advantage = 0
        for delta, done in zip(reversed(deltas), reversed(dones)):
            running_advantage = delta + self.gamma * running_advantage * (1 - done)
            advantages = advantages.write(advantages.size(), running_advantage)
        return tf.reverse(advantages.stack(), axis=[0])

    def update(self, states, actions, rewards, old_log_probs, next_states, dones):
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor([float(done) for done in dones], dtype=tf.float32)

        _, next_values = self.actor_critic(next_states)
        _, values = self.actor_critic(states)

        advantages = self.compute_advantages(rewards, values, next_values, dones)
        targets = advantages + values

        for _ in range(10):  # Update multiple epochs
            with tf.GradientTape() as tape:
                action_probs, values = self.actor_critic(states)
                action_dist = tfp.distributions.Categorical(probs=action_probs)
                log_probs = action_dist.log_prob(actions)

                # Ratio for PPO
                ratios = tf.exp(log_probs - old_log_probs)
                clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)
                actor_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))

                critic_loss = tf.reduce_mean((targets - values) ** 2)
                entropy = tf.reduce_mean(action_dist.entropy())

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            gradients = tape.gradient(loss, self.actor_critic.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.actor_critic.trainable_variables))


# Main Training Loop with Animation
if __name__ == "__main__":
    env = MissileEnv()
    agent = PPOAgent(state_dim=env.state_space, action_dim=env.action_space)

    # Initialize Matplotlib for Animation
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        global state
        action, _ = agent.select_action(state)
        state, _, done, _ = env.step(action)
        env.render(ax)
        if done:
            state = env.reset()  # Reset environment on episode end

    state = env.reset()
    ani = FuncAnimation(fig, update, frames=200, interval=50)  # Update every 50ms
    plt.show()
