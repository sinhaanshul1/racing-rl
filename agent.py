import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import os

# --- ASSUMPTION: The ValueNetwork and CriticNetwork classes are defined ---
# --- ASSUMPTION: The ActorNetwork class is also defined and available ---
# (We will use the definitions you provided and the structure we built)
from networks import CriticNetwork, ValueNetwork, ActorNetwork
from replay_buffer import ReplayBuffer # Assuming all are saved in networks.py

class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=(101, 101, 2), 
                 tau=0.005, env=None, gamma=0.99, n_actions=1, max_size=1000000, 
                 layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2,
                 checkpoint_dir='tmp/sac'):
        
        # Hyperparameters and Environment
        self.gamma = gamma
        self.tau = tau
        self.scale = reward_scale
        self.batch_size = batch_size
        self.n_actions = n_actions
        
        # --- 1. Instantiate Networks ---
        
        # Value Networks (Current and Target)
        self.value = ValueNetwork(input_shape=input_dims, name='value', checkpoint_dir=checkpoint_dir)
        self.target_value = ValueNetwork(input_shape=input_dims, name='target_value', checkpoint_dir=checkpoint_dir)
        
        # Critic Networks (Twin Q-Functions)
        self.critic_1 = CriticNetwork(input_shape=input_dims, num_actions=n_actions, name='critic_1', checkpoint_dir=checkpoint_dir)
        self.critic_2 = CriticNetwork(input_shape=input_dims, num_actions=n_actions, name='critic_2', checkpoint_dir=checkpoint_dir)
        
        # Actor Network (Policy)
        self.actor = ActorNetwork(input_shape=input_dims, n_actions=n_actions, name='actor', checkpoint_dir=checkpoint_dir)
        
        # --- 2. Optimizers ---
        
        # Critic and Value networks share the same learning rate (alpha)
        self.value_optimizer = Adam(learning_rate=alpha)
        self.critic_optimizer = Adam(learning_rate=alpha)
        
        # Actor network uses its own learning rate (beta)
        self.actor_optimizer = Adam(learning_rate=beta)
        
        # --- 3. Initialize Target Network Weights ---
        # The target network starts with the same weights as the current value network
        self.update_target_networks(tau=1.0) # tau=1.0 means a hard copy
        
        # --- 4. Placeholder for Replay Buffer ---
        # Assuming you will instantiate and assign a ReplayBuffer instance later
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

    def choose_action(self, observation):
        # 1. Add Batch Dimension: [observation] -> (1, 101, 101)
        # 2. Convert to Tensor
        state = tf.convert_to_tensor(observation, dtype=tf.float32)
        
        # 3. CRITICAL FIX: Add the Channel Dimension (axis=-1)
        # Shape goes from (101, 101) to (101, 101, 1)
        state = tf.expand_dims(state, axis=-1)
        
        # 4. Add the Batch Dimension at the start
        # Shape goes from (101, 101, 1) to (1, 101, 101, 1)
        state = tf.expand_dims(state, axis=0) 
        
        # Alternatively, the simplest way to add both:
        # state = tf.expand_dims(tf.expand_dims(observation, axis=-1), axis=0)
        
        # If your observation is already a 3D array (101, 101, 2) from the environment:
        # state = tf.convert_to_tensor([observation], dtype=tf.float32) # (1, 101, 101, 2)
        # (Based on the error, your observation is only 2D: 101x101)

        actions, _ = self.actor(state)
        return actions.numpy()[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_target_networks(self, tau=None):
        """Performs a soft update of the target value network weights."""
        if tau is None:
            tau = self.tau
            
        # Get trainable weights of the current and target value networks
        weights = []
        targets = self.target_value.weights
        
        # Apply the soft update formula: theta_target = tau*theta + (1 - tau)*theta_target
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
            
        self.target_value.set_weights(weights)

    def save_models(self):
        """Saves all network weights to their respective checkpoint files."""
        print('... saving models ...')
        self.value.save_weights()
        self.target_value.save_weights()
        self.critic_1.save_weights()
        self.critic_2.save_weights()
        self.actor.save_weights()

    def load_models(self):
        """Loads all network weights from their respective checkpoint files."""
        print('... loading models ...')
        # NOTE: Models MUST be built (called once) before loading weights!
        self.value.load_weights()
        self.target_value.load_weights()
        self.critic_1.load_weights()
        self.critic_2.load_weights()
        self.actor.load_weights()

    # The core learning logic is implemented as a TensorFlow function for performance
    @tf.function
    def learn(self, state, action, reward, new_state, done):
        
        # --- 1. Update Value Network ($V(s)$) ---
        with tf.GradientTape() as tape:
            # Current V-value: The prediction V(s)
            v_current = self.value(state)
            
            # New Policy (Action & Log Prob): Sample action a' and its log probability
            actions, log_probs = self.actor(state)

            # Q-values for the new policy: Q1(s, a') and Q2(s, a')
            q1_new_policy = self.critic_1([state, actions])
            q2_new_policy = self.critic_2([state, actions])
            q_min = tf.math.minimum(q1_new_policy, q2_new_policy)

            # V-Target: Q_min(s, a') - alpha * log_pi(a'|s)
            v_target = q_min - log_probs * self.scale
            
            # Value Loss: Squared error loss between current V and target V
            value_loss = 0.5 * tf.keras.losses.MSE(v_current, v_target) 
        
        # Apply Gradients
        value_gradient = tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradient, self.value.trainable_variables))

        
        # --- 2. Update Q-Networks ($Q(s, a)$) ---
        with tf.GradientTape() as tape:
            # Target V-value: V_target(s') for the next state
            v_prime = self.target_value(new_state)
            
            # Q-Target: r + gamma * (1 - done) * V_target(s')
            q_target = reward + self.gamma * v_prime * (1 - done)
            
            # Current Q-values: Q1(s, a) and Q2(s, a)
            q1_current = self.critic_1([state, action])
            q2_current = self.critic_2([state, action])
            
            # Q-Loss: Sum of MSE for Q1 and Q2
            q1_loss = 0.5 * tf.keras.losses.MSE(q1_current, q_target)
            q2_loss = 0.5 * tf.keras.losses.MSE(q2_current, q_target)
            q_loss = q1_loss + q2_loss
        
        # Apply Gradients (to both Critics simultaneously)
        q_vars = self.critic_1.trainable_variables + self.critic_2.trainable_variables
        q_gradient = tape.gradient(q_loss, q_vars)
        self.critic_optimizer.apply_gradients(zip(q_gradient, q_vars))

        
        # --- 3. Update Policy Network ($\pi(a|s)$) ---
        with tf.GradientTape() as tape:
            # New Policy (Action & Log Prob): Sample a new action a' from the Actor for the current state
            new_actions, log_probs = self.actor(state)
            
            # Q-values for new policy: Q1(s, a') and Q2(s, a')
            q1_new_policy = self.critic_1([state, new_actions])
            q2_new_policy = self.critic_2([state, new_actions])
            q_min = tf.math.minimum(q1_new_policy, q2_new_policy)
            
            # Actor Loss (Policy Loss): E[alpha * log_pi - Q_min]
            actor_loss = self.scale * log_probs - q_min
            actor_loss = tf.math.reduce_mean(actor_loss)
            
        # Apply Gradients
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

    # --- Wrapper for the learn function to handle NumPy input from the buffer ---
    def update_agent(self):
        if self.memory is None or self.memory.counter < self.batch_size:
            return

        # Sample data from the buffer
        state_np, action_np, reward_np, new_state_np, done_np = self.memory.sample_buffer(self.batch_size)

        # Convert NumPy arrays to TensorFlow tensors
        state = tf.convert_to_tensor(state_np, dtype=tf.float32)
        new_state = tf.convert_to_tensor(new_state_np, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward_np, dtype=tf.float32)
        action = tf.convert_to_tensor(action_np, dtype=tf.float32)
        done = tf.convert_to_tensor(done_np, dtype=tf.float32)

        state = tf.expand_dims(state, axis=-1)
        new_state = tf.expand_dims(new_state, axis=-1)

        # Call the decorated TensorFlow function
        self.learn(state, action, reward, new_state, done)
        
        # Update the target network
        self.update_target_networks()