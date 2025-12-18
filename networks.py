import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Concatenate, Input
import os

class CriticNetwork(tf.keras.Model):
    def __init__(self, input_shape=(101, 101, 1), num_actions=1, name='Critic', checkpoint_dir='tmp/sac'):
        super(CriticNetwork, self).__init__(name=name)
        self.checkpoint_dir = checkpoint_dir
        # Creates a unique path for this critic instance (e.g., tmp/sac/Critic_1_sac)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                                            name + '_sac')
        
        
        # --- 1. Define Visual Branch (CNN Layers) ---
        
        # Note: You only define the layers once in __init__
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.flatten = Flatten()
        
        # --- 2. Define MLP Layers (for combined features) ---
        
        # The first Dense layer's input size will be determined dynamically
        # after concatenation (e.g., CNN_output_size + num_actions)
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(256, activation='relu')
        
        # Output layer for the Q-value (single scalar)
        self.q_output = Dense(1, activation='linear')
        
        # Note: We don't define an explicit action layer; the action is simply
        # concatenated with the CNN output features.

    def call(self, inputs):
        # inputs is expected to be a tuple/list: (visual_input, action_input)
        visual_input, action_input = inputs
        
        # --- 1. Forward Pass through CNN ---
        x = self.conv1(visual_input)
        x = self.conv2(x)
        x = self.conv3(x)
        visual_features = self.flatten(x)
        
        # --- 2. Concatenation ---
        # Combine the CNN feature vector and the action scalar
        combined = Concatenate(axis=1)([visual_features, action_input])
        
        # --- 3. Forward Pass through MLP ---
        x = self.d1(combined)
        x = self.d2(x)
        
        q_value = self.q_output(x)
        
        return q_value
    def save_weights(self):
        """Saves the network's weights to its assigned checkpoint file."""
        print(f'... saving weights to {self.checkpoint_file}')
        # Ensure the directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        # TensorFlow's built-in save method
        self.save_weights(self.checkpoint_file)

    def load_weights(self):
        """Loads the network's weights from its assigned checkpoint file."""
        print(f'... loading weights from {self.checkpoint_file}')
        try:
            # TensorFlow's built-in load method
            self.load_weights(self.checkpoint_file)
        except tf.errors.NotFoundError:
            print(f'Error: Checkpoint file not found at {self.checkpoint_file}. Starting from scratch.')
        except Exception as e:
            print(f'An unexpected error occurred during loading: {e}')


class ValueNetwork(tf.keras.Model):
    def __init__(self, input_shape=(101, 101, 2), name='Value', checkpoint_dir='tmp/sac'):
        # The Value Network only needs the state input shape, not the action count
        super(ValueNetwork, self).__init__(name=name)
        
        # --- File Path Setup ---
        self.checkpoint_dir = checkpoint_dir
        # Creates a unique path for this Value network instance (e.g., tmp/sac/Value_sac)
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                                            name + '_sac') 
        
        # --- 1. Define Visual Branch (CNN Layers) ---
        # These are identical to the Critic's CNN layers as they process the same visual input
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.flatten = Flatten()
        
        # --- 2. Define MLP Layers (for state features) ---
        # No concatenation is needed here, just the CNN output features
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(256, activation='relu')
        
        # Output layer for the V-value (single scalar)
        self.v_output = Dense(1, activation='linear')

    def call(self, inputs):
        # The inputs argument is expected to be only the visual state tensor
        visual_input = inputs  # No unpacking needed if only one input is passed
        
        # --- 1. Forward Pass through CNN ---
        x = self.conv1(visual_input)
        x = self.conv2(x)
        x = self.conv3(x)
        visual_features = self.flatten(x) # Output is (Batch, CNN_Output_Size)
        
        # --- 2. Forward Pass through MLP ---
        x = self.d1(visual_features)
        x = self.d2(x)
        
        v_value = self.v_output(x)
        
        return v_value

    def save_weights(self):
        """Saves the network's weights to its assigned checkpoint file."""
        print(f'... saving weights to {self.checkpoint_file}')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.save_weights(self.checkpoint_file)

    def load_weights(self):
        """Loads the network's weights from its assigned checkpoint file."""
        print(f'... loading weights from {self.checkpoint_file}')
        try:
            # Note: Must ensure the model is built before loading!
            self.load_weights(self.checkpoint_file)
        except tf.errors.NotFoundError:
            print(f'Error: Checkpoint file not found at {self.checkpoint_file}. Starting from scratch.')
        except Exception as e:
            print(f'An unexpected error occurred during loading: {e}')



# Constant for log probability calculation
LOG_STD_MAX = 2.0
LOG_STD_MIN = -20.0
EPS = 1e-6  # Epsilon for numerical stability

class ActorNetwork(tf.keras.Model):
    def __init__(self, input_shape=(101, 101, 2), n_actions=1, name='Actor', checkpoint_dir='tmp/sac'):
        super(ActorNetwork, self).__init__(name=name)
        
        # --- File Path Setup ---
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac') 
        self.n_actions = n_actions

        # --- 1. Define Visual Branch (CNN Layers) ---
        self.conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
        self.conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.flatten = Flatten()
        
        # --- 2. Define MLP Layers ---
        self.d1 = Dense(256, activation='relu')
        self.d2 = Dense(256, activation='relu')
        
        # --- 3. Output Heads (for Mean and Log Std) ---
        self.mu = Dense(self.n_actions, activation='linear')
        self.log_std = Dense(self.n_actions, activation='linear')
        
        # This will be used in the call method for the final action bounds
        self.action_scale = tf.constant(1.0, dtype=tf.float32) 
        self.action_bias = tf.constant(0.0, dtype=tf.float32)

    def call(self, visual_input):
        # --- 1. Forward Pass through CNN ---
        x = self.conv1(visual_input)
        x = self.conv2(x)
        x = self.conv3(x)
        visual_features = self.flatten(x)
        
        # --- 2. Forward Pass through MLP ---
        x = self.d1(visual_features)
        x = self.d2(x)
        
        # --- 3. Output Heads ---
        mu = self.mu(x)
        log_std = self.log_std(x)
        
        # --- 4. Log Standard Deviation Clipping ---
        # Clip log_std for numerical stability, crucial for SAC
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tf.exp(log_std)
        
        # --- 5. Sampling the Action (Stochasticity) ---
        # Draw a sample from the standard normal distribution
        noise = tf.random.normal(shape=tf.shape(mu))
        
        # Action before squashing: pi_unconstrained
        pi_unconstrained = mu + noise * std 
        
        # --- 6. TanH Squashing ---
        # Squashing the action to the range (-1, 1)
        actions = tf.tanh(pi_unconstrained)
        
        # Scale and bias actions if necessary (e.g., if range is not [-1, 1])
        # For simple steering [-1, 1], action_scale=1, action_bias=0
        actions = actions * self.action_scale + self.action_bias

        # --- 7. Log Probability Calculation (Change of Variables Formula) ---
        
        # The log probability of the unconstrained Gaussian sample (pi_unconstrained)
        gaussian_log_prob = -0.5 * (((pi_unconstrained - mu) / (std + EPS))**2 + 2 * log_std + tf.math.log(2.0 * np.pi))
        
        # Correction term for the tanh squashing (summed across action dimensions)
        # log(1 - tanh(x)^2) = log(1 - a^2)
        # The sum is over the action dimensions (axis=1 or axis=-1)
        log_prob_correction = tf.math.log(self.action_scale * (1.0 - actions**2) + EPS)
        
        # Total log probability
        log_probs = tf.reduce_sum(gaussian_log_prob - log_prob_correction, axis=1, keepdims=True)
        
        return actions, log_probs

    def save_weights(self):
        """Saves the network's weights to its assigned checkpoint file."""
        print(f'... saving weights to {self.checkpoint_file}')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.save_weights(self.checkpoint_file)

    def load_weights(self):
        """Loads the network's weights from its assigned checkpoint file."""
        print(f'... loading weights from {self.checkpoint_file}')
        try:
            self.load_weights(self.checkpoint_file)
        except tf.errors.NotFoundError:
            print(f'Error: Checkpoint file not found at {self.checkpoint_file}. Starting from scratch.')
        except Exception as e:
            print(f'An unexpected error occurred during loading: {e}')