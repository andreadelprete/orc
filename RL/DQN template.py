import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randint, uniform

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
 
def np2tf(y):
    ''' convert from numpy to tensorflow '''
    out = tf.expand_dims(tf.convert_to_tensor(y), 0).T
    return out
    
def tf2np(y):
    ''' convert from tensorflow to numpy '''
    return tf.squeeze(y).numpy()


def get_critic(nx, nu):
    ''' Create the neural network to represent the Q function '''
    inputs = layers.Input(shape=(nx+nu,1))
    state_out1 = layers.Dense(16, activation="relu")(inputs) 
    state_out2 = layers.Dense(32, activation="relu")(state_out1) 
    state_out3 = layers.Dense(64, activation="relu")(state_out2) 
    state_out4 = layers.Dense(64, activation="relu")(state_out3)
    outputs = layers.Dense(1)(state_out4) 

    model = tf.keras.Model(inputs, outputs)

    return model

def update(xu_batch, cost_batch, xu_next_batch):
    ''' Update the weights of the Q network using the specified batch of data '''
    # all inputs are tf tensors
    with tf.GradientTape() as tape:         
        # Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".
        # Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. 
        # Tensors can be manually watched by invoking the watch method on this context manager.
        target_values = Q_target(xu_next_batch, training=True)   
        # Compute 1-step targets for the critic loss
        y = cost_batch + DISCOUNT*target_values                            
        # Compute batch of Values associated to the sampled batch of states
        Q_value = Q(xu_batch, training=True)                         
        # Critic's loss function. tf.math.reduce_mean() computes the mean of elements across dimensions of a tensor
        Q_loss = tf.math.reduce_mean(tf.math.square(y - Q_value))  
    # Compute the gradients of the critic loss w.r.t. critic's parameters (weights and biases)
    Q_grad = tape.gradient(Q_loss, Q.trainable_variables)          
    # Update the critic backpropagating the gradients
    critic_optimizer.apply_gradients(zip(Q_grad, Q.trainable_variables))    

nx = 2
nu = 1
QVALUE_LEARNING_RATE = 1e-3
DISCOUNT = 0.99

# Create critic and target NNs
Q = get_critic(nx, nu)
Q_target = get_critic(nx, nu)

Q.summary()

# Set initial weights of targets equal to those of the critic
Q_target.set_weights(Q.get_weights())

# Set optimizer specifying the learning rates
critic_optimizer = tf.keras.optimizers.Adam(QVALUE_LEARNING_RATE)

w = Q.get_weights()
for i in range(len(w)):
    print("Shape Q weights layer", i, w[i].shape)
    
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))
    
print("\nDouble the weights")
for i in range(len(w)):
    w[i] *= 2
Q.set_weights(w)

w = Q.get_weights()
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))

print("\nSave NN weights to file (in HDF5)")
Q.save_weights("namefile.h5")

print("Load NN weights from file\n")
Q_target.load_weights("namefile.h5")

w = Q_target.get_weights()
for i in range(len(w)):
    print("Norm Q weights layer", i, np.linalg.norm(w[i]))
    