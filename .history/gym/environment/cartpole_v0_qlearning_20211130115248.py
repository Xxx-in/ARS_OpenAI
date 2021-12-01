# Adapted from: https://medium.com/@tuzzer/cart-pole-balancing-with-q-learning-b54c6068d947

import gym
from gym.core import ActionWrapper
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plot

# from time import sleep
import gym
import numpy as np
import random
import math
from time import sleep

## Initialize the "Cart-Pole" environment
env = gym.make("CartPole-v0")

# Initializing the random number generator
np.random.seed(int(time.time()))

## Defining the environment related constants

# Number of discrete states (bucket) per state dimension
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')
# Number of discrete actions
NUM_ACTIONS = env.action_space.n  # (left, right)
# Bounds for each discrete state
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS)
print('action index:'ACTION_INDEX)

## Creating a Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

## Learning related constants
# Continue here
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
TEST_RAND_PROB = 0.2

## Defining the simulation related constants
NUM_TRAIN_EPISODES = 2  # how many training episodes - 180
NUM_TEST_EPISODES = 1  # how many testing episodes (should ) - 1
MAX_TRAIN_T = 200  # how many steps for 1 training episode - 200
MAX_TEST_T = 200  # '' - 200
STREAK_TO_END = 100  # how many consecutive wins to end
SOLVED_T = 195  # how many rewards/steps/time pole can stay upright to consider as 1 win episode
VERBOSE = True  # print data


def train():

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(0)
    explore_rate = get_explore_rate(0)
    discount_factor = 0.99  # since the world is unchanging

    num_train_streaks = 0
    
    # complete training with _ no. of episode
    for episode in range(NUM_TRAIN_EPISODES):
        # Reset the environment
        obv = env.reset()

        # the initial state
        state_0 = state_to_bucket(obv)
        
        print('learning_rate: ', learning_rate)
        print('explore_rate: ', explore_rate)
            
        # complete 1 episode
        for t in range(MAX_TRAIN_T):
            # t = current timestep in episode [0-199]
            # env.render()

            # Select an action
            action = select_action(state_0, explore_rate)

            # Execute the action
            obv, reward, done, _ = env.step(action)

            # Observe the result
            state = state_to_bucket(obv)

            # Update the Q based on the result
            best_q = np.amax(q_table[state])    #best q-value of action in a certain state 
            q_table[state_0 + (action,)] += learning_rate * (reward + discount_factor * (best_q) - q_table[state_0 + (action,)])

            # Setting up for the next iteration
            state_0 = state

            # Print data
            if VERBOSE:
                print("\nEpisode = %d" % episode)
                print("t = %d" % t)
                print("Action: %d" % action)
                print("State: %s" % str(state))
                print("Reward: %f" % reward)
                print("Observation: %s" % str(obv))
                print("Best Q: %f" % best_q)
                print("Explore rate: %f" % explore_rate)
                print("Learning rate: %f" % learning_rate)
                print("Streaks: %d" % num_train_streaks)

                print("")

            if done:
                print("Episode %d finished after %f time steps" % (episode, t))
                if t >= SOLVED_T:
                    num_train_streaks += 1
                else:
                    num_train_streaks = 0
                print("Number of consecutive wins: ", num_train_streaks)
                break
            
        # It's considered done when it's solved over _ times consecutively
        if num_train_streaks > STREAK_TO_END:
            print('Training done at (episode -100):', episode-100)
            print('Q-table:', q_table)
            break

        # Update parameters at end of every episode
        explore_rate = get_explore_rate(episode)
        learning_rate = get_learning_rate(episode)


# def test():

#     num_test_streaks = 0

#     for episode in range(NUM_TEST_EPISODES):

#         # Reset the environment
#         obv = env.reset()

#         # the initial state
#         state_0 = state_to_bucket(obv)

#         # basic initializations
#         tt = 0
#         done = False

#         while ((abs(obv[0]) < 2.4) & (abs(obv[2]) < 45)):
#         # while not(done):
#             tt += 1
#             # env.render()

#             # Select an action
#             action = select_action(state_0, 0)
#             # action = select_action(state_0, TEST_RAND_PROB)
#             # action = select_action(state_0, 0.01)

#             # Execute the action
#             obv, reward, done, _ = env.step(action)

#             # Observe the result
#             state_0 = state_to_bucket(obv)

#             print("Test episode %d; time step %f." % (episode, tt))


def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


# explore rate = choose between bigger value of min_explore_rate vs smaller value of ( 1 vs 1-log(current_timestep+1) )
# 1-log(current_timestep+1) decreases as timestep t increases (from <1 --> >1)
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t + 1) / 25)))


def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t + 1) / 25)))


def state_to_bucket(state):
    bucket_indice = []
    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == "__main__":
    print("Training ...")
    train()
    # print('Testing ...')
    # test()
