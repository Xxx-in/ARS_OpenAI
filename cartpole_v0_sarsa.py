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

# Number of discrete states (bins/bucket) per state dimension 
NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')

# Number of discrete actions
NUM_ACTIONS = env.action_space.n  # (left, right)

# Bounds for each discrete state
# STATE_BOUNDS = [(LB of cart position, UB of cart position), (cart velocity), (pole angle), (pole tip velocity)]
STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
STATE_BOUNDS[1] = (-0.5, 0.5)
STATE_BOUNDS[3] = (-math.radians(50), math.radians(50))
# Index of the action
ACTION_INDEX = len(NUM_BUCKETS) #4

## Creating a (global) Q-Table for each state-action pair
q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

rewards_record = np.zeros(100) # array to store reward obtained in latest 100 episode
num_train_streaks = 0   # total consecutive episodes with reward >= 195

## Learning related constants
# Continue here
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
TEST_RAND_PROB = 0.2

## Defining the simulation related constants
MAX_TRAIN_T = 200  # how many steps for 1 training episode - 200
MAX_TEST_T = 200  # how many steps for 1 testing episode - 200
STREAK_TO_END = 100  # how many consecutive wins to end
SOLVED_T = 195  # how many rewards/ time steps which pole stayed upright to consider as 1 successful episode
VERBOSE = False  # print data

#train 1 episode
def train(episode, q_table):

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(episode)
    explore_rate = get_explore_rate(episode)
    discount_factor = 0.99  # since the world is unchanging

    # Reset the environment
    obv = env.reset()

    # the initial state
    state_0 = state_to_bucket(obv)

    # Select an action
    action_0 = select_action(state_0, explore_rate)
    
    #for each timestep in 1 episode
    for t in range(MAX_TRAIN_T):
        # env.render()
        
        # Execute the action
        obv, reward, done, _ = env.step(action_0)

        # Observe the result
        state_1 = state_to_bucket(obv)
        
        # Select an action
        action_1 = select_action(state_1, explore_rate)

        # Update the Q based on the result
        q_table[state_0 ][action_0] += learning_rate*(reward + discount_factor*(q_table[state_1][action_1]) - q_table[state_0][action_0])

        # Setting up for the next iteration
        state_0 = state_1
        action_0 = action_1

        # Print data
        if (VERBOSE):
            print("\nEpisode = %d" % episode)
            print("t = %d" % t)
            print("Action: %d" % action)
            print("State: %s" % str(state))
            print("Reward: %f" % reward)
            print("Best Q: %f" % best_q)
            print("Explore rate: %f" % explore_rate)
            print("Learning rate: %f" % learning_rate)
        
        if done:
            rewards_record[episode%100] = t+1
            return q_table
        
#test if q-table has reached convergence
def test(episode, rewards_record, num_train_streaks):

    # record total consecutive successful episode
    if (rewards_record[episode%100] >= SOLVED_T):
        num_train_streaks += 1
    else:
        num_train_streaks = 0

    # problem considered trained when q-table can achieve 100 consecutive successful episodes  
    if num_train_streaks > STREAK_TO_END:
        print("Model trained after %d episodes" % (episode+1-100))

    average = np.mean(rewards_record)
    print('Average:', average)
    
    if average >= 195 and episode > 100:
        problem_solved = True
    else:
        problem_solved = False
        
    return problem_solved, num_train_streaks
         
def select_action(state, explore_rate):
    # Select a random action
    if random.random() < explore_rate:
        action = env.action_space.sample()
    # Select the action with the highest q
    else:
        action = np.argmax(q_table[state])
    return action


def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - math.log10((t+1)/25)))

def get_learning_rate(t):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0 - math.log10((t+1)/25)))

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
            offset = (NUM_BUCKETS[i]-1)*STATE_BOUNDS[i][0]/bound_width
            scaling = (NUM_BUCKETS[i]-1)/bound_width
            bucket_index = int(round(scaling*state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)

if __name__ == "__main__":
    episode = 0
    problem_solved = False  #agent successfully trained
    while (not(problem_solved)):
        print('Training episode', episode, '...')
        train(episode,q_table)
        print('Testing episode', episode, '...')
        problem_solved, num_train_streaks = test(episode,rewards_record,num_train_streaks)
        episode += 1
    print("Episodes before solved: {}".format(episode-100))
    # for i in range(2):
    #     print('Training episode', episode, '...')
    #     train(episode)
    #     episode += 1
    print("Q-table:", q_table)