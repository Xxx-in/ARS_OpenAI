import gym
from gym.core import ActionWrapper
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plot

# --------- INITIALIZE ENV -----------------
# extend CartPole environment to add new modified reset() function 
# to start at specific environment for fitness testing
from gym.envs.classic_control import CartPoleEnv
class ExtendedCartPoleEnv(CartPoleEnv):
    def fitness_test_reset(self,state):
        self.state = state
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)
    
env = ExtendedCartPoleEnv() # to use fitness_test_reset() function
env2 = gym.make("CartPole-v0")  # to use original step() function 
                                # extended class of CartPole env is not registered in OpenAI Gym, hence ExtendedCartPoleEnv.step() function is inaccurate
    
# Initializing the random number generator
np.random.seed(int(time.time()))


# --------------- DEFINING CONSTANTS -----------------

##ENV RELATED CONSTANTS
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


## LEARNING RELATED CONSTANTS
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
TEST_RAND_PROB = 0.2


## SIMULATED RELATED CONSTANTS
MAX_TRAIN_T = 200  # how many steps for 1 training episode - 200
MAX_TEST_T = 200  # how many steps for 1 testing episode - 200
STREAK_TO_END = 100  # how many consecutive wins to end
SOLVED_T = 195  # how many rewards/ time steps which pole stayed upright to consider as 1 successful episode

# ---------------- RL FUNCTIONS -----------------

# train 1 episode using sarsa method
def sarsa_train(episode, q_table):

    # Instantiating the learning related parameters
    learning_rate = get_learning_rate(episode)
    explore_rate = get_explore_rate(episode)
    discount_factor = 0.99  # since the world is unchanging

    # Reset the environment
    obv = env.reset()

    # the initial state
    state_0 = state_to_bucket(obv)

    # Select an action
    action_0 = select_action(state_0, explore_rate, q_table)
    
    #for each timestep in 1 episode
    for t in range(MAX_TRAIN_T):
        # env.render()
        
        # Execute the action
        obv, reward, done, _ = env.step(action_0)

        # Observe the result
        state_1 = state_to_bucket(obv)
        
        # Select an action
        action_1 = select_action(state_1, explore_rate, q_table)

        # Update the Q based on the result
        q_table[state_0 ][action_0] += learning_rate*(reward + discount_factor*(q_table[state_1][action_1]) - q_table[state_0][action_0])

        # Setting up for the next iteration
        state_0 = state_1
        action_0 = action_0
        
        if done:
            break
    return q_table, t+1
        
#train 1 episode using ql method
def ql_train(episode,q_table):

    ## Instantiating the learning related parameters
    learning_rate = get_learning_rate(episode)
    explore_rate = get_explore_rate(episode)
    discount_factor = 0.99  # since the world is unchanging

    # Reset the environment
    obv = env.reset()

    # the initial state
    state_0 = state_to_bucket(obv)

    #for each timestep in 1 episode
    for t in range(MAX_TRAIN_T):
        # env.render()

        # Select an action
        action = select_action(state_0, explore_rate, q_table)

        # Execute the action
        obv, reward, done, _ = env.step(action)

        # Observe the result
        state = state_to_bucket(obv)

        # Update the Q based on the result
        best_q = np.amax(q_table[state])
        q_table[state_0 + (action,)] += learning_rate*(reward + discount_factor*(best_q) - q_table[state_0 + (action,)])

        # Setting up for the next iteration
        state_0 = state
        
        if done:
            break
        
    return q_table, t+1
         

def select_action(state, explore_rate, q_table):
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

# --------------- GENE ALGO FUNCTIONS -----------------

# take avg reward obtained in recent 10 rounds as fitness value 
def fitness_function(rewards_record, episode):
    x = []
    
    # extract reward of recent 10 rounds & calcualte average
    for i in range(10):
        x.append(rewards_record[((episode-9)%100)+i])
    return np.mean(x)

def q_weighted_crossover (parent1_q_table, parent2_q_table):
    child1_q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
    child2_q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
    for x in range(NUM_BUCKETS[2]):
        for y in range(NUM_BUCKETS[3]):
            for z in range(NUM_ACTIONS): 
                if (random.random() <= 0.95):    # determine if crossover happens 
                    #if q1[state] = 0 vs q2[state] > 0
                    if (parent1_q_table[0][0][x][y][z] == 0):
                        child1_q_table[0][0][x][y][z] = parent2_q_table[0][0][x][y][z]
                        child2_q_table[0][0][x][y][z] = parent2_q_table[0][0][x][y][z]
                        break
                    #if q1[state] > 0 vs q2[state] = 0
                    if (parent2_q_table[0][0][x][y][z] == 0):
                        child1_q_table[0][0][x][y][z] = parent1_q_table[0][0][x][y][z]
                        child2_q_table[0][0][x][y][z] = parent1_q_table[0][0][x][y][z]
                        break
                            
                    if (parent1_q_table[0][0][x][y][z] < parent2_q_table[0][0][x][y][z]):
                        weight1 = parent1_q_table[0][0][x][y][z] / parent2_q_table[0][0][x][y][z]
                        weight2 = (1 - weight1)
                        child1_q_table[0][0][x][y][z] = np.min([weight1,weight2]) * parent1_q_table[0][0][x][y][z] + np.max([weight1,weight2]) * parent2_q_table[0][0][x][y][z]
                        child2_q_table[0][0][x][y][z] = np.min([weight1,weight2]) * parent1_q_table[0][0][x][y][z] + np.max([weight1,weight2]) * parent2_q_table[0][0][x][y][z]
                    elif (parent1_q_table[0][0][x][y][z] > parent2_q_table[0][0][x][y][z]):
                        weight1 = parent2_q_table[0][0][x][y][z] / parent1_q_table[0][0][x][y][z]
                        weight2 = (1 - weight1)
                        child1_q_table[0][0][x][y][z] = np.max([weight1,weight2]) * parent1_q_table[0][0][x][y][z] + np.min([weight1,weight2]) * parent2_q_table[0][0][x][y][z]
                        child2_q_table[0][0][x][y][z] = np.max([weight1,weight2]) * parent1_q_table[0][0][x][y][z] + np.min([weight1,weight2]) * parent2_q_table[0][0][x][y][z]
                    else:
                        child1_q_table[0][0][x][y][z] = parent1_q_table[0][0][x][y][z]
                        child2_q_table[0][0][x][y][z] = parent1_q_table[0][0][x][y][z]


        return child1_q_table, child2_q_table

    
def selection(q1,q2,q3,q4,avgq1,avgq2,avgq3,avgq4):
    reward_list = [np.mean(avgq1),np.mean(avgq2),np.mean(avgq3),np.mean(avgq4)] # list of average fitness value of all solution in population
    q_list = [q1, q2, q3, q4] # list of all solution in population
    top1_index = reward_list.index(max(reward_list)) # find index of highest average reward 
    top1_q = q_list[top1_index]
    reward_list.pop(top1_index) # when found, remove from list (to find 2nd highest)
    q_list.pop(top1_index)
    
    
    top2_index = reward_list.index(max(reward_list)) # find index of highest average reward 
    top2_q = q_list[top2_index]

    return top1_q, top2_q

def mutation(child_q_table):
    for x in range(NUM_BUCKETS[2]):
        for y in range(NUM_BUCKETS[3]):
            for z in range(NUM_ACTIONS): 
                if (random.random() <= 0.001):
                    child_q_table[0][0][x][y][z] *= round(random.uniform(-1.25,1.25),2) 
    return child_q_table
    
   
if __name__ == "__main__":
        
    q = open("geneAlgo_qWeightCO_ff2_qtable.txt", "w")
    r = open("geneAlgo_qWeightCO_ff2_reward.txt", "w")
    e = open("geneAlgo_qWeightCO_ff2_episode.txt", "a")
    episode = 0
    
    #initialize population
    parent1_q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))    #parent1 is fittest q_table
    child1_q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))     #parent2 is 2nd fittest q_table
    child2_q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))
    parent2_q_table = np.zeros(NUM_BUCKETS + (NUM_ACTIONS,))

    parent1_rewards_record = np.zeros(100)
    parent2_rewards_record = np.zeros(100)
    child1_rewards_record = np.zeros(100)
    child2_rewards_record = np.zeros(100)

    parent1_fitness = []
    parent2_fitness = []
    child1_fitness = []
    child2_fitness = []

    #initial population = 1 sarsa + 1 q-learning + 2 children
    parent1_q_table, parent1_rewards_record[episode%100] = ql_train(episode, parent1_q_table)
    parent2_q_table, parent1_rewards_record[episode%100] = sarsa_train(episode, parent2_q_table)
    child1_q_table,child2_q_table = q_weighted_crossover(parent1_q_table, parent2_q_table) # crossover to form another 2 solution
    
    # train population
    for i in range(10):
        if i == 0 and episode%10 == 0:
            pass
        else: 
            episode += 1
        
        child1_q_table, child1_rewards_record[episode%100] = ql_train(episode, child1_q_table)
        child2_q_table, child2_rewards_record[episode%100]  = sarsa_train(episode, child2_q_table)
        parent1_q_table, parent1_rewards_record[episode%100]  = ql_train(episode, parent1_q_table)
        parent2_q_table, parent2_rewards_record[episode%100]  = sarsa_train(episode, parent2_q_table)

    # compute fitness
    parent1_fitness= fitness_function(parent1_rewards_record, episode)
    parent2_fitness = fitness_function(parent2_rewards_record, episode)
    child1_fitness = fitness_function(child1_rewards_record, episode)
    child2_fitness  = fitness_function(child2_rewards_record, episode)
    
     #--loop--
    #training is done if best q_table parent 1 can yield average reward of 195 in 100 consecutive rows  
    while(np.mean(parent1_rewards_record) < 195):
    # for i in range (1000):
        # selection: q_table with highest average return becomes parent, regenerate child
        parent1_q_table, parent2_q_table = selection(parent1_q_table,parent2_q_table,child1_q_table,child2_q_table, parent1_fitness, parent2_fitness, child1_fitness, child2_fitness)

        #crossover
        child1_q_table, child2_q_table = q_weighted_crossover(parent1_q_table, parent2_q_table)
        
        # mutation of child
        child1_q_table = mutation(child1_q_table)
        child2_q_table = mutation(child2_q_table)

        # train population
        for i in range(10):
            
            if i == 0 and episode%10 == 0:
                pass
            else: 
                episode += 1
            
            child1_q_table, child1_rewards_record[episode%100] = ql_train(episode, child1_q_table)
            child2_q_table, child2_rewards_record[episode%100]  = sarsa_train(episode, child2_q_table)
            parent1_q_table, parent1_rewards_record[episode%100]  = ql_train(episode, parent1_q_table)
            parent2_q_table, parent2_rewards_record[episode%100]  = sarsa_train(episode, parent2_q_table)

        # compute fitness
        parent1_fitness= fitness_function(parent1_rewards_record, episode)
        parent2_fitness = fitness_function(parent2_rewards_record, episode)
        child1_fitness = fitness_function(child1_rewards_record, episode)
        child2_fitness  = fitness_function(child2_rewards_record, episode)
        
        print('Episode:', episode, '//  Fitness value of fittest q_table：', round(np.mean(parent1_fitness),4), '//  Reward yield in this episode by fittest q_table：', parent1_rewards_record[episode%100])

        q.write('episode {}\n'. format(episode))
        q.write('Q-table \n Parent1:\n {}\n\n {} \n{} \n\n{}\n {}\n\n {}\n {}\n\n '.format(parent1_q_table, 'Parent2: ', parent2_q_table, 'Child1: ', child1_q_table, 'Child2:', child2_q_table))
        
        r.write('episode {}\n'. format(episode))
        r.write('Avg reward of latest 100:\n {} {} {} {} \n'.format(np.mean(parent1_rewards_record), np.mean(parent2_rewards_record), np.mean(child1_rewards_record), np.mean(child2_rewards_record)))
        
        episode += 1
        
    print('Episode before solved: {}'.format(episode-100))
    
    
    q.write('Episode before solved: {}'.format(episode-100))
    q.close()
    r.write('Episode before solved: {}'.format(episode-100))
    r.close()
    e.write('\n {}'.format(str(episode-100)))
    e.close()
            
        