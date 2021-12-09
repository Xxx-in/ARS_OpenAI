import gym
from gym.core import ActionWrapper
import numpy as np
import random
import math
import time
import matplotlib.pyplot as plot
from operator import attrgetter

# from time import sleep
import gym
import numpy as np
import random
import math
from time import sleep

#import base q-learning and sarsa training file
import cartpole_v0_qlearning as ql
import cartpole_v0_sarsa as sarsa

#add new modified reset() function to start at specific environment for fitness testing
from gym.envs.classic_control import CartPoleEnv

class ExtendedCartPoleEnv(CartPoleEnv):
    def fitness_test_reset(self,state):
        self.state = state
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

# env = gym.make("CartPole-v0-mod")
env = ExtendedCartPoleEnv()

class solution_chromosome:
    def __init__(self):
        self.q_table = np.zeros(ql.NUM_BUCKETS + (ql.NUM_ACTIONS,))
        self.rewards_record = np.zeros([ql.NUM_BUCKETS[2] , ql.NUM_BUCKETS[3]])
        self.avg_reward = 0
        self.current_train_ep_reward = 0
    
#fitness function = average reward obtained when starting from all 18 diff states
def fitness_function(q_table):
    #array to store observation values that corresponding to each bucket of pole velocity (after state discretization)
    pole_angle = [-1, -0.2, 0,	0.1, 0.2, 0.4]
    #array to store observation values that corresponding to each bucket of  angular velocity at pole tip (after state discretization)
    pole_tip_velocity = [-1, -0.4, 0.6]
    #create array which corresponds to size of discretized state space to hold total reward obtained when starting from each state
    rewards_record = np.zeros([ql.NUM_BUCKETS[2] , ql.NUM_BUCKETS[3]])

    # when reset, env is set to each corresponding discretized state bucket
    for x in range(ql.NUM_BUCKETS[2]):
        for y in range(ql.NUM_BUCKETS[3]):
            
            # Reset the environment
            obv = env.fitness_test_reset([0, 0, pole_angle[x], pole_tip_velocity[y]])

            # the initial state
            state_0 = ql.state_to_bucket(obv)
            done = False 
            
            # complete 1 testing episode
            while(not(done)):
            # env.render()

                # Select an action, policy = select action with highest q-value
                action = np.argmax(q_table[state_0])

                # Execute the action
                obv, reward, done, _ = env.step(action)

                # Observe the result
                state = ql.state_to_bucket(obv)
                
                # Setting up for the next iteration
                state_0 = state
                rewards_record[x][y] += reward

    return rewards_record, np.mean(rewards_record)

def uniform_crossover (parent1, parent2):
    child1 = solution_chromosome()
    child2 = solution_chromosome()
    for x in range(ql.NUM_BUCKETS[2]):
        for y in range(ql.NUM_BUCKETS[3]):
            for z in range(ql.NUM_ACTIONS): 
                if (random.random() <= 0.95):    # determine if crossover happens
                    if(random.random() < 0.5):  # determines if inherit which parent's 'gene'  
                        child1.q_table[0][0][x][y][z] = parent1.q_table[0][0][x][y][z]
                        child2.q_table[0][0][x][y][z] = parent2.q_table[0][0][x][y][z]
                    else:
                        child1.q_table[0][0][x][y][z] = parent2.q_table[0][0][x][y][z]
                        child2.q_table[0][0][x][y][z] = parent1.q_table[0][0][x][y][z]
            z = 0 
            y += 1
        y = 0
        x += 1
    return child1, child2
    
def selection(q1,q2,q3,q4):
    reward_list = [q1.avg_reward, q2.avg_reward, q3.avg_reward, q4.avg_reward] # list of average reward of all solution in population
    q_list = [q1, q2, q3, q4] # list of all solution in population
    top1_avg_reward = max(reward_list) # find highest average reward 
    reward_list.remove(top1_avg_reward) # when found, remove from list (to find 2nd highest)
    
    # find corresponding q_table with highest total reward
    for i in q_list: 
        if (i.avg_reward == top1_avg_reward):
            top1_q = i 
            break
            
    top2_avg_reward = max(reward_list)
    # find corresponding q_table with 2nd highest total reward
    for i in q_list:
        if (i.avg_reward == top2_avg_reward):
            top2_q = i 
            break

    return top1_q, top2_q

def mutation(child):
    for x in range(ql.NUM_BUCKETS[2]):
        for y in range(ql.NUM_BUCKETS[3]):
            for z in range(ql.NUM_ACTIONS): 
                if (random.random() <= 0.001):
                    child.q_table[0][0][x][y][z] *= round(random.uniform(-1.25,1.25),2) # mutation for exploration: increase 
    return child.q_table

def converge(avg_record_list): 
    x = (avg_record_list == avg_record_list[0]).sum()
    if x == len(avg_record_list):
        return True
    else: 
        return False
       
    
if __name__ == "__main__":
    episode = 0
    parent1_avg_reward_return = np.zeros(100)
    convergence = False
    # intiialize population
    parent1 = solution_chromosome() 
    parent2 = solution_chromosome()
    child1 = solution_chromosome()
    child2 = solution_chromosome()
    
    #initial population = 1 sarsa + 1 q-learning + 2 children
    parent1.q_table, parent1.current_train_ep_reward = ql.train(episode, parent1.q_table)
    parent2.q_table, parent1.current_train_ep_reward = sarsa.train(episode, parent2.q_table)
    child1,child2 = uniform_crossover(parent1, parent2) # crossover to form another 2 solution chromosome as population
    
    #compute fitness
    parent1.rewards_record, parent1.avg_reward = fitness_function(parent1.q_table)
    parent2.rewards_record, parent2.avg_reward  = fitness_function(parent2.q_table)
    child1.rewards_record, child1.avg_reward  = fitness_function(child1.q_table)
    child2.rewards_record, child2.avg_reward  = fitness_function(child2.q_table)

    #--loop--
    # while(not(convergence)):
    for i in range (200):
        # selection: q_table with highest average return becomes parent, regenerate child
        parent1,parent2 = selection(parent1,parent2,child1,child2)
        # print('parent1:',parent1.q_table,'parent1:', parent2.q_table)
        #crossover
        child1, child2 = uniform_crossover(parent1, parent2)
        
        # mutation of child
        child1.q_table = mutation(child1)
        child2.q_table = mutation(child2)

        #train child
        child1.q_table, child1.current_train_ep_reward = ql.train(episode, child1.q_table)
        child2.q_table, child2.current_train_ep_reward  = sarsa.train(episode, child2.q_table)
        parent1.q_table, parent1.current_train_ep_reward  = ql.train(episode, parent1.q_table)
        parent2.q_table, parent2.current_train_ep_reward  = sarsa.train(episode, parent2.q_table)
        
        #compute fitness function for trained child
        child1.rewards_record, child1.avg_reward  = fitness_function(child1.q_table)
        child2.rewards_record, child2.avg_reward  = fitness_function(child2.q_table)

        #check convergence
        parent1_avg_reward_return[episode%100] = parent1.avg_reward
        convergence = converge(parent1_avg_reward_return)
        episode += 1
        # print('episode', episode) 
        # print(episode, 'table:', parent1.q_table)  
        # print(convergence)
        # print('rewards_record',parent1.rewards_record)
        # print('avg_record',parent1.avg_reward)
        # print(episode, parent1.current_train_ep_reward)
        # if episode == 50:
        #     print(parent1.q_table) 
        #     print(parent1.avg_reward, parent2.avg_reward, child1.avg_reward, child2.avg_reward)
        # print (parent1.q_table)
        # print(parent1.avg_reward, parent2.avg_reward, child1.avg_reward, child2.avg_reward)
        




