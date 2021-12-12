PT 1 - INTRO:

This is the work of Lim Zhi Xin (student ID: 20204972//OWA: hpyz1) to study the effect of combining SARSA and Q-learning
Reinforment Learning approaches using genetic algorithm as template to combine both methods. 

A generic template for genetic algorithm to combine SARSA + Q-Learning:
--------------------------------------
START
Generate the initial population 		// initial population size = 4: 1 ql, 1 sarsa, 2 child ql+sarsa
Compute fitness					// fitness function is calculated as total reward when each ep starts at diff state
REPEAT
    Selection					// select 2 solution as chromosome out of 4 population
    Crossover					// refer to part 2 for diff types of crossover
    Mutation
    Train each population with QL/SARSA
    Compute fitness
UNTIL population has converged
STOP
---------------------------------------

Training is considered done when agent successfully obtain total reward > 195 in 100 consecutive episodes

_____________________________________________________________________________________________________________________________

PART 2 - OPENAI ENV USED

OpenAIGym Environment used: CartPole-v0

__________________________________________________________________________________________________________________________

PT 3 - FILES INCLUDED IN FOLDER:

.gym						: OpenAi gym repository
cartpole_v0_qlearning.py			: Applying traditional Q-learning methods in env
cartpole_v0_sarsa.py   				: Applying traditional SARSA methods in env
cartpole-gene-uniformCO.py			: Applying genetic algorithm & @ crossover, child q_table has 50% to inherit 
						  q_value(state,action) from either parent q_table 1 or parent q_table2
cartpole-gene-meanMergeCO.py			: Applying genetic algorithm & @ crossover, child inherits mean of q_value(state,action) 
						  of both parent q_table **
cartpole-gene-mergeRandWeightCO.py		: Applying genetic algorithm & @ crossover, child q_table inherits random weightage x of 
						  q_value(action, state) from parent q_table1 and (1-x) from q_table2
cartpole-gene-qWeightCO.py				: Applying genetic algorithm $ @ crossver, child inherits (x/y) depending on ratio of 
						  q_value in q_table1 to q_table_2 **
repeat.py					: To run all above files for 50 times

**Although child q_table will be the same due to crossover, but the population size is still  kept at 4 due to diff training methods 
on child q_table
(child1_q_table is trained with QL // child1_q_table is trained with SARSA)

___________________________________________________________________________________________________________________________

PT 4 - RUNNING .PY FILES

*** Prerequisite: Make sure you have Python3 installed on your device. 
(Follow this link to setup Python: 
https://www.digitalocean.com/community/tutorials/how-to-install-python-3-and-set-up-a-local-programming-environment-on-windows-10)

To check if you have python installed, open command-line interface and type 'python'
Python is already installed in your system if you can see the output below:
	
----------------------------------------------------------------------
Python 2.4.3 (#1, May 18 2006, 07:40:45) 
Type "help", "copyright", "credits" or "license" for more information.
>>
----------------------------------------------------------------------------

1. Type in command 'pip install gym' to install OpenAI gym environment.
2. Open command-line interface. 
3. Navigate to the directory of this folder with 'cd /path/'
4. Run .py file with 'python filename.py'
___________________________________________________________________________________________________________________________

PT 5 - OUTPUT FILE

After running the .py files, some .txt. files are generated to store outcome of the training 

Naming format:

<algorihm>_<crossoverMethodCO>_<data>

Note: <crossoverMethodCO> is only applicable for files generated by genetic algorithm

There are 3 types of possible <data>:
episode : Each int represents no. of episodes required before agent is successfully trained for each trial
q_table: Each 1x1x6x3 array represetns the final optimal q_table at the end of each trial
reward: Each line represents the average of total reward received in the past 100 episodes when following diff q_tables

Note: reward is only applicable for files generated by genetic algorithm
___________________________________________________________________________________________________________________________
PT 6 - HARDWARE USED

OS: Microsoft Windows 10 Pro
Version: 10.0.190643 Build 19043
Model: HP Envy Notebook
System type: x64-based PC
Processor: Intel(R) Core(TM) i7-7500U CPU @ 2.70GHz, 2904 Mhz, 2 Core(s), 4 Logical Processor(s)
Memory: 16GB RAM