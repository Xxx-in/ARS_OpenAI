import os 
for i in range(25):
    os.system("cartpole_v0_qlearning.py")
    os.system("cartpole_v0_sarsa.py")    
    os.system("cartpole_v0_gene_meanWeightCO_ff1.py")
    os.system("cartpole_v0_gene_qWeightCO_ff1.py")
    os.system("cartpole_v0_gene_randWeightCO_ff1.py")
    os.system("cartpole_v0_gene_uniformCO_ff1.py")
print ('done')
