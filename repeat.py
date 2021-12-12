import subprocess 
for i in range(25):
    subprocess.call(['python', "cartpole_v0_qlearning.py"])
    subprocess.call(['python', "cartpole_v0_sarsa.py"])    
    subprocess.call(['python', "cartpole_v0_gene_meanWeightCO_ff1.py"])
    subprocess.call(['python', "cartpole_v0_gene_randWeightCO_ff1.py"])
    subprocess.call(['python', "cartpole_v0_gene_uniformCO_ff1.py"])
    subprocess.call(['python', "cartpole_v0_gene_meanWeightCO_ff2.py"])
    subprocess.call(['python', "cartpole_v0_gene_randWeightCO_ff2.py"])
    subprocess.call(['python', "cartpole_v0_gene_uniformCO_ff2.py"])
print ('done')
