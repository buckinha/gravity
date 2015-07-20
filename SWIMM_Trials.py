"""SWIMM Trials"""

import MDP, SWIMM, HKB_Heuristics, random, numpy
from MDP_PolicyOptimizer import *

def MDP_optimization(pathway_count=300, timesteps=100, start_ID=0, policy=[0,0,0]):
    """Create a SWIMM pathway set and attempt a basic policy optimization"""
    
    #creating pathways
    pathways = [None] * pathway_count
    for i in range(pathway_count):
        pw = SWIMM.simulate(timesteps=timesteps, random_seed=start_ID+i, policy=policy, SILENT=True)
        pathways[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(pw)
    
    
    opt = MDP_PolicyOptimizer(2)
    opt.pathway_set = pathways
    
    print("")
    print("Optimizing")
    print(opt.optimize_policy())
    
def opt_with_HKB_simple_gradient(pathway_count=300, timesteps=100, start_ID=0, policy=[0,0,0]):
    """Create a SWIMM pathway set and attempt a basic policy optimization with HKB simple gradient"""
    
    #creating pathways
    pathways = [None] * pathway_count
    for i in range(pathway_count):
        pw = SWIMM.simulate(timesteps=timesteps, random_seed=start_ID+i, policy=policy, SILENT=True)
        pathways[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(pw)
    
    
    opt = MDP_PolicyOptimizer(2)
    opt.pathway_set = pathways
    #opt.normalize_pathways()
    
    print("")
    print("Optimizing with simple gradient method")
    x0 = [0,0]
    bounds = [[-20,20],[-100,100]]
    HKB_Heuristics.simple_gradient(opt.calc_obj_fn, opt.calc_obj_FPrime, x0, bounds=bounds, step_size=0.1, MINIMIZING=True, USE_RELATIVE_STEP_SIZES=False)
      
def opt_with_HKB_threshold(pathway_count=300, timesteps=100, start_ID=0, policy=[0,0,0]):
    """Create a SWIMM pathway set and attempt a basic policy optimization with HKB simple gradient"""
    
    #creating pathways
    pathways = [None] * pathway_count
    for i in range(pathway_count):
        pw = SWIMM.simulate(timesteps=timesteps, random_seed=start_ID+i, policy=policy, SILENT=True)
        pathways[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(pw)
    
    
    opt = MDP_PolicyOptimizer(2)
    opt.pathway_set = pathways
    #opt.normalize_pathways()
    
    print("")
    print("Optimizing")
    x0 = [0,0]
    bounds = [[-20,20],[-100,100]]   
    
    
    HKB_Heuristics.threshold(opt.calc_obj_fn, x0, bounds=bounds, iter_cap=500, tolerance=1.1, MINIMIZING=True, SILENT=False)
    
    
def scattered_policies(pathway_count=500, timesteps=1000, OLD_POLICY_STYLE=True):
    """return the average step value from pathways generated under diverse policies"""
    
    pws = [None] * pathway_count
    pols = [None] * pathway_count

    for i in range(pathway_count):
        pol = None
        if OLD_POLICY_STYLE:
            pol = [random.uniform(-2500,-500), random.uniform(0,30), 0]
        else:
            pol = [0, random.uniform(0,20), random.uniform(-100,0)]
        
        pws[i] = SWIMM.simulate(timesteps, policy=pol, random_seed=i, SILENT=True)

        #remember policy
        pols[i] = pol[:]
    
    print("\'Best Policy\' Simulation Complete - Pathway " + str(pathway_count+1))
    SWIMM.simulate(timesteps, policy=[0,20,-75], random_seed = pathway_count+1)

    print("")
    print("\'Let Burn\' Simulation Complete - Pathway " + str(pathway_count+1))
    SWIMM.simulate(timesteps, policy=[-20,0,0], random_seed = pathway_count+1)

    print("")
    print("\'Suppress All\' Simulation Complete - Pathway " + str(pathway_count+1))
    SWIMM.simulate(timesteps, policy=[20,0,0], random_seed = pathway_count+1)
    
    print("")
    print("Policies and their Average Pathway Values")
    for i in range(pathway_count):
        print(str(pols[i]) + ", " + str(pws[i]["Average State Value"]))
    
def scattered_near_best(pathway_count=500, timesteps=1000, OLD_POLICY_STYLE=True, best_pol=None):
    pws = [None] * pathway_count

    #save values greater than 1.2
    better_pols = []

    if OLD_POLICY_STYLE and not best_pol:
        best_pol = [-1590,20,0]
    elif not OLD_POLICY_STYLE and not best_pol:
        best_pol = [0,20,-80]

    for i in range(pathway_count):
        pol = None
        if OLD_POLICY_STYLE:
            pol = [best_pol[0] + random.uniform(-50,50), best_pol[1] + random.uniform(-5,5), 0]
        else:
            pol = [0, best_pol[1] + random.uniform(-5,5), best_pol[2] + random.uniform(-15,15)]
        
        pws[i] = SWIMM.simulate(timesteps, policy=pol, random_seed=i, SILENT=True)
        #save policies that produce better values than average
        if pws[i]["Average State Value"] > 1.2:
            better_pols.append(pol[:])
    
    print("\'Best Policy\' Simulation Complete - Pathway " + str(pathway_count+1))
    SWIMM.simulate(timesteps, policy=[0,20,-75], random_seed = pathway_count+1)

    print("")
    print("\'Let Burn\' Simulation Complete - Pathway " + str(pathway_count+1))
    SWIMM.simulate(timesteps, policy=[-20,0,0], random_seed = pathway_count+1)

    print("")
    print("\'Suppress All\' Simulation Complete - Pathway " + str(pathway_count+1))
    SWIMM.simulate(timesteps, policy=[20,0,0], random_seed = pathway_count+1)
    
    for i in range(pathway_count):
        print(pws[i]["Average State Value"])

    print("")
    print("Better Policies")
    print(better_pols)

def simulate_from_policy_seeds(pathway_count_per_seed=1000, timesteps=500, seeds=[]):

    averages = [None] * len(seeds)

    for s in range(len(seeds)):

        vals = [None] * pathway_count_per_seed
        pol = seeds[s]

        for i in range(pathway_count_per_seed):
            
        
            pw = SWIMM.simulate(timesteps, policy=pol, random_seed=i, SILENT=True)
            vals[i] = pw["Average State Value"]

        averages[s] = numpy.mean(vals)
    
  

    print("")
    print("Average Timestep Value for each Policy")
    for s in range(len(seeds)):
        print(str(seeds[s]) + ", " + str(averages[s]))

