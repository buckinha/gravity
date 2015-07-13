"""SWIMM Trials"""

import MDP, SWIMM, HKB_Heuristics, random
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
    
    for i in range(pathway_count):
        pol = None
        if OLD_POLICY_STYLE:
            pol = [random.uniform(-100,100), random.uniform(-20,20), 0]
        else:
            pol = [0, random.uniform(-20,20), random.uniform(-100,100)]
        
        pws[i] = SWIMM.simulate(timesteps, policy=pol, random_seed=i, SILENT=True)
        
    SWIMM.simulate(timesteps, policy=[0,20,-75], random_seed = pathway_count+1)
    
    for i in range(pathway_count):
        print(pws[i]["Average State Value"])
    
    