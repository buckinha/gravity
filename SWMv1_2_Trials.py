"""SWM v1.2 Trials"""

import MDP, MDP_opt, SWMv1_2, HKB_Heuristics, random, numpy, datetime, HKB_Heuristics


def pathway_value_graph_1(pathway_count_per_point, timesteps, p0_range=[-20,20], p1_range=[-20,20], p0_step=0.5, p1_step=0.5, PROBABILISTIC_CHOICES=True, OUTPUT_FOR_SCILAB=True):
    """Step through the policy space and get the monte carlo net values at each policy point"""

    start_time = "Started:  " + str(datetime.datetime.now())

    #get step counts and starting points
    p0_step_count = int(  abs(p0_range[1] - p0_range[0]) / p0_step  ) + 1
    p1_step_count = int(  abs(p1_range[1] - p1_range[0]) / p1_step  ) + 1
    p0_start = p0_range[0]
    if p0_range[1] < p0_range[0]: p0_start = p0_range[1]
    p1_start = p1_range[0]
    if p1_range[1] < p1_range[0]: p1_start = p1_range[1]


    #create the rows/columns structure
    p1_rows = [None] * p1_step_count
    for i in range(p1_step_count):
        p1_rows[i] = [None] * p0_step_count


    #step through the polcies and generate monte carlo rollouts, and save their average value
    pathways = [None]*pathway_count_per_point

    for row in range(p1_step_count):
        for col in range(p0_step_count):
            p0_val = p0_start + col*p0_step
            p1_val = p1_start + row*p1_step
          
            for i in range(pathway_count_per_point):
                pathways[i] = SWMv1_2.simulate(timesteps=timesteps, policy=[p0_val,p1_val,0], random_seed=(5000+i), SILENT=True, PROBABILISTIC_CHOICES=PROBABILISTIC_CHOICES)

            #get the average value
            val_sum = 0.0
            for i in range(pathway_count_per_point):
                val_sum += pathways[i]["Average State Value"]

            val_avg = val_sum / pathway_count_per_point

            p1_rows[row][col] = val_avg


    end_time = "Finished: " + str(datetime.datetime.now())

    #finished gathering output strings, now write them to the file
    f = open('pathway_value_graph_1.txt', 'w')

    #Writing Header
    f.write("SWMv1_2_Trials.pathway_value_graph_1()\n")
    f.write(start_time + "\n")
    f.write(end_time + "\n")
    f.write("Pathways per Point: " + str(pathway_count_per_point) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("P0 Range: " + str(p0_range) +"\n")
    f.write("P1 Range: " + str(p1_range) +"\n")

    #writing model parameters from whatever's still in the pathway set.
    # the model parameters don't change from one M.C batch to another.
    f.write("\n")
    f.write("Vulnerability Min: " + str(pathways[0]["Vulnerability Min"]) + "\n")
    f.write("Vulnerability Max: " + str(pathways[0]["Vulnerability Max"]) + "\n")
    f.write("Vulnerability Change After Suppression: " + str(pathways[0]["Vulnerability Change After Suppression"]) + "\n")
    f.write("Vulnerability Change After Mild: " + str(pathways[0]["Vulnerability Change After Mild"]) + "\n")
    f.write("Vulnerability Change After Severe: " + str(pathways[0]["Vulnerability Change After Severe"]) + "\n")
    f.write("Timber Value Min: " + str(pathways[0]["Timber Value Min"]) + "\n")
    f.write("Timber Value Max: " + str(pathways[0]["Timber Value Max"]) + "\n")
    f.write("Timber Value Change After Suppression: " + str(pathways[0]["Timber Value Change After Suppression"]) + "\n")
    f.write("Timber Value Change After Mild: " + str(pathways[0]["Timber Value Change After Mild"]) + "\n")
    f.write("Timber Value Change After Severe: " + str(pathways[0]["Timber Value Change After Severe"]) + "\n")
    f.write("Suppression Cost - Mild: " + str(pathways[0]["Suppression Cost - Mild"]) + "\n")
    f.write("Suppression Cost - Severe: " + str(pathways[0]["Suppression Cost - Severe"]) + "\n")
    f.write("Severe Burn Cost: " + str(pathways[0]["Severe Burn Cost"]) + "\n")
    f.write("\n")

    if not OUTPUT_FOR_SCILAB:
        #Writing Data for Excel
        f.write(",,Parameter 0\n")
        f.write(",,")
        for i in range(p0_step_count):
            f.write( str( p0_start + i*p0_step ) + ",")
        f.write("\n")
        f.write("Paramter 1")
        for row in range(p1_step_count):
            f.write(",")
            #write the p1 value
            f.write( str( p1_start + row*p1_step ) + "," )

            for col in range(p0_step_count):
                f.write( str(p1_rows[row][col]) + "," )
            f.write("\n")
    else:
        #Writing Data for Scilab
        f.write("Scilab Matrix\n")
        for row in range(p1_step_count):

            for col in range(p0_step_count):
                f.write( str(p1_rows[row][col]) + " " )
            f.write("\n")



    f.close()

def obj_fn_graph_1(pathway_count_per_point, timesteps, starting_policy, objective_function='J1', p0_range=[-20,20], p1_range=[-20,20], p0_step=0.5, p1_step=0.5, OUTPUT_FOR_SCILAB=True):
    """ Calculates obj.fn. values throughout the given policy space
    """

    start_time = "Started:  " + str(datetime.datetime.now())

    #assign policy
    pol = SWMv1_2.sanitize_policy(starting_policy)


    #set the objective function. Default to J1
    obj_fn = MDP_opt.J1
    if   objective_function == 'J2': obj_fn = MDP_opt.J2
    elif objective_function == 'J3': obj_fn = MDP_opt.J3
    elif objective_function == 'J4': obj_fn = MDP_opt.J4


    #get step counts and starting points
    p0_step_count = int(  abs(p0_range[1] - p0_range[0]) / p0_step  ) + 1
    p1_step_count = int(  abs(p1_range[1] - p1_range[0]) / p1_step  ) + 1
    p0_start = p0_range[0]
    if p0_range[1] < p0_range[0]: p0_start = p0_range[1]
    p1_start = p1_range[0]
    if p1_range[1] < p1_range[0]: p1_start = p1_range[1]


    #create the rows/columns structure
    rows = [None] * p1_step_count
    for i in range(p1_step_count):
        rows[i] = [None] * p0_step_count


    #create pathway set
    random.seed(0)
    pathways = [None] * pathway_count_per_point
    for i in range(pathway_count_per_point):

        #check for policies that are meant to be varied
        p=[0,0,0]
        if starting_policy == "MIXED_CT":
            p[0] = random.uniform(-2,2)
            p[1] = random.uniform(-2,2)
        elif starting_policy == "MIXED_ALL":
            p[0] = random.uniform(p0_range[0],p0_range[1])
            p[1] = random.uniform(p1_range[0],p1_range[1])
        else: p = pol[:]


        pw = SWMv1_2.simulate(timesteps=timesteps, policy=p, random_seed=(6500+i), SILENT=True)
        pathways[i] = SWMv1_2.convert_to_MDP_pathway(pw)

    #get a sample pathway to pull file header information from
    sample_pw = SWMv1_2.simulate(1,pol,random_seed=0,SILENT=True)


    #loop over all rows and columns and populate each point with its obj. fn. value
    for row in range(p1_step_count):
        for col in range(p0_step_count):
            #set policy
            p0_val = p0_start + col*p0_step
            p1_val = p1_start + row*p1_step
            rows[row][col] = obj_fn(policy_vector=[p0_val,p1_val], pathways=pathways, FEATURE_NORMALIZATION=False)


    end_time = "Finished: " + str(datetime.datetime.now())

    #finished gathering output strings, now write them to the file
    f = open('objective_function_graph_1.txt', 'w')

    #Writing Header
    f.write("SWMv1_2_Trials.objective_function_graph_1()\n")
    f.write(start_time + "\n")
    f.write(end_time + "\n")
    f.write("Pathways per Point: " + str(pathway_count_per_point) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("P0 Range: " + str(p0_range) +"\n")
    f.write("P1 Range: " + str(p1_range) +"\n")
    f.write("P0 Step Size: " + str(p0_step) + "\n")
    f.write("P1 Step Size: " + str(p1_step) + "\n")
    f.write("Objective Function: " + str(objective_function) + "\n")

    #writing model parameters from whatever's still in the pathway set.
    # the model parameters don't change from one M.C batch to another.
    f.write("\n")
    f.write("SIMULATION PARAMETERS\n")
    f.write("Vulnerability Min: " + str(sample_pw["Vulnerability Min"]) + "\n")
    f.write("Vulnerability Max: " + str(sample_pw["Vulnerability Max"]) + "\n")
    f.write("Vulnerability Change After Suppression: " + str(sample_pw["Vulnerability Change After Suppression"]) + "\n")
    f.write("Vulnerability Change After Mild: " + str(sample_pw["Vulnerability Change After Mild"]) + "\n")
    f.write("Vulnerability Change After Severe: " + str(sample_pw["Vulnerability Change After Severe"]) + "\n")
    f.write("Timber Value Min: " + str(sample_pw["Timber Value Min"]) + "\n")
    f.write("Timber Value Max: " + str(sample_pw["Timber Value Max"]) + "\n")
    f.write("Timber Value Change After Suppression: " + str(sample_pw["Timber Value Change After Suppression"]) + "\n")
    f.write("Timber Value Change After Mild: " + str(sample_pw["Timber Value Change After Mild"]) + "\n")
    f.write("Timber Value Change After Severe: " + str(sample_pw["Timber Value Change After Severe"]) + "\n")
    f.write("Suppression Cost - Mild: " + str(sample_pw["Suppression Cost - Mild"]) + "\n")
    f.write("Suppression Cost - Severe: " + str(sample_pw["Suppression Cost - Severe"]) + "\n")
    f.write("Severe Burn Cost: " + str(sample_pw["Severe Burn Cost"]) + "\n")
    f.write("\n")

    if not OUTPUT_FOR_SCILAB:
        #Writing Data for Excel
        f.write(",,Parameter 0\n")
        f.write(",,")
        for i in range(p0_step_count):
            f.write( str( p0_start + i*p0_step ) + ",")
        f.write("\n")
        f.write("Paramter 1")
        for row in range(p1_step_count):
            f.write(",")
            #write the p1 value
            f.write( str( p1_start + row*p1_step ) + "," )

            for col in range(p0_step_count):
                f.write( str(rows[row][col]) + "," )
            f.write("\n")
    else:
        #Writing Data for Scilab
        f.write("Scilab Matrix\n")
        for row in range(p1_step_count):

            for col in range(p0_step_count):
                f.write( str(rows[row][col]) + " " )
            f.write("\n")



    f.close()

def SWM_simple_hill_climb(pathway_count=200, timesteps=150, policy="MIXED_CT", objective_function="J3"):

    #sanitize policy
    pol = SWMv1_2.sanitize_policy(policy)

    #create pathways
    pathways = [None] * pathway_count
    for i in range(pathway_count):

        #set up policy
        p0 = pol[0]
        p1 = pol[1]
        #check for interesting policies
        if policy == "MIXED_CT":
            p0 = random.uniform(-2,2)
            p1 = random.uniform(-2,2)
        elif policy == "MIXED_ALL":
            p0 = random.uniform(-20,20)
            p1 = random.uniform(-20,20)
        p = [p0,p1]

        #simulate has a signature of:
        #simulate(timesteps, policy=[0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True)
        pw = SWMv1_2.simulate(timesteps, p, 6500+i, {}, True, True)
        pathways[i] = SWMv1_2.convert_to_MDP_pathway(pw)


    #set up objfn and fprime
    opt = MDP_opt.Optimizer(2)
    opt.pathway_set = pathways
    opt.set_obj_fn(objective_function)
    objfn = opt.calc_obj_fn
    fprime = opt.calc_obj_FPrime

    bounds = [[-20,20],[-20,20]]
    
    x0 = [-1,1]

    #start hill-climb

    #simple gradient has a signature of:
    #def simple_gradient(objfn, fprime, x0, bounds=None, step_size=0.05, MINIMIZING=True, USE_RELATIVE_STEP_SIZES=False, max_steps=200)

    result = HKB_Heuristics.simple_gradient(objfn, fprime, x0, bounds=bounds, step_size=0.1, MINIMIZING=False, USE_RELATIVE_STEP_SIZES=True, max_steps=20)

    
    #finished gathering output strings, now write them to the file
    f = open('simple_hill_climb.txt', 'w')

    #Writing Header
    f.write("SWMv1_2_Trials.simple_hill_climb()\n")
    f.write("Pathways per Point: " + str(pathway_count) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("Objective Function: " + str(objective_function) + "\n")
    f.write("\n")

    f.write("P0 P1 ObjFnVal\n")
    for i in range(len(result["Step Path"])):
        f.write(str(result["Step Path"][i][0]) + " ")
        f.write(str(result["Step Path"][i][1]) + " ")
        f.write(str(result["Value Path"][i]) + "\n")


    f.close()
    
def SWM_simpler_hill_climb(pathway_count=200, timesteps=150, climbing_steps=20, step_size=0.2, policy="MIXED_CT", objective_function="J3", MINIMIZING=False):
    #sanitize policy
    pol = SWMv1_2.sanitize_policy(policy)

    #create pathways
    pathways = [None] * pathway_count
    for i in range(pathway_count):

        #set up policy
        p0 = pol[0]
        p1 = pol[1]
        #check for interesting policies
        if policy == "MIXED_CT":
            p0 = random.uniform(-2,2)
            p1 = random.uniform(-2,2)
        elif policy == "MIXED_ALL":
            p0 = random.uniform(-20,20)
            p1 = random.uniform(-20,20)
        p = [p0,p1]

        #simulate has a signature of:
        #simulate(timesteps, policy=[0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True)
        pw = SWMv1_2.simulate(timesteps, p, 6500+i, {}, True, True)
        pathways[i] = SWMv1_2.convert_to_MDP_pathway(pw)
    
    #default to J3
    objfn = MDP_opt.J3
    fprime = MDP_opt.J3prime
    if objective_function == "J1":
        objfn = MDP_opt.J1
        fprime = MDP_opt.J1prime
    
    
    x0 = [0,0]
    #signature is
    #simpler_hill_climb(        objfn, fprime,    x0, step_size=0.5, MINIMIZING=False, max_steps=20,     objfn_args=None,     fprime_args=None):
    result = HKB_Heuristics.simpler_hill_climb(objfn, fprime,    x0, step_size=step_size, MINIMIZING=MINIMIZING, max_steps=climbing_steps, objfn_args=pathways, fprime_args=pathways)
    
    #finished gathering output strings, now write them to the file
    f = open('SIMPLER_hill_climb.txt', 'w')

    #Writing Header
    f.write("SWMv1_2_Trials.simpler_hill_climb()\n")
    f.write("\n")
    f.write("PATHWAY SET INFORMATION:\n")
    f.write("Pathways Count: " + str(pathway_count) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("Policy: " + str(policy) + "\n")
    f.write("x0: " + str(x0) + "\n")
    f.write("\n")
    f.write("HILLCLIMBING INFORMATION:\n")
    f.write("Hill-climbing steps: " + str(climbing_steps) + "\n")
    f.write("Step Size: " + str(step_size) + "\n")
    f.write("Objective Function: " + str(objective_function) + "\n")
    if MINIMIZING:
        f.write("HKB_Heuristics.simpler_hill_climb() is set to MINIMIZE\n")
    else:
        f.write("HKB_Heuristics.simpler_hill_climb() is set to MAXIMIZE\n")
    f.write("\n")

    f.write("P0 P1 ObjFnVal\n")
    for i in range(len(result["Path"])):
        f.write(str(result["Path"][i][0]) + " ")
        f.write(str(result["Path"][i][1]) + " ")
        f.write(str(result["Values"][i]) + "\n")


    f.close()
    

def SWM_hill_climb(pathway_count=200, timesteps=150, climbing_steps=20, step_size=0.2, small_step_size=0.04, policy="MIXED_CT", objective_function="J3", MINIMIZING=False):

    

    #sanitize policy
    pol = SWMv1_2.sanitize_policy(policy)


    #create pathways
    random.seed(0)
    pathways = [None] * pathway_count
    for i in range(pathway_count):

        #set up policy
        p0 = pol[0]
        p1 = pol[1]
        #check for interesting policies
        if policy == "MIXED_CT":
            p0 = random.uniform(-2,2)
            p1 = random.uniform(-2,2)
        elif policy == "MIXED_ALL":
            p0 = random.uniform(-20,20)
            p1 = random.uniform(-20,20)
        p = [p0,p1]

        #simulate has a signature of:
        #simulate(timesteps, policy=[0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True)
        pw = SWMv1_2.simulate(timesteps, p, 6500+i, {}, True, True)
        pathways[i] = SWMv1_2.convert_to_MDP_pathway(pw)
    
    #default to J3
    objfn = MDP_opt.J3
    fprime = MDP_opt.J3prime
    if objective_function == "J1":
        objfn = MDP_opt.J1
        fprime = MDP_opt.J1prime
    
    
    x0 = [0,0]
    #signature is
    #                       hill_climb(objfn, x0, step_size=0.1,       small_step_size=0.02,            greatest_disimprovement=0.95, MINIMIZING=False,      max_steps=20,             objfn_arg=None)
    result = HKB_Heuristics.hill_climb(objfn, x0, step_size=step_size, small_step_size=small_step_size, greatest_disimprovement=0.9, MINIMIZING=MINIMIZING, max_steps=climbing_steps, objfn_arg=pathways) 
    
    #finished gathering output strings, now write them to the file
    f = open('SWM_hill_climb.txt', 'w')

    #Writing Header
    f.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f.write("\n")
    f.write("PATHWAY SET INFORMATION:\n")
    f.write("Pathways Count: " + str(pathway_count) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("Policy: " + str(policy) + "\n")
    f.write("\n")
    f.write("HILLCLIMBING INFORMATION:\n")
    f.write("Hill-climbing steps: " + str(climbing_steps) + "\n")
    f.write("Step Size: " + str(step_size) + "\n")
    f.write("Small Step Size: " + str(small_step_size) + "\n")
    f.write("Objective Function: " + str(objective_function) + "\n")
    f.write("x0: " + str(x0) + "\n")
    if MINIMIZING:
        f.write("HKB_Heuristics.hill_climb() is set to MINIMIZE\n")
    else:
        f.write("HKB_Heuristics.hill_climb() is set to MAXIMIZE\n")
    f.write("\n")

    f.write("P0 P1 ObjFnVal\n")
    for i in range(len(result["Path"])):
        #check to see if there is, in fact, a path position here
        if result["Path"][i]:
            f.write(str(result["Path"][i][0]) + " ")
            f.write(str(result["Path"][i][1]) + " ")
            f.write(str(result["Values"][i]) + "\n")


    f.close()