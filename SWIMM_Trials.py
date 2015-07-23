"""SWIMM Trials"""

import MDP, SWIMM, HKB_Heuristics, random, numpy, MDP_opt

def MDP_optimization(pathway_count=300, timesteps=100, start_ID=0, policy=[0,0,0]):
    """Create a SWIMM pathway set and attempt a basic policy optimization"""
    
    #creating pathways
    pathways = [None] * pathway_count
    for i in range(pathway_count):
        pw = SWIMM.simulate(timesteps=timesteps, random_seed=start_ID+i, policy=policy, SILENT=True)
        pathways[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(pw)
    
    
    opt = MDP_opt.Optimizer(2)
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
    
    
    opt = MDP_opt.Optimizer(2)
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
    
    
    opt = MDP_opt.Optimizer(2)
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

def derivitive_graph_1(pathway_count_per_point, timesteps, p1_range=[-1,1], p2_range=[-500,-2500], p1_step=1, p2_step=100, best_pol=[-1590,20,0]):
    """Outputs data for building a graph of the behavior of the derivitive near the optimal policy

    Output File: derivitive_graph_1_output.txt

    """

    total_steps_p1 = int (   abs(p1_range[1] - p1_range[0]) / p1_step   )
    total_steps_p2 = int (   abs(p2_range[1] - p2_range[0]) / p2_step   )

    #setting start-point to the lower of the two
    p1_start = p1_range[0]
    if p1_range[1] < p1_range[0]: p1_start = p1_range[1]
    p2_start = p2_range[0]
    if p2_range[1] < p2_range[0]: p2_start = p2_range[1]

    #to collect output data for writing to derivitive_graph_1_output.csv
    output_strings = []
    start_time = "Beginning Analysis:" + str(datetime.datetime.now())
    print(start_time)

    #to calculate the derivitives, use an optimizer object
    opt = MDP_opt.Optimizer(2)

    for p1 in range(total_steps_p1 + 1):
        for p2 in range(total_steps_p2 + 1):

            p1_val = p1_start + p1_step*p1
            p2_val = p2_start + p2_step*p2

            pol = [p1_val,p2_val,0]

            pathways = [None] * pathway_count_per_point

            for i in range(pathway_count_per_point):
                pathways[i] = SWIMM.simulate(timesteps, policy=pol, random_seed=(p1*1000 + p2), SILENT=True)
                pathways[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(pathways[i])

            #get the derivitive
            opt.pathway_set = pathways
            opt.Policy.set_params(pol[:2])
            deriv = opt.calc_obj_FPrime()


            #Basic Outputs
            s = str(p1_val) + "," + str(p2_val) + "," + str(deriv[0]) + "," + str(deriv[1]) 

            #Derived Outputs
            # 2 means derivitive is zero, at the optima
            # 1 means derivitive pointing correctly toward the optima
            # 0 means derivitive is zero other than at the optima
            # -1 means derivitive is pointing incorrectly (including non-zero at the optima)

            #check derivitive 0
            if (p1_val < best_pol[0]) and (round(deriv[0],8) > 0):
                #pointing up, from beneath =)
                s = s + "," + str(1)
            elif (p1_val < best_pol[0]) and (round(deriv[0],8) < 0):
                #pointing down, from beneath =(
                s = s + "," + str(-1)
            elif (p1_val > best_pol[0]) and (round(deriv[0],8) > 0):
                #pointing up, from above =(
                s = s + "," + str(-1)
            elif (p1_val > best_pol[0]) and (round(deriv[0],8) < 0):
                #pointing down, from above =)
                s = s + "," + str(1)
            elif (p1_val == best_pol[0]):
                if (round(deriv[0],8) == 0):
                    #at optima, derivitive neutral =)
                    s = s + "," + str(2)
                else:
                    #at optima, derivitive non-neutral =(
                    s = s + "," + str(-1)
            elif (round(deriv[0], 8) == 0):
                #derivitive is neutral, somewhere away from the optima
                s = s + "," + str(0)


            #check derivitive 1
            if (p2_val < best_pol[1]) and (round(deriv[1],8) > 0):
                #pointing up, from beneath =)
                s = s + "," + str(1)
            elif (p2_val < best_pol[1]) and (round(deriv[1],8) < 0):
                #pointing down, from beneath =(
                s = s + "," + str(-1)
            elif (p2_val > best_pol[1]) and (round(deriv[1],8) > 0):
                #pointing up, from above =(
                s = s + "," + str(-1)
            elif (p2_val > best_pol[1]) and (round(deriv[1],8) < 0):
                #pointing down, from above =)
                s = s + "," + str(1)
            elif (p2_val == best_pol[1]):
                if (round(deriv[1],8) == 0):
                    #at optima, derivitive neutral =)
                    s = s + "," + str(2)
                else:
                    #at optima, derivitive non-zero =(
                    s = s + "," + str(-1)
            elif (round(deriv[1], 8) == 0):
                #derivitive is neutral, somewhere away from the optima
                s = s + "," + str(0)
             
            output_strings.append(s)

    end_time = "Finished:" + str(datetime.datetime.now())
    print(end_time)
    
    #finished gathering output strings, now write them to the file
    f = open('derivitive_graph_1_output.txt', 'w')
    f.write("SWIMM_Trials.derivitive_graph_1()\n")
    f.write(start_time + "\n")
    f.write(end_time + "\n")
    f.write("Pathways per Point: " + str(pathway_count_per_point) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("P1 Range: " + str(p1_range) +"\n")
    f.write("P2 Range: " + str(p2_range) +"\n")
    f.write("Best Policy: " + str(best_pol) +"\n")
    f.write("\n")
    f.write("DATA\n")
    f.write("P1,P2,dP1/dObj,dP2/dObj,P1_STATUS,P2_STATUS\n")
    for i in range(len(output_strings)):
        f.write(output_strings[i] + "\n")
    f.close

def obj_fn_graph_1(pathway_count_per_point, timesteps, starting_policy, objective_function='J1', p0_range=[-1,1], p1_range=[-1,1], p0_step=0.1, p1_step=0.1):
    """ Calculates obj.fn. values throughout the given policy space
    """

    start_time = "Finished:" + str(datetime.datetime.now())

    #assign policy
    pol = []
    if isinstance(starting_policy, list):
        if len(starting_policy) == 2:
            #it's length-2, so add the shift parameter
            pol = starting_policy + [0]
        else:
            #it's probably length-3, so just assign it
            pol = starting_policy
    else:
        #it's not a list, so find out what string it is
        if starting_policy == 'LB':    pol = [-20,0,0]
        elif starting_policy == 'SA':  pol = [ 20,0,0]
        elif starting_policy == 'CT':  pol = [  0,0,0]


    #set the objective function. Default to J1
    obj_fn = MDP_opt.J1
    if   objective_function == 'J2': obj_fn = MDP_opt.J2
    elif objective_function == 'J3': obj_fn = MDP_opt.J3
    elif objective_function == 'J4': obj_fn = MDP_opt.J4


    #get step counts and starting points
    p0_step_count = (  abs(p0_range[1] - p0_range[0]) / p0_step  ) + 1
    p1_step_count = (  abs(p1_range[1] - p1_range[0]) / p1_step  ) + 1
    p0_start = p0_range[0]
    if p0_range[1] < p0_range[0]: p0_start = p0_range[1]
    p1_start = p1_range[0]
    if p1_range[1] < p1_range[0]: p1_start = p1_range[1]


    #create the rows/columns structure
    p1_rows = [None] * p1_step_count
    for i in range(p1_step_count):
        p1_rows[i] = [None] * p0_step_count


    #create pathway set
    pathways = [None] * pathway_count_per_point
    for i in range(pathway_count_per_point):
        pw = SWIMM.simulate(timesteps=timesteps, policy=pol, random_seed=(5000+i))
        pathways[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(pw)


    #loop over all rows and columns and populate each point with its obj. fn. value
    for row in range(p1_step_count):
        for col in range(p0_step_count):
            #set policy
            p0_val = p0_start + col*p0_step
            p1_val = p1_start + row*p1_step
            p1_rows[row][col] = obj_fn(policy=[p0_val,p1_val], pathways=pathways, FEATURE_NORMALIZATION=False)


    end_time = "Finished:" + str(datetime.datetime.now())

    #finished gathering output strings, now write them to the file
    f = open('objective_function_graph_1.txt', 'w')

    #Writing Header
    f.write("SWIMM_Trials.objective_function_graph_1()\n")
    f.write(start_time + "\n")
    f.write(end_time + "\n")
    f.write("Pathways per Point: " + str(pathway_count_per_point) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("P1 Range: " + str(p1_range) +"\n")
    f.write("P2 Range: " + str(p2_range) +"\n")
    f.write("Starting Policy: " + str(starting_policy) +"\n")
    f.write("\n")

    #Writing Data
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

    f.close