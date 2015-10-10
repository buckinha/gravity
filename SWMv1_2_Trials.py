"""SWM v1.2 Trials"""

import MDP, MDP_opt, SWMv1_2, HKB_Heuristics, random, numpy, datetime, HKB_Heuristics
import os.path

def standard_MDP_set(pathway_count, timesteps, policy):
    """
    Generates a set of SWM v1.2 pathways and returns them as a list of MDP pathway objects
    """

    pathways = [None]*pathway_count

    for i in range(pathway_count):
        #first, check for interesting policies. I can't use santize_policy in this case because
        #each pathway needs one and only one policy, and I don't want to upset the random
        #number generation sequence by having them draw their own MIXED_CT or MIXED_ALL policies
        pol = [0.0,0.0,0.0]
        if policy == "MIXED_CT":
            pol[0] = random.uniform(-2.0,2.0)
            pol[1] = random.uniform(-2.0,2.0)
        elif policy == "MIXED_ALL":
            pol[0] = random.uniform(-20.0,20.0)
            pol[1] = random.uniform(-20.0,20.0)
        else:
            #its not one of the strings which imply changing policies, so now use sanitize_policy
            # in case it's "CT", "SA", "LB", etc...
            pol = SWMv1_2.sanitize_policy(policy)


        pw = SWMv1_2.simulate(timesteps,pol,random_seed=i+8500,SILENT=True)
        pathways[i] = SWMv1_2.convert_to_MDP_pathway(pw)

    return pathways

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

def obj_fn_graph_1(pathway_count_per_point, timesteps, starting_policy, objective_function='J3', p0_range=[-20,20], p1_range=[-20,20], p0_step=0.5, p1_step=0.5, pathways=None, OUTPUT_FOR_SCILAB=True, folder=None):
    """ Calculates obj.fn. values throughout the given policy space
    """

    start_time = "Started:  " + str(datetime.datetime.now())

    #assign policy
    pol = SWMv1_2.sanitize_policy(starting_policy)


    #set the objective function. Default to J3
    obj_fn = MDP_opt.J3
    if   objective_function == 'J1': obj_fn = MDP_opt.J1
    elif objective_function == 'J2': obj_fn = MDP_opt.J2
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


    #create pathway set, if none was path
    if not pathways:
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
    #if there's a folder argument, use it
    filename = 'objective_function_graph_1.txt'
    if folder:
        filename = os.path.join(folder, filename)
    f = open(filename, 'w')

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

def obj_fn_graph_from_pathways(pathways, starting_policy_label, objective_function='J3', p0_range=[-20,20], p1_range=[-20,20], p0_step=0.5, p1_step=0.5, OUTPUT_FOR_SCILAB=True,folder=None):
    """Takes a set of pathways and builds a map of the objective function in policy space"""
    #get needed arguments
    pw_ct = len(pathways)
    ts = len(pathways[0].events)

    #call obj_fn_graph_1
    obj_fn_graph_1(pathway_count_per_point=pw_ct,
                   timesteps=ts,
                   starting_policy=starting_policy_label,
                   objective_function=objective_function,
                   p0_range=p0_range,
                   p1_range=p1_range,
                   p0_step=p0_step,
                   p1_step=p1_step,
                   pathways=pathways,
                   OUTPUT_FOR_SCILAB=OUTPUT_FOR_SCILAB,
                   folder=folder)


def obj_fn_graph_2(pathways, objective_function='J3', p0_range=[-20,20], p1_range=[-20,20], p0_step=0.5, p1_step=0.5, OUTPUT_FOR_SCILAB=True, folder="obj_fn_graph_2_outputs"):
    """ Calculates obj.fn. values and weight variances, etc... throughout the policy space

    ARGUEMENTS

    pathways: a list of MDP pathway objects
    objective_function
    p0_range
    p1_range
    p0_step
    p1_step
    OUTPUT_FOR_SCILAB
    folder
    """

    start_time = "Started:  " + str(datetime.datetime.now())

    #set the objective function. Default to J3
    obj_fn = MDP_opt.J3
    if   objective_function == 'J1': obj_fn = MDP_opt.J1
    elif objective_function == 'J2': obj_fn = MDP_opt.J2
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
    w_var = [None] * p1_step_count
    w_std = [None] * p1_step_count
    w_max = [None] * p1_step_count
    w_min = [None] * p1_step_count
    w_ave = [None] * p1_step_count
    for i in range(p1_step_count):
        rows[i] = [0.0] * p0_step_count
        w_var[i] = [0.0] * p0_step_count
        w_std[i] = [0.0] * p0_step_count
        w_max[i] = [0.0] * p0_step_count
        w_min[i] = [0.0] * p0_step_count
        w_ave[i] = [0.0] * p0_step_count

    #get the pathways' starting policies
    starting_pols = [None] * len(pathways)
    for i in range(len(pathways)):
        starting_pols[i] = pathways[i].generation_policy_parameters[:]


    #loop over all rows and columns and populate each point with its obj. fn. value
    for row in range(p1_step_count):
        for col in range(p0_step_count):
            #set policy
            p0_val = p0_start + col*p0_step
            p1_val = p1_start + row*p1_step
            result = obj_fn(policy_vector=[p0_val,p1_val], pathways=pathways, FEATURE_NORMALIZATION=False, VALUE_NORMALIZATION=False, SILENT=True, RETURN_WEIGHTS=True)
            rows[row][col] = result[0]
            w_var[row][col] = numpy.var(result[1])
            w_std[row][col] = numpy.std(result[1])
            w_max[row][col] = max(result[1])
            w_min[row][col] = min(result[1])
            w_ave[row][col] = numpy.mean(result[1])



    end_time = "Finished: " + str(datetime.datetime.now())

    #finished gathering output strings; writing them to the various output files

    #if there's a folder argument, use it
    if folder:
        #check if output folder exists:
        if not os.path.exists(folder):
            os.makedirs(folder)

    f_details = open(os.path.join(folder,"details.txt"), 'w')
    f_objfn = open(os.path.join(folder,"objfn.txt"), 'w')
    f_w_var = open(os.path.join(folder,"weights_variance.txt"), 'w')
    f_w_std = open(os.path.join(folder,"weights_stddev.txt"), 'w')
    f_w_min = open(os.path.join(folder,"weights_min.txt"), 'w')
    f_w_max = open(os.path.join(folder,"weights_max.txt"), 'w')
    f_w_ave = open(os.path.join(folder,"weights_mean.txt"), 'w')
    f_start_pols = open(os.path.join(folder,"starting_policies.txt"),'w')

    #Writing Header
    f_details.write("SWMv1_2_Trials.objective_function_graph_2()\n")
    f_details.write(start_time + "\n")
    f_details.write(end_time + "\n")
    f_details.write("Pathways per Point: " + str(len(pathways)) +"\n")
    f_details.write("Timesteps per Pathway: " + str(len(pathways[0].events)) +"\n")
    f_details.write("P0 Range: " + str(p0_range) +"\n")
    f_details.write("P1 Range: " + str(p1_range) +"\n")
    f_details.write("P0 Step Size: " + str(p0_step) + "\n")
    f_details.write("P1 Step Size: " + str(p1_step) + "\n")
    f_details.write("Objective Function: " + str(objective_function) + "\n")

    #writing model parameters from whatever's still in the pathway set.
    # the model parameters don't change from one M.C batch to another.
    f_details.write("\n")
    f_details.write("SIMULATION PARAMETERS\n")
    f_details.write("Vulnerability Min: " + str(pathways[0].metadata["Vulnerability Min"]) + "\n")
    f_details.write("Vulnerability Max: " + str(pathways[0].metadata["Vulnerability Max"]) + "\n")
    f_details.write("Vulnerability Change After Suppression: " + str(pathways[0].metadata["Vulnerability Change After Suppression"]) + "\n")
    f_details.write("Vulnerability Change After Mild: " + str(pathways[0].metadata["Vulnerability Change After Mild"]) + "\n")
    f_details.write("Vulnerability Change After Severe: " + str(pathways[0].metadata["Vulnerability Change After Severe"]) + "\n")
    f_details.write("Timber Value Min: " + str(pathways[0].metadata["Timber Value Min"]) + "\n")
    f_details.write("Timber Value Max: " + str(pathways[0].metadata["Timber Value Max"]) + "\n")
    f_details.write("Timber Value Change After Suppression: " + str(pathways[0].metadata["Timber Value Change After Suppression"]) + "\n")
    f_details.write("Timber Value Change After Mild: " + str(pathways[0].metadata["Timber Value Change After Mild"]) + "\n")
    f_details.write("Timber Value Change After Severe: " + str(pathways[0].metadata["Timber Value Change After Severe"]) + "\n")
    f_details.write("Suppression Cost - Mild: " + str(pathways[0].metadata["Suppression Cost - Mild"]) + "\n")
    f_details.write("Suppression Cost - Severe: " + str(pathways[0].metadata["Suppression Cost - Severe"]) + "\n")
    f_details.write("Severe Burn Cost: " + str(pathways[0].metadata["Severe Burn Cost"]) + "\n")
    f_details.write("\n")


    f_objfn.write("SWMv1_2_Trials.objective_function_graph_2()\n")
    f_w_var.write("SWMv1_2_Trials.objective_function_graph_2()\n")
    f_w_std.write("SWMv1_2_Trials.objective_function_graph_2()\n")
    f_w_min.write("SWMv1_2_Trials.objective_function_graph_2()\n")
    f_w_max.write("SWMv1_2_Trials.objective_function_graph_2()\n")
    f_w_ave.write("SWMv1_2_Trials.objective_function_graph_2()\n")
    f_start_pols.write("SWMv1_2_Trials.objective_function_graph_2()\n")

    f_objfn.write("Objective Function Values\n\n")
    f_w_var.write("Variance of Pathway Weights\n\n")
    f_w_std.write("Standard Deviation of Pathway Weights\n\n")
    f_w_min.write("Minimum of all the Pathway Weights\n\n")
    f_w_max.write("Maximum of all the Pathway Weights\n\n")
    f_w_ave.write("Mean of the Pathway Weights\n\n")
    f_start_pols.write("Starting Policies of the Underlying Pathway Set\n\n")


    #writing map values into all files
    if not OUTPUT_FOR_SCILAB:
        #Writing Data for Excel

        #writing column labels
        f_objfn.write(",,Parameter 0\n,,")
        f_w_var.write(",,Parameter 0\n,,")
        f_w_std.write(",,Parameter 0\n,,")
        f_w_min.write(",,Parameter 0\n,,")
        f_w_max.write(",,Parameter 0\n,,")
        f_w_ave.write(",,Parameter 0\n,,")
        for i in range(p0_step_count):
            f_objfn.write( str( p0_start + i*p0_step ) + ",")
            f_w_var.write( str( p0_start + i*p0_step ) + ",")
            f_w_std.write( str( p0_start + i*p0_step ) + ",")
            f_w_min.write( str( p0_start + i*p0_step ) + ",")
            f_w_max.write( str( p0_start + i*p0_step ) + ",")
            f_w_ave.write( str( p0_start + i*p0_step ) + ",")

        #writing primary row label
        f_objfn.write("\nParamter 1")
        f_w_var.write("\nParamter 1")
        f_w_std.write("\nParamter 1")
        f_w_min.write("\nParamter 1")
        f_w_max.write("\nParamter 1")
        f_w_ave .write("\nParamter 1")

        for row in range(p1_step_count):
            #write the p1 value (individual row labels)
            f_objfn.write("," + str( p1_start + row*p1_step ) + "," )
            f_w_var.write("," + str( p1_start + row*p1_step ) + "," )
            f_w_std.write("," + str( p1_start + row*p1_step ) + "," )
            f_w_min.write("," + str( p1_start + row*p1_step ) + "," )
            f_w_max.write("," + str( p1_start + row*p1_step ) + "," )
            f_w_ave.write("," + str( p1_start + row*p1_step ) + "," )

            for col in range(p0_step_count):
                #write the grid values
                f_objfn.write( str(rows[row][col]) + "," )
                f_w_var.write( str(w_var[row][col]) + "," )
                f_w_std.write( str(w_std[row][col]) + "," )
                f_w_min.write( str(w_min[row][col]) + "," )
                f_w_max.write( str(w_max[row][col]) + "," )
                f_w_ave.write( str(w_ave[row][col]) + "," )

            #close the row
            f_objfn.write("\n")
            f_w_var.write("\n")
            f_w_std.write("\n")
            f_w_min.write("\n")
            f_w_max.write("\n")
            f_w_ave.write("\n")

    else:
        #Writing Data for Scilab
        f_objfn.write("Scilab Matrix\n")
        f_w_var.write("Scilab Matrix\n")
        f_w_std.write("Scilab Matrix\n")
        f_w_min.write("Scilab Matrix\n")
        f_w_max.write("Scilab Matrix\n")
        f_w_ave.write("Scilab Matrix\n")

        for row in range(p1_step_count):

            for col in range(p0_step_count):
                f_objfn.write( str(rows[row][col]) + " " )
                f_w_var.write( str(w_var[row][col]) + " " )
                f_w_std.write( str(w_std[row][col]) + " " )
                f_w_min.write( str(w_min[row][col]) + " " )
                f_w_max.write( str(w_max[row][col]) + " " )
                f_w_ave.write( str(w_ave[row][col]) + " " )

            #close the row
            f_objfn.write("\n")
            f_w_var.write("\n")
            f_w_std.write("\n")
            f_w_min.write("\n")
            f_w_max.write("\n")
            f_w_ave.write("\n")


    #writing starting policies
    for i in range(len(starting_pols)):
        for j in range(len(starting_pols[i])):
            f_start_pols.write(str(starting_pols[i][j]) + " ")
        #close the row
        f_start_pols.write("\n")


    f_details.close()
    f_objfn.close()
    f_w_var.close()
    f_w_std.close()
    f_w_min.close()
    f_w_max.close()
    f_w_ave.close()
    f_start_pols.close()


def SWM_hill_climb(pathway_count=100, timesteps=150, climbing_steps=20, step_size=0.2, small_step_size=0.04, policy="MIXED_CT", objective_function="J3", MINIMIZING=False, OUTPUT_OBJ_FN_MAP=True):

    start_time = "Started:  " + str(datetime.datetime.now())

    #sanitize policy
    pol = SWMv1_2.sanitize_policy(policy)

    #create pathways
    print("")
    print("Creating pathway set")
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
        #fprime = MDP_opt.J1prime
    

    
    x0 = [0,0]
    #signature ishill_climb(objfn, 
               #            x0, 
               #            objfn_arg=None, 
               #            bounds=None, 
               #            step_size=0.1, 
               #            small_step_size=0.02, 
               #            greatest_disimprovement=0.95,
               #            addl_expl_vectors=10,
               #            starburst_vectors=10,
               #            starburst_mag=2.0,  
               #            MINIMIZING=False, 
               #            max_steps=20):
    
    print("Beginning hill-climbing algorithm")
    print("..time is " + str(datetime.datetime.now()))
    result = HKB_Heuristics.hill_climb(objfn=objfn, 
                                       x0=x0, 
                                       objfn_arg=pathways,
                                       bounds=[[-20,20],[-20,20]],
                                       step_size=step_size, 
                                       small_step_size=small_step_size, 
                                       greatest_disimprovement=0.9, 
                                       addl_expl_vectors=10,
                                       starburst_vectors=20,
                                       starburst_mag=2.0,
                                       MINIMIZING=MINIMIZING, 
                                       max_steps=climbing_steps) 
    
    end_time = "Ended:  " + str(datetime.datetime.now())
    #finished gathering output strings, now write them to the file

    #check if output folder exists:
    folder = "SWM_HC_Outputs" 
    if not os.path.exists(folder):
        os.makedirs(folder)


    if OUTPUT_OBJ_FN_MAP:
        print("Building objective function map")
        print("..time is " + str(datetime.datetime.now()))
        obj_fn_graph_from_pathways(pathways=pathways, 
                                   starting_policy_label=policy, 
                                   objective_function=objective_function, 
                                   p0_range=[-20,20], 
                                   p1_range=[-20,20], 
                                   p0_step=0.5, 
                                   p1_step=0.5, 
                                   OUTPUT_FOR_SCILAB=True,
                                   folder=folder)

    print("")
    print("Process Complete... writing output files")


    #check if output folder exists:
    folder = "SWM_HC_Outputs" 
    if not os.path.exists(folder):
        os.makedirs(folder)

    f_details = open(os.path.join(folder,"details.txt"),'w')
    f_path = open(os.path.join(folder,"path.txt"), 'w')
    f_explore = open(os.path.join(folder,"exploration.txt"),'w')
    f_expl_dis = open(os.path.join(folder,"explr_dis.txt"),'w')
    f_expl_impr = open(os.path.join(folder,"explr_impr.txt"),'w')
    f_starburst = open(os.path.join(folder,"starburst.txt"),'w')
    f_star_impr = open(os.path.join(folder,"star_impr.txt"),'w')
    f_star_dis = open(os.path.join(folder,"star_dis.txt"),'w')

    #Writing Details
    f_details.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_details.write("\n")
    f_details.write(start_time + "\n")
    f_details.write(end_time + "\n")
    f_details.write("\n")
    f_details.write("PATHWAY SET INFORMATION:\n")
    f_details.write("Pathways Count: " + str(pathway_count) +"\n")
    f_details.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f_details.write("Policy: " + str(policy) + "\n")
    f_details.write("\n")
    f_details.write("HILLCLIMBING INFORMATION:\n")
    f_details.write("Hill-climbing steps: " + str(climbing_steps) + "\n")
    f_details.write("Step Size: " + str(step_size) + "\n")
    f_details.write("Small Step Size: " + str(small_step_size) + "\n")
    f_details.write("Objective Function: " + str(objective_function) + "\n")
    f_details.write("x0: " + str(x0) + "\n")
    if MINIMIZING:
        f_details.write("HKB_Heuristics.hill_climb() is set to MINIMIZE\n")
    else:
        f_details.write("HKB_Heuristics.hill_climb() is set to MAXIMIZE\n")
    f_details.close()

    #Writing Pathway information
    f_path.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_path.write("Pathway of the Ascent/Descent\n")
    f_path.write("(Points are duplicated for use in Scilab.xarrows function)")
    f_path.write("\n")
    f_path.write("\n")
    f_path.write("P0 P1 ObjFnVal\n")
    for i in range(len(result["Path"])):
        f_path.write(str(result["Path"][i][0]) + " ")
        f_path.write(str(result["Path"][i][1]) + " ")
        f_path.write(str(result["Values"][i]) + "\n")

        #if this is not the first or the last entry, record the position twice
        #Scilab uses pairs in the vector to define the start and stop position
        #of each arrow, so the format looks like this:
        # [arrow1_start_x, arrow1_end_x, arrow2_start_x, arrow2_end_x, etc..]
        #so in this case, where each arrow starts where the previous ends, 
        # those coordinates will be repeated twice, except for the first and last
        #arrows.
        if (not i==0) and (not i==len(result["Path"])-1):
            f_path.write(str(result["Path"][i][0]) + " ")
            f_path.write(str(result["Path"][i][1]) + " ")
            f_path.write(str(result["Values"][i]) + "\n")

    f_path.close()

    #writing exploration sets

    f_explore.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_explore.write("Exploration Vectors")
    f_explore.write("\n")
    f_expl_impr.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_expl_impr.write("Exploration Improving Vectors")
    f_expl_impr.write("\n")
    f_expl_dis.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_expl_dis.write("Exploration Disimproving Vectors")
    f_expl_dis.write("\n")

    #The key "Exploration History" contains a list, and each element is a dictionary
    #Each dictionary has the following format:
    #
    # exp_hist = {
    #    "Step" : i,
    #    "Origin" : x_current[:],
    #    "Origin Value" : value_current,
    #    "Vectors" : explore_set[:],
    #    "Values" : explore_vals[:]
    #    }

    #loop over each list member
    for group in result["Exploration History"]:
        #for each member, loop over each of it's vectors, and write the start and end points
        # and their associated values

        for v in range(len(group["Vectors"])):

            #check for improvement/disimprovement

            if (
                ((not MINIMIZING) and ( group["Origin Value"] > group["Values"][v] )) or
                ((    MINIMIZING) and ( group["Origin Value"] < group["Values"][v] ))
               ):
                #this was an improving vector

                #write the origin point
                for k in range(len(group["Origin"])):
                    f_expl_dis.write(str(group["Origin"][k]) + " ")
                f_expl_dis.write("\n")

                #write the target point
                for k in range(len(group["Vectors"][v])):
                    f_expl_dis.write(str(group["Vectors"][v][k]) + " ")
                f_expl_dis.write("\n")

            else:
                #this was a disimproving vector

                #write the origin point
                for k in range(len(group["Origin"])):
                    f_expl_impr.write(str(group["Origin"][k]) + " ")
                f_expl_impr.write("\n")

                #write the target point
                for k in range(len(group["Vectors"][v])):
                    f_expl_impr.write(str(group["Vectors"][v][k]) + " ")
                f_expl_impr.write("\n")



            #either way, write the vector to the general output
            for k in range(len(group["Origin"])):
                f_explore.write(str(group["Origin"][k]) + " ")
            f_explore.write("\n")

            #write the target point
            for k in range(len(group["Vectors"][v])):
                f_explore.write(str(group["Vectors"][v][k]) + " ")
            f_explore.write("\n")


    f_explore.close()
    f_expl_dis.close()
    f_expl_impr.close()


    #writing starburst vectors

    f_starburst.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_starburst.write("Starburst Vectors")
    f_starburst.write("\n")
    f_star_impr.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_star_impr.write("Starburst Improving Vectors")
    f_star_impr.write("\n")
    f_star_dis.write("SWMv1_2_Trials.SWM_hill_climb()\n")
    f_star_dis.write("Starburst Disimproving Vectors")
    f_star_dis.write("\n")


    #The key "Starburst History" contains a list, and each element is a dictionary
    #Each dictionary has the following format:
    #
    # star_hist = {
    #  "Step": i,
    #  "Origin": x_current[:],
    #  "Origin Value" : value_current,
    #  "Vectors" : starbursts[:],
    #  "Vector Values" : starburst_values[:]
    # }

    #loop over each list member
    for group in result["Starburst History"]:
        #for each member, loop over each of it's vectors, and write the start and end points
        # and their associated values

        for v in range(len(group["Vectors"])):

            #check for improvement/disimprovement

            if (
                ((not MINIMIZING) and ( group["Origin Value"] > group["Vector Values"][v] )) or
                ((    MINIMIZING) and ( group["Origin Value"] < group["Vector Values"][v] ))
               ):
                #this was an improving vector

                #write the origin point
                for k in range(len(group["Origin"])):
                    f_star_impr.write(str(group["Origin"][k]) + " ")
                f_star_impr.write("\n")

                #write the target point
                for k in range(len(group["Vectors"][v])):
                    f_star_impr.write(str(group["Vectors"][v][k]) + " ")
                f_star_impr.write("\n")

            else:
                #this was a disimproving vector

                #write the origin point
                for k in range(len(group["Origin"])):
                    f_star_dis.write(str(group["Origin"][k]) + " ")
                f_star_dis.write("\n")

                #write the target point
                for k in range(len(group["Vectors"][v])):
                    f_star_dis.write(str(group["Vectors"][v][k]) + " ")
                f_star_dis.write("\n")



            #either way, write the vector to the general output
            for k in range(len(group["Origin"])):
                f_starburst.write(str(group["Origin"][k]) + " ")
            f_starburst.write("\n")

            #write the target point
            for k in range(len(group["Vectors"][v])):
                f_starburst.write(str(group["Vectors"][v][k]) + " ")
            f_starburst.write("\n")


    f_starburst.close()
    f_star_impr.close()
    f_star_dis.close()

def SWM_multi_hill_climb(pathways, x0_lists, climbing_steps=20, step_size=0.2, small_step_size=0.04, objective_function="J3", MINIMIZING=False, OUTPUT_OBJ_FN_MAP=True):

    """
    ARGUEMENTS

    pathways: a list of MDP pathways, typically generated from standard_MDP_set(...)

    x0_lists: a list containing the starting policies for each restart of the hillclimbing algorithm

    climbing_steps: how many iterations each climb is allowed to take before forcing it to terminate

    step_size: the constant stepsize for the main axis of the hillclimb's route

    small_step_size: the constant stepsize for the the secondary axes of the hillclimb's route

    objective_function: currently allowed to be "J1" "J2" "J3" etc... as per the MDP software

    MINIMIZING: whether to tell the MDP software to miminize (True) or maximize (False)

    OUTPUT_OBJ_FN_MAP: if set to True, at the end of the hillclimbings, a Monte Carlo-based map of the
     true objective function will be constructed (rather than the surrogate objfn's used elsewhere)
    """

    start_time = "Started:  " + str(datetime.datetime.now())
    
    #default to J3
    objfn = MDP_opt.J3
    fprime = MDP_opt.J3prime
    if objective_function == "J1":
        objfn = MDP_opt.J1
        #fprime = MDP_opt.J1prime
    

    #signature is multi_hill_climb(objfn, 
                               # x0_lists, 
                               # objfn_arg=None,
                               # bounds=None, 
                               # step_size=0.1, 
                               # small_step_size=0.02, 
                               # greatest_disimprovement=0.95,
                               # addl_expl_vectors=10,
                               # starburst_vectors=10,
                               # starburst_mag=2.0,  
                               # MINIMIZING=False, 
                               # max_steps=20)
    
    print("Beginning multi-hill-climbing algorithm")
    print("..time is " + str(datetime.datetime.now()))
    results = HKB_Heuristics.multi_hill_climb(objfn=objfn, 
                                       x0_lists=x0_lists, 
                                       objfn_arg=pathways,
                                       bounds=[[-20,20],[-20,20]],
                                       step_size=step_size, 
                                       small_step_size=small_step_size, 
                                       greatest_disimprovement=0.9, 
                                       addl_expl_vectors=10,
                                       starburst_vectors=20,
                                       starburst_mag=2.0,
                                       MINIMIZING=MINIMIZING, 
                                       max_steps=climbing_steps) 
    
    end_time = "Ended:  " + str(datetime.datetime.now())
    #finished gathering output strings, now write them to the file

    #check if output folder exists:
    folder = "SWM_HC_Outputs" 
    if not os.path.exists(folder):
        os.makedirs(folder)


    if OUTPUT_OBJ_FN_MAP:
        print("Building objective function map")
        print("..time is " + str(datetime.datetime.now()))
        #obj_fn_graph_2(pathways, objective_function='J3', p0_range=[-20,20], p1_range=[-20,20], 
                                                           #p0_step=0.5, p1_step=0.5, 
                                                           #OUTPUT_FOR_SCILAB=True, 
                                                           #folder="obj_fn_graph_2_outputs")
        obj_fn_graph_2(pathways=pathways,  
                                   objective_function=objective_function, 
                                   p0_range=[-20,20], 
                                   p1_range=[-20,20], 
                                   p0_step=0.5, 
                                   p1_step=0.5, 
                                   OUTPUT_FOR_SCILAB=True,
                                   folder=folder)


    print("")
    print("Process Complete... writing output files")


    #check if output folder exists:
    folder = "SWM_HC_Outputs" 
    if not os.path.exists(folder):
        os.makedirs(folder)

    f_details = open(os.path.join(folder,"details.txt"),'w')
    f_path = open(os.path.join(folder,"path.txt"), 'w')
    f_explore = open(os.path.join(folder,"exploration.txt"),'w')
    f_expl_dis = open(os.path.join(folder,"explr_dis.txt"),'w')
    f_expl_impr = open(os.path.join(folder,"explr_impr.txt"),'w')
    f_starburst = open(os.path.join(folder,"starburst.txt"),'w')
    f_star_impr = open(os.path.join(folder,"star_impr.txt"),'w')
    f_star_dis = open(os.path.join(folder,"star_dis.txt"),'w')

    #more files for line graphs of individual climbs
    f_climbs=[None]*len(x0_lists)
    for i in range(len(x0_lists)):
        f_climbs[i] = open(os.path.join(folder,"climb_"+str(i)+".txt"),'w')

    #Writing Details
    f_details.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_details.write("\n")
    f_details.write(start_time + "\n")
    f_details.write(end_time + "\n")
    f_details.write("\n")
    f_details.write("PATHWAY SET INFORMATION:\n")
    f_details.write("Pathways Count: " + str(len(pathways)) +"\n")
    f_details.write("Timesteps per Pathway: " + str(len(pathways[0].events)) +"\n")
    #f_details.write("Policy: " + str(policy) + "\n")
    f_details.write("\n")
    f_details.write("HILLCLIMBING INFORMATION:\n")
    f_details.write("Hill-climbing steps: " + str(climbing_steps) + "\n")
    f_details.write("Step Size: " + str(step_size) + "\n")
    f_details.write("Small Step Size: " + str(small_step_size) + "\n")
    f_details.write("Objective Function: " + str(objective_function) + "\n")
    for i in range(len(x0_lists)):
        f_details.write("x0[" + str(i) + "]: " + str(x0_lists[i]) + "\n")
    if MINIMIZING:
        f_details.write("HKB_Heuristics.hill_climb() is set to MINIMIZE\n")
    else:
        f_details.write("HKB_Heuristics.hill_climb() is set to MAXIMIZE\n")
    f_details.close()

    #Writing Pathway information
    f_path.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_path.write("Pathway of the Ascent/Descent\n")
    f_path.write("(Points are duplicated for use in Scilab.xarrows function)\n")
    f_path.write("\n")
    f_path.write("P0 P1 ObjFnVal\n")

    #writing headers for individual climb files
    for i in range(len(f_climbs)):
        f_climbs[i].write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
        f_climbs[i].write("Climb " + str(i) + "\n")
        f_climbs[i].write("\n")
        f_climbs[i].write("P0 P1 Value Var logVar STD Ave\n")

    for p in range(len(results)):
        #record all pathway steps as one long list...
        #because scilab will treat each in pairs, it'll draw the independent
        #paths separately if I write them properly
        for i in range(len(results[p]["Path"])):
            f_path.write(str(results[p]["Path"][i][0]) + " ")
            f_path.write(str(results[p]["Path"][i][1]) + " ")
            f_path.write(str(results[p]["Values"][i]) + "\n")

            #if this is not the first or the last entry, record the position twice
            #Scilab uses pairs in the vector to define the start and stop position
            #of each arrow, so the format looks like this:
            # [arrow1_start_x, arrow1_end_x, arrow2_start_x, arrow2_end_x, etc..]
            #so in this case, where each arrow starts where the previous ends, 
            # those coordinates will be repeated twice, except for the first and last
            #arrows.
            if (not i==0) and (not i==len(results[p]["Path"])-1):
                f_path.write(str(results[p]["Path"][i][0]) + " ")
                f_path.write(str(results[p]["Path"][i][1]) + " ")
                f_path.write(str(results[p]["Values"][i]) + "\n")


            #Also write to individual climb files
            f_climbs[p].write(str(results[p]["Path"][i][0]) + " ")
            f_climbs[p].write(str(results[p]["Path"][i][1]) + " ")
            f_climbs[p].write(str(results[p]["Values"][i]) + " ")
            f_climbs[p].write(str(results[p]["Weights Variance"][i]) + " ")
            f_climbs[p].write(str(results[p]["Weights log(Variance)"][i]) + " ")
            f_climbs[p].write(str(results[p]["Weights STD"][i]) + " ")
            f_climbs[p].write(str(results[p]["Weights Average"][i]) + "\n")

    f_path.close()

    #close the individual climb files
    for i in range(len(f_climbs)):
        f_climbs[i].close()


    #writing exploration sets

    f_explore.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_explore.write("Exploration Vectors")
    f_explore.write("\n")
    f_expl_impr.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_expl_impr.write("Exploration Improving Vectors")
    f_expl_impr.write("\n")
    f_expl_dis.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_expl_dis.write("Exploration Disimproving Vectors")
    f_expl_dis.write("\n")

    #The key "Exploration History" contains a list, and each element is a dictionary
    #Each dictionary has the following format:
    #
    # exp_hist = {
    #    "Step" : i,
    #    "Origin" : x_current[:],
    #    "Origin Value" : value_current,
    #    "Vectors" : explore_set[:],
    #    "Values" : explore_vals[:]
    #    }

    for p in range(len(results)):
        #loop over each list member
        for group in results[p]["Exploration History"]:
            #for each member, loop over each of it's vectors, and write the start and end points
            # and their associated values

            for v in range(len(group["Vectors"])):

                #check for improvement/disimprovement

                if (
                    ((not MINIMIZING) and ( group["Origin Value"] > group["Values"][v] )) or
                    ((    MINIMIZING) and ( group["Origin Value"] < group["Values"][v] ))
                   ):
                    #this was an improving vector

                    #write the origin point
                    for k in range(len(group["Origin"])):
                        f_expl_dis.write(str(group["Origin"][k]) + " ")
                    f_expl_dis.write("\n")

                    #write the target point
                    for k in range(len(group["Vectors"][v])):
                        f_expl_dis.write(str(group["Vectors"][v][k]) + " ")
                    f_expl_dis.write("\n")

                else:
                    #this was a disimproving vector

                    #write the origin point
                    for k in range(len(group["Origin"])):
                        f_expl_impr.write(str(group["Origin"][k]) + " ")
                    f_expl_impr.write("\n")

                    #write the target point
                    for k in range(len(group["Vectors"][v])):
                        f_expl_impr.write(str(group["Vectors"][v][k]) + " ")
                    f_expl_impr.write("\n")



                #either way, write the vector to the general output
                for k in range(len(group["Origin"])):
                    f_explore.write(str(group["Origin"][k]) + " ")
                f_explore.write("\n")

                #write the target point
                for k in range(len(group["Vectors"][v])):
                    f_explore.write(str(group["Vectors"][v][k]) + " ")
                f_explore.write("\n")


    f_explore.close()
    f_expl_dis.close()
    f_expl_impr.close()


    #writing starburst vectors

    f_starburst.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_starburst.write("Starburst Vectors")
    f_starburst.write("\n")
    f_star_impr.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_star_impr.write("Starburst Improving Vectors")
    f_star_impr.write("\n")
    f_star_dis.write("SWMv1_2_Trials.SWM_multi_hill_climb()\n")
    f_star_dis.write("Starburst Disimproving Vectors")
    f_star_dis.write("\n")


    #The key "Starburst History" contains a list, and each element is a dictionary
    #Each dictionary has the following format:
    #
    # star_hist = {
    #  "Step": i,
    #  "Origin": x_current[:],
    #  "Origin Value" : value_current,
    #  "Vectors" : starbursts[:],
    #  "Vector Values" : starburst_values[:]
    # }

    for p in range(len(results)):
        #loop over each list member
        for group in results[p]["Starburst History"]:
            #for each member, loop over each of it's vectors, and write the start and end points
            # and their associated values

            for v in range(len(group["Vectors"])):

                #check for improvement/disimprovement

                if (
                    ((not MINIMIZING) and ( group["Origin Value"] > group["Vector Values"][v] )) or
                    ((    MINIMIZING) and ( group["Origin Value"] < group["Vector Values"][v] ))
                   ):
                    #this was an improving vector

                    #write the origin point
                    for k in range(len(group["Origin"])):
                        f_star_impr.write(str(group["Origin"][k]) + " ")
                    f_star_impr.write("\n")

                    #write the target point
                    for k in range(len(group["Vectors"][v])):
                        f_star_impr.write(str(group["Vectors"][v][k]) + " ")
                    f_star_impr.write("\n")

                else:
                    #this was a disimproving vector

                    #write the origin point
                    for k in range(len(group["Origin"])):
                        f_star_dis.write(str(group["Origin"][k]) + " ")
                    f_star_dis.write("\n")

                    #write the target point
                    for k in range(len(group["Vectors"][v])):
                        f_star_dis.write(str(group["Vectors"][v][k]) + " ")
                    f_star_dis.write("\n")



                #either way, write the vector to the general output
                for k in range(len(group["Origin"])):
                    f_starburst.write(str(group["Origin"][k]) + " ")
                f_starburst.write("\n")

                #write the target point
                for k in range(len(group["Vectors"][v])):
                    f_starburst.write(str(group["Vectors"][v][k]) + " ")
                f_starburst.write("\n")


    f_starburst.close()
    f_star_impr.close()
    f_star_dis.close()











#######################
##    DEPRECATING    ##
#######################
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
    



