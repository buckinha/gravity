"""SWIMMv1_1 Trials"""

import MDP, MDP_opt, SWIMMv1_1, HKB_Heuristics, random, numpy, datetime


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
                pathways[i] = SWIMMv1_1.simulate(timesteps=timesteps, policy=[p0_val,p1_val,0], random_seed=(5000+i), SILENT=True, PROBABILISTIC_CHOICES=PROBABILISTIC_CHOICES)

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
    f.write("SWIMMv1_1_Trials.pathway_value_graph_1()\n")
    f.write(start_time + "\n")
    f.write(end_time + "\n")
    f.write("Pathways per Point: " + str(pathway_count_per_point) +"\n")
    f.write("Timesteps per Pathway: " + str(timesteps) +"\n")
    f.write("P0 Range: " + str(p0_range) +"\n")
    f.write("P1 Range: " + str(p1_range) +"\n")

    #writing model parameters from whatever's still in the pathway set.
    # the model parameters don't change from one M.C batch to another.
    f.write("\n")
    f.write("Constant Reward: " + str(pathways[0]["Constant Reward"]) + "\n")
    f.write("Condition Change After Suppression: " + str(pathways[0]["Condition Change After Suppression"]) + "\n")
    f.write("Condition Change After Mild: " + str(pathways[0]["Condition Change After Mild"]) + "\n")
    f.write("Condition Change After Severe: " + str(pathways[0]["Condition Change After Severe"]) + "\n")
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