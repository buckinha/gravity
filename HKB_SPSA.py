#HKB_SPSA
#Implementing a Simultaneous Perturbation Stochastic Approximation (SPSA) estimate.
import random, MDP, datetime, os.path, numpy, SWMv1_2

def SPSA(x0, alpha, objfn, objfn_args=None, random_seed=None):
    """Returns a vector containing the SPSA gradient estimate

    ARGUEMENTS
    x0: a vector containing the policy around which you want the SPSA estimate

    alpha: scalar step size for each parameter perturbation

    objfn: a function handle with signature objfn(x, objfn_args)

    objfn_args: an object (probably a list of pathways, etc...) to pass to objfn()

    random_seed: the seed to give to python's random number generator. If None, then this is ignored.

    """

    #set the random number generator, if desired.
    if random_seed:
        random.seed(random_seed)

    n = len(x0)
    v_plus = [0.0] * n
    v_minus = [0.0] * n

    #create perturbation vectors
    for i in range(n):
        v = random.choice([-1,1])

        v_plus[i] = x0[i] + (alpha * v)
        v_minus[i] = x0[i] - (alpha * v)


    #get obj fn values at the two perturbations
    v_p_val = 0.0
    v_m_val = 0.0
    if objfn_args:
        v_p_val = objfn(v_plus, objfn_args)
        v_m_val = objfn(v_minus, objfn_args)
    else:
        v_p_val = objfn(v_plus)
        v_m_val = objfn(v_minus)


    #compute the approximate gradient
    SPSA_grad =[0.0] * n
    diff = v_p_val - v_m_val
    for i in range(n):
        SPSA_grad[i] = ( diff ) / ( v_plus[i] - v_minus[i] )



    return SPSA_grad


def SPSA_KLD_Climb(objfn, x0, alpha=0.1, gamma=None, epsilon = 0.0001, max_steps=50, retries=10, KLD_constraint=None, objfn_args=None, MINIMIZING=False):
    """Hill-climbing using an SPSA gradient apprximation, with optional KL Divergence constraint

    This function does a single hill climb using Spall's Simultaneous Perturbation Stochastic 
    Approximation (SPSA) method (see http://www.jhuapl.edu/spsa/index.html). The algorithm starts
    the climb at x0 and makes steps according to the gradient approximation. Step size can be set
    to decay throughout the climb, or to remain constant. 

    If desired, and if the objfn_args is a list of MDP pathways, a climbing constraint can be 
    imposed such that when the KL Divergence of the newest policy vs the one that generated
    the pathways originally is greater than the constraint, climbing terminates. Otherwise, climbing
    will continue untill the climbing steps constraint has been met, or when the gradient 
    approximation becomes too flat.

    ARGUEMENTS
    objfn: a function handle for the objective to be climbed. It must have one of the following two
     signatures:  objfn(x)   or   objfn(x, objfn_args), where x is a vector of some set length, "n"
    
    x0: a vector of length "n", which is the starting position of the climb, and can be directly
     input into objfn(.)

    alpha: the scalar step-size for each parameter. That is, when each parameter is perturbed, it will
     be by an amount equal to +/- alpha

    gamma: the decay rate on step-size. If set to 1.0, or "None", the step size will remain constant.
     Otherwise, the step size (that is, the perturbation size on each parameter) at iteration "k"
     will be (+/-)alpha * gamma^k, where the first iteration has k=0. Gamma values must be between 0
     and 1, or "None", or they will be ignored.

    epsilon: After each iteration, if the difference between the previous objective function value and 
     the current one is less than epsilon, it will be rounded to 0, i.e. no improvement.

    max_steps: The maximum number of hill-climbing steps that the algorithm can take.

    retries: After a step is made, if the new objective function value is less than the old one, or 
     if it zero (that is, between 0 and epsilon), the change will be reverted and a new climbing 
     step will be tried. This can happen up to the value of "retries", after which, if an improvement
     has not been found, the agorithm will terminate.

    KLD_constraint: the maximum value that the KL Divergence can be at any step. If after an improving
     step, the KL Divergence is higher than this value, the algorithm will terminate.

    objfn_args: any additional arguments to pass to the objective function. This is typically a list
     of MDP pathways.

    MINIMIZING: A boolean flag, indicating whether the algorithm is to minimize a loss fucntion (set to 
     True) or to maximize (set to False)




    RETURNS

    a dictionary with the following elements:
    --KEY--             --VALUE--
    "Final Value"       the final objective function for the last step to be completed
    "Final Position"    the final position of the policy vector
    "Path"              a list containing the vectors of each step, starting with x0 and ending with
                          the final policy
    "Path Values"       a list containing the objective function values at each step
    "KL Divergences"     a list containing the KL Divergence value at each step
    "Steps Taken"             the total number of steps taken during the climb
    "Message"           a string containing a description of how and why the algorithm ended
    "Start Time"        the system time when the algorithm began
    "End Time"          the system time when the algorithm completed
    "Alpha"             the input arguement "alpha"
    "Gamma"             the input arguement "gamma"
    "Epsilon"           the input arguement "epsilon"
    "KLD Constraint"    the input arguement "KLD_constraint"
    "Retries"           the input arguement "retries"

    """

    start_time = datetime.datetime.now()

    #Sanitize Inputs
    if not gamma: gamma = 1.0
    if (gamma <= 0) or (gamma > 1.0):
        print ("SPSA_KLD_Climb(.) WARNING: gamma out of range 0 < gamma < 1; setting gamma = 1.0")
        gamma = 1.0


    #start the random number generator
    random.seed(datetime.datetime.now())

    #Create Arrays and Other Important Values
    n = len(x0)
    step_positions = [None] * max_steps
    step_values = [None] * max_steps
    step_KLD = [0.0] * max_steps


    #Compute the values for the alpha vector
    alpha_vec = [0.0] * max_steps
    for i in range(max_steps):
        alpha_vec[i] = alpha * pow(gamma,i)


    #copy starting position
    x_current = x0[:]

    #get the objfn value at the starting position
    value_current = 0.0
    if objfn_args:
        value_current = objfn(x0, objfn_args)
    else:
        value_current = objfn(x0)

    #the comparisons in the loop are written for maximizations, so if we're minimizing, just
    # flip the signs:
    if MINIMIZING: value_current *= -1.0

    #set up some return-value and loop-control variables
    message = ""
    step_count = -1
    KLD_binding = False



    #Begin Iterations
    for step in range(max_steps):
        step_count += 1

        #we either just began the first iteration, or are starting a new step. Either way,
        # record the current position to the step_position list
        step_positions[step] = x_current[:]
        step_values[step] = value_current

        #if a KLD constraint has been given, compute it
        if KLD_constraint:
            step_KLD[step] = MDP.KLD(objfn_args, x_current)

            #and enforce KLD constraint if necessary
            if step_KLD[step] >= KLD_constraint:
                #we're out of bounds, so set the message and terminate the algorithm
                message = "KL Divergence has exceeded the constraint at step: " + str(step)

                #to remind ourselves at the end of the loop:
                KLD_binding = True

                #exit the entire climb loop
                break


        IMPROVEMENT_FOUND = False
        for attempt in range(retries):

            #compute the gradient
            new_gradient = SPSA(x_current, alpha_vec[i], objfn, objfn_args)

            #make the step
            new_position = vector_add(new_gradient, x_current)

            #get the new objfn value
            new_value = 0.0
            if objfn_args:
                new_value = objfn(new_position, objfn_args)
            else:
                new_value = objfn(new_position)

            #the comparisons below are for maximization, so if we're minimizing, just flip the signs
            if MINIMIZING: new_value += -1.0

            #compare the new and old objfn values and choose result
            if new_value > value_current + epsilon:
                #this is an improvement

                #set current value and position to the new one

                value_current = new_value
                x_current = new_position[:]
                IMPROVEMENT_FOUND = True

                break

            else:
                #this is a disimprovement or else is close enough to zero to be considered zero

                #just continue with retries to see if we can find a way out
                pass

        #at this point, the current step is done with it's retries, or found an improvement
        # along the approximate gradient. Check which one happened, and continue accordingly
        if not IMPROVEMENT_FOUND:
            #all retries were exhausted with no improvments, so the algorithm is done.
            message = "No improvements could be found: Step " + str(step)
            break
        else:
            #an improvement was found, so keep climbing 
            pass

    #and now we've exited the climb loop entirely. This can happen either because:
    # 1) no improvements were found, and the associated break statement was hit
    # 2) the KL Divergence constraint became binding and the associated break statement was hit
    # 3) the for-loop finished at max_steps

    end_time = datetime.datetime.now()

    #if it was 1 or 2, the message will already be set appropriately. Check for 3.
    if (step_count >= max_steps - 1) and not KLD_binding:
        message = "Maximum step iterations reached with no other stop conditions."

    #trim the arrays down, if necessary
    if step_count < max_steps -1:
        step_positions = step_positions[:step_count+1]
        step_values = step_values[:step_count+1]
        step_KLD = step_KLD[:step_count+1]

    #Prepare return value
    summary = {}
    summary["Final Value"] = value_current
    summary["Final Position"] = x_current
    summary["Path"] = step_positions
    summary["Path Values"] = step_values
    summary["KL Divergences"] = step_KLD
    summary["Steps Taken"] = step_count
    summary["Message"] = message
    summary["Start Time"] = start_time
    summary["End Time"] = end_time
    summary["Alpha"] = alpha
    summary["Gamma"] = gamma
    summary["Epsilon"] = epsilon
    summary["KLD Constraint"] = KLD_constraint
    summary["Retries"] = retries


    return summary


def add_SWM_MC(SPSA_result, reps, years):
    """ Takes the output of SPSA_KLD_Climb and runs SWM Monte Carlo sims on each position in the climb
    
    ARGUEMENTS
    SPSA_result: a dictionary which was returned by SPSA_KLD_Climb(...)

    reps: how many pathways to run for each policy in the hill-climbing path over which to take
     an average.

    years: how many years each pathway should run


    RETURNS
    the same dictionary, but with the added keys "MC Values" and "MC STD"

    """

    climb_MC_values = [0.0] * len(SPSA_result["Path"])
    climb_MC_STD = [0.0] * len(SPSA_result["Path"])

    for position in range(len(SPSA_result["Path"])):
        
        sim_results = [0.0] * reps

        for rep in range(reps):

            seed = (position*5000 + rep)

            #signature for simulate() is:
            #simulate(timesteps, policy=[0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True)
            sim = SWMv1_2.simulate(years, SPSA_result["Path"][position], random_seed=seed, SILENT=True)
            sim_results[rep] = sim["Average State Value"]
        

        climb_MC_values[position] = numpy.mean(sim_results)
        climb_MC_STD[position] = numpy.std(sim_results)


    #now add these lists to the summary that was given, and return it
    SPSA_result["MC Values"] = climb_MC_values
    SPSA_result["MC STD"] = climb_MC_STD

    return SPSA_result



def output_to_file(filename, summary):
    """Takes a summary from HKB_SPSA(.) and writes the output to a file"""

    """
    HKB_SPSA returns the following dictionary:
    summary["Final Value"] = value_current
    summary["Final Position"] = x_current
    summary["Path"] = step_positions
    summary["Path Values"] = step_values
    summary["KL Divergences"] = step_KLD
    summary["Steps Taken"] = step_count
    summary["Message"] = message
    summary["Start Time"]
    summary["End Time"]
    summary["Alpha"]
    summary["Gamma"]
    summary["Epsilon"]
    summary["Retries"]
    summary["KLD Constraint"]

    and if the output has been run through the MC evaluator, 
    summary["MC Average"]
    summary["MC STD"]
    """

    #check if output folder exists:
    folder = "HKB_SPSA_Outputs" 
    if not os.path.exists(folder):
        os.makedirs(folder)


    f_details = open(os.path.join(folder, str(filename) + "_details.txt"),'w')
    f_climb = open(os.path.join(folder,filename), 'w')

    #Writing Details
    f_details.write("HKB_SPSA.SPSA_KLD_Climb()\n")
    f_details.write("\n")
    f_details.write("Started: " + str(summary["Start Time"]) + "\n")
    f_details.write("Ended:   " + str(summary["End Time"]) + "\n")
    f_details.write("\n")
    f_details.write("CLIMB INFORMATION:\n")
    f_details.write("Climbing steps: " + str(summary["Steps Taken"]) + "\n")
    f_details.write("Alpha: " + str(summary["Alpha"]) + "\n")
    f_details.write("Gamma: " + str(summary["Gamma"]) + "\n")
    f_details.write("Epsilon: " + str(summary["Epsilon"]) + "\n")
    f_details.write("KLD Constraint: " + str(summary["KLD Constraint"]) + "\n")
    f_details.write("Retries: " + str(summary["Retries"]) + "\n")
    f_details.write("Objective Function: J4\n")
    f_details.write("\nTermination Message: " + summary["Message"])
    f_details.close()

    #Writing Climb information
    f_climb.write("HKB_SPSA.SPSA_KLD_Climb()\n")
    f_climb.write("\n")
    for i in range(len(summary["Path"][0])):
        f_climb.write("P" + str(i) + " ")
    f_climb.write("Value KLD")
    if "MC Values" in summary.keys(): f_climb.write(" MC_Val MC_STD")
    f_climb.write("\n")


    for i in range(len(summary["Path"])):

        #write this step's policy parameters
        for j in range(len(summary["Path"][0])):
            f_climb.write( str( summary["Path"][i][j] ) + " ")

        #write this step's details
        f_climb.write( str( summary["Path Values"][i]) + " ")
        f_climb.write( str( summary["KL Divergences"][i]))

        #check if MC results have been appended, and if so, print them too
        if "MC Values" in summary.keys():
            f_climb.write( " " + str( summary["MC Values"][i]) )
        if "MC STD" in summary.keys():
            f_climb.write( " " + str( summary["MC STD"][i]) )

        #close the current line
        f_climb.write("\n")

    f_climb.close()


def vector_add(a, b):
    """Adds two vectors together, piecewise.

    Note: If one vector is longer, a warnign is printed to screen, but the function continues
    by concatenating a zero-vector at the end of the shorter vector to make it equal in length
    to the longer vector.

    """
    a_len = len(a)
    b_len = len(b)

    if not a_len==b_len:
        print("vector_add(a,b) in HKB_SPSA : WARNING, vectors a and b are not of the same length...")

    if a_len >= b_len:
        for i in range(b_len):
            a[i] += b[i]
        return a

    else:
        for i in range(a_len):
            b[i] += a[i]
        return b

