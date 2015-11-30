import SWMv2_1_Trials as SWMT
#import numpy as np
import scipy
import statsmodels.api as stats



def reg_heur(pol_0, sampling_radius=0.5, pw_count=500, years=100, minimum_pathways=100, alpha_step=1, p_val_limit=0.05, max_steps=5,PRINT_R_PLOTTING=False, SILENT=False, random_seed=0):
    """Uses multivariable regressions to choose step directions for a SWMv2.1 Hill-climb

    ARGUEMENTS
    pol_0: a python list: The policy from which to start the search.

    sampling_radius: the radius of the spherical region around the current policy in which new 
     policies can be drawn.

    pw_count: how many pathways to attempt draw at each step

    years: how many years each pathway should be simulated for 

    minimum_pathways: the minimum number of pathways allowed without termination. Since pathways
     that have suppression rates of either 0 or 100 percent are rejected, it is possible under 
     many policies that SWMT.limited_MDP_set will not return the full amount given by pw_count. 

    alpha_step: The step-size to use. Steps length along each dimension are scaled such that the
     parameter with the largest magnitude regression coefficient will have a step size of alpha_step
     and all others will have smaller step sizes but keeping the ratio of the coefficients to each 
     other equal to the ratio of the step sizes to each other. 

    p_val_limit: A value between 0 and 1. At least one regression coefficient must be considered 
     statistically significant at p_val_limit or less for the heuristic to continue

    max_steps: The maximum number of hill-climbing steps before cutting off the algorithm.

    PRINT_R_PLOTTING: Boolean. If True, will output R code to be used for plotting the climbs



    RETURNS
    a dictionary with the following elements, where N is the number of steps:
    ["Path"] = A list of length N+1 of the policies along the pathway, including the first and last
    ["Values"] = A list of length N+1 containing the constant of the regression, indicating the
       average value of the simulations from each step, taking variation/noise into account.
    ["Steps"] = A list of length N containing the steps made from each position in "Path" excepting 
         the last (from which there was no step)
    ["R Plotting"] = a list of length N of strings containing the related R code for plotting each segment 
        of the climb
    ["Exit Type"] = One of the following strings: "MAX_STEPS" "INITIAL_PLATEAU" "PLATEAU" "NO_POWER"
    """


    #prepare the summary dictionary to return
    summary = {}
    summary["Path"] = None
    summary["Values"] = None
    summary["Steps"] = None
    summary["R Plotting"] = None
    summary["Exit Type"] = ""
    summary["Message"] = ""

    #get initial set
    pw = SWMT.limited_MDP_set(pw_count, years, pol_0, random_seed=random_seed, sampling_radius=sampling_radius)
    if len(pw) < minimum_pathways:
        if not SILENT: print("Initial Policy produced too many LB or SA pathways... quitting.")
        summary["Message"] = "Initial Policy produced too many LB or SA pathways."
        summary["Exit Type"] = "INITIAL_PLATEAU"
        return summary

    #remember the current pathway count so that later, we can re-use the X matrix without 
    # reinstantiating so long as the pw set hasn't changed size
    current_pw_length = len(pw)

    #constructing the X matrix as a 2D list
    #X = [None] * len(pol_0)
    #for b in range(len(pol_0)):
    #    X[b] = [0.0] * len(pw)
    X = [None] * len(pw)
    for p in range(len(pw)):
        X[p] = [0.0] * len(pol_0)

    #construct the Y vector
    y    = [0.0] * len(pw)

    #fill the X matrix and Y vector from the pathway data
    #we're restructuring the policy parameters such that they are in terms of distance from 
    # the corresponsing parameter in pol_0
    for p in range(len(pw)):
    
        #get the y value
        y[p] =    pw[p].net_value / years

        #get the x values
        for b in range(len(pol_0)):
            X[p][b] = pw[p].generation_policy_parameters[b] - pol_0[b]

    #convert the X matrix to a scipy matrix, and add a constant term
    X = scipy.matrix(X)#.transpose()
    X = stats.add_constant(X)

    # Fit regression model
    results = stats.OLS(y, X).fit()

    #Inspect the results
    print results.summary()

    # print("")
    # print("")
    # print("")
    print("Coeffs: " + str(results.params))
    print("p-vals: " + str(results.pvalues))

    #check if regression parameters are in-bounds
    OOB = True
    for c in results.params[1:]: #excluding the constant
        if c <= p_val_limit: OOB = False

    if OOB:
        #none of the initial regression parameters were in-bounds
        if not SILENT: 
            print("None of the initial regression parameters had p values less than " + str(p_val_limit) + "... quitting.")
        summary["Message"] = "None of the initial regression parameters had p values less than " + str(p_val_limit) + "."
        summary["Exit Type"] = "NO_POWER"
        return summary


    #prepare for the loop
    center_pol = pol_0[:]
    pol_history = []
    pol_history.append(center_pol[:])
    val_history = []
    val_history.append(results.params[0] + 0) #the constant is closely related to the MC average value
    step_history = []
    exit_message = ""

    #print("")
    #print("regression constant was: " + str(results.params[0]))
    #print("pathway mean is: " + str(SWMT.stats(pw)["Mean"]))

    #now enter the loop
    for step in range(max_steps):

        #create the new center policy
        #start by finding the maximum of the absolute values of the parameters
        max_coef = float("-inf")

        #record the current step to add to the step_history list
        this_step = [None] * len(center_pol)

        for i in range(1, len(results.params)): #excluding the constant
            if abs(results.params[i]) > max_coef: max_coef = abs(results.params[i])

        for b in range(len(center_pol)):
            #the policy vectors have a constant term as their first item, but this is not the
            # same constant as the REGRESSION constant. So then, when the regressors are
            # [policy_Constant, policy_param1, policy_param2, ..., policy_paramB], the regression
            # will fit a coefficient to EACH, includign the policy constant, and then add ITS OWN
            # constant term as well. Thus, the regression "params" list is one larger than the
            # policy params list, and the first of them is the regression constant. So in the 
            # following step, we just shift up one, to match the coefficients with their proper
            # policy terms.

            #putting individual checks on p-values
            #if results.pvalues[b] <= p_val_limit:
            if True:
                center_pol[b] = center_pol[b] + alpha_step * ( results.params[b+1] / max_coef )
            else:
                #the p-value for this term isn't good enough, so just leave the policy parameter alone
                #which is the same as:
                #center_pol[b] = center_pol[b]
                pass

            #record the step direction and magnitude as well, in case it's needed
            this_step[b] =                  alpha_step * ( results.params[b+1] / max_coef )

        #add a copy of this policy to the history
        pol_history.append(center_pol[:])

        #add a copy of the step to the step history
        step_history.append(this_step[:])

        #generate a new limited_MDP_set
        pw = SWMT.limited_MDP_set(pw_count, years, center_pol, random_seed=random_seed+step, sampling_radius=sampling_radius)

        #check for pathway sets that are too short (because of plateau policies)
        if len(pw) < minimum_pathways:
            if not SILENT: print("Exit Condition: The current policy could not produce enough non-LB/SA pathways.")
            exit_message = "Step " + str(step) + ": The current policy could not produce enough non-LB/SA pathways."
            summary["Message"] = "The current policy could not produce enough non-LB/SA pathways."
            summary["Exit Type"] = "PLATEAU"

            #since we're breaking here, add the constant to the value list since we'll miss that step otherwise
            val_history.append(results.params[0])

            break


        #we have new pathways, so prepare for the next regression

        #if the number of pathways has changed, re-instantiate X and Y
        if not (len(pw) == current_pw_length):
            current_pw_length = len(pw)

            #re-constructing the X matrix as a 2D list
            X = [None] * len(pw)
            for p in range(len(pw)):
                X[p] = [0.0] * len(pol_0)

            #construct the Y vector
            y    = [0.0] * len(pw)

        #fill the X matrix and Y vector from the new pathway data
        #Note: we're restructuring the policy parameters such that they are in terms of distance from 
        # the corresponsing parameter in pol_0
        for p in range(len(pw)):
            #get the y value
            y[p] =    pw[p].net_value / years

            #get the x values
            for b in range(len(pol_0)):
                X[p][b] = pw[p].generation_policy_parameters[b] - pol_0[b]

        #convert the X matrix to a scipy matrix, and add a constant term
        X = scipy.matrix(X)
        X = stats.add_constant(X)

        #do the regression
        results = stats.OLS(y, X).fit()
        #add the constant to the value list
        val_history.append(results.params[0])


        #check if at least one regression parameter is in-bounds:
        OOB = True
        for c in results.params[1:]: #excluding the constant
            if c <= p_val_limit: OOB = False

        if OOB:
            #none of the regression parameters are in-bounds
            if not SILENT: print("Exit Condition: None of the regression parameters have p values less than " + str(p_val_limit) + "... quitting.")
            exit_message = "Step " + str(step) + "None of the regression parameters have p values less than " + str(p_val_limit)
            summary["Message"] = "None of the regression parameters have p values less than " + str(p_val_limit) + "."
            summary["Exit Type"] = "NO_POWER"
            break

    #we've exited the loop
    if exit_message == "":
        #we fell off the loop without otherwise breaking, so we hit the "max_steps" limit
        message = "Maximum number of steps reached."
        summary["Message"] = "Maximum number of steps reached."
        if not SILENT: print("Maximum number of steps reached.")


    

    r_lines = None
    if (not SILENT) and (PRINT_R_PLOTTING):
        #this is just printing the first two policy parameters
        print("")
        print("")
        r_lines = ["segments("] * (len(val_history)-1)
        for i in range(len(val_history)-1):
            #the plots are always flip-flopped axis, so make sure to do that here, too
            #adding starting point
            r_lines[i] += str(pol_history[i][1]) + "," + str(pol_history[i][0])
            #adding ending point
            r_lines[i] += "," + str(pol_history[i+1][1]) + "," + str(pol_history[i+1][0])
            #adding the color, based on the value at the ending point
            r_lines[i] += ", col="
            if val_history[i+1] < -4:
                r_lines[i] += "'purple')"
            elif val_history[i+1] < 0:
                r_lines[i] += "'blue')"
            elif val_history[i+1] < 4:
                r_lines[i] += "'green')"
            elif val_history[i+1] < 6.1:
                r_lines[i] += "'yellow')"
            elif val_history[i+1] < 7:
                r_lines[i] += "'orange')"
            else:
                r_lines[i] += "'red')"
            
            #go ahead and print it here, for now
            print(r_lines[i])

    #general output
    if not SILENT:
        print("")
        print("")
        print("RESULTS")
        print("STEPS: " + str(len(pol_history)))

        header_str = "VALUE"
        for b in range(len(pol_history[0])):
            header_str += " P" + str(b)
        print(header_str)

        for i in range(len(val_history)):
            this_line = str(val_history[i])
            for b in range(len(pol_history[0])):
                this_line = this_line + " " + str(pol_history[i][b])

            print(this_line)


    #finish the summary dictionary to return
    #it should already have "Message" and "Exit Type" added
    summary["Path"] = pol_history
    summary["Values"] = val_history
    summary["Steps"] = step_history
    summary["R Plotting"] = r_lines

    return summary


def RH_Climb(policy_set=None, policy_range=[[-25,25],[-25,25]], plateau_function=None, max_climbs=25, max_steps=25, sampling_radius=1.0, step_alpha=1.0, pathway_count=[100,200,300,400,500], pw_count_minimum=20, p_value=0.05):
    """ Conducts multiple runs through the regression heuristic in an attempt to find a global optima.

    DESCRIPTION:

    ARGUEMENTS:

    RETURNS:

    """
    policy_length = 0

    #SET UP THE POLICY SET

    #check if the policy_set is a list of lists (multiple policies)
    if not policy_set:
        #no policies were passed in, so set up policy_set with None values as a flag to generate
        # them within the climbs loop
        policy_set = [None] * max_climbs

        #set policy length to match the bounds given by policy_range
        policy_length = len(policy_range)

    else:
        if isinstance(policy_set[0], list):
            #the first element of policy_set is itself a list, so this should be a multiple policy argument
            #We'll just leave them be then, and set max_climbs to the length of the list, ignoring whatever
            #it might have otherwise been set to.
            max_climbs = len(policy_set)

            #set policy_length to match the policies given
            policy_length = len(policy_set[0])

        else:
            #the first element is not a list, so we'll assume this is a single policy
            #In this case, we'll use this policy as the first one attempted, and fill in the
            # rest with None values as a flag to generate them as we go
            
            #remember the first policy
            first_policy = policy_set[:]

            #set policy_length to match the policy given
            policy_length = len(policy_set)

            #make policy_set into a list of the appropriate length
            policy_set = [None] * max_climbs

            #set the first policy
            policy_set[0] = first_policy



    #set up the return dictionary
    summary = {}
    summary["Individual Climbs"] = []
    summary["Best Policy"] = None
    summary["Best Value"] = float("-inf")



    #START LOOPING THROUGH THE CLIMBS
    for climb_number in range(max_climbs):

        #check if there is a policy already, or if one needs to be created
        if policy_set[climb_number]:
            #there is a policy, so don't worry about anything
            pass
        else:
            #this policy is set to None, so generate a new one within the given bounds
            policy_set[climb_number] = [0.0] * policy_length
            for b in range(policy_length):
                policy_set[climb_number][b] = random.uniform(policy_range[b][0], policy_range[b][1])


        #filter the current policy, and if necessary, draw a new one
        for redraw in range(100):
            #limiting redraw attempts...
            #TODO
            pass



def dummy_filter(policy_to_check, training_set):
    """Returns True if this policy is "allowed" by the rule learned upon training_set"""

    return True