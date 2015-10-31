import SWMv1_3_Trials as SWMT
import numpy, scipy
import statsmodels.api as stats



def reg_heur(pol_0, sampling_radius=1.0, pw_count=500, years=100, minimum_pathways=100, alpha_step=1, p_val_limit=0.1, max_steps=20,PRINT_R_PLOTTING=False, SILENT=False):
    """Uses multivariable regressions to choose step directions for a SWMv1.3 Hill-climb

    ARGUEMENTS
    pol_0: a python list of length 2 (for SWMv1.3): The policy from which to start the search.

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
    summary.["Path"] = None
    summary.["Values"] = None
    summary.["Steps"] = None
    summary.["R Plotting"] = None
    summary.["Exit Type"] = ""
    summary.["Message"] = ""

    #get initial set
    pw = SWMT.limited_MDP_set(pw_count, years, pol_0, random_seed=0, sampling_radius=sampling_radius)
    if len(pw) < 100:
        if not SILENT: print("Initial Policy produced too many LB or SA pathways... quitting.")
        summary["Message"] = "Initial Policy produced too many LB or SA pathways."
        summary["Exit Type"] = "INITIAL_PLATEAU"
        return summary

    #build X and Y vectors
    x_p0 = [0.0] * len(pw)
    x_p1 = [0.0] * len(pw)
    x_both = [None] * len(pw)
    y    = [0.0] * len(pw)
    for i in range(len(pw)):
        x_p0[i] = pw[i].generation_policy_parameters[0] - pol_0[0]
        x_p1[i] = pw[i].generation_policy_parameters[1] - pol_0[1]
        x_both[i] = [x_p0[i], x_p1[i]]
        y[i] =    pw[i].net_value / years

    X = scipy.matrix(x_both)
    X = stats.add_constant(X)
    #print(X)

    # Fit regression model
    results = stats.OLS(y, X).fit()

    #Inspect the results
    #print results.summary()

    # print("")
    # print("")
    # print("")
    #print("Coeffs: " + str(results.params))
    #print("p-vals: " + str(results.pvalues))

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
            center_pol[b] = center_pol[b] + alpha_step * ( results.params[b+1] / max_coef )
            this_step[b] =                  alpha_step * ( results.params[b+1] / max_coef )

        #add a copy of this policy to the history
        pol_history.append(center_pol[:])

        #add a copy of the step to the step history
        step_history.append(this_step[:])

        #generate a new limited_MDP_set
        pw = SWMT.limited_MDP_set(pw_count, years, center_pol, random_seed=step, sampling_radius=sampling_radius)

        #check for pathway sets that are too short (because of plateau policies)
        if len(pw) < minimum_pathways:
            if not SILENT: print("Exit Condition: The current policy could not produce enough non-LB/SA pathways.")
            exit_message = "Step " + str(step) + ": The current policy could not produce enough non-LB/SA pathways."
            summary["Message"] = "The current policy could not produce enough non-LB/SA pathways."
            summary["Exit Type"]] = "PLATEAU"

            #since we're breaking here, add the constant to the value list since we'll miss that step otherwise
            val_history.append(results.params[0])

            break

        #we have new pathways, so prepare for the next regression
        x_p0 = [0.0] * pw_count
        x_p1 = [0.0] * pw_count
        x_both = [None] * pw_count
        y    = [0.0] * pw_count
        for i in range(pw_count):
            x_p0[i] = pw[i].generation_policy_parameters[0] - center_pol[0]
            x_p1[i] = pw[i].generation_policy_parameters[1] - center_pol[1]
            x_both[i] = [x_p0[i], x_p1[i]]
            y[i] =    pw[i].net_value / years

        X = scipy.matrix(x_both)
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

    #we've exitted the loop
    if exit_message == "":
        #we fell off the loop without ending, so we hit max_steps
        message = "Maximum number of steps reached."
        summary["Message"] = "Maximum number of steps reached."
        if not SILENT: print("Maximum number of steps reached.")


    

    r_lines = None
    if (not SILENT) and (PRINT_R_PLOTTING):
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
    if not SILENT:
        print("")
        print("")
        print("RESULTS")
        print("STEPS: " + str(len(pol_history)))

        print("VALUE PO P1")
        for i in range(len(val_history)):
            print(str(val_history[i]) + " " + str(pol_history[i][0]) + " " + str(pol_history[i][1]) )


    #finish the summary dictionary to return
    #it should already have "Message" and "Exit Type" added
    summary.["Path"] = pol_history
    summary.["Values"] = val_history
    summary.["Steps"] = step_history
    summary.["R Plotting"] = r_lines

    return summary