import SWMv1_3_Trials as SWMT
import numpy, scipy
import statsmodels.api as stats



def reg_heur(pol_0, wiggle, pw_count=500, years=100, alpha_step=1, p_val_limit=0.1, max_steps=20,PRINT_R_PLOTTING=False):

    #get initial set
    pw = SWMT.limited_MDP_set(pw_count, years, pol_0, random_seed=0, policy_wiggle=1.0)
    if len(pw) < 100:
        print("Initial Policy produced too many LB or SA pathways... quitting.")
        return -1

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
        print("None of the initial regression parameters had p values less than " + str(p_val_limit) + "... quitting.")
        return -1


    #prepare for the loop
    center_pol = pol_0[:]
    pol_history = []
    pol_history.append(center_pol[:])
    val_history = []
    val_history.append(results.params[0] + 0) #the constant is closely related to the MC average value
    exit_message = ""

    #now enter the loop
    for step in range(max_steps):

        #create the new center policy
        #start by finding the maximum of the absolute values of the parameters
        max_coef = float("-inf")
        for i in range(1, len(results.params)): #excluding the constant
            if abs(results.params[i]) > max_coef: max_coef = abs(results.params[i])

        for b in range(len(center_pol)):
            center_pol[b] = center_pol[b] + alpha_step * ( results.params[b+1] / max_coef )

        #add a copy of this policy to the history
        pol_history.append(center_pol[:])

        #generate a new limited_MDP_set
        pw = SWMT.limited_MDP_set(pw_count, years, center_pol, random_seed=step, policy_wiggle=1.0)

        #if that worked: (else what??)
        if len(pw) < pw_count:
            print("Exit Condition: The current policy could not produce enough non-LB/SA pathways.")
            exit_message = "Step " + str(step) + ": The current policy could not produce enough non-LB/SA pathways."
            val_history.append(None)
            break

        #prepare for the regression
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
        val_history.append(results.params[0])

        #check if at least one regression parameter is in-bounds:
        OOB = True
        for c in results.params[1:]: #excluding the constant
            if c <= p_val_limit: OOB = False

        if OOB:
            #none of the regression parameters are in-bounds
            print("Exit Condition: None of the regression parameters have p values less than " + str(p_val_limit) + "... quitting.")
            exit_message = "Step " + str(step) + "None of the regression parameters have p values less than " + str(p_val_limit)
            break

    #we've exitted the loop
    if exit_message == "":
        #we fell off the loop without ending, so we hit max_steps
        message = "Maximum number of steps reached."
        print("Maximum number of steps reached.")


    


    if PRINT_R_PLOTTING:
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
    if True:
        print("")
        print("")
        print("RESULTS")
        print("STEPS: " + str(len(pol_history)))

        print("VALUE PO P1")
        for i in range(len(val_history)):
            print(str(val_history[i]) + " " + str(pol_history[i][0]) + " " + str(pol_history[i][1]) )
