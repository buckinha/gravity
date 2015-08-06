"""Functions for heuristic solving """
import random
import numpy


def threshold(objfn, x0, bounds=None, iter_cap=1000, tolerance=1.1, MINIMIZING=True, SILENT=False):
    """A Continuous Threshold Acceptance Algorithm

    ARGUEMENTS
    objfn:      the function to be mimimized. Must take one list arguement of the same size as x0
    x0:         the initial state. Must be the same size as the argument list expected by objfn
    bounds:     a list of lists; each sub-list containing two elements. The first is the lower bound
                allowed for that element, and the second is the upper bound. The heuristic will not
                try values outside of these boundaries
    iter_cap:   how many iterations to run. 
    tolerance:  if miniizing (default behavior), the value of any objfn return must be less than or
                equal to tolerance*global_best or else it will be rejected. If Mmaximizing, any objfn
                return value must be greater than or equal to tolerance*global_best.
    MINIMIZING: Boolean. If True (default), the function will seek to minimize objfn. Otherwise, it will
                try to maximize objfn

    """


    element_count = len(x0)
    x_current = x0[:]
    initial_val = objfn(x0)
    global_best = float("inf")
    global_best_solution = [None]*element_count


    #setting up bounds for elements
    if not bounds:
        bounds = [None] * element_count
        for i in range(element_count):
            bounds[i] = [-1.0,1.0]


    if not MINIMIZING:
        #if this is set up as a maximization, then tolerance will be < 1
        # so, we want to switch that around appropriately. If tolerance is 0.95, allowing
        # 5% disimprovements, change that to a 1.05, which also alows 5% disimprovement on the
        # minimization side. It's not exact, but it'll work fine.

        tolerance = 1 + (1-tolerance) 



    #beginning search
    if not SILENT: print("BEGINNING THRESHOLD ACCEPTANCE ALGORITHM")
    for i in range(iter_cap):

        #choose an element of x_current to modify
        changing_index = random.randint(0,element_count-1)

        #choose a new value for that element
        #new_val = random.uniform(bounds[changing_index][0], bounds[changing_index][1])
        new_val = random.normalvariate(x_current[changing_index], 0.05)
        #checking bounds
        if new_val < bounds[changing_index][0]:new_val = bounds[changing_index][0]
        if new_val > bounds[changing_index][1]:new_val = bounds[changing_index][1]

        #remember old value
        old_val = x_current[changing_index]

        #set the value
        x_current[changing_index] = new_val

        #evaluate
        objfn_val = objfn(x_current)

        if not MINIMIZING:
            objfn_val *= -1.0



        #check threshhold
        if objfn_val > (tolerance * global_best):
            #reject this change

            #reset value
            x_current[changing_index] = old_val

        else:
            #accept the new element, but check if it's a new global best or not
            if objfn_val < global_best:
                #the change created a new global best so remember it
                global_best = objfn_val
                global_best_solution = x_current[:]
                if not SILENT: print("..new global best: val=" + str(round(global_best,3)) + ", iter=" + str(i))
            else:
                #the change is a disimprovement, but within the tolerance threshold, so allow it but
                # do not update the global_vest variable
                if not SILENT: print("..disimprovement accepted: val=" + str(round(objfn_val,3)) + ", iter=" + str(i))
                #pass

    #iterations complete
    if not SILENT: print("ITERATIONS COMPLETE")
    if not SILENT: print("Initial Value: " + str(round(initial_val,3)) + "  at solution: " + str(x0))
    if not SILENT: print("Final Value: " + str(round(global_best,3)) + "  at solution: " + str(global_best_solution))

    return global_best_solution

def genetic(objfn, vector_length, bounds=None, iter_cap=500, generation_size=10, mutation_rate=0.10, MINIMIZING=True, seeds=None):
    """A Continuous Genetic Algorithm
    """

    #expanding generation size so that all seeds can be used.
    if seeds:
        if generation_size < len(seeds):
            generation_size = len(seeds)

    mother_set = [None] * generation_size
    daughter_set = [None] * generation_size
    mother_vals = [None] * generation_size
    daughter_vals = [None] * generation_size

    #setting bounds, if not passed in as an argument
    if not bounds:
        bounds = [None] * vector_length
        for i in range(vector_length):
            bounds[i] = [-10,10]

    #setting seeds, if present
    seed_count = 0
    if seeds:
        seed_count = len(seeds)
    for i in range(seed_count):
        mother_set[i] = seeds[i][:]

    #generate random mothers
    for i in range(seed_count, generation_size):
        mother_set[i] = [None] * vector_length
        for j in range(vector_length):
            mother_set[i][j] = random.uniform(bounds[j][0], bounds[j][1])

    
    for i in range(generation_size):
        #evaluate this one
        mother_vals[i] = objfn(mother_set[i])
    
    #Beginning Algorithm
    print("Beginning Continuous Genetic Algorithm")
    for i in range(iter_cap):
        if i == iter_cap-1:
            print ("\r...100%")
        elif i == round(iter_cap * 0.8):
            print("\r...80%")
        elif i == round(iter_cap * 0.6):
            print("\r...60%")
        elif i == round(iter_cap * 0.4):
            print("\r...40%")
        elif i == round(iter_cap * 0.2):
            print("\r...20%")
        #clear old daughters
        daughter_set = [None] * generation_size


        #generate new daughters
        for d in range(generation_size):
            #choose two mothers
            mama1 = random.randint(0,generation_size-1)
            mama2 = random.randint(0,generation_size-1)
            while mama1 == mama2:
                mama2 = random.randint(0,generation_size-1)

            #set daughter elements
            daughter_set[d] = [None] * vector_length
            for e in range(vector_length):
                if random.randint(0,1) == 0:
                    daughter_set[d][e] = mother_set[mama1][e]
                else:
                    daughter_set[d][e] = mother_set[mama2][e]

                #check mutation
                if random.random() < mutation_rate:
                    #mutate this trait
                    daughter_set[d][e] = random.normalvariate(daughter_set[d][e], 0.2)
                    #check bounds
                    if daughter_set[d][e] < bounds[e][0]: daughter_set[d][e] = bounds[e][0]
                    if daughter_set[d][e] > bounds[e][1]: daughter_set[d][e] = bounds[e][1]

        #check the value of each daughter and mother
        for d in range(generation_size):
            daughter_vals[d] = objfn(daughter_set[d])
            mother_vals[d] = objfn(mother_set[d])


        #choose the best of all the mothers and daughters for the new generation
        for d in range(generation_size):
            #for this mother, check if there's a daughter that's better
            best_daughter = -1
            for e in range(generation_size):
                if (MINIMIZING and (daughter_vals[e] < mother_vals[d])) or ((not MINIMIZING) and (daughter_vals[e] > mother_vals[d])):

                        #this daughter is better, so set the mother equal to it
                        mother_set[d] = daughter_set[e][:]

                        #remember this value for elimination, if necessary
                        best_daughter = e

            #finished looking through the daughters, and have assigned the best one to the current mother,
            #  if the best one is an improvement over the current mother
            #so now eliminate that daughter from fture consideration
            if not best_daughter == -1:
                if MINIMIZING:
                    daughter_vals[best_daughter] = float("inf")
                else:
                    daughter_vals[best_daughter] = float("-inf")

    print("..CGA complete")
    print("Best Member of the final generation:")
    best_ind = -1
    best_val = float("inf")
    if not MINIMIZING: best_val = float("-inf")
    for i in range(generation_size):
        if (MINIMIZING and (mother_vals[i] < best_val)) or ((not MINIMIZING) and (mother_vals[i] > best_val)):
            best_val = mother_vals[i]
            best_ind = i
    print("Value: " + str(mother_vals[i]))
    print("Solution: " + str(mother_set[i]))

    return mother_set[i]

def simple_gradient(objfn, fprime, x0, bounds=None, step_size=0.05, MINIMIZING=True, USE_RELATIVE_STEP_SIZES=False, max_steps=200):

    print("")
    x1 = x0[:]
    x2 = x0[:]
    a_little = step_size
    val0 = objfn(x0)

    #creating an array to hold each step position and obj. fn. value
    step_path = []
    value_path = []

    if not bounds:
        #there weren't any bounds passed in, so use the default ones
        bounds = [None] * len(x0)
        for i in range(len(bounds)):
            bounds[i] = [-1,1]
        bounds[0] = [-10,10]

    step_count = 0
    while True:
        step_count += 1
        if step_count > max_steps:
            print("..iteration cap reached, exiting...")
            break

        val1 = objfn(x1)
        grad1 = fprime(x1)

        step_path.append(x1)
        value_path.append(val1)

        grad_max = 0.0
        if USE_RELATIVE_STEP_SIZES:
            for i in range(len(grad1)):
                if numpy.abs(grad1[i]) > numpy.abs(grad_max): grad_max = grad1[i]


        #take a small step in the direction of grad1
        for i in range(len(grad1)):
            #add or subtract, as necessary
            if (MINIMIZING and grad1[i] > 0) or ((not MINIMIZING) and (grad1[i] < 0)):
                if USE_RELATIVE_STEP_SIZES:
                    x2[i] = x1[i] - a_little * numpy.abs(grad1[i]/grad_max)
                else:
                    x2[i] = x1[i] - a_little
            elif (MINIMIZING and grad1[i] < 0) or ((not MINIMIZING) and (grad1[i] > 0)):
                if USE_RELATIVE_STEP_SIZES:
                    x2[i] = x1[i] + a_little * numpy.abs(grad1[i]/grad_max)
                else:
                    x2[i] = x1[i] + a_little
            else:
                # grad1[i] == 0
                x2[i] = x1[i]

        #check bounds
        for i in range(len(x2)):
            if x2[i] < bounds[i][0]: x2[i] = bounds[i][0]
            if x2[i] > bounds[i][1]: x2[i] = bounds[i][1]

        #and now check if that just reset all our changes
        IDENTICAL=True
        for i in range(len(x2)):
            if not x1[i] == x2[i]: IDENTICAL = False
        if IDENTICAL:
            #this means that we were already at the boundaries, changed things, and then got reset back
            # to the boundaries, and if we don't stop now, it'll just keep doing that same thing
            print("..NOTE: Bounds are restricting further improvement... exiting (step " + str(step_count) + ")")
            break




        #see what that did
        val2 = objfn(x2)

        if (MINIMIZING and (val2 < val1))   or   ((not MINIMIZING) and (val2 > val1)) :
            #we've done what we wanted, so accept this change, and loop
            x1 = x2[:]
            continue
        else:
            print("...full gradient step failed to improve, beginning piecewise steps (step " +str(step_count)+ ")")
            #the small step didn't improve anything...
            #try incremental improvements
            IMPROVEMENT_FOUND = False
            for i in range(len(x1)):

                #reset x2 to try the next parameter
                x2 = x1[:]

                #add or subtract, as necessary
                if (MINIMIZING and grad1[i] > 0) or ((not MINIMIZING) and (grad1[i] < 0)):
                    if USE_RELATIVE_STEP_SIZES:
                        x2[i] = x1[i] - a_little * numpy.abs(grad1[i]/grad_max)
                    else:
                        x2[i] = x1[i] - a_little
                elif (MINIMIZING and grad1[i] < 0) or ((not MINIMIZING) and (grad1[i] > 0)):
                    if USE_RELATIVE_STEP_SIZES:
                        x2[i] = x1[i] + a_little * numpy.abs(grad1[i]/grad_max)
                    else:
                        x2[i] = x1[i] + a_little

                #check bounds
                if ( x2[i] < bounds[i][0] ) or ( x2[i] > bounds[i][1] ):
                    #that improvemetn isn't allowed, so continue
                    continue

                #did that improve?
                val2 = objfn(x2)
                if (MINIMIZING and (val2<val1)) or ((not MINIMIZING) and (val2>val1)):
                    #improvement, so keep this, and continue
                    x1 = x2[:]
                    IMPROVEMENT_FOUND = True
                    print("... ... improving on parameter " + str(i))
                    break
                else:
                    #didn't improve, so just continue
                    pass

            #loop has exited, check to see if it was because an improvement was found, or
            #  if it just rolled off the end
            if IMPROVEMENT_FOUND:
                #well, hey!
                #go ahead and continue the gradient decsent
                continue
            else:
                #so following the whole gradient failed,
                # and so did following each parameter's gradient individually, so 
                # stop and take stock
                print("..gradient is disimproving, even piecewise; exiting...")
                break


    #exited outermost loop

    val_final = round(objfn(x1),3)

    print("")
    print("FINAL: No further improvements could be made..")
    print("")
    print("x0 was valued at: " + str(round(val0,3)))
    print(" at a solution of:")
    print(str(x0))
    print("")
    print("Final solution was valued at: " + str(val_final))
    print(" at a solution of:")
    print(str(x1))


    summary = {}
    summary["Step Path"] = step_path
    summary["Value Path"] = value_path
    summary["Starting Value"] = round(val0,3)
    summary["Final Value"] = val_final
    summary["Starting Position"] = x0
    summary["Final Position"] = x1
    
    return summary
    
def simpler_hill_climb(objfn, fprime, x0, step_size=0.2, MINIMIZING=False, max_steps=20, objfn_args=None, fprime_args=None):
    """An extemely simple hill-climbing algorithm with will blindly follow the derivitive for a set number of steps.
    
    """
    
    #a list to hold the position values at each step
    path_list = [None] * max_steps
    
    #a list to hold the objective function values at each step. 
    # These values are not used in the hill-climb, they are just for reference
    value_list = [None] * max_steps

    #a list for the continually updated "position" of the objective function
    x_c = x0[:]
    
    for i in range(max_steps):
        #add the current position to the list that holds the hill-climbing "path"
        path_list[i] = x_c[:]
        
        #compute the current objective function value and record it
        v_c=0
        if objfn_args:
            v_c = objfn(x_c, objfn_args)
        else:
            v_c = objfn(x_c)
        value_list[i] = v_c
        
        #compute the gradient
        g_c=[]
        if fprime_args:
            g_c = fprime(x_c, fprime_args)
        else:
            g_c = fprime(x_c)
        
        #find the largest gradient component
        max_g = float("-inf")
        for j in range(len(g_c)):
            if abs(g_c[j]) > max_g: max_g = abs(g_c[j])
            
        #update the current position appropriately
        if MINIMIZING:
            #MINIMIZING
            #for each gradient component, 
            #  if the component is negative, take a step in the positive direction (thus becoming more negative)
            #  if the component is positive, take a step in the negative direction (thus becoming more negative)
            #scale steps so that the largest gradient component gets a step of step_size 
            # and every other component gets a proportionally smaller step
            for c in range(len(g_c)):
                if g_c[c] > 0:
                    #x_c[c] -= abs(g_c[c] / max_g) * step_size
                    x_c[c] -= step_size
                elif g_c[c] < 0:
                    #x_c[c] += abs(g_c[c] / max_g) * step_size
                    x_c[c] += step_size
                else:
                    #the gradient equals zero...
                    pass
        else:
            #MAXIMIZING
            #for each gradient component, 
            #  if the component is positive, take a step in the positive direction
            #  if the component is negative, take a step in the negative direction (thus becoming more positive)
            #scale steps so that the largest gradient component gets a step of step_size 
            # and every other component gets a proportionally smaller step
            for c in range(len(g_c)):
                if g_c[c] > 0:
                    #x_c[c] += abs(g_c[c] / max_g) * step_size
                    x_c[c] += step_size
                elif g_c[c] < 0:
                    #x_c[c] -= abs(g_c[c] / max_g) * step_size
                    x_c[c] -= step_size
                else:
                    #the gradient equals zero...
                    pass
                
    #finished with all the steps we're allowed to take, so report the pathway and value lists
    summary={}
    summary["Path"] = path_list
    summary["Values"] = value_list
       
    return summary
        

def hill_climb(objfn, 
               x0, 
               step_size=0.1, 
               small_step_size=0.02, 
               greatest_disimprovement=0.95,
               addl_expl_vectors=10,
               starburst_vectors=10,
               starburst_mag=2.0,  
               MINIMIZING=False, 
               max_steps=20, 
               objfn_arg=None):

    """Uses the objective function to test the nearby area and selects the best choice, or the least disimprovement.

    DESCRIPTION
    At each step, a series of changes will be tried to the objective function. For each component of
    the objective function, a new try will be made by increasing that component by a random positive
    amount (up to step_size in magnitude), with all other components being varied at random up to
    +/- small_step_size. For that same component, a second try will be made except with the component 
    being decreased by a random amount up to step_size in magnitude.

    In this way, tries will be made such that all components are varied, at the minimum, in both the
    positive and negative directions.

    The algorithm will then measure the objective function at each of those locations, and if any of
    them are improvements, it will chose the greatest improvement. If all locations are disimprovements
    then, but the best of them is at least better than (greatest_disimprovement * current_value), then
    the best one will be taken as the next move. 

    If none of the exploration vectors have values that pass the greatest_disimprovement test, then
    a 'startburst' step is initiated. A new set of vectors are generated with random components. 
    Each compononent can fall between +/- (starburst_magnitude * step_size), where starburst_magnitude
    is intended on being greater than 1. In this way, when the algorithm otherwise can make no progress
    it makes an amplififed search of the surrounding policy space in all directions. If it can find an
    improvement, it will take it. (At the moment, it will not accept disimprovements)

    The algorithm also terminates if it iterates max_steps times.


    ARGUEMENTS
    objfn: the objective function to be evaluated. Must have signature objfn(x) or objfn(x, arg)
    
    x0: starting position, and defines the length of the vector that the algorithm will use.
    
    step_size: Each component of the objective function will be varied, both positively and negatively
    
    small_step_size: The amount each 'other' component will be varied by (in either +/- directions) 
    
    greatest_disimprovement: a percentage. If disimprovements are not above this value * current objfn
     value, then they will not be accepted.
    
    addl_expl_vectors: how many additional exploration vectors to add to each exploration set. These
     vectors will have random components, each between +/-step_size

    starburst_vectors: how many vectors to use during starburst steps

    starburst_mag: vectors generated during a starburst step have random components, each between
     +/- (starburst_mag * step_size)

    MINIMIZING: A boolean representing whether the algorithm is minimizing the objective function. 
    If not, then it will maximize (default behavior)
    
    max_steps: The maximum number of iterations the algorithm can try before automatically terminating.
    
    objfn_arg: A single arguemnt that may be passed into the objective function (in addition to the x vector.)


    RETURNS
    Dictionary with the following elements:
    "Path": a list of the values of the x vector, in order, that the algorithm tried.
    "Values": a list of the objective function values of each x vector location

    """

    vector_length = len(x0)

    explore_set = [None] * (addl_expl_vectors + vector_length * 2)
    explore_vals = [None] * (addl_expl_vectors + vector_length * 2)


    #record_keeping
    path_list = [None] * max_steps
    value_list = [None] * max_steps
    starburst_history = []
    exploration_history = []


    #set starting position
    x_current = x0[:]
    value_current = None
    if objfn_arg:
        value_current = objfn(x0,objfn_arg)
    else:
        value_current = objfn(x0)


    #iterate
    for i in range(max_steps):

        #either the loop just began, or the last iteration found an allowed move;
        # Either way, add the current position and value to the lists
        path_list[i] = x_current[:]
        value_list[i] = value_current


        ##### STEP 1 - Build Exploration Set

        #loop over the exploration set vectors. There are twice as many as this, but 
        # we're adding them two at a time
        for j in range(vector_length):
            #get a postive value for the jth component, and random values for all others
            #reset the current vector at this position
            explore_set[j] = [None]*vector_length

            for k in range(vector_length):
                #looping over each component
                if j==k:
                    #this is the jth component, so change it POSITIVELY within step_size
                    explore_set[j][k] = x_current[k] + random.uniform(0,step_size)
                else:
                    #this is one of the other components, so change it random +/- small_step_size
                    explore_set[j][k] = x_current[k] + random.uniform((-1.0 * small_step_size), small_step_size)


            #get a negative value for the jth component, and random values for all others
            #reset the current vector at this position
            explore_set[j+vector_length] = [None]*vector_length

            for k in range(vector_length):
                #looping over each component
                if j==k:
                    #this is the jth component, so change it NEGATIVELY within step_size
                    explore_set[j+vector_length][k] = x_current[k] - random.uniform(0,step_size)
                else:
                    #this is one of the other components, so change it random +/- small_step_size
                    explore_set[j+vector_length][k] = x_current[k] + random.uniform((-1.0 * small_step_size), small_step_size)

        #add additional random vectors
        for j in range(addl_expl_vectors):
            addl_vec = [None] * vector_length
            for k in range(vector_length):
                addl_vec[k] = random.uniform(-1*step_size , step_size)
            explore_set[2*vector_length + j] = addl_vec[:]



        ##### STEP 2 - Get Objfn values for each member of the exploration set
        for j in range(addl_expl_vectors + vector_length*2):
            if objfn_arg:
                #an arguement was given, so use it
                explore_vals[j] = objfn(explore_set[j], objfn_arg)
            else:
                #no arguement was given, so just pass x_current
                explore_vals[j] = objfn(explore_set[j])

        #record-keeping: record this exploration set
        exp_hist = {
                   "Step" : i,
                   "Origin" : x_current[:],
                   "Origin Value" : value_current,
                   "Vectors" : explore_set[:],
                   "Values" : explore_vals[:]
                   }
        exploration_history.append(exp_hist)


        ##### STEP 3a - Check for improvements and choose the best
        best_val = None
        best_index = None
        if MINIMIZING: best_val = float("inf")
        if not MINIMIZING: best_val = float("-inf")

        for j in range(addl_expl_vectors + vector_length*2):
            if MINIMIZING:
                if explore_vals[j] < best_val:
                    best_val = explore_vals[j]
                    best_index = j
            else:
                if explore_vals[j] > best_val:
                    best_val = explore_vals[j]
                    best_index = j

        IMPROVEMENT_FOUND = False
        if MINIMIZING:
            if best_val < value_current:
                IMPROVEMENT_FOUND = True
        else:
            if best_val > value_current:
                IMPROVEMENT_FOUND = True

        ##### STEP 3b - If there are no improvements, check if the best value is an allowed disimprovement
        if not IMPROVEMENT_FOUND:
            if MINIMIZING:
                if value_current > 0:
                    #current value is positive, so...
                    #i.e., if greatest_disimprovement = 0.5, then we'll allow up to 1.5
                    # and if greatest_dis... = 0.9, we'll allow up to 1.1
                    if best_val < value_current * (1 + ( 1 - greatest_disimprovement ) ):
                        IMPROVEMENT_FOUND = True
                else: #current value is negative...
                    if best_val < value_current * greatest_disimprovement:
                        IMPROVEMENT_FOUND = True

            else: #MAXIMIZING            
                if value_current < 0:
                    #current value is negative, so...
                    #i.e., if greatest_disimprovement = 0.5, then we'll allow up to 1.5
                    # and if greatest_dis... = 0.9, we'll allow up to 1.1
                    if best_val > value_current * (1 + ( 1 - greatest_disimprovement ) ):
                        IMPROVEMENT_FOUND = True
                else: #current value is positive...
                    if best_val > value_current * greatest_disimprovement:
                        IMPROVEMENT_FOUND = True


        ##### STEP 4 - If there are not even any allowed disimprovements, try the starburst step
        if not IMPROVEMENT_FOUND:
            #generate starburst vectors
            starbursts = [None] * starburst_vectors
            for s in range(starburst_vectors):
                starbursts[s] = [None] * vector_length
                for c in range(vector_length):
                    starbursts[s][c] = random.uniform(-1 * starburst_mag * step_size, starburst_mag * step_size)

            #calculate the values of each starburst vector
            starburst_values = [None] * starburst_vectors
            best_sbv = float("-inf")
            best_sbv_index = None
            if MINIMIZING: best_sbv = float("inf")

            for s in range(starburst_vectors):
                if objfn_arg:
                    starburst_values[s] = objfn(starbursts[s], objfn_arg)
                else:
                    starburst_values[s] = objrn(starbursts[s])

                #remember the value if it's the best of the starbursts
                if MINIMIZING:
                    if starburst_values[s] < best_sbv:
                        best_sbv = starburst_values[s]
                        best_sbv_index = s
                else:
                    if starburst_values[s] > best_sbv:
                        best_sbv = starburst_values[s]
                        best_sbv_index = s

            #check if the best of the starburst vectors is better than the current position
            #TODO, if desired, add allowance for disimprovement

            #record_keeping, record these starburst vectors
            star_hist = {
                         "Step": i,
                         "Origin": x_current[:],
                         "Origin Value" : value_current,
                         "Vectors" : starbursts[:],
                         "Vector Values" : starburst_values[:]
                        }
            starburst_history.append(star_hist)


            if MINIMIZING:
                if best_sbv < value_current:
                    IMPROVEMENT_FOUND = True
                    x_current = starbursts[best_sbv_index][:]
                    value_current = best_sbv
            else:
                if best_sbv > value_current:
                    IMPROVEMENT_FOUND = True
                    x_current = starbursts[best_sbv_index][:]
                    value_current = best_sbv


        #### STEP 5 - Terminate if no acceptable moves have been found
        if not IMPROVEMENT_FOUND:
            #there are no improvements or allowed disimprovements, even after a starburst
            #so, terminate.
            print("")
            print("..hill_climb() cannot find improvements or allowed disimprovments... terminating")
            print("..step: " + str(i+1) + " of " + str(max_steps))
            #print("CURRENT EXPLORATION SET:")
            #print(str(explore_set))
            #print("")
            #print("SET VALUES")
            #print(str(explore_vals))
            break


        #### STEP 4 - An improvement or allowed disimprovement was found, so update the path and value
        # lists, and update the current position
        x_current = explore_set[best_index][:]
        value_current = best_val



    #FINISHED
    #either we ran out of steps or the algorithm quit on its own because it couldn't find
    # any improvements or allowed disimprovements.
    summary={}
    summary["Path"] = path_list
    summary["Values"] = value_list
    summary["Final Position"] = x_current
    summary["Final Value"] = value_current
    summary["Exploration History"] = exploration_history
    summary["Starburst History"] = starburst_history
    
    return summary
        

def parabola_1(x):
    return x[0]*x[0] - x[1]

def parabola_1_prime(x):
    fx0 = 2*x[0]
    fx1 = -1
    return [fx0, fx1] 

def hyp_par_5_var(x):
    return (x[0]*x[0]) + ((x[1]-1)*(x[1]-1)) + ((x[2]-2)*(x[2]-2)) + ((x[3]-3)*(x[3]-3)) + ((x[4]-4)*(x[4]-4))

def hyp_par_5_var_prime(x):
    fx0 = 2*x[0] 
    fx1 = 2*x[1] - 2*1
    fx2 = 2*x[2] - 2*2
    fx3 = 2*x[3] - 2*3
    fx4 = 2*x[4] - 2*4
    return [fx0, fx1, fx2, fx3, fx4]
    
    
def hyp_par_5_var_max(x):
    return (-1.0*x[0]*x[0]) + (-1.0*(x[1]-1)*(x[1]-1)) + (-1.0*(x[2]-2)*(x[2]-2)) + (-1.0*(x[3]-3)*(x[3]-3)) + (-1.0*(x[4]-4)*(x[4]-4))

def hyp_par_5_var_max_prime(x):
    fx0 = -2.0*x[0] 
    fx1 = -2.0*x[1] + 2*1
    fx2 = -2.0*x[2] + 2*2
    fx3 = -2.0*x[3] + 2*3
    fx4 = -2.0*x[4] + 2*4
    return [fx0, fx1, fx2, fx3, fx4]