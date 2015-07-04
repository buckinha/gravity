"""Functions for heuristic solving"""
import random


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






def parabola_1(x):
    return x[0]*x[0] - x[1]

def hyper_parabola_5_var(x):
    return (x[0]*x[0]) + ((x[1]-1)*(x[1]-1)) + ((x[2]-2)*(x[2]-2)) + ((x[3]-3)*(x[3]-3)) + ((x[4]-4)*(x[4]-4))