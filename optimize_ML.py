import MDP, MDP_opt, os

def optimize_ML(starting_policy_filename, objective_fn="J3"):
    """This function repeats short gradient decents until new pathways need to be generated.



    """
    
    #Load the two pathway sets
    print("")
    print("Loading training and holdout sets...")
    holdout_set = load_pathway_set("ML_holdout_set")
    training_set = load_pathway_set("ML_training_set")
    print("--- " + str(len(training_set)) + " training set pathways loaded")
    print("--- " + str(len(holdout_set)) + " holdout set pathways loaded")
    

    #when evaluating each new policy against the holdout set, some dissimprovement can be allowed
    #this amount will be added (a disimprovement) to the previous best value. New values must fall
    #below the sum, or else the policy will be considered invalid against the holdout set.
    holdout_wiggle = 0

    #flags
    opt_fail = False
    holdout_fail = False

    #failsafe counter: ML loop will exit if it goes around this many times, irregardless of improvements
    fail_safe_count = 10

    #iterations need to be counted
    iter_count = 0

    #creating policy objects. 
    ML_pol = load_policy(starting_policy_filename)
    holdout_pol = MDP.MDP_Policy(len(ML_pol.get_params()))
    #copying values from ML_pol to holdout_pol without self-references
    holdout_pol.set_params(ML_pol.get_params()[:])

    #print("Creating optimizer objects...")
    #create a FireGirlPolicyOptimizer object and load up the info it needs
    opt = MDP_opt.Optimizer(len(ML_pol.b))
    opt.pathway_set = training_set
    opt.Policy = ML_pol
    #populating initial weights
    opt.calc_pathway_weights()
    opt.set_obj_fn(objective_fn)

    #create a second optimizer object 
    opt_holdout = MDP_opt.Optimizer(len(holdout_pol.b))
    opt_holdout.pathway_set = holdout_set
    def calc_pathway_average_prob(self, pathway):
        """Returns the average probability this pathway's actions, given the current policy"""
        
        sum_of_probs = 0
        
        for ev in pathway.events:
            #use the current policy to calculate a new probability with the original features
            sum_of_probs += self.Policy.calc_action_prob(ev)
        
        #now that we've summed all the probabilities, divide by the total number of events
        ave_prob = sum_of_probs / len(pathway.events)
        
        return ave_prob

    opt_holdout.Policy = holdout_pol
    #populating initial weights
    opt_holdout.calc_pathway_weights()
    opt_holdout.set_obj_fn(objective_fn)
    
    #Get the current values against the given policy
    best_holdout_val = opt_holdout.calc_obj_fn()
    best_opt_val = opt.calc_obj_fn()
    
    #how many gradient decent steps to allow?
    descent_steps = 1
    
    print("")
    print("Initial training set obj.fn. val: " + str(round(best_opt_val)) )
    print("Initial holdout set obj.fn. val:  " + str(round(best_holdout_val)) )
    
    #Begining Machine Learning Loop
    print("")
    print("Beginning ML loop...")
    while True:
        #checking failsafe, to avoid infinite loops
        iter_count += 1
        if iter_count > fail_safe_count:
            print("optimize_ML() has reached its failsafe limit on iterations, and is exiting")
            break
        
        print("  l-bfgs-b pass " + str(iter_count))
        opt_result = opt.optimize_policy(descent_steps)

        #pulling out individual results
        #(ignoring the original values and just pulling the resultant values)
        opt_result_b = opt_result[0][1]
        opt_result_val = opt_result[1][1]
        opt_result_dict = opt_result[2]


        #checking for improvements gained by this policy
        if opt_result_val <= best_opt_val:
            #improvement was found, so record it
            best_opt_val = opt_result_val
        else:
            #no improvement was found, so exit the loop
            opt_fail = True

        #checking for improvements gained against the holdout set
        #setting params to the new ones
        opt_holdout.Policy.set_params(opt_result_b[:])
        new_holdout_val = opt_holdout.calc_obj_fn()
        if new_holdout_val < best_holdout_val + holdout_wiggle:
            #improvement was found, so record it
            best_holdout_val = new_holdout_val
        else:
            #no improvement was found, so exit the loop
            holdout_fail = True

        #if improvements were found in BOTH the training AND the holdout set, record the new policy and continue
        if (not opt_fail) and (not holdout_fail):
            #improvements in both, so record the policy for the next iteration
            ML_pol.set_params(opt_result_b[:])
        else:
            #one of the two failed, so exit the loop
            break


    #Machine Learning Loop has exited
    print("")
    if opt_fail:
        print("scipy.optimize.fmin_l_bfgs_b has found a local optima")

    if holdout_fail:
        print("The final policy did not improve the expectation on the holdout set")

    #print final values
    print("Final training set obj.fn. val: " + str(round(best_opt_val)) )
    print("Final holdout set obj.fn. val:  " + str(round(best_holdout_val)) )
    
    opt.save_policy("ML_policies" + os.sep + "ML_" + objective_fn + "_from_" + str(len(training_set)) + "_pathways.policy")



def load_pathway_set(subfolder):
    """Loads a saved set of pathways from current_directory/subfolder
    """
    
    pathway_set = []
    
    #look at every *.pathways file in the training set folder
    for f in os.listdir(subfolder):
        if f.endswith(".pathways"): 
            f_name = subfolder + os.sep + f
            pkl_file = open(f_name, 'rb')
            this_set = pickle.load(pkl_file)
            pkl_file.close()
            
            #force each pathway to update their values
            #for pw in this_set:
            #    pw.update_net_value()
                
            #and add these to the training set
            pathway_set = pathway_set + this_set           
                
    return pathway_set


def load_policy(filename):
    """Loads a policy from the given folder"""
        
    pkl_file = open(filename, 'rb')
    pol = pickle.load(pkl_file)
    pkl_file.close()
    
    return pol
    


