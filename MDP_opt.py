import math, scipy, pickle
from scipy.optimize import *
import MDP
from FireGirlPathway import *

class Optimizer:
    def __init__(self,policy_length):
        """
        Arguements
        policy_length: integer: how many elements are in each state/policy for this MDP
        """

        #A list to hold all the pathways in a data set
        self.pathway_set = []
                
        #A list to hold the "probability weights" of each pathway
        self.pathway_weights = []                       #J1 weights
        self.pathway_weights_normalized = []            #J1.1 weights
        self.pathway_weights_averaged = []              #J2 weights
        self.pathway_weights_importance_sampling = []   #J3 weights
        #self.pathway_weights_generation = []   #denominators for J3 weights

        
        #Boundaries for what the parameters can be set to during scipy's optimization routine:
        self.b_bounds = [[-1.0,1.0]] * policy_length

        #and set the constant to be able to vary more freely
        self.b_bounds[0] = [-1000.0, 1000.0]


        #Flag: Use log(probabilities)  -  If we want to force sums of log(probs), set to True
        #                                 To just multiply probabilities, set to False
        self.USE_LOG_PROB = False
        
        #Flag: use normalized weights for the objective function
        self.NORMALIZED_WEIGHTS_OBJ_FN = False
        #Flag: use normalized weights for F prime
        self.NORMALIZED_WEIGHTS_F_PRIME = False
        #Flag: use averaged weights for the objective function
        self.AVERAGED_WEIGHTS_OBJ_FN = False
        #Flag: use averaged weights for F prime
        self.AVERAGED_WEIGHTS_F_PRIME = False

        #Flag: Use importance sampling weights (J3)
        self.IMPORTANCE_SAMPLING = True

        #Flag: Normalize Pathway net values
        self.NORMALIZE_PATHWAY_NET_VALUES = True #(J4)


        #Flag: Suppress all outputs that originate from this object
        self.SILENT = False
        #Flag: Suppress all outputs that originate from pathways this object creates/uses
        self.MUTE_PATHWAYS = True


        #The policy that is controlling each pathway's actions
        self.Policy = MDP.MDP_Policy(policy_length)
        self.policy_length = policy_length


    ##########################
    # Optimization Functions #
    ##########################

    def normalize_pathways(self):
        n = self.normalize_all_features()
        self.normalize_pathway_values()

    def normalize_all_features(self, rng=1.0):
        #This function normalizes each state(feature) variable of each event in each pathway to fall
        #  between its rng and -rng arguments.

        #arrays to hold the max and min of each feature over ALL ignitions in ALL pathways
        feature_max = [float("-inf")] * self.policy_length
        feature_min = [float("inf") ] * self.policy_length

        #look in every pathway
        for pw in self.pathway_set:
            #look in every event in this pathway
            for ev in pw.events:
                #look at each feature in this ignition
                for f in range(ev.state_length):

                    #if this feature is higher than the previously found max, remember it
                    if ev.state[f] > feature_max[f]:
                        feature_max[f] = ev.state[f]
                    #if this feature is lower than the previously found min, remember it
                    if ev.state[f] < feature_min[f]:
                        feature_min[f] = ev.state[f]

        #calculate means
        feature_mean = []
        for f in range(len(feature_max)):
            feature_mean.append(   (feature_max[f] + feature_min[f]) / 2.0   )

        #calculate normalization magnitude
        #this is the value that we'll divide each feature by, once its been recentered around
        # a mean of zero
        feature_norm_mag = []
        for f in range(len(feature_max)):
            feature_norm_mag.append(  (feature_max[f] - feature_mean[f]) / rng )


        #now normalize each feature value to fall between "max" and "min"
        #in every pathway
        for pw in self.pathway_set:
            #in every ignition in this pathway
            for ev in pw.events:
                #look at each feature in this ignition
                for f in range(ev.state_length):

                    #normalize step 1: shift the mean to zero
                    ev.state[f] -= feature_mean[f]

                    #normalize step 2: change magnitude to be within +/- rng
                    #but check for 0's:  Any feature that is absolutely constant (like the constant) will 
                    # have a feature_norm_mag value of zero. When that's the case, just set the feature
                    # to one (rather than zero, because a one will allow the constant parameter to actually
                    #     do something)
                    if feature_norm_mag[f] == 0:
                        ev.state[f] = 1.0
                    else:
                        ev.state[f] = ev.state[f] / feature_norm_mag[f]

            #record normalization values, in case the pathway ever needs to convert back to original values
            pw.normalized = True
            pw.normalization_mags = feature_norm_mag
            pw.normalization_means = feature_mean


        #in case it's desired, pass back the max, min, mean, and norm values
        return_dict = {}
        return_dict["Max Values"] = feature_max
        return_dict["Ave Values"] = feature_mean
        return_dict["Min Values"] = feature_min
        return_dict["Normalization Magnitude"] = feature_norm_mag
        
        return return_dict

    def normalize_pathway_values(self, rng=1.0):
        """Rescales the net-values of the pathways in self.pathway_set to fall between +/-rng"""

        #find the max and min
        #and find the average
        val_max = float("-inf")
        val_min = float("inf")
        val_sum = 0.0

        for pw in self.pathway_set:
            if pw.net_value > val_max: val_max = pw.net_value
            if pw.net_value < val_min: val_min = pw.net_value
            val_sum += pw.net_value

        val_avg = val_sum / len(self.pathway_set)
        norm_mag = rng / (val_max - val_avg)

        #re-scale
        for pw in self.pathway_set:
            pw.normalized_value = True 
            pw.normalized_value_mag = norm_mag
            pw.normalized_value_avg = val_avg

            pw.net_value = ( pw.net_value - val_avg ) * norm_mag




    def calc_pathway_joint_prob(self, pathway, USE_LOG_PROB=False):
        """Calculates the joint probability of events in an MDP_Pathway object

        #This function looks through each ignition event and computes the 
        #   product of all the action probabilities.  If the
        #   USE_LOG_PROB flag is set, it will sum the logged probabilities.
        """

        total_prob = 0
        #variables for use within the loop (so as not to instantiate them repeatedly)
        action_prob = 0
        product = 1
        summation = 0
        break_loop = False

        if USE_LOG_PROB == False:
            #according to the flag, we're not using sum(log(probs)) so just
            #  compute the product

            #reseting values for a new loop iteration
            product = 1.0
            break_loop = False
                
            for ev in pathway.events:
                try:
                    #use the current policy to calculate a new probability with the original features
                    #   of each event
                    action_prob = self.Policy.calc_action_prob(ev)
                        
                    product *= action_prob
                    
                #except (TypeError):
                #    print("FGPathway.calcTotalProb() encountered a TypeError:")
                except (ArithmeticError):
                    #it is possible that a long series of multiplications over fractions
                    #  could result in an underflow conidtion. Here I'm assuming that is
                    #  the type of ArithmeticError that we've run into.

                    #since the value underflowed, it is clearly VERY small, so just
                    # set it to zero
                    product = 0.0

                    #and report it
                    print("a FireGirlPathway object reports an underflow condition during a PRODUCT calculation")

                    #and don't bother with any more multiplications
                    break_loop = True

                if break_loop == True: break 

            total_prob = product

        else:
            #according to the flag, we ARE using sum(log(probs)) so do so:
            
            #resetting values for a new loop iteration
            summation = 0.0

            for ev in pathway.events:
                #use the current policy to calculate a new probability with the original features
                #   of each ignition
                action_prob = self.Policy.calc_action_prob(ev)
                    
                summation += math.ln(action_prob)

            try:
                total_prob = math.exp(summation)
            except (ArithmeticError):
                #I'm assuming here that an underflow condition is the only type of
                #  arithmeticError I'm likely to find...

                #If it underflows, the probability total is very small, so just set
                #  it to zero
                total_prob = 0.0

                #and report it
                print("a FireGirlPathway object reports an underflow condition during a SUMMATION calculation")


        return total_prob

    def calc_pathway_average_prob(self, pathway):
        """Returns the average probability this pathway's actions, given the current policy"""
        
        sum_of_probs = 0
        
        for ev in pathway.events:
            #use the current policy to calculate a new probability with the original features
            sum_of_probs += self.Policy.calc_action_prob(ev)
        
        #now that we've summed all the probabilities, divide by the total number of events
        ave_prob = sum_of_probs / len(pathway.events)
        
        return ave_prob


    def calc_pathway_weights(self):
        """This function looks through each event of a given pathway and applies the current
        #  policy to the features of each one. 

        The resulting 'probability' from the policy 
        #  function is either multiplied or 'log-summed' be the others to produce the final 
        #  weighting value. This is done for every pathway in the pathway_set, and each
        #  weight is assigned to pathway_weights[] at the same index as their pathway in 
        #  the pathway_set list
        """

        #clearing old weights
        pw_count = len(self.pathway_set)
        self.pathway_weights = [None] * pw_count                       #J1 weights
        self.pathway_weights_normalized = [None] * pw_count            #J1.1 weights
        self.pathway_weights_averaged = [None] * pw_count              #J2 weights
        self.pathway_weights_importance_sampling = [None] * pw_count   #J3 weights

        #iterating over each pathway and appending each new weigh to the list
        for i in range(pw_count):
            
            #calculate all three weighting regimes
            p1 = self.calc_pathway_joint_prob(self.pathway_set[i])     #J1 weights - "joint probability"
            p2 = self.calc_pathway_average_prob(self.pathway_set[i])   #J2 weights - averaged probabilities
            
            self.pathway_weights[i] = p1                  #J1 weights
            self.pathway_weights_averaged[i] = p2         #J2 weights
        
                    
        #calculate sum of ALL weights for J1.1
        weight_sum = 0.0
        for w in self.pathway_weights:
            weight_sum += w
        
        #computing normalization of each pathway weight over the sum of ALL pathway weights
        for i in range(pw_count):
            self.pathway_weights_normalized[i] = self.pathway_weights[i] / weight_sum  #J1.1 weights

        #calculating J3 weights
        #if generation weights have not been otherwised assigned, they will be equal to 1, 
        #   and therefore J3 weights will be identical to J1.0 weights
        for i in range(pw_count):

            #assigning weights
            self.pathway_weights_importance_sampling[i] = self.pathway_weights[i] / self.pathway_set[i].generation_joint_prob
                    
    def calc_obj_fn(self, b=None):
        #This function contains the optimization objective function. It operates
        #  on the current list of pathways. If any values for 'b' are passed in,
        #  (most likely by scipy.optimize.fmin_l_bfgs_b(), then they are assigned
        #  to the current HKBFire_Policy object so that subsequent function calls
        #  will use the correct ones.


        #variable to hold the final value
        obj_fn_val = 0    
        
        # checking for new beta parameters (which will be set by scipy.fmin.l_bfgs_b)
        if not b == None:
            self.Policy.set_params(b)

        #Note: self.calcPathwayWeights will assign this policy to each pathway

        # Calculate the weights... these will use the current Policy
        #    rather than whatever policy the pathways used during simulation.
        self.calc_pathway_weights()
        
        #a variable to hold the sum of ALL the probability-weighted pathway values
        total_value = 0

        #loop over all the values/weights and sum them
        for i in range(len(self.pathway_set)):
        
            #check which weights to use:
            #Importance Sampling (J3) supercedes Normalization (J1.1)
            if self.IMPORTANCE_SAMPLING:
                #using J3 weights
                #TODO: Verfiy that this is how to do this???
                total_value += self.pathway_set[i].net_value * self.pathway_weights_importance_sampling[i]

            #Normalization (J1.1) supercedes averaging (J2)
            elif self.NORMALIZED_WEIGHTS_OBJ_FN:
                #using J1.1 normalized weights
                total_value += self.pathway_set[i].net_value * self.pathway_weights_normalized[i]
            else:
                #using "un-normalized" weights... check averaging...
                if self.AVERAGED_WEIGHTS_OBJ_FN:
                    #using J2 averaged weights
                    total_value += self.pathway_set[i].net_value * self.pathway_weights_averaged[i]
                else:
                    #using straight J1.0 "joint probability weights"
                    total_value += self.pathway_set[i].net_value * self.pathway_weights[i]


        #NOTE:
        #any final checks/modifications to total_val can go here:

        #since scipy fmin... is a minimization routine, return the negative
        #obj_fn_val = -1.0 * total_value          
        #return obj_fn_val
        return total_value

    def FP_delta_prob(self, beta, pw, evnt):
        #this function calculates the inner "delta_prob" value for each calculation of the derivitive.
        # caclObjFPrime() loops over it repeatedly, and passes in which parameter (beta), pathway (pw) and
        # event (evnt) to calculate value, which is then summed in caclObjFPrime()
        
        #get the action choice of this pathway at this event
        action = self.pathway_set[pw].events[evnt].action
        
        #get the feature of this pathway and this event for this beta
        flik = self.pathway_set[pw].events[evnt].state[beta]

        prob = self.Policy.calc_prob(self.pathway_set[pw].events[evnt].state)
        delta_lgstc = flik * prob * (1-prob)

        delta_prob = delta_lgstc
        if action:
            #action was taken, so leave it alone
            pass
        else:
            #action was not taken, so:
            delta_prob *= -1


        return delta_prob
    
    def calc_obj_FPrime(self, betas=None):
        #This function returns the gradient of the objective function

        #The scipy documentation describes the fprime arguement as:
        #fprime : callable fprime(x,*args)
        #The gradient of func. If None, then func returns the function value and the 
        #  gradient (f, g = func(x, *args)), unless approx_grad is True in which case func returns only f.

        #The return value should probably just be a list of the derivitive values with respect to each 
        #   b-parameter in the policy

        #  d Obj()/d b_k value is 
        #
        #   Sum over pathways [ val_l * product over ignitions [ prob_i ] * sum over ignitions [d wrt prob / prob] ]
        #
        #   where d wrt prob =  sup_i * d(logistic(b*f)) + (1 - sup_i)(-1) (d(logistic(b*f)))
        #
        #   and where d(logistic(b*f)) = f_l,i,k * logistic(f_l,i,k * b_k) * (1 - logistic(f_l,i,k * b_k))

        #Assign values to b. I don't think this function will ever be called
        #  outside of the l_bfgs function, but if so, handle it:
        b = None
        if betas == None:
            b = self.Policy.get_params()
        else:
            b = betas
            self.Policy.set_params(b)

        # list to hold the final values, one per parameter
        d_obj_d_bk = []
        for i in range(len(b)):
            d_obj_d_bk.append(0.0)
            
        #get the weight (total or ave) for each pathway decision sequence using the 
        #   current policy (which is possibly being varied by l_bfgs, etc...)
        
        self.calc_pathway_weights()

        #variables for use within the loop
        prob = 1.0
        delta_prob = 0.0
        sum_delta_prob = 0.0
        sum_delta_prob_AVE = 0.0
        action = False
        sup = 0
        
        #iterate over each beta and evaluate the gradient along it
        for beta in range(len(b)):

            #SEE MATHEMATICS DOCUMENTATION FOR A DETAILED EXPLANATION OF ALL THAT FOLLOWS

            for pw in range(len(self.pathway_set)):

                #reset value for this pathway
                sum_delta_prob = 0.0
                sum_delta_prob_AVE = 0.0

                for evnt in range(len(self.pathway_set[pw].events)):

                    #get the action choice of this pathway at this event
                    action = self.pathway_set[pw].events[evnt].action

                    #get the probability of actually making this action in this event
                    prob = self.pathway_set[pw].events[evnt].decision_prob

                    #checking for unreasonably small probabilities
                    if prob == 0: prob = 0.00001

                    delta_prob = self.FP_delta_prob(beta, pw, evnt)

                    sum_delta_prob_AVE += delta_prob #for use in the J2 "averaged" calculation
                    sum_delta_prob += delta_prob / prob #for use in J1 "standard" and J1.1 "normalized"

                
                #finished adding up sum_delta_prob for all the ignitions in this pathway, so
                # calculate the d/dx value:
                
                
                #check which weights to use, and to the derivative appropriately
                if self.IMPORTANCE_SAMPLING:
                    #using J3 Importance Sampling Weights
                    d_obj_d_bk[beta] += self.pathway_set[pw].net_value * self.pathway_weights_importance_sampling[pw] * sum_delta_prob
                elif self.NORMALIZED_WEIGHTS_F_PRIME:
                    #using normalized weights inside the derivative calculation (J1.1)
                    d_obj_d_bk[beta] += self.pathway_set[pw].net_value * self.pathway_weights_normalized[pw] * sum_delta_prob
                else:
                    if self.AVERAGED_WEIGHTS_F_PRIME:
                        #doing average-weight calculation (J2)
                        invI = (1.0 / len(self.pathway_set[pw].events) )
                        d_obj_d_bk[beta] += self.pathway_set[pw].net_value * invI * sum_delta_prob_AVE                            
                    else:
                        #using standard joint-probability math (J1.0)
                        d_obj_d_bk[beta] += self.pathway_set[pw].net_value * self.pathway_weights[pw] * sum_delta_prob
                    
                    

                #going on to the next pathway

            #going on to the next beta

        #finished with all betas

        #because this is a minimization routine, and the objective function is being flipped, so too
        #  should be the derivatives
        #for b in range(len(d_obj_d_bk)):
        #    d_obj_d_bk[b] *= -1.0


        # And Finally, return the list
        return scipy.array(d_obj_d_bk)
   
    def optimize_policy(self, max_iterations=15000):
        """Does a single l_bfgs_b gradient decent on the current pathway set.

        Arguements:
        max_iterations: the total number of steps to let scipy.fmin_l_bfgs_b to take

        Returns
        TODO: fix calling functions to re-work the return values
        A list containing two elements
        -Element 1: A list of the policy parameters used in each iteration 
        -Element 2: A list of the optimized objective function value for each iteration

        Note: there are no "iterations" anymore. A single pass is made, so only one two elements
        are present in each list: the starting parameters/value, and the resulting parameters/value.
        """
        
        #record the first 'optimization value' which is really just a placeholder 
        #  to keep indices even
        start_val = self.calc_obj_fn()
        #record the first parameter set
        start_params = self.Policy.get_params()
        
        #tell scipy to optimize our objective function (calcObjectiveFn())
        #  given a certain set of parameters...
        
        #running the scipy gradient descent method
        #signature is:
        #scipy.optimize.fmin_l_bfgs_b(func, x0, fprime=None, args=(), approx_grad=0, bounds=None, m=10, 
        #                             factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=-1, maxfun=15000,
        #                             maxiter=15000, disp=None, callback=None)
        # func is a function name, at is referenceing the objective function that it will use
        # x0 is the starting parameters that fmin_l_bfgs_b() will use
        # fprime="None" is telling fmin...() that we are not passing in a function to calculate derivitave values
        # args=() is asking for any input arguments needed for the objective function, other than the beta parameters
        # approx_grad is telling fmin...() to approximate it's own derivatives, or to use some other gradient
        # bounds should be a list of upper and lower bound pairs. See scipy.optimize.fmin_l_bfgs_b documentation.
        # m is the maximum number of variable metric corrections used to define the limited memory matrix. 
        #     (The limited memory BFGS method does not store the full hessian but uses this many terms in an 
        #      approximation to it.)
        # factr is the iteration stopping point. When (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= factr * eps, where eps 
        #      is the machine precision, which is automatically generated by the code. Typical values for factr are: 
        #      1e12 for low accuracy; 1e7 for moderate accuracy; 10.0 for extremely high accuracy.
        # The rest of the arguments are left as defaults.
        

        #setting x0 to the initial guess, which can just be whatever our current policy parameters are.
        #converting to numpy arrays

        x0 = scipy.array(self.Policy.get_params())

        #DEBUG
        if not (len(self.b_bounds) == len(x0)):
            print("error in MDP_PolicyOptimizer.optimizePolicy...")
            print("x0 = " + str(x0))
            print("bounds = " + str(self.b_bounds))



        #                                                   arg names:                
        output_policy = fmin_l_bfgs_b(self.calc_obj_fn,       #func
                                      x0,                   #x0
                                      self.calc_obj_FPrime,   #fprime
                                      args=[],              #args
                                      approx_grad=False,    #approx_grad
                                      bounds=self.b_bounds, #bounds
                                      factr=1,              #factr
                                      maxiter=max_iterations)#maxiter
        
        #the output of fmin_l_bfgs_b() has the following structure: [x, f, d], where:
        #   x : array_like
        #       Estimated position of the minimum.
        #   f : float
        #       Value of func at the minimum.
        #   d : dict
        #       Information dictionary.
        
        
        #record the new parameter set
        end_params = output_policy[0]
        
        #record the final objective function value
        end_val = output_policy[1]
        
        output_dict = output_policy[2]
        
      
        #take the new parameter set and assign them back to the policy
        self.Policy.set_params(output_policy[0])
        

        return [[start_params, end_params], [start_val, end_val], output_dict]
    
    def set_obj_fn(self, objfn):
        """Sets the optimizer's flags to set it up for the given objective function type.

        Arguements
        objfn: a string indicating which objective function to use.
        --options are:
        -- "J1" for the J1.1 objective function (original joint-probability method with normalization)
        -- "J2" for the J2 objective function (Hailey's averaging method)
        -- (J1.0 is ineffective, so it is not represented here)
        """

        if objfn == "J1":
            #set flags for J1.0
            self.IMPORTANCE_SAMPLING = False
            self.NORMALIZED_WEIGHTS_OBJ_FN = False
            self.NORMALIZED_WEIGHTS_F_PRIME = False
            self.AVERAGED_WEIGHTS_OBJ_FN = False
            self.AVERAGED_WEIGHTS_F_PRIME = False
            #set flags for J1.1          
            # self.IMPORTANCE_SAMPLING = False
            # self.NORMALIZED_WEIGHTS_OBJ_FN = True
            # self.NORMALIZED_WEIGHTS_F_PRIME = True
            # self.AVERAGED_WEIGHTS_OBJ_FN = False
            # self.AVERAGED_WEIGHTS_F_PRIME = False
        elif objfn == "J2":
            #set flags for J2
            self.IMPORTANCE_SAMPLING = False
            self.NORMALIZED_WEIGHTS_OBJ_FN = False
            self.NORMALIZED_WEIGHTS_F_PRIME = False
            self.AVERAGED_WEIGHTS_OBJ_FN = True
            self.AVERAGED_WEIGHTS_F_PRIME = True
        elif objfn == "J3":
            self.IMPORTANCE_SAMPLING = True
            self.NORMALIZED_WEIGHTS_OBJ_FN = False
            self.NORMALIZED_WEIGHTS_F_PRIME = False
            self.AVERAGED_WEIGHTS_OBJ_FN = False
            self.AVERAGED_WEIGHTS_F_PRIME = False

        else:
            #undefined behavior... default to J1.10
            print("Undefined objective function: " + str(objfn) + "   Setting J1.0 by default")
            self.IMPORTANCE_SAMPLING = False
            self.NORMALIZED_WEIGHTS_OBJ_FN = False
            self.NORMALIZED_WEIGHTS_F_PRIME = False
            self.AVERAGED_WEIGHTS_OBJ_FN = False
            self.AVERAGED_WEIGHTS_F_PRIME = False

    def print_opt_output(self, output):
        #takes the outputs from the optimize() function and prints them in a nicer way
        params = output[0]
        obj_vals = output[1]
        
        print("         ObjFn Val,     Params.....")
        print("                        CONS   date    date2    temp   wind   timb   timb8  timb24  fuel  fuel8  fuel24")
        for v in range(len(obj_vals)):
            if v == 0:
                print("before: "),
            else:
                print("after:  "),
            print(str(round(obj_vals[v],10)) + "  "),
            
            for p in range(len(params[v])):
                print(" " + str(round(params[v][p],5))),
            
            print("") #to end the line
                
 
    ###################
    # Other Functions #
    ###################
    
    def create_and_convert_firegirl_pathways(self, pathway_count, years, start_ID):
        """Creates a set of FireGirl pathways, and then converts them into MDP pathways for use in the optimizer"""
        
        #setting up initial lists
        fg_pathways = [None] * pathway_count
        self.pathway_set = [None] * pathway_count
        
        for i in range(pathway_count):
            fg_pathways[i] = FireGirlPathway(i+start_ID)
            fg_pathways[i].Policy.b = self.Policy.b[:]
            fg_pathways[i].generateNewLandscape()
            fg_pathways[i].doYears(years)
            fg_pathways[i].updateNetValue()
            self.pathway_set[i] = MDP.convert_firegirl_pathway_to_MDP_pathway(fg_pathways[i])
        
        #normalizing pathways
        self.normalize_all_features()

        #normalizing net values
        if self.NORMALIZE_PATHWAY_NET_VALUES:
            self.normalize_pathway_values()

        #populate initial weights
        self.calc_pathway_weights()
        
        
    def save_pathways(self, filename):
        output = open(filename, 'wb')

        # Pickle dictionary using default protocol 0.
        pickle.dump(self.pathway_set, output)

        output.close()

    def load_pathways(self, filename):
        #This function loads a saved set of FireGirl pathways

        pkl_file = open(filename, 'rb')

        self.pathway_set = []
        self.pathway_set = pickle.load(pkl_file)

        pkl_file.close()

    def strip_nonessential_pathway_data(self):
        for pw in self.pathway_set:
            pw.strip_metadata()

    def reset_policy(self, policy_length):
        #This function resets the policy to a 50/50 coin-toss
        self.Policy = MDP.MDP_Policy(policy_length)

    def save_policy(self, filename):
        """Saves the current policy object"""
        output = open(filename, 'wb')

        # Pickle dictionary using default protocol 0.
        pickle.dump(self.Policy, output)

        output.close()
        
    def load_policy(self, filename):
        """Loads a saved policy and assigns it to this optimization object"""

        pkl_file = open(filename, 'rb')

        self.Policy = pickle.load(pkl_file)

        pkl_file.close()



###################################
# Stand-alone Objective Functions #
###################################


def normalize_pathway_features(pathways, rng=1.0):
    """Returns a pathway set with normalized features

    This function normalizes each state(feature) variable of each event in each pathway to fall
    between its rng and -rng arguments.

    Any pathway that reports having already been normalized, is 'un-normalized' first
    """

    #un-normalize anythign that needs it
    #TODO

    feature_count = pathways[0].policy_length

    #arrays to hold the max and min of each feature over ALL ignitions in ALL pathways
    feature_max = [float("-inf")] * feature_count
    feature_min = [float("inf") ] * feature_count

    #look in every pathway
    for pw in pathways:
        #look in every event in this pathway
        for ev in pw.events:
            #look at each feature in this ignition
            for f in range(ev.state_length):

                #if this feature is higher than the previously found max, remember it
                if ev.state[f] > feature_max[f]:
                    feature_max[f] = ev.state[f]
                #if this feature is lower than the previously found min, remember it
                if ev.state[f] < feature_min[f]:
                    feature_min[f] = ev.state[f]

    #calculate means
    feature_mean = [None] * feature_count
    for f in range(len(feature_max)):
        feature_mean[f] =  (feature_max[f] + feature_min[f]) / 2.0   

    #calculate normalization magnitude
    #this is the value that we'll divide each feature by, once its been recentered around
    # a mean of zero
    feature_norm_mag = [None] * feature_count
    for f in range(len(feature_max)):
        feature_norm_mag[f] =  (feature_max[f] - feature_mean[f]) / rng 


    #now normalize each feature value to fall between "max" and "min"
    #in every pathway
    for pw in pathways:
        #in every ignition in this pathway
        for ev in pw.events:
            #look at each feature in this ignition
            for f in range(ev.state_length):

                #normalize step 1: shift the mean to zero
                ev.state[f] -= feature_mean[f]

                #normalize step 2: change magnitude to be within +/- rng
                #but check for 0's:  Any feature that is absolutely constant (like the constant) will 
                # have a feature_norm_mag value of zero. When that's the case, just set the feature
                # to one (rather than zero, because a one will allow the constant parameter to actually
                #     do something)
                if feature_norm_mag[f] == 0:
                    ev.state[f] = 1.0
                else:
                    ev.state[f] = ev.state[f] / feature_norm_mag[f]

        #record normalization values, in case the pathway ever needs to convert back to original values
        pw.normalized = True
        pw.normalization_mags = feature_norm_mag
        pw.normalization_means = feature_mean

    #No return value... the pathways are already updated

def normalize_pathway_net_values(pathways, rng=1.0):
    """Rescales the net-values of the pathways in self.pathway_set to fall between +/-rng"""

    #find the max and min
    #and find the average
    val_max = float("-inf")
    val_min = float("inf")
    val_sum = 0.0

    for pw in pathways:
        if pw.net_value > val_max: val_max = pw.net_value
        if pw.net_value < val_min: val_min = pw.net_value
        val_sum += pw.net_value

    val_avg = val_sum / len(pathways)
    norm_mag = rng / (val_max - val_avg)

    #re-scale
    for pw in pathways:
        pw.normalized_value = True 
        pw.normalized_value_mag = norm_mag
        pw.normalized_value_avg = val_avg

        pw.net_value = ( pw.net_value - val_avg ) * norm_mag


def J1(policy_vector, pathways, FEATURE_NORMALIZATION=True, VALUE_NORMALIZATION=False, SILENT=True, RETURN_WEIGHTS=False):
    """Calculate the J1 objective function given a policy and a pathway set

    CALCULATION TYPE 
    -Using the pathway's average net_value per timestep
    -Normalizing features
    -Objective function value is divided by the number of pathways

    NOTES:
    -Assumes all pathways have the same number of timesteps as the first one in the list
    """

    timesteps = len(pathways[0].events)

    #Normalize Feature Values
    if FEATURE_NORMALIZATION:
        normalize_pathway_features(pathways)

    #Normalize Pathway Values
    if VALUE_NORMALIZATION:
        normalize_pathway_net_values(pathways)


    #Gather the pathway values
    values = [None] * len(pathways)
    for p in range(len(pathways)):
        #using value averaged over each timestep
        values[p] = (pathways[p].net_value / timesteps)

    
    #Calculate joint probabilities for each pathway
    weights = [None] * len(pathways)

    for p in range(len(pathways)):
        #in J1, the weight is simply the joint probability of this policy having
        # resulted in the pathway's series of decisions
        pw_joint_prob = 1.0

        for e in pathways[p].events:
            #get this policy's recommendation for this event
            policy_value = MDP.logistic(MDP.crossproduct(e.state, policy_vector))

            #check the action taken in the pathway,
            #  if it's suppress (action=True) the probability of THIS policy making the
            #    same decision is the actual policy_value calculated above
            #  otherwise, it's 1 - policy_value (the probability of choosing let-burn)

            if e.action:
                #action = True, i.e. suppress
                pw_joint_prob *= policy_value
            else:
                #action = False, i.e let-burn
                pw_joint_prob *= (1-policy_value)

        #the joint probability is fully-calculated, so add it to the list of weights
        weights[p] = pw_joint_prob


    #In J1, the objective function is the cross-product of the joint probabilities (weights) and net values
    obj_fn_val = MDP.crossproduct(values, weights)

    #and to make it comparabile to obj fn vals from sets with different pathway numbers, divide by
    #  the number of pathways
    obj_fn_val = obj_fn_val / len(pathways)

    if not SILENT:
        print("J1 Weights:")
        print(str(weights))

    if RETURN_WEIGHTS:
        return [obj_fn_val, weights]
    else:
        return obj_fn_val

def J2(policy_vector, pathways, FEATURE_NORMALIZATION=True, VALUE_NORMALIZATION=False, SILENT=True, RETURN_WEIGHTS=False):
    """Calculate the J2 objective function given a policy and a pathway set

    CALCULATION TYPE 
    -Using the pathway's average net_value per timestep
    -Normalizing features
    -Objective function value is divided by the number of pathways

    NOTES:
    -Assumes all pathways have the same number of timesteps as the first one in the list
    """

    timesteps = len(pathways[0].events)

    #Normalize Feature Values
    if FEATURE_NORMALIZATION:
        normalize_pathway_features(pathways)
    
    #Normalize Pathway Values
    if VALUE_NORMALIZATION:
        normalize_pathway_net_values(pathways)


    #Gather the pathway values
    values = [None] * len(pathways)
    for p in range(len(pathways)):
        #using value averaged over each timestep
        values[p] = (pathways[p].net_value / timesteps)

    
    #Calculate joint probabilities for each pathway
    weights = [None] * len(pathways)

    for p in range(len(pathways)):
        #in J2, the weight is the average of all the individual decision probabilities 
        # that this current policy gives for each event.

        probability_sum = 0.0

        for e in pathways[p].events:
            #get this policy's recommendation for this event
            policy_value = MDP.logistic(MDP.crossproduct(e.state, policy_vector))

            #check the action taken in the pathway,
            #  if it's suppress (action=True) the probability of THIS policy making the
            #    same decision is the actual policy_value calculated above
            #  otherwise, it's 1 - policy_value (the probability of choosing let-burn)

            if e.action:
                #action = True, i.e. suppress
                probability_sum += policy_value
            else:
                #action = False, i.e let-burn
                probability_sum += (1-policy_value)

        #the joint probability is fully-calculated, so add it to the list of weights
        weights[p] = probability_sum


    #In J2, the objective function is the cross-product of the average probabilities (weights) and net values
    obj_fn_val = MDP.crossproduct(values, weights)

    #and to make it comparabile to obj fn vals from sets with different pathway numbers, divide by
    #  the number of pathways
    obj_fn_val = obj_fn_val / len(pathways)

    if not SILENT:
        print("J2 Weights:")
        print(str(weights))

    if RETURN_WEIGHTS:
        return [obj_fn_val, weights]
    else:
        return obj_fn_val

def J3(policy_vector, pathways, FEATURE_NORMALIZATION=True, VALUE_NORMALIZATION=False, SILENT=True, RETURN_WEIGHTS=False):
    """Calculate the J3 objective function given a policy and a pathway set

    CALCULATION TYPE 
    -Using the pathway's average net_value per timestep
    -Normalizing features
    -Objective function value is divided by the number of pathways

    NOTES:
    -Assumes all pathways have the same number of timesteps as the first one in the list
    """

    timesteps = len(pathways[0].events)

    #Normalize Feature Values
    if FEATURE_NORMALIZATION:
        normalize_pathway_features(pathways)
    
    #Normalize Pathway Values
    if VALUE_NORMALIZATION:
        normalize_pathway_net_values(pathways)


    #Gather the pathway values
    values = [None] * len(pathways)
    for p in range(len(pathways)):
        #using value averaged over each timestep
        values[p] = (pathways[p].net_value / timesteps)

    
    #Calculate joint probabilities for each pathway
    weights = [None] * len(pathways)

    for p in range(len(pathways)):
        #in J3, the weight is  the joint probability of this policy having
        # resulted in the pathway's series of decisions, divided by the same value
        # that the ORIGINAL policy gave (the policy that was used when the pathway
        # was generated)

        pw_joint_prob = 1.0

        for e in pathways[p].events:
            #get this policy's recommendation for this event
            policy_value = MDP.logistic(MDP.crossproduct(e.state, policy_vector))

            #check the action taken in the pathway,
            #  if it's suppress (action=True) the probability of THIS policy making the
            #    same decision is the actual policy_value calculated above
            #  otherwise, it's 1 - policy_value (the probability of choosing let-burn)

            if e.action:
                #action = True, i.e. suppress
                pw_joint_prob *= policy_value
            else:
                #action = False, i.e let-burn
                pw_joint_prob *= (1.0-policy_value)

        #the joint probability is fully-calculated, so add it to the list of weights
        weights[p] = pw_joint_prob / pathways[p].generation_joint_prob


    #In J3, the objective function is the cross-product of the probabiltiy weights and net values
    obj_fn_val = MDP.crossproduct(values, weights)

    #and to make it comparabile to obj fn vals from sets with different pathway numbers, divide by
    #  the number of pathways
    obj_fn_val = obj_fn_val / len(pathways)


    if not SILENT:
        print("J3 Weights:")
        print(str(weights))
    
    if RETURN_WEIGHTS:
        return [obj_fn_val, weights]
    else:
        return obj_fn_val

def J4(policy_vector, pathways, FEATURE_NORMALIZATION=True, VALUE_NORMALIZATION=False, SILENT=True, RETURN_WEIGHTS=False):
    """Calculate the J4 objective function given a policy and a pathway set

    CALCULATION TYPE 
    -Using the pathway's average net_value per timestep
    -Normalizing features
    -Objective function value is divided by the number of pathways

    NOTES:
    -Assumes all pathways have the same number of timesteps as the first one in the list
    """

    timesteps = len(pathways[0].events)

    #Normalize Feature Values
    if FEATURE_NORMALIZATION:
        normalize_pathway_features(pathways)
    
    #Normalize Pathway Values
    if VALUE_NORMALIZATION:
        normalize_pathway_net_values(pathways)


    #Gather the pathway values
    values = [None] * len(pathways)
    for p in range(len(pathways)):
        #using value averaged over each timestep
        values[p] = (pathways[p].net_value / timesteps)

    
    #Calculate joint probabilities for each pathway
    J1_weights = [None] * len(pathways)
    weights = [None] * len(pathways)
    weight_sum = 0.0

    for p in range(len(pathways)):
        #in J4, the weight is the joint probability of this policy having
        # resulted in the pathway's series of decisions, divided by the sum of all
        # the joint probabilities, making it a zero-sum game

        pw_joint_prob = 1.0

        for e in pathways[p].events:
            #get this policy's recommendation for this event
            policy_value = MDP.logistic(MDP.crossproduct(e.state, policy_vector))

            #check the action taken in the pathway,
            #  if it's suppress (action=True) the probability of THIS policy making the
            #    same decision is the actual policy_value calculated above
            #  otherwise, it's 1 - policy_value (the probability of choosing let-burn)

            if e.action:
                #action = True, i.e. suppress
                pw_joint_prob *= policy_value
            else:
                #action = False, i.e let-burn
                pw_joint_prob *= (1-policy_value)

        #the joint probability is fully-calculated, so add it to the list of weights
        J1_weights[p] = pw_joint_prob
        weight_sum += pw_joint_prob

    #now use the J1 weights and the sum to calculate the J4 weights
    for p in range(len(pathways)):
        weights[p] = J1_weights[p] / weight_sum



    #In J3, the objective function is the cross-product of the probabiltiy weights and net values
    obj_fn_val = MDP.crossproduct(values, weights)

    #and to make it comparabile to obj fn vals from sets with different pathway numbers, divide by
    #  the number of pathways
    obj_fn_val = obj_fn_val / len(pathways)

    if not SILENT:
        print("J4 Weights:")
        print(str(weights))

    if RETURN_WEIGHTS:
        return [obj_fn_val, weights]
    else:
        return obj_fn_val

def J1prime(policy_vector, pathways, FEATURE_NORMALIZATION=True, VALUE_NORMALIZATION=False, SILENT=True):
    """
    Calculates the derivitive of the J1 objective function and returns it as a list

    Arguements
    policy_vector : a list of the parameters of the logistic policy function
    pathways : a list of MDP pathways on which to calculate
    FEATURE_NORMALIZATION : Boolean; Whether or not to re-scale feature values to fall between 1 and -1
    VALUE_NORMALIZATION : Boolean; Whether or not to re-scale pathway values to fall between 1 and -1
    SILENT: Boolean; whether to report results to standard out

    """
    vector_size = len(policy_vector)
    pathway_count = len(pathways)
    timesteps = len(pathways[0].events)

    # list to hold the final values, one per parameter
    d_obj_d_bk = [0.0] * vector_size

    #Normalize Feature Values
    if FEATURE_NORMALIZATION:
        normalize_pathway_features(pathways)

    #Normalize Pathway Values
    if VALUE_NORMALIZATION:
        normalize_pathway_net_values(pathways)

    #Gather the pathway values
    values = [None] * len(pathways)
    for p in range(len(pathways)):
        #using value averaged over each timestep
        values[p] = (pathways[p].net_value / timesteps)

    
    #Calculate joint probabilities for each pathway
    weights = [None] * len(pathways)

    for p in range(len(pathways)):
        #in J1, the weight is simply the joint probability of this policy having
        # resulted in the pathway's series of decisions
        pw_joint_prob = 1.0

        for e in pathways[p].events:
            #get this policy's recommendation for this event
            policy_value = MDP.logistic(MDP.crossproduct(e.state, policy_vector))

            #check the action taken in the pathway,
            #  if it's suppress (action=True) the probability of THIS policy making the
            #    same decision is the actual policy_value calculated above
            #  otherwise, it's 1 - policy_value (the probability of choosing let-burn)

            if e.action:
                #action = True, i.e. suppress
                pw_joint_prob *= policy_value
            else:
                #action = False, i.e let-burn
                pw_joint_prob *= (1-policy_value)

        #the joint probability is fully-calculated, so add it to the list of weights
        weights[p] = pw_joint_prob
            

    #creating and MDP_Policy object to do calculations
    MDPpol = MDP.MDP_Policy(vector_size)
    MDPpol.set_params(policy_vector)

    #variables for use within the loop
    prob = 1.0
    delta_prob = 0.0
    sum_delta_prob = 0.0
    sum_delta_prob_AVE = 0.0
    action = False
    sup = 0
    
    #iterate over each beta and evaluate the gradient along it
    for beta in range(vector_size):

        for pw in range(pathway_count):

            #reset value for this pathway
            sum_delta_prob = 0.0
            sum_delta_prob_AVE = 0.0

            for evnt in range(timesteps):

                #get the action choice of this pathway at this event
                action = pathways[pw].events[evnt].action

                #get the probability of actually making this action in this event
                prob = pathways[pw].events[evnt].decision_prob

                #checking for unreasonably small probabilities
                if prob == 0: prob = 0.00001
      
                #get the feature of this pathway and this event for this beta
                flik = pathways[pw].events[evnt].state[beta]

                #calculate THIS policy's probability of suppressing at THIS state of THIS pathway
                prob = MDPpol.calc_prob(pathways[pw].events[evnt].state)
                        
                delta_lgstc = flik * prob * (1-prob)

                delta_prob = delta_lgstc
                if action:
                    #action was taken, so leave it alone
                    pass
                else:
                    #action was not taken, so:
                    delta_prob *= -1

                sum_delta_prob += delta_prob / prob #J1

            
            #finished adding up sum_delta_prob for all the ignitions in this pathway, so
            # calculate the d/dx value:
            
            
            # #check which weights to use, and to the derivative appropriately
            # if self.IMPORTANCE_SAMPLING:
            #     #using J3 Importance Sampling Weights
            #     d_obj_d_bk[beta] += self.pathway_set[pw].net_value * self.pathway_weights_importance_sampling[pw] * sum_delta_prob
            # elif self.NORMALIZED_WEIGHTS_F_PRIME:
            #     #using normalized weights inside the derivative calculation (J1.1)
            #     d_obj_d_bk[beta] += self.pathway_set[pw].net_value * self.pathway_weights_normalized[pw] * sum_delta_prob
            # else:
            #     if self.AVERAGED_WEIGHTS_F_PRIME:
            #         #doing average-weight calculation (J2)
            #         invI = (1.0 / len(self.pathway_set[pw].events) )
            #         d_obj_d_bk[beta] += self.pathway_set[pw].net_value * invI * sum_delta_prob_AVE                            
            #     else:
            #         #using standard joint-probability math (J1.0)
            #         d_obj_d_bk[beta] += self.pathway_set[pw].net_value * self.pathway_weights[pw] * sum_delta_prob
            d_obj_d_bk[beta] += pathways[pw].net_value * weights[pw] * sum_delta_prob
                

            #going on to the next pathway

        #going on to the next beta

    #finished with all betas
    
    # And Finally, return the list
    return scipy.array(d_obj_d_bk)

def J3prime(policy_vector, pathways, FEATURE_NORMALIZATION=True, VALUE_NORMALIZATION=False, SILENT=True):
    """
    Calculates the derivitive of the J3 objective function and returns it as a list

    Arguements
    policy_vector : a list of the parameters of the logistic policy function
    pathways : a list of MDP pathways on which to calculate
    FEATURE_NORMALIZATION : Boolean; Whether or not to re-scale feature values to fall between 1 and -1
    VALUE_NORMALIZATION : Boolean; Whether or not to re-scale pathway values to fall between 1 and -1
    SILENT: Boolean; whether to report results to standard out

    """
    vector_size = len(policy_vector)
    pathway_count = len(pathways)
    timesteps = len(pathways[0].events)

    # list to hold the final values, one per parameter
    d_obj_d_bk = [0.0] * vector_size

    #Normalize Feature Values
    if FEATURE_NORMALIZATION:
        normalize_pathway_features(pathways)

    #Normalize Pathway Values
    if VALUE_NORMALIZATION:
        normalize_pathway_net_values(pathways)

    #Gather the pathway values
    values = [None] * len(pathways)
    for p in range(len(pathways)):
        #using value averaged over each timestep
        values[p] = (pathways[p].net_value / timesteps)

    
    #Calculate joint probabilities for each pathway
    weights = [None] * len(pathways)

    for p in range(len(pathways)):
        #in J3, the weight is  the joint probability of this policy having
        # resulted in the pathway's series of decisions, divided by the same value
        # that the ORIGINAL policy gave (the policy that was used when the pathway
        # was generated)

        pw_joint_prob = 1.0

        for e in pathways[p].events:
            #get this policy's recommendation for this event
            policy_value = MDP.logistic(MDP.crossproduct(e.state, policy_vector))

            #check the action taken in the pathway,
            #  if it's suppress (action=True) the probability of THIS policy making the
            #    same decision is the actual policy_value calculated above
            #  otherwise, it's 1 - policy_value (the probability of choosing let-burn)

            if e.action:
                #action = True, i.e. suppress
                pw_joint_prob *= policy_value
            else:
                #action = False, i.e let-burn
                pw_joint_prob *= (1.0-policy_value)

        #the joint probability is fully-calculated, so add it to the list of weights
        weights[p] = pw_joint_prob / pathways[p].generation_joint_prob
            

    #creating and MDP_Policy object to do calculations
    MDPpol = MDP.MDP_Policy(vector_size)
    MDPpol.set_params(policy_vector)

    #variables for use within the loop
    prob = 1.0
    delta_prob = 0.0
    sum_delta_prob = 0.0
    sum_delta_prob_AVE = 0.0
    action = False
    sup = 0
    
    #iterate over each beta and evaluate the gradient along it
    for beta in range(vector_size):

        for pw in range(pathway_count):

            #reset value for this pathway
            sum_delta_prob = 0.0
            sum_delta_prob_AVE = 0.0

            for evnt in range(timesteps):

                #get the action choice of this pathway at this event
                action = pathways[pw].events[evnt].action

                #get the probability of actually making this action in this event
                prob = pathways[pw].events[evnt].decision_prob

                #checking for unreasonably small probabilities
                if prob == 0: prob = 0.00001
      
                #get the feature of this pathway and this event for this beta
                flik = pathways[pw].events[evnt].state[beta]

                #calculate THIS policy's probability of suppressing at THIS state of THIS pathway
                prob = MDPpol.calc_prob(pathways[pw].events[evnt].state)
                        
                delta_lgstc = flik * prob * (1-prob)

                delta_prob = delta_lgstc
                if action:
                    #action was taken, so leave it alone
                    pass
                else:
                    #action was not taken, so:
                    delta_prob *= -1

                sum_delta_prob += delta_prob / prob #J1

            
            #finished adding up sum_delta_prob for all the ignitions in this pathway, so
            # calculate the d/dx value:
            
            d_obj_d_bk[beta] += pathways[pw].net_value * weights[pw] * sum_delta_prob
                

            #going on to the next pathway

        #going on to the next beta

    #finished with all betas

    # And Finally, return the list
    return scipy.array(d_obj_d_bk)

def J4primeWARNING(policy_vector, pathways, FEATURE_NORMALIZATION=True, VALUE_NORMALIZATION=False, SILENT=True):
    """
    WARNING: This function is not correct yet... The derivative needs to be recalculated for J4


    Calculates the derivitive of the J4 objective function and returns it as a list

    Arguements
    policy_vector : a list of the parameters of the logistic policy function
    pathways : a list of MDP pathways on which to calculate
    FEATURE_NORMALIZATION : Boolean; Whether or not to re-scale feature values to fall between 1 and -1
    VALUE_NORMALIZATION : Boolean; Whether or not to re-scale pathway values to fall between 1 and -1
    SILENT: Boolean; whether to report results to standard out

    """
    vector_size = len(policy_vector)
    pathway_count = len(pathways)
    timesteps = len(pathways[0].events)

    # list to hold the final values, one per parameter
    d_obj_d_bk = [0.0] * vector_size

    #Normalize Feature Values
    if FEATURE_NORMALIZATION:
        normalize_pathway_features(pathways)

    #Normalize Pathway Values
    if VALUE_NORMALIZATION:
        normalize_pathway_net_values(pathways)

    #Gather the pathway values
    values = [None] * len(pathways)
    for p in range(len(pathways)):
        #using value averaged over each timestep
        values[p] = (pathways[p].net_value / timesteps)

    
    #Calculate joint probabilities for each pathway
    J1_weights = [None] * len(pathways)
    weights = [None] * len(pathways)
    weight_sum = 0.0

    for p in range(len(pathways)):
        #in J4, the weight is the joint probability of this policy having
        # resulted in the pathway's series of decisions, divided by the sum of all
        # the joint probabilities, making it a zero-sum game

        pw_joint_prob = 1.0

        for e in pathways[p].events:
            #get this policy's recommendation for this event
            policy_value = MDP.logistic(MDP.crossproduct(e.state, policy_vector))

            #check the action taken in the pathway,
            #  if it's suppress (action=True) the probability of THIS policy making the
            #    same decision is the actual policy_value calculated above
            #  otherwise, it's 1 - policy_value (the probability of choosing let-burn)

            if e.action:
                #action = True, i.e. suppress
                pw_joint_prob *= policy_value
            else:
                #action = False, i.e let-burn
                pw_joint_prob *= (1-policy_value)

        #the joint probability is fully-calculated, so add it to the list of weights
        J1_weights[p] = pw_joint_prob
        weight_sum += pw_joint_prob

    #now use the J1 weights and the sum to calculate the J4 weights
    for p in range(len(pathways)):
        weights[p] = J1_weights[p] / weight_sum
            

    #creating and MDP_Policy object to do calculations
    MDPpol = MDP.MDP_Policy(vector_size)
    MDPpol.set_params(policy_vector)

    #variables for use within the loop
    prob = 1.0
    delta_prob = 0.0
    sum_delta_prob = 0.0
    sum_delta_prob_AVE = 0.0
    action = False
    sup = 0
    
    #iterate over each beta and evaluate the gradient along it
    for beta in range(vector_size):

        for pw in range(pathway_count):

            #reset value for this pathway
            sum_delta_prob = 0.0
            sum_delta_prob_AVE = 0.0

            for evnt in range(timesteps):

                #get the action choice of this pathway at this event
                action = pathways[pw].events[evnt].action

                #get the probability of actually making this action in this event
                prob = pathways[pw].events[evnt].decision_prob

                #checking for unreasonably small probabilities
                if prob == 0: prob = 0.00001
      
                #get the feature of this pathway and this event for this beta
                flik = pathways[pw].events[evnt].state[beta]

                #calculate THIS policy's probability of suppressing at THIS state of THIS pathway
                prob = MDPpol.calc_prob(pathways.events[evnt].state)
                        
                delta_lgstc = flik * prob * (1-prob)

                delta_prob = delta_lgstc
                if action:
                    #action was taken, so leave it alone
                    pass
                else:
                    #action was not taken, so:
                    delta_prob *= -1

                sum_delta_prob += delta_prob / prob #J1 / J3

            
            #finished adding up sum_delta_prob for all the ignitions in this pathway, so
            # calculate the d/dx value:
                        
            d_obj_d_bk[beta] += pathways[pw].net_value * weights[pw] * sum_delta_prob


            #going on to the next pathway

        #going on to the next beta

    #finished with all betas

    # And Finally, return the list
    return scipy.array(d_obj_d_bk)



