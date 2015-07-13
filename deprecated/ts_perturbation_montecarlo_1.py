#This script can check either J1.1 or J2.
#It will find an optimal policy, and then do N rollouts of 22 perturbed policies, and N rollouts from the optimal policy,
#  and compare the actual pathway net values at the end of each rollout (rather than the objective function values

import random, math
from FireGirlOptimizer import *
FGPO = FireGirlPolicyOptimizer()

pathway_count = 200
ignition_count = 100
perturb_percent = 0.5 #this is the percent increase or decrease that will be applied to each 
#                       perturbed parameter in the perturbed polcies
new_paths = 100       #the number of new pathways to generate under each policy

### STEP 1 ######################################################
#   Generate a set of pathways (100s) using a coin-toss policy and find an optimal policy

print("Creating original pathways with coin-toss policy")
FGPO.createFireGirlPathways(pathway_count,ignition_count)
FGPO.normalizeAllFeatures()

#setting flags
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = True
FGPO.NORMALIZED_WEIGHTS_F_PRIME = True
FGPO.AVERAGED_WEIGHTS_OBJ_FN = False
FGPO.AVERAGED_WEIGHTS_F_PRIME = False


print("Beginning Policy Optimzation")
results = FGPO.optimizePolicy()
print("Policy Optimization Complete: Parameters are:")
print(str(FGPO.Policy.getParams()))


### STEP 2 ######################################################
#   Do 100 pathways using the optimized policy and take the average of their net values

print(" ")
print("Beginning Monte Carlo pathway generation for the optimzed policy.")
total_val = 0
optimal_pol_average = 0
    
for i in range(new_paths):
    #generate a new pathway with this policy
    #NOTE: this will erase the original set used in step 1
    #NOTE: passing in "i" as the ID number ensures that each pathway starts from a different landscape
    #      otherwise, they'll always be exactly the same as each other.
    #NOTE: leaving the optional argument, Policy, off, lets the optimizer use its current policy, which
    #      is always saved at the end of an optimization function call. So this function will use the 
    #      optimal policy we generated in step 1
    FGPO.createFireGirlPathways(1,ignition_count, i)
    
    #add the net_value of this landscape to the total for this perturbed policy
    total_val += FGPO.pathway_set[0].net_value  #I'm just accessing the member directly

optimal_pol_average = total_val / new_paths


### STEP 3 ######################################################
#   Create a large set of perturbations of that optimal policy

#copy the values of the optimal policy
pol_optim = []
for i in range(11):
    #copying the data from the optimzied policy (and doing + - 1 to ensure the VALUE is saved, not a reference)
    pol_optim.append(FGPO.Policy.b[i] + 1.0 - 1.0)

perturbed_policies = []
for i in range(11):
    #make new copies of the optimal policy
    pol_copy1 = []
    pol_copy2 = []
    for j in range(11):
        #copying the values of the optimal policy to the perturbed policies.
        # again, the + - 1 is to ensure no pythony references get used. I need to be absolutely sure
        # that none of the later policies overwrite the changes in previous policies, etc...
        pol_copy1.append(pol_optim[j] + 1.0 - 1.0)
        pol_copy2.append(pol_optim[j] + 1.0 - 1.0)
    
    #perturb the copied policies at index i
    pol_copy1[i] *= (1 + perturb_percent)  # increase by this percent
    pol_copy2[i] *= (1 - perturb_percent)  # decrease by this percent
    
    #add the perturbed copies to the list
    perturbed_policies.append(pol_copy1)
    perturbed_policies.append(pol_copy2)
    
#print(str(perturbed_policies))


### STEP 4 ######################################################
#   For each perturbed policy do a MonteCarlo evaluation of the net values produced when generating NEW pathways under that policy

#For each perturbed policy, we want to generate 100 new pathways, and take the average of their net values
#  rather than any of the objective function values

pert_pols_average_values = []
new_policy = FireGirlPolicy()

for pert_pol in perturbed_policies:
    print("Beginning Monte Carlo pathway generation for perturbed policy " + str(perturbed_policies.index(pert_pol)))
    #variable to hold the sum of all the net_values from the pathways generated under this perturbed policy
    total_val = 0
    
    for i in range(new_paths):
        #generate a new pathway with this policy
        #NOTE: this will erase the original set used in step 1
        #NOTE: passing in "i" as the ID number ensures that each pathway starts from a different landscape
        #      otherwise, they'll always be exactly the same as each other.
        #NOTE: adding 2000 to the ID number so that the new rollouts are NOT the same pathways that generated
        #      the policy. Otherwise, overfit will be an advantage.
        new_policy.setParams(pert_pol)
        FGPO.createFireGirlPathways(1,ignition_count, 2000 + i, new_policy)
        
        #add the net_value of this landscape to the total for this perturbed policy
        FGPO.pathway_set[0].updateNetValue()
        total_val += FGPO.pathway_set[0].net_value  #I'm just accessing the member directly
    
    pert_pols_average_values.append(total_val / new_paths)

    
### STEP 5 ######################################################
#   Compare the average net values of each perturbed policy to those generated under the optimal policy

print(" ")
print("Average value of pathways generated under the optimal policy:")
print("(higher values are better)")
print(str(optimal_pol_average))
print("Average value of pathways generated under perturbed policies:")
print("Value           Delta")
for i in range(len(pert_pols_average_values)):
    print(str(pert_pols_average_values[i]) + "  " + str( pert_pols_average_values[i] - optimal_pol_average ) ), #comma to tell python not to end the line
    if pert_pols_average_values[i] > optimal_pol_average:
        print(" <-- IMPROVEMENT")
    else:
        print(" ") #to end the line
print(" ")
print("Policy Generated by L-BFGS-B:")
print(str(pol_optim))