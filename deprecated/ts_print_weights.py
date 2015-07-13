import random, math
from FireGirlOptimizer import *
FGPO = FireGirlPolicyOptimizer()

pathway_count = 10
ignition_count = 50
perturb_percent = 0.5 #this is the percent increase or decrease that will be applied to each 
#                       perturbed parameter in the perturbed polcies
new_paths = pathway_count       #the number of new pathways to generate under each policy



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
print(str(FGPO.Policy.b))

FGPO.calcPathwayWeights()

print("J1 weights")
print(str(FGPO.pathway_weights))
print("J1.1 weights")
print(str(FGPO.pathway_weights_normalized))
print("J2 weights")
print(str(FGPO.pathway_weights_averaged))


#OUTPUT
# J1 weights
# [1.01925701e-16, 6.9073255e-19, 1.09303907e-17, 4.0224007e-18, 3.6003607e-20, 3.4877044e-18, 6.1468839e-19, 8.05808255e-19, 7.4057867e-18, 2.08952108e-14]
# J1.1 weights
# [0.0048478035, 3.28527124e-05, 0.00051987268, 0.000191313951, 1.71240831e-06, 0.000165882655, 2.9235890e-05, 3.8325958e-05, 0.00035223500, 0.99382076]
# J2 weights
# [0.53386453, 0.488437121, 0.50994820, 0.495848486, 0.46861579, 0.49723926, 0.48370738, 0.480546396, 0.50017182, 0.58428294]