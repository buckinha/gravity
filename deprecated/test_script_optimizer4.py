from FireGirlOptimizer import *
FGPO = FireGirlPolicyOptimizer()

###To create, uncomment the following two lines:
FGPO.createFireGirlPathways(10,50)
FGPO.normalizeAllFeatures()

#FGPO.saveFireGirlPathways("FG_pathways_20x50.fgl")

###To load (already created data), uncomment the following line
#FGPO.loadFireGirlPathways("FG_pathways_20x50.fgl")


FGPO.Policy.setParams([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
FGPO.calcPathwayWeights()
print(" ")
print("Pathway Net Values: ")
print(str(FGPO.pathway_net_values))
print("Total-Prob Weights")
print(str(FGPO.pathway_weights))
print("Normalized Weights")
print(str(FGPO.pathway_weights_normalized))
print("Averaged Weights")
print(str(FGPO.pathway_weights_averaged))

#Setting Flags for J1
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = False
FGPO.NORMALIZED_WEIGHTS_F_PRIME = False
FGPO.AVERAGED_WEIGHTS_OBJ_FN = False
FGPO.AVERAGED_WEIGHTS_F_PRIME = False
FGPO.Policy.setParams([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
print(" ")
print("DETAILS for J1")
print("objfn: " + str(FGPO.calcObjFn()))
print("fprme: " + str(FGPO.calcObjFPrime()))
print("Beginning Optimization Routine J1")
#FGPO.resetPolicy()
output=FGPO.optimizePolicy()
FGPO.printOptOutput(output)


#Setting Flags for J1.1a (obj fn AND derivative)
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = True
FGPO.NORMALIZED_WEIGHTS_F_PRIME = True
FGPO.AVERAGED_WEIGHTS_OBJ_FN = False
FGPO.AVERAGED_WEIGHTS_F_PRIME = False
FGPO.Policy.setParams([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
print(" ")
print("DETAILS for J1.1a - normalizing BOTH functions")
print("objfn: " + str(FGPO.calcObjFn()))
print("fprme: " + str(FGPO.calcObjFPrime()))
print("Beginning Optimization Routine J1.1a")
#FGPO.resetPolicy()
output=FGPO.optimizePolicy()
FGPO.printOptOutput(output)
#for reference, printing the J1.0 obj fn value
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = False
FGPO.NORMALIZED_WEIGHTS_F_PRIME = False
FGPO.AVERAGED_WEIGHTS_OBJ_FN = False
FGPO.AVERAGED_WEIGHTS_F_PRIME = False
print("obj fn under J1.0: " + str(FGPO.calcObjFn()))
#for reference, printing the J2 obj fn value
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = False
FGPO.NORMALIZED_WEIGHTS_F_PRIME = False
FGPO.AVERAGED_WEIGHTS_OBJ_FN = True
FGPO.AVERAGED_WEIGHTS_F_PRIME = True
print("obj fn under J2: " + str(FGPO.calcObjFn()))

#Setting Flags for J1.1b (derivative only)
# FGPO.NORMALIZED_WEIGHTS_OBJ_FN = False
# FGPO.NORMALIZED_WEIGHTS_F_PRIME = True
# FGPO.AVERAGED_WEIGHTS_OBJ_FN = False
# FGPO.AVERAGED_WEIGHTS_F_PRIME = False
# FGPO.Policy.setParams([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
# print(" ")
# print("DETAILS for J1.1b - normalizing ONLY the derivative")
# print("objfn: " + str(FGPO.calcObjFn()))
# print("fprme: " + str(FGPO.calcObjFPrime()))
# print("Beginning Optimization Routine J1.1b")
# #FGPO.resetPolicy()
# output=FGPO.optimizePolicy()
# FGPO.printOptOutput(output)

#Setting Flags for J2 (averaged weights for BOTH)
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = False
FGPO.NORMALIZED_WEIGHTS_F_PRIME = False
FGPO.AVERAGED_WEIGHTS_OBJ_FN = True
FGPO.AVERAGED_WEIGHTS_F_PRIME = True
FGPO.Policy.setParams([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
print(" ")
print("DETAILS for J2")
print("objfn: " + str(FGPO.calcObjFn()))
print("fprme: " + str(FGPO.calcObjFPrime()))
print("Beginning Optimization Routine J2")
#FGPO.resetPolicy()
output=FGPO.optimizePolicy()
FGPO.printOptOutput(output)
#for reference, printing the J1.0 obj fn value
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = False
FGPO.NORMALIZED_WEIGHTS_F_PRIME = False
FGPO.AVERAGED_WEIGHTS_OBJ_FN = False
FGPO.AVERAGED_WEIGHTS_F_PRIME = False
print("obj fn under J1.0: " + str(FGPO.calcObjFn()))
#for reference, printing the J1.1a obj fn value
FGPO.NORMALIZED_WEIGHTS_OBJ_FN = True
FGPO.NORMALIZED_WEIGHTS_F_PRIME = True
FGPO.AVERAGED_WEIGHTS_OBJ_FN = False
FGPO.AVERAGED_WEIGHTS_F_PRIME = False
print("obj fn under J1.1a: " + str(FGPO.calcObjFn()))


