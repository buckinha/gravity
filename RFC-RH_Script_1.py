import SWMv2_1 as SWM
import numpy as np
from sphere_dist import sphere_dist
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from SWMv2_1_RH import reg_heur


training_set_size=500
test_set_size = 2000
RFC_depth = 2
RFC_estimators = 20
RH_climbs = 20


#######################
# Create Starting Set #
#######################
raw = [SWM.simulate(timesteps=200, policy=sphere_dist([0,0,0,0,0,0],25,random_seed=i), random_seed=i, SILENT=True) for i in range(training_set_size)]

clean = [ [raw[i]["Suppression Rate"]] + raw[i]["Generation Policy"] + [raw[i]["Average State Value"]] for i in range(len(raw))]

#add columns for boolean SA and boolean LB
#                 Supp=True            LB=True 
training_set = [ [clean[i][0]>=0.995,  clean[i][0]<=0.005] + clean[i]    for i in range(len(clean))]

#and finally. make it into a numpy array
training_set = np.array(training_set)




########################
#   Instantiate RFCs   #
########################
RFC_SA = RandomForestClassifier(max_depth=RFC_depth, n_estimators=RFC_estimators)
RFC_LB = RandomForestClassifier(max_depth=RFC_depth, n_estimators=RFC_estimators)



########################
#      Train RFCs      #
########################
#training one RFC on SA
RFC_SA.fit(training_set[0::,3:9], training_set[0::,0])

#training another RFC on LB
RFC_LB.fit(training_set[0::,3:9], training_set[0::,1])

#if desired: saving to disk
if False:
    from sklearn.externals import joblib
    joblib.dump(RFC_SA, 'RFC_models/SWMv2_1_RFC_SA.pkl') 
    joblib.dump(RFC_LB, 'RFC_models/SWMv2_1_RFC_LB.pkl') 

#Later you can load back the pickled model (possibly in another Python process) with:
#RFC_SA2 = joblib.load('RFC_models/SWMv2_1_RFC_SA.pkl') 
#RFC_LB2 = joblib.load('RFC_models/SWMv2_1_RFC_LB.pkl') 





################################################
# Filter the Starting Set to Non-edge Policies #
################################################

# x[0] and x[1] are the 0/1 flags for LB or SA, so if they sum to zero, this is a middle policy
non_edge = filter(lambda x: x[0] + x[1] == 0, training_set)




############################################
# Sort the Remaining Starting Set by Value #
############################################

#the last value of each row, x[-1], is the average state value
non_edge_sorted = sorted(non_edge, key=lambda x: x[-1], reverse=True)





######################################################
# Starting at the Highest Valued Sim, Run reg_heur() #
######################################################


#function handle is
#reg_heur(pol_0, sampling_radius=0.1, pw_count=500, years=200, minimum_pathways=100, alpha_step=0.1, p_val_limit=0.10, max_steps=15,PRINT_R_PLOTTING=False, SILENT=True, random_seed=0)

climb_results = [ reg_heur(non_edge_sorted[i][3:9], random_seed=i)    for i in range(6)]

#[climb_results[i]["Values"] for i in range(len(climb_results))]
#[climb_results[i]["Path"][-1] for i in range(len(climb_results))]