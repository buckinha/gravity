
import SWMv2_1 as SWM
import numpy as np
from sphere_dist import sphere_dist
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


training_set_size=500
test_set_size = 2000
RFC_depth = 2
RFC_estimators = 20



#create training set

raw = [SWM.simulate(timesteps=200, policy=sphere_dist([0,0,0,0,0,0],25,random_seed=i), random_seed=i, SILENT=True) for i in range(training_set_size)]

clean = [ [raw[i]["Suppression Rate"]] + raw[i]["Generation Policy"]  for i in range(len(raw))]

#add columns for boolean SA and boolean LB
#                 Supp=True            LB=True      
training_set = [ [clean[i][0]>=0.995,  clean[i][0]<=0.005] + clean[i]    for i in range(len(clean))]

#and finally. make it into a numpy array
training_set = np.array(training_set)





#instantiate random forest classfiers
RFC_SA = RandomForestClassifier(max_depth=RFC_depth, n_estimators=RFC_estimators)
RFC_LB = RandomForestClassifier(max_depth=RFC_depth, n_estimators=RFC_estimators)

#train one RFC on SA
RFC_SA.fit(training_set[0::,3::], training_set[0::,0])

#train another RFC on LB
RFC_LB.fit(training_set[0::,3::], training_set[0::,1])



#create testing set of policies
test_pols = [sphere_dist([0,0,0,0,0,0],25,random_seed=i) for i in range(test_set_size)]

#make predictions on the testing policies
output_SA = RFC_SA.predict(test_pols)
output_LB = RFC_LB.predict(test_pols)


#run the simulations of the testing policies
test_sims = [SWM.simulate(timesteps=200, policy=test_pols[i], random_seed=i+9999, SILENT=True) for i in range(test_set_size)] 


#show accuracy over all testing simulations
results = [  [ test_sims[i]["Suppression Rate"], output_SA[i], output_LB[i] ] + test_pols[i] for i in range(test_set_size) ]
results = np.array(results)

res_sim_SA = np.array(filter(lambda x: x[0]>=0.995, results))
res_sim_LB = np.array(filter(lambda x: x[0]<=0.005, results))
res_sim_mid = np.array(filter(lambda x: (x[0]>=0.005 and x[0]<=0.995), results))





#saving to disk
if False:
    from sklearn.externals import joblib
    joblib.dump(RFC_SA, 'RFC_models/SWMv2_1_RFC_SA.pkl') 
    joblib.dump(RFC_LB, 'RFC_models/SWMv2_1_RFC_LB.pkl') 

#Later you can load back the pickled model (possibly in another Python process) with:
#RFC_SA2 = joblib.load('RFC_models/SWMv2_1_RFC_SA.pkl') 
#RFC_LB2 = joblib.load('RFC_models/SWMv2_1_RFC_LB.pkl') 









#graphing


size_1 = 25
size_2 = 45



#make bigger circles when a central sim was classified as EITHER LB or SA
mid_area = map(lambda x,y: size_1 + size_2*max(x,y), res_sim_mid[0::,1], res_sim_mid[0::,2])
#make a bigger circle when an SA sim was NOT classified as SA
sa_area = map(lambda x: size_1 + size_2*(1-x), res_sim_SA[0::,1])
#make a bigger circle when an LB sim was NOT classified as LB
lb_area = map(lambda x: size_1 + size_2*(1-x), res_sim_LB[0::,2])



#plt.scatter(x, y, s=area, c=colors, alpha=0.5)
#Plotting heat vs constant
plt.scatter(res_sim_mid[0::,4],res_sim_mid[0::,3], s=mid_area, c='green', alpha=0.5)
plt.scatter(res_sim_SA[0::,4],res_sim_SA[0::,3], s=mid_area, c='red', alpha=0.5)
plt.scatter(res_sim_LB[0::,4],res_sim_LB[0::,3], s=mid_area, c='blue', alpha=0.5)

plt.title("RFC. of " + str(test_set_size) + " SWM-6D Tests (projected in 2D); Trained on " + str(training_set_size) + " sims.")

plt.xlabel("Policy Parameter on Heat")
plt.ylabel("Policy Constant")

plt.show()






#Plotting heat vs humidity
plt.scatter(res_sim_mid[0::,4],res_sim_mid[0::,5], s=mid_area, c='green', alpha=0.5)
plt.scatter(res_sim_SA[0::,4],res_sim_SA[0::,5], s=mid_area, c='red', alpha=0.5)
plt.scatter(res_sim_LB[0::,4],res_sim_LB[0::,5], s=mid_area, c='blue', alpha=0.5)

plt.title("RFC. of " + str(test_set_size) + " SWM-6D Tests (projected in 2D); Trained on " + str(training_set_size) + " sims.")

plt.xlabel("Policy Parameter on Heat")
plt.ylabel("Policy Parameter on Humidity")

plt.show()





#Plotting heat vs timber
plt.scatter(res_sim_mid[0::,4],res_sim_mid[0::,6], s=mid_area, c='green', alpha=0.5)
plt.scatter(res_sim_SA[0::,4],res_sim_SA[0::,6], s=mid_area, c='red', alpha=0.5)
plt.scatter(res_sim_LB[0::,4],res_sim_LB[0::,6], s=mid_area, c='blue', alpha=0.5)

plt.title("RFC. of " + str(test_set_size) + " SWM-6D Tests (projected in 2D); Trained on " + str(training_set_size) + " sims.")

plt.xlabel("Policy Parameter on Heat")
plt.ylabel("Policy Parameter on Timber")

plt.show()





#Plotting heat vs vulnerability
plt.scatter(res_sim_mid[0::,4],res_sim_mid[0::,7], s=mid_area, c='green', alpha=0.5)
plt.scatter(res_sim_SA[0::,4],res_sim_SA[0::,7], s=mid_area, c='red', alpha=0.5)
plt.scatter(res_sim_LB[0::,4],res_sim_LB[0::,7], s=mid_area, c='blue', alpha=0.5)

plt.title("RFC. of " + str(test_set_size) + " SWM-6D Tests (projected in 2D); Trained on " + str(training_set_size) + " sims.")

plt.xlabel("Policy Parameter on Heat")
plt.ylabel("Policy Parameter on Vulnerabilty")

plt.show()





#Plotting humidity vs vulnerability
plt.scatter(res_sim_mid[0::,5],res_sim_mid[0::,7], s=mid_area, c='green', alpha=0.5)
plt.scatter(res_sim_SA[0::,5],res_sim_SA[0::,7], s=mid_area, c='red', alpha=0.5)
plt.scatter(res_sim_LB[0::,5],res_sim_LB[0::,7], s=mid_area, c='blue', alpha=0.5)

plt.title("RFC. of " + str(test_set_size) + " SWM-6D Tests (projected in 2D); Trained on " + str(training_set_size) + " sims.")

plt.xlabel("Policy Parameter on Humidity")
plt.ylabel("Policy Parameter on Vulnerabilty")

plt.show()






#Plotting timber vs constant
plt.scatter(res_sim_mid[0::,6],res_sim_mid[0::,3], s=mid_area, c='green', alpha=0.5)
plt.scatter(res_sim_SA[0::,6],res_sim_SA[0::,3], s=mid_area, c='red', alpha=0.5)
plt.scatter(res_sim_LB[0::,6],res_sim_LB[0::,3], s=mid_area, c='blue', alpha=0.5)

plt.title("RFC. of " + str(test_set_size) + " SWM-6D Tests (projected in 2D); Trained on " + str(training_set_size) + " sims.")

plt.xlabel("Policy Parameter on Timber")
plt.ylabel("Policy Constant")

plt.show()





#Plotting humidity vs constant
plt.scatter(res_sim_mid[0::,5],res_sim_mid[0::,3], s=mid_area, c='green', alpha=0.5)
plt.scatter(res_sim_SA[0::,5],res_sim_SA[0::,3], s=mid_area, c='red', alpha=0.5)
plt.scatter(res_sim_LB[0::,5],res_sim_LB[0::,3], s=mid_area, c='blue', alpha=0.5)

plt.title("RFC. of " + str(test_set_size) + " SWM-6D Tests (projected in 2D); Trained on " + str(training_set_size) + " sims.")

plt.xlabel("Policy Parameter on Humidity")
plt.ylabel("Policy Constant")

plt.show()