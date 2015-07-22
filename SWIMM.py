"""A Simple Wildfire-inspired MDP model"""

import random, math, numpy

def simulate(timesteps, policy=[0,0,0], random_seed=0, model_parameters={}, SILENT=False):
    
    random.seed(random_seed)
    
    constant_reward = 2
    if "Constant Reward" in model_parameters.keys(): constant_reward = model_parameters["Constant Reward"]

    #range of the randomly drawn, uniformally distributed "event"
    #this is the only so-called state "feature" in this MDP and
    #is comparable to a wildfire
    event_max = 1.0
    event_min = 0.0
    
    timesteps = int(timesteps)


    #cost of suppression in a mild event
    supp_cost_mild = 1
    if "Suppression Cost - Mild Event" in model_parameters.keys(): supp_cost_mild = model_parameters["Suppression Cost - Mild Event"]

    #cost of suppresion in a severe event
    supp_cost_severe = 4
    if "Suppression Cost - Severe Event" in model_parameters.keys(): supp_cost_severe = model_parameters["Suppression Cost - Severe Event"]

    #cost of a severe fire on the next timestep
    burn_cost = 6
    if "Severe Burn Cost" in model_parameters.keys(): burn_cost = model_parameters["Severe Burn Cost"]


    threshold_suppression = 0.8
    threshold_mild = 0.8
    threshold_severe = 0.8
    if "Threshold After Suppression" in model_parameters.keys(): threshold_suppression = model_parameters["Threshold After Suppression"]
    if "Threshold After Mild Event" in model_parameters.keys(): threshold_mild = model_parameters["Threshold After Mild Event"]
    if "Threshold After Severe Event" in model_parameters.keys(): threshold_severe = model_parameters["Threshold After Severe Event"]

    PROBABALISTIC_CHOICES = True
    if "Probabalistic Choices" in model_parameters.keys(): PROBABALISTIC_CHOICES = model_parameters["Probabalistic Choices"]


    #setting 'enums'
    POST_SUPPRESSION = 0
    POST_MILD = 1
    POST_SEVERE = 2

    MILD=0
    SEVERE=1



    #starting simulations
    states = [None] * timesteps

    #start current condition randomly among the three states
    current_condition = random.randint(0,2)

    for i in range(timesteps):

        #event value is the single "feature" of events in this MDP
        ev = random.uniform(event_min, event_max)

        #severity is meant to be a hidden, "black box" variable inside the MDP
        # and not available to the logistic function as a parameter
        severity = MILD
        if current_condition == POST_SUPPRESSION:
            if ev >= threshold_suppression:
                severity = SEVERE
        elif current_condition == POST_MILD:
            if ev >= threshold_mild:
                severity = SEVERE
        elif current_condition == POST_SUPPRESSION:
            if ev >= threshold_severe:
                severity = SEVERE


        #logistic function for the policy choice
        #policy_crossproduct = policy[0] + policy[1]*ev
        #modified logistic policy function
        #                     CONSTANT       COEFFICIENT       SHIFT
        policy_crossproduct = policy[0] + ( policy[1] * (ev + policy[2]) )
        if policy_crossproduct > 100: policy_crossproduct = 100
        if policy_crossproduct < -100: policy_crossproduct = -100

        policy_value = 1 / (1 + math.exp(-1*(policy_crossproduct)))

        choice_roll = random.uniform(0,1)
        #assume let-burn
        choice = False
        choice_prob = 1 - policy_value
        #check for suppress, and update values if necessary
        if PROBABALISTIC_CHOICES:
            if choice_roll < policy_value:
                choice = True
                choice_prob = policy_value
        else:
            if policy_value >= 0.5:
                choice = True
                choice_prob = policy_value


        ### CALCULATE REWARD ###
        supp_cost = 0
        if choice:
            #suppression was chosen
            if severity == MILD:
                supp_cost = supp_cost_mild
            elif severity == SEVERE: 
                supp_cost = supp_cost_severe

        burn_penalty = 0
        if current_condition == POST_SEVERE:
            burn_penalty = burn_cost


        current_reward = constant_reward - supp_cost - burn_penalty
                

        states[i] = [ev, choice, choice_prob, policy_value, current_reward, i]



        ### TRANSITION ###
        if not choice:
            #no suppression
            if severity == SEVERE:
                current_condition = POST_SEVERE
            elif severity == MILD:
                current_condition = POST_MILD
        else:
            #suppression
            current_condition = POST_SUPPRESSION



        


    #finished simulations, report some values
    vals = []
    suppressions = 0
    joint_prob = 1
    prob_sum = 0
    for i in range(timesteps):
        if states[i][1]: suppressions += 1
        joint_prob *= states[i][2]
        prob_sum += states[i][2]
        vals.append(states[i][4])
    ave_prob = prob_sum / timesteps

    summary = {
                "Average State Value": round(numpy.mean(vals),1),
                "Total Pathway Value": round(numpy.sum(vals),0),
                "STD State Value": round(numpy.std(vals),1),
                "Suppressions": suppressions,
                "Suppression Rate": round((float(suppressions)/timesteps),2),
                "Joint Probability": joint_prob,
                "Average Probability": round(ave_prob, 3),
                "ID Number": random_seed,
                "Timesteps": timesteps,
                "Generation Policy": policy
              }

    if not SILENT:
        print("")
        print("Simulation Complete - Pathway " + str(random_seed))
        print("Average State Value: " + str(round(numpy.mean(vals),1)) + "   STD: " + str(round(numpy.std(vals),1)))
        print("Suppressions: " + str(suppressions))
        print("Suppression Rate: " + str(round((float(suppressions)/timesteps),2)))
        print("Joint Probability:" + str(joint_prob))
        print("Average Probability: " + str(round(ave_prob, 3)))
        print("")


    summary["States"] = states

    return summary


def simulate_all_policies(timesteps=10000, start_seed=0):


    result_CT =    simulate(timesteps, policy=[  0,  0, 0.0], random_seed=start_seed, SILENT=True)
    result_LB =    simulate(timesteps, policy=[-20,  0, 0.0], random_seed=start_seed, SILENT=True)
    result_SA =    simulate(timesteps, policy=[ 20,  0, 0.0], random_seed=start_seed, SILENT=True)
    result_KNOWN = simulate(timesteps, policy=[  0, 20,-0.8], random_seed=start_seed, SILENT=True)

    result_CT["Name"] = "Coin-Toss:    "
    result_SA["Name"] = "Suppress-All: "
    result_LB["Name"] = "Let-burn:     "
    result_KNOWN["Name"] = "Known:        "

    results = [result_CT, result_SA, result_LB, result_KNOWN]
    print("Policy            Ave    STD    SupRate  AveProb  JointProb")
    for r in results:
        print(r["Name"] + "   "),
        print(str(r["Average State Value"]) + "   "),
        print(str(r["STD State Value"]) + "   "),
        print(str(r["Suppression Rate"]) + "      "),
        print(str(r["Average Probability"]) + "    "),
        print(str(r["Joint Probability"]))

