"""A Simple Wildfire-inspired MDP model. Version 1.1"""

import random, math, numpy

def simulate(timesteps, policy=[0,0,0], random_seed=0, model_parameters={}, SILENT=False, PROBABILISTIC_CHOICES=True):
    
    random.seed(random_seed)
    
    #range of the randomly drawn, uniformally distributed "event" that corresponds to fire severity
    event_max = 1.0
    event_min = 0.0
    
    timesteps = int(timesteps)


    #sanitize policy
    policy = sanitize_policy(policy)


    #REWARD STRUCTURE
    constant_reward = 5
    if "Constant Reward" in model_parameters.keys(): constant_reward = model_parameters["Constant Reward"]

    #cost of suppression in a mild event
    supp_cost_mild = 15
    if "Suppression Cost - Mild Event" in model_parameters.keys(): supp_cost_mild = model_parameters["Suppression Cost - Mild Event"]

    #cost of suppresion in a severe event
    supp_cost_severe = 30
    if "Suppression Cost - Severe Event" in model_parameters.keys(): supp_cost_severe = model_parameters["Suppression Cost - Severe Event"]

    #cost of a severe fire on the next timestep
    burn_cost = 50
    if "Severe Burn Cost" in model_parameters.keys(): burn_cost = model_parameters["Severe Burn Cost"]

    condition_change_after_suppression = -0.01
    condition_change_after_mild = 0.02
    condition_change_after_severe = 0.00
    if "Condition Change After Suppression" in model_parameters.keys(): condition_change_after_suppression = model_parameters["Condition Change After Suppression"]
    if "Condition Change After Mild" in model_parameters.keys(): condition_change_after_mild = model_parameters["Condition Change After Mild"]
    if "Condition Change After Severe" in model_parameters.keys(): condition_change_after_severe = model_parameters["Condition Change After Severe"]


    if "Probabilistic Choices" in model_parameters.keys():
        if model_parameters["Probabilistic Choices"] == "True":
            PROBABILISTIC_CHOICES = True
        else:
            PROBABILISTIC_CHOICES = False



    #starting_condition = 0.8
    starting_condition = random.uniform(0.2,0.8)
    if "Starting Condition" in model_parameters.keys(): starting_condition = model_parameters["Starting Condition"]


    #setting 'enums'
    MILD=0
    SEVERE=1


    #starting simulations
    states = [None] * timesteps

    #start current condition randomly among the three states
    current_condition = starting_condition

    for i in range(timesteps):

        #event value is the single "feature" of events in this MDP
        ev = random.uniform(event_min, event_max)

        #severity is meant to be a hidden, "black box" variable inside the MDP
        # and not available to the logistic function as a parameter
        severity = MILD
        if ev >= current_condition: severity = SEVERE


        #logistic function for the policy choice
        #policy_crossproduct = policy[0] + policy[1]*ev
        #modified logistic policy function
        #                     CONSTANT       COEFFICIENT       SHIFT
        policy_crossproduct = policy[0] + ( policy[1] * (ev + policy[2]) )
        if policy_crossproduct > 100: policy_crossproduct = 100
        if policy_crossproduct < -100: policy_crossproduct = -100

        policy_value = 1.0 / (1.0 + math.exp(-1*(policy_crossproduct)))

        choice_roll = random.uniform(0,1)
        #assume let-burn
        choice = False
        choice_prob = 1.0 - policy_value
        #check for suppress, and update values if necessary
        if PROBABILISTIC_CHOICES:
            if choice_roll < policy_value:
                choice = True
                choice_prob = policy_value
        else:
            if policy_value >= 0.5:
                choice = True
                choice_prob = policy_value


        ### CALCULATE REWARD ###
        supp_cost = 0
        burn_penalty = 0
        if choice:
            #suppression was chosen
            if severity == MILD:
                supp_cost = supp_cost_mild
            elif severity == SEVERE: 
                supp_cost = supp_cost_severe
        else:
            #suppress was NOT chosen
            if severity == SEVERE:
                #set this timestep's burn penalty to the value given in the overall model parameter
                #this is modeling the timber values lost in a large fire.
                burn_penalty = burn_cost

        


        current_reward = constant_reward - supp_cost - burn_penalty
                

        states[i] = [current_condition, ev, choice, choice_prob, policy_value, current_reward, i]



        ### TRANSITION ###
        if not choice:
            #no suppression
            if severity == SEVERE:
                current_condition += condition_change_after_severe
            elif severity == MILD:
                current_condition += condition_change_after_mild
        else:
            #suppression
            current_condition += condition_change_after_suppression


        #Enforce max/min
        if current_condition > 1.0: current_condition = 1.0
        if current_condition < 0.0: current_condition = 0.0
        


    #finished simulations, report some values
    vals = []
    suppressions = 0.0
    joint_prob = 1.0
    prob_sum = 0.0
    for i in range(timesteps):
        if states[i][2]: suppressions += 1
        joint_prob *= states[i][3]
        prob_sum += states[i][3]
        vals.append(states[i][5])
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
                "Generation Policy": policy,
                "Version": "1.1",
                "Constant Reward": constant_reward,
                "Condition Change After Suppression": condition_change_after_suppression,
                "Condition Change After Mild": condition_change_after_mild,
                "Condition Change After Severe": condition_change_after_severe,
                "Suppression Cost - Mild": supp_cost_mild,
                "Suppression Cost - Severe": supp_cost_severe,
                "Severe Burn Cost": burn_cost,
                "Starting Condition": starting_condition
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

def sanitize_policy(policy):
    pol = []
    if isinstance(policy, list):
        if len(policy) == 2:
            #it's length-2, so add the shift parameter
            pol = policy + [0]
        else:
            #it's probably length-3, so just assign it
            pol = policy
    else:
        #it's not a list, so find out what string it is
        if policy == 'LB':    pol = [-20,0,0]
        elif policy == 'SA':  pol = [ 20,0,0]
        elif policy == 'CT':  pol = [  0,0,0]

    return pol
