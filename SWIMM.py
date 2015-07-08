"""A Simple Wildfire-inspired MDP model"""

import random, math, numpy

def simulate(timesteps, policy=[0,0,0], SILENT=False):
    
    basic_state_value = 2

    event_max = 100
    event_min = 0


    #cost of suppression in a low-severity fire
    action_1_cost_sev_low = 1

    #cost of suppresion in a high-severity fire
    action_1_cost_sev_high = 4

    #cost of a severe fire on the next timestep
    burn_cost = 6


    severity_switchpoint_after_suppression = 80
    severity_switchpoint_after_low = 80
    severity_switchpoint_after_high = 80 



    states = [None] * timesteps

    previous_burn_cost = 0
    previous_burn = 'none'

    for i in range(timesteps):

        #event value is the single "feature" of events in this MDP
        ev = random.uniform(event_min, event_max)

        #severity is meant to be a hidden, "black box" variable inside the MDP
        sev = 'low'
        if previous_burn == 'none':
            if ev > severity_switchpoint_after_suppression: sev = 'high'
        elif previous_burn == "low":
            if ev > severity_switchpoint_after_low: sev = 'high'
        elif previous_burn == 'high':
            if ev > severity_switchpoint_after_high: sev = 'high'


        #logistic function for the policy choice
        #policy_crossproduct = policy[0] + policy[1]*ev
        #modified logistic policy function
        #                     CONSTANT       COEFFICIENT       SHIFT
        policy_crossproduct = policy[0] + ( policy[1] * (ev + policy[2]) )
        if policy_crossproduct > 100: policy_crossproduct = 100
        if policy_crossproduct < -100: policy_crossproduct = -100

        policy_value = 1 / (1 + math.exp(-1*(policy_crossproduct)))

        choice_roll = random.uniform(0,1)
        choice = False
        choice_prob = 1 - policy_value
        if choice_roll < policy_value:
            choice = True
            choice_prob = policy_value

        #now check the choice vs severity combinations, and update things for the next state
        if sev == 'low':
            if choice:
                #low severity, with suppression
                previous_burn_cost = 0
                previous_burn = 'none'
            else:
                #low severity, without suppression
                previous_burn_cost = 0
                previous_burn = 'low'
            
        else: #sev == 'high'
            if choice:
                #high severity, with suppression
                previous_burn_cost = 0
                previous_burn = 'none'
            else:
                #high severity, without suppression
                previous_burn_cost = burn_cost
                previous_burn = 'high'


        this_state_value = basic_state_value - previous_burn_cost
        if choice:
            if sev == 'high':
                this_state_value -= action_1_cost_sev_high
            else: #sev == 'low'
                this_state_value -= action_1_cost_sev_low


        states[i] = [ev, choice, choice_prob, this_state_value]


    #finished simulations, report some values
    vals = []
    suppressions = 0
    joint_prob = 1
    prob_sum = 0
    for i in range(timesteps):
        if states[i][1]: suppressions += 1
        joint_prob *= states[i][2]
        prob_sum += states[i][2]
        vals.append(states[i][3])
    ave_prob = prob_sum / timesteps

    summary = {
                "Average State Value": round(numpy.mean(vals),1),
                "STD State Value": round(numpy.std(vals),1),
                "Suppressions": suppressions,
                "Suppression Rate": round((float(suppressions)/timesteps),2),
                "Joint Probability": joint_prob,
                "Average Probability": round(ave_prob, 3)
              }

    if not SILENT:
        print("")
        print("Simulation Complete")
        print("Average State Value: " + str(round(numpy.mean(vals),1)) + "   STD: " + str(round(numpy.std(vals),1)))
        print("Suppressions: " + str(suppressions))
        print("Suppression Rate: " + str(round((float(suppressions)/timesteps),2)))
        print("Joint Probability:" + str(joint_prob))
        print("Average Probability: " + str(round(ave_prob, 3)))
        print("")


    summary["States"] = states

    return summary


def simulate_all_policies(timesteps=10000):
    result_CT =    simulate(timesteps, policy=[  0,  0,  0], SILENT=True)
    result_LB =    simulate(timesteps, policy=[-20,  0,  0], SILENT=True)
    result_SA =    simulate(timesteps, policy=[ 20,  0,  0], SILENT=True)
    result_KNOWN = simulate(timesteps, policy=[  0, 20,-80], SILENT=True)

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

