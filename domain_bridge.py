import MDP, MDP_opt, SWIMM

# Keep track of file numbers so they don't repeat
server_file_counter = 0
def file_number_str():
    global server_file_counter 
    server_file_counter += 1
    return server_file_counter

def initialize():
    """
    Return the initialization object for the SWIMM domain.
    """
    return {
                "reward": [
                            {"name": "Constant Reward",
                             "description":"Reward given at each state regardless of state variables or actions.",
                             "current_value": 2, "max": 100, "min": -100, "units": "~"},
                            {"name": "Suppression Cost - Severe Event",
                             "description":"The cost of suppressing a severe fire event.",
                             "current_value": 4, "max": 100, "min": -100, "units": "~"},
                            {"name": "Suppression Cost - Mild Event",
                             "description":"The cost of suppressing a common, low-severity event.",
                             "current_value": 1, "max": 100, "min": -100, "units": "~"},
                            {"name": "Severe Burn Cost",
                             "description":"The future cost of not suppressing a severe fire event.",
                             "current_value": 6, "max": 100, "min": -100, "units": "~"}
                            ],
                "transition": [
                            {"name": "Simulations",
                              "description": "how many separate MDP simulations to run",
                              "current_value": 200, "max": 10000, "min": 0, "units": "Y"},
                            {"name": "Timesteps",
                              "description": "how many steps into the future each simulation should take",
                              "current_value": 200, "max": 10000, "min": 0, "units": "Y"},
                            {"name": "Threshold After Suppression",
                              "description": "events above this threshold will be considered severe fires if they occur in the next timestep after a suppression",
                              "current_value": 0.8, "max": 1.0, "min": 0, "units": "Y"},
                            {"name": "Threshold After Low-Severity Event",
                              "description": "events above this threshold will be considered severe fires if they occur in the next timestep after a low-severity event",
                              "current_value": 0.8, "max": 1.0, "min": 0, "units": "Y"},
                            {"name": "Threshold After High-Severity Event",
                              "description": "events above this threshold will be considered severe fires if they occur in the next timestep after a high-severity event",
                              "current_value": 0.8, "max": 1.0, "min": 0, "units": "Y"}
                             ],
                "policy": [
                            {"name": "Constant",
                             "description":"for the intercept",
                             "current_value": 0, "max": 20, "min":-20, "units": ""},
                            {"name": "Severity",
                             "description":"for the current step's event severity",
                             "current_value": 0, "max": 20, "min":-20, "units": ""}

                          ]
                    }

def optimize(query):
    """
    Return a newly optimized query.
    """
    dict_reward = query["reward"]
    dict_transition = query["transition"]
    dict_policy = query["policy"]

    #how many, how long?
    simulation_count = dict_transition["Simulations"]
    timesteps = dict_transition["Timesteps"]
    
    #compiling the parameters that are needed by SWIMM (the extras will be ignored)
    dict_SWIMM = {}
    dict_SWIMM.update(dict_reward)
    dict_SWIMM.update(dict_transition)
    
    #setting policy
    pol = [dict_policy["Constant"], dict_policy["Severity"], 0]

    #creating pathways
    SWIMM_pws = [None] * dict_transition["Simulations"]
    for i in range(dict_transition["Simulations"]):
        new_sim = SWIMM.simulate(timesteps, policy=pol, random_seed=i, model_parameters=dict_SWIMM, SILENT=True)
        SWIMM_pws[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(new_sim)

    #creating optimization objects
    opt = MDP_opt.Optimizer()

    #giving pathways to the optimizer
    opt.pathway_set = SWIMM_pws
    
    #set desired objective function
    #TODO, defaulting to J3 

    #doing one round of optimization
    #opt.normalize_pathways()
    opt.optimize_policy()

    #pulling the policy variables back out
    learned_params = opt.Policy.get_params()

    #TODO make this robust to FireWoman policies
    dict_new_pol = {}
    dict_new_pol["Constant"] = learned_params[0]
    dict_new_pol["Severity"] = learned_params[1]


    return dict_new_pol

def rollouts(query):
    """
    Return a set of rollouts for the given parameters.


    The return structure should be as follows:

    [[{},{},{}],[{},{},{}],[{},{},{}]]

    Where each dictionary represents a single event, and each inner list contains the dictionaries
    associated with an individual pathway.

    """
    dict_reward = query["reward"]
    dict_transition = query["transition"]
    dict_policy = query["policy"] 

    simulations = int(dict_transition["Simulations"])
    timesteps = int(dict_transition["Timesteps"])
    start_ID = 0

    #compiling the parameters that are needed by SWIMM (the extras will be ignored)
    dict_SWIMM = {}
    dict_SWIMM.update(dict_reward)
    dict_SWIMM.update(dict_transition)
    
    #setting policy
    pol = [dict_policy["Constant"], dict_policy["Severity"], 0]

    #creating pathways
    SWIMM_pws = [None] * simulations
    for i in range(simulations):
        new_sim = SWIMM.simulate(timesteps, policy=pol, random_seed=i, model_parameters=dict_SWIMM, SILENT=True)
        #SWIMM_pws[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(new_sim)
        SWIMM_pws[i] = new_sim

    #outermost list to collect one sub-list for each pathway, etc...
    return_list = []

    #parse the data needed...
    for pw in SWIMM_pws:
        #new ignition events list for this pathway
        pw_values = []
        for ev in pw["States"]:
        
            #SWIMM simulations return a dictionary constructed as follows:
            
            #SWIMM_state = [ev, choice, choice_prob, policy_value, this_state_value, i]    
            #SWIMM_pathway = {
            #    "Average State Value": round(numpy.mean(vals),1),
            #    "Total Pathway Value": round(numpy.sum(vals),0),
            #    "STD State Value": round(numpy.std(vals),1),
            #    "Suppressions": suppressions,
            #    "Suppression Rate": round((float(suppressions)/timesteps),2),
            #    "Joint Probability": joint_prob,
            #    "Average Probability": round(ave_prob, 3),
            #    "ID Number": random_seed,
            #    "Timesteps": timesteps,
            #    "Generation Policy": policy,
            #    "States": SWIMM_state
            #  }
        
            features = {}
            features["Event Severity"]     = ev[0]
            features["Action"]             = ev[1]
            features["Action Probability"] = ev[2]
            features["Policy Probability"] = ev[3]
            features["Reward"]             = ev[4]
            features["Timestep"]           = ev[5]
            
            
            #adding this dictionary to this pathway's list of dictionaries.
            #it's just a re-arrangement of the same information
            pw_values.append(features)

        #the events list for this pathway has been filled, so add it to the return list
        return_list.append(pw_values)

    #done with all pathways

    return return_list

def state(query):
    """
    Return a series of images up to the requested event number.
    """
    event_number = int(query["Event Number"])
    pathway_number = int(query["Pathway Number"])
    dict_reward = query["reward"]
    dict_transition = query["transition"]
    dict_policy = query["policy"] 

    #compiling the parameters that are needed by SWIMM (the extras will be ignored)
    dict_SWIMM = {}
    dict_SWIMM.update(dict_reward)
    dict_SWIMM.update(dict_transition)
    
    #setting policy
    pol = [dict_policy["Constant"], dict_policy["Severity"], 0]

    #creating pathway
    spw = SWIMM.simulate(timesteps=event_number, policy=pol, random_seed=pathway_number, model_parameters=dict_SWIMM, SILENT=True)

    #SWIMM simulations return a dictionary constructed as follows:
    
    #SWIMM_state = [ev, choice, choice_prob, policy_value, this_state_value, i]    
    #SWIMM_pathway = {
    #    "Average State Value": round(numpy.mean(vals),1),
    #    "Total Pathway Value": round(numpy.sum(vals),0),
    #    "STD State Value": round(numpy.std(vals),1),
    #    "Suppressions": suppressions,
    #    "Suppression Rate": round((float(suppressions)/timesteps),2),
    #    "Joint Probability": joint_prob,
    #    "Average Probability": round(ave_prob, 3),
    #    "ID Number": random_seed,
    #    "Timesteps": timesteps,
    #    "Generation Policy": policy,
    #    "States": SWIMM_state
    #  }


    returnObj = {
            "statistics": 
            {
              "Average State Value": spw["Average State Value"],
              "Total Pathway Value": spw["Total Pathway Value"],
              "STD State Value": spw["STD State Value"],
              "Suppressions": spw["Suppressions"],
              "Suppression Rate": spw["Suppression Rate"],
              "Joint Probability": spw["Joint Probability"],
              "Average Probability": spw["Average Probability"],
              "ID Number": spw["ID Number"],
              "Generation Policy": spw["Generation Policy"],
              "Event Severity": spw["States"][0],
              "Event Choice": spw["States"][1],
              "Choice Probability": spw["States"][2],
              "Policy Probability": spw["States"][3],
              "Reward": spw["States"][4]
            },
            "images": []
            }
    return returnObj
