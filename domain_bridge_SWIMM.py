from MDP_PolicyOptimizer import *
import MDP, SWIMM

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
                             "current_value": 2, "max": 100, "min": -100, "units": "~"},
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
                              "current_value": 80, "max": 100, "min": 0, "units": "Y"},
                            {"name": "Threshold After Low-Severity Event",
                              "description": "events above this threshold will be considered severe fires if they occur in the next timestep after a low-severity event",
                              "current_value": 80, "max": 100, "min": 0, "units": "Y"},
                            {"name": "Threshold After High-Severity Event",
                              "description": "events above this threshold will be considered severe fires if they occur in the next timestep after a high-severity event",
                              "current_value": 80, "max": 100, "min": 0, "units": "Y"}
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
    opt = MDP_PolicyOptimizer()

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
    SWIMM_pws = [None] * dict_transition["Simulations"]
    for i in range(dict_transition["Simulations"]):
        new_sim = SWIMM.simulate(timesteps, policy=pol, random_seed=i, model_parameters=dict_SWIMM, SILENT=True)
        SWIMM_pws[i] = MDP.convert_SWIMM_pathway_to_MDP_pathway(new_sim)

    #outermost list to collect one sub-list for each pathway, etc...
    return_list = []

    #parse the data needed...
    for pw in SWIMM_pws:
        #new ignition events list for this pathway
        pw_values = []
        for ign in pw.ignition_events:
        
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
            features["Event Severity"]     = pw["States"][0]
            features["Action"]             = pw["States"][1]
            features["Action Probability"] = pw["States"][2]
            features["Policy Probability"] = pw["States"][3]
            features["Reward"]             = pw["States"][4]
            features["Timestep"]           = pw["States"][5]
            
            
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
    
    show_count = 50
    step = 1
    if "Past Events to Show" in query.keys():
        show_count = 1 + int(query["Past Events to Show"])
    if "Past Events to Step Over" in query.keys():
        step = 1 + int(query["Past Events to Step Over"])

    #sanitizing
    if step < 1: step = 1
    if show_count < 1: show_count = 1


    #creating optimization objects
    opt = FireGirlPolicyOptimizer()

    #giving the simulation parameters to opt, so that it can pass
    # them on to it's pathways as it creates them
    opt.setFireGirlModelParameters(dict_transition, dict_reward)

    #setting policy as well
    #TODO make this robust to FireWoman policies
    pol = FireGirlPolicy()
    pol.setParams([dict_policy["Constant"],
                   dict_policy["Date"],
                   dict_policy["Days Left"],
                   dict_policy["Temperature"],
                   dict_policy["Wind Speed"],
                   dict_policy["Timber Value"],
                   dict_policy["Timber Value 8"],
                   dict_policy["Timber Value 24"],
                   dict_policy["Fuel Load"],
                   dict_policy["Fuel Load 8"],
                   dict_policy["Fuel Load 24"],
                  ])

    #assigning the policy to opt, so that it can use it in simulations.
    opt.setPolicy(pol)

    #Setting opt to tell it's pathway(s) to remember their histories
    #un-needed, since we're just re-creating the pathway of interest anyway
    #opt.PATHWAYS_RECORD_HISTORIES = True 

    opt.SILENT = True

    #creating image name list
    names = [[],[],[],[]]

    #creating pathway with no years... this will generate the underlying landscape and set
    #  all the model parameters that were assigned earlier.
    opt.createFireGirlPathways(1, 0, pathway_number)

    #now incrementing the years
    #because we start with the final year, and then skip backward showing every few landscapes,
    #we may have to skip over several of the first landscapes before we start showing any
    start = event_number - (step * (show_count -1))

    #checking for negative numbers, in case the users has specified too many past landscapes to show
    while start < 0:
        start += step

    #manually telling the pathway to do the first set of years
    opt.pathway_set[0].doYears(start)

    #get new names
    timber_name = "static/timber_" + str(file_number_str()) + ".png"
    fuel_name = "static/fuel_" + str(file_number_str()) + ".png"
    composite_name = "static/composite_" + str(file_number_str()) + ".png"
    burn_name = "static/burn_" + str(file_number_str()) + ".png"

    #and save it's images
    opt.pathway_set[0].saveImage(timber_name, "timber")
    opt.pathway_set[0].saveImage(fuel_name, "fuel")
    opt.pathway_set[0].saveImage(composite_name, "composite")
    opt.pathway_set[0].saveImage(burn_name, "timber", 10)

    #add these names to the lists
    names[0].append(timber_name)
    names[1].append(fuel_name)
    names[2].append(composite_name)
    names[3].append(burn_name)


    #now loop through the rest of the states
    for i in range(start, event_number+1, step):
        #do the next set of years
        opt.pathway_set[0].doYears(step)

        #create a new image filenames
        timber_name = "static/timber_" + str(file_number_str()) + ".png"
        fuel_name = "static/fuel_" + str(file_number_str()) + ".png"
        composite_name = "static/composite_" + str(file_number_str()) + ".png"
        burn_name = "static/burn_" + str(file_number_str()) + ".png"

        #save the images
        opt.pathway_set[0].saveImage(timber_name, "timber")
        opt.pathway_set[0].saveImage(fuel_name, "fuel")
        opt.pathway_set[0].saveImage(composite_name, "composite")
        opt.pathway_set[0].saveImage(burn_name, "timber", 10)

        #add these names to the lists
        names[0].append(timber_name)
        names[1].append(fuel_name)
        names[2].append(composite_name)
        names[3].append(burn_name)

    timber_stats = pathway_summary(opt.pathway_set[0],"timber")
    fuel_stats = pathway_summary(opt.pathway_set[0],"fuel")
    total_growth = opt.pathway_set[0].getGrowthTotal()
    total_suppression = opt.pathway_set[0].getSuppressionTotal()
    total_harvest = opt.pathway_set[0].getHarvestTotal()
    total_timber_loss = opt.pathway_set[0].getTimberLossTotal()

    returnObj = {
            "statistics": {
              "Event Number": int(query["Event Number"]),
              "Pathway Number": int(query["Pathway Number"]),
              "Average Timber Value": int(timber_stats[0]),
              "Timber Value Std.Dev.": int(timber_stats[1]),
              "Average Timber Value - Center": int(timber_stats[2]),
              "Timber Value Std.Dev. - Center": int(timber_stats[3]),
              "Average Fuel Load": int(fuel_stats[0]),
              "Fuel Load Std.Dev.": int(fuel_stats[1]),
              "Average Fuel Load - Center": int(fuel_stats[2]),
              "Fuel Load Std.Dev. - Center": int(fuel_stats[3]),
              "Cumulative Harvest":total_harvest,
              "Cumulative Suppression Cost": total_suppression,
              "Cumulative Timber Loss":total_timber_loss,
              "Cumulative Timber Growth":total_growth,
             },
            "images": names
            }
    return returnObj
