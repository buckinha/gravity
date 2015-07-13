from FireGirlOptimizer import *
from FireGirlStats import *

# Keep track of file numbers so they don't repeat
server_file_counter = 0
def file_number_str():
    global server_file_counter 
    server_file_counter += 1
    return server_file_counter

def initialize():
    """
    Return the initialization object for the FireGirl domain.
    """
    return {
                "reward": [
                            {"name": "Discount",
                             "description":"The per-year discount",
                             "current_value": 1, "max": 1, "min": 0, "units": "~"},
                            {"name": "Suppression Fixed Cost",
                             "description":"cost per day of suppression",
                             "current_value": 500, "max": 999999, "min": 0, "units": "$"},
                            {"name": "Suppression Variable Cost",
                             "description":"cost per hectare of suppression",
                             "current_value": 500, "max": 999999, "min": 0, "units": "$"}
                            ],
                "transition": [
                             {"name": "Years to simulate",
                              "description": "how far to look into the future",
                              "current_value": 10, "max": 150, "min": 0, "units": "Y"},
                             {"name": "Futures to simulate",
                              "description": "how many stochastic futures to generate",
                              "current_value": 25, "max": 1000, "min": 0, "units": "#"},
                             {"name": "Landscape Size",
                              "description": "how many cells wide and tall should the landscape be. Min:9, Max:129",
                              "current_value": 21, "max": 129, "min": 9, "units": "#"},
                             {"name": "Harvest Percent",
                              "description": "timber harvest rate as a percent of annual increment",
                              "current_value": 0.95, "max": 1, "min": 0, "units": "%"},
                             {"name": "Minimum Timber Value",
                              "description":"the minimum timber value required before harvest is allowed",
                              "current_value": 50, "max":9999, "min": 0, "units": "$"},
                             {"name": "Slash Remaning",
                              "description": "the amount of fuel load (slash) left after a harvest",
                              "current_value": 10, "max":9999, "min": 0, "units": "#"},
                             {"name": "Fuel Accumulation",
                              "description": "the amount of fuel load that accumulates each year",
                              "current_value": 2, "max":9999, "min": 0, "units": "#"},
                             {"name": "Suppression Effect",
                              "description": "the reduction in fire spread rate as the result of suppression",
                              "current_value": 0.5, "max":1, "min": 0, "units": "%"},
                              {"name": "Use Original Bugs",
                               "description": "set to 0 to use original bugs. 1 (or non-zero) to use the patches.",
                               "current_value": 0, "max":1, "min": 0, "units": "~"},
                              {"name": "Growth Model",
                               "description": "set to 1 to use original model; or 2 for updated model.",
                               "current_value": 1, "max":2, "min": 1, "units": "~"}
                             ],
                "policy": [
                            {"name": "Constant",
                             "description":"for the intercept",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Date",
                             "description":"for each day of the year",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Days Left",
                             "description":"for each day left in the year",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name":"Temperature",
                             "description":"for air temperature at the time of an ignition",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Wind Speed",
                             "description":"for wind speed at the time of an ignition",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Timber Value",
                             "description":"for the timber value at an ignition location",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Timber Value 8",
                             "description":"for the average timber value in the 8 neighboring stands",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Timber Value 24",
                             "description":"for the average timber value in the 24 neighboring stands",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Fuel Load",
                             "description":"for the fuel load at an ignition location",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Fuel Load 8",
                             "description":"for the average fuel load in the 8 neighboring stands",
                             "current_value": 0, "max": 10, "min":-10, "units": ""},
                            {"name": "Fuel Load 24",
                             "description":"for the average fuel load in the 24 neighboring stands",
                             "current_value": 0, "max": 10, "min":-10, "units": ""}

                          ]
                    }

def optimize(query):
    """
    Return a newly optimized query.
    """
    dict_reward = query["reward"]
    dict_transition = query["transition"]
    dict_policy = query["policy"]

    #some variables
    #pathway_count = 5 #how many pathways to use in the optimization
    #years = 5  #how many years to simulate for each pathway

    pathway_count = dict_transition["Futures to simulate"]
    years = dict_transition["Years to simulate"]


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


    #creating pathways
    opt.createFireGirlPathways(int(pathway_count),int(years))

    #set desired objective function
    if "Objective Function" in dict_transition.keys():
        opt.setObjFn(dict_transition["Objective Function"])

    #doing one round of optimization
    opt.optimizePolicy()

    #pulling the policy variables back out
    learned_params = opt.Policy.getParams()

    #TODO make this robust to FireWoman policies
    dict_new_pol = {}
    dict_new_pol["Constant"] = learned_params[0]
    dict_new_pol["Date"] = learned_params[1]
    dict_new_pol["Days Left"] = learned_params[2]
    dict_new_pol["Temperature"] = learned_params[3]
    dict_new_pol["Wind Speed"] = learned_params[4]
    dict_new_pol["Timber Value"] = learned_params[5]
    dict_new_pol["Timber Value 8"] = learned_params[6]
    dict_new_pol["Timber Value 24"] = learned_params[7]
    dict_new_pol["Fuel Load"] = learned_params[8]
    dict_new_pol["Fuel Load 8"] = learned_params[9]
    dict_new_pol["Fuel Load 24"] = learned_params[10]


    return dict_new_pol

def rollouts(query):
    """
    Return a set of rollouts for the given parameters.
    """
    dict_reward = query["reward"]
    dict_transition = query["transition"]
    dict_policy = query["policy"] 

    pathway_count = int(dict_transition["Futures to simulate"])
    years = int(dict_transition["Years to simulate"])
    start_ID = 0

    #generate 100 rollouts
    opt = FireGirlPolicyOptimizer()
    opt.setObjFn("J1")
    #opt.setObjFn("J2")
    opt.SILENT = True
    
    #setting policy...
    #This is brittle, and will not work directly with FireWoman data... or with future versions
    # of FireGirl if new features get added...
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

    #setting the policy in the optimizer, which will pass it to each created pathway
    opt.setPolicy(pol)

    #giving the optimizer custom model parameters
    opt.setFireGirlModelParameters(dict_transition,dict_reward)

    #creating landscapes. The function will enforce the custom model parameters
    opt.createFireGirlPathways(pathway_count,years,start_ID)

    #outermost list to collect one sub-list for each pathway, etc...
    return_list = []

    #parse the data needed...
    for pw in opt.pathway_set:
        #new ignition events list for this pathway
        year_values = []
        for ign in pw.ignition_events:

            #get the dictionary representation of the ignition
            features = ign.getDictionary()

            #fill the total's dictionary
            features["Harvest Value"] = pw.getHarvest(ign.year)
            #features["Suppression Cost"] = pw.getSuppressionCost(ign.year) #already reported in ign.getDictionary()
            features["Growth"] = pw.getGrowth(ign.year)

            #TODO - Fix for Discount Rate
            features["Discounted Reward"] = features["Harvest Value"] - features["Suppression Cost"]

            features["Event Number"] = ign.year

            #NOTE:  This will be the same number for all ignitions in this pathway. It's the
            # id number that a pathway uses to instantiate its random seed 
            features["Pathway Number"] = pw.ID_number

            #adding cumulative measurements, from the start, up to this year
            features["Cumulative Harvest Value"] = pw.getHarvestFrom(0, ign.year)
            features["Cumulative Growth"] = pw.getGrowthFrom(0, ign.year)
            features["Cumulative Timber Loss"] = pw.getTimberLossFrom(0, ign.year)
            features["Cumulative Suppression Cost"] = pw.getSuppressionFrom(0, ign.year)


            #add this ignition event + year details to this pathway's list of dictionaries
            year_values.append(features)

        #the events list for this pathway has been filled, so add it to the return list
        return_list.append(year_values)

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
