"""SWMv2.1 Trials"""

import MDP, MDP_opt, HKB_Heuristics, random, numpy, datetime
import os.path
from sphere_dist import sphere_dist
import  SWMv2_1 as SWM

def standard_MDP_set(pathway_count, timesteps, policy, random_seed=0, VALUE_ON_HABITAT=False, percentage_habitat=0, sampling_radius=0.0):
    """
    Generates a set of SWM v2.1 pathways and returns them as a list of MDP pathway objects

    ARGUEMENTS
    pathway_count: integer - how many pathways to create
    timesteps: integer  - how many timesteps each pathway should be simulated for
    policy: a list of policy values denoting the center of the set's distribution
    random_seed: the initial random seed from which to work. Each pathway will have it's
      own random seed derived from this one. Giving the same parameters and random seed will
      always result in the same set of pathways (bugs notwithsdanding)
    VALUE_ON_HABITAT: boolean, When set to True, the pathways will have the habitat value index as 
      their "value" rather than the logging/suppression budget values
    sampling_radius: float, etc...: when each pathway is simulated, the policy it is given is shifted 
      to some point around the orginal policy within this distance.
    """

    pathways = [None]*pathway_count

    pol_length = len(policy)

    #setting up the starting policy
    original_pol = SWM.sanitize_policy(policy)


    for i in range(pathway_count):

        #do policy perturbation
        pol = sphere_dist(center=original_pol, radius=sampling_radius, random_seed=random_seed+i)

        #run a simulation
        pw = SWM.simulate(timesteps, pol, random_seed=i+8500+random_seed, SILENT=True)

        #add the simulation to the list after converting it's form
        pathways[i] = SWM.convert_to_MDP_pathway(pw, VALUE_ON_HABITAT=VALUE_ON_HABITAT, percentage_habitat=percentage_habitat)

    return pathways

def limited_MDP_set(pathway_count, timesteps, policy, random_seed=0, VALUE_ON_HABITAT=False, percentage_habitat=0, sampling_radius=0.0, fail_at_count=2000):
    """
    Generates as per the standard set, but rejects pathways with suppression_rate = 0 or 1

    ARGUEMENTS
    pathway_count: integer - how many pathways to create
    timesteps: integer  - how many timesteps each pathway should be simulated for
    policy: either a list of policy values, or a string "CT" "SA" "LB" "MIXED_CT", "MIXED_ALL"
    random_seed: the initial random seed from which to work. Each pathway will have it's
      own random seed derived from this one. Giving the same parameters and random seed will
      always result in the same set of pathways (bugs notwithsdanding)
    VALUE_ON_HABITAT: boolean, When set to True, the pathways will have the habitat value index as 
      their "value" rather than the logging/suppression budget values
    sampling_radius: float, etc...: when each pathway is simulated, the policy it is given is shifted 
      by up to this amount in either the + or - direction, but limited such that the perturbation vector
      is within the spherical region with this radius around the initial policy .
    fail_at_count: an integer reflecting the maximum number of attempts the function can make at finding
      policies/pathways that have suppression rates other than 0 or 1
    """

    pathways = [None]*pathway_count

    pol_length = len(policy)
    
    original_pol = SWM.sanitize_policy(policy)

    good_pathways = 0
    retries = 0
    for count in range(fail_at_count):

        #do policy perturbation
        #retries is being added to the random seed because otherwise, once we hit an LB or SA pathway,
        # it'll just keep re-simulating the same one with the same random seed.
        pol = sphere_dist(center=original_pol, radius=sampling_radius, random_seed=random_seed+retries+good_pathways)

        #run the pathway simulation
        pw = SWM.simulate(timesteps, pol, random_seed=good_pathways+8500+random_seed+retries, SILENT=True)


        #check the suppression rate
        if (pw["Suppression Rate"] == 0) or (pw["Suppression Rate"] == 1):
            #this pathway needs to be rejected:
            #  keep i as it is, and continue the loop
            #print("skipping a pathway, pw['Suppression Rate'] = " + str(pw["Suppression Rate"]))
            #print("  Policy is " + str(pol))

            #increment retries to make the next attempt have a new random seed. Otherwise we'll just 
            # generate the same sample over and over again
            retries += 1

        else:
            #this pathway can be accepted
            pathways[good_pathways] = SWM.convert_to_MDP_pathway(pw,VALUE_ON_HABITAT=VALUE_ON_HABITAT,percentage_habitat=percentage_habitat)
            good_pathways += 1

            #reset retries
            retries=0

            #check for i being in range
            if good_pathways >= pathway_count:
                #we've filled the array, so quit
                break

    #we've exitted the loop. Check to see if the array was filled:
    if good_pathways < pathway_count:
        #we didn't fill the set, so... do what?
        print("WARNING: limited_MDP_set() could only find " + str(good_pathways) + " non-plateau policies...")
        pathways = pathways[:good_pathways]

    return pathways

def filtered_MDP_set(filters, pathway_count, timesteps, policy, random_seed=0, sampling_radius=0.0, fail_at_count=2000, VALUE_ON_HABITAT=False, percentage_habitat=0):
    """Returns a set of SWMv2.1 MDP objects with filtered random policies around the one given.

    ARGUEMENTS
    filter - a function handle or list of function handles. Function should return True to EXCLUDE a policy
    pathway_count: integer - how many pathways to create
    timesteps: integer  - how many timesteps each pathway should be simulated for
    policy: either a list of policy values, or a string "CT" "SA" "LB" "MIXED_CT", "MIXED_ALL"
    random_seed: the initial random seed from which to work. Each pathway will have it's
      own random seed derived from this one. Giving the same parameters and random seed will
      always result in the same set of pathways (bugs notwithsdanding)
    VALUE_ON_HABITAT: boolean, When set to True, the pathways will have the habitat value index as 
      their "value" rather than the logging/suppression budget values
    sampling_radius: float, etc...: when each pathway is simulated, the policy it is given is shifted 
      by up to this amount in either the + or - direction, but limited such that the perturbation vector
      is within the spherical region with this radius around the initial policy .
    fail_at_count: an integer reflecting the maximum number of attempts the function can make at finding
      policies/pathways that have suppression rates other than 0 or 1
    """

    #check to see if the filter is a list, or just a singlet
    if isinstance(filters, list):
        #it's already a list
        pass
    else:
        #it's not a list, so we'll assume it's a singlet and make it into a length 1 list.
        filters = [filters]

    #sanitize the input policy
    original_pol = SWM.sanitize_policy(policy)


    #an array for the policies we'll simulate on
    pols = [ [0.0] * len(original_pol) for i in range(pathway_count) ]

    #instantiate new_pol
    new_pol = [0.0] * len(original_pol)

    #counter for good pathways
    good_pathways = 0

    #try getting policies that satisfy the filters
    for i in range(fail_at_count):
        new_pol = sphere_dist(center=original_pol, radius=sampling_radius, random_seed=i+random_seed)

        #vet the policy with any filters that have been passed
        discard = False
        for f in filters:
            if f([new_pol]):
                print("Filtering Policy:" + str(new_pol))
                discard = True
            #else:
             #   print(str(f(new_pol)) + "   " + str(new_pol))

        #add this policy to the list if it passed the filters
        if not discard:
            #copy new_pol
            pols[good_pathways] = new_pol[:]
            #increment count of good pathways
            good_pathways += 1

            #check for exit condition
            if good_pathways >= pathway_count:
                break

    #we've finished the loop, so check to see if we found enough good policies
    if good_pathways < pathway_count:
        #we didn't fill the set, so... do what?
        print("WARNING: filtered_MDP_set() could only find " + str(good_pathways) + " pathways that passed the filter(s)...")


    #create the simulations
    sims = [None] * good_pathways
    for i in range(good_pathways):
        pw = SWM.simulate(timesteps, pols[i], random_seed=good_pathways+9200+random_seed, SILENT=True)
        sims[i] = SWM.convert_to_MDP_pathway(pw,VALUE_ON_HABITAT=VALUE_ON_HABITAT,percentage_habitat=percentage_habitat)
    

    return sims



def stats(pathway_set, SILENT=False):
    """Prints several descriptive statistics about a pathway set to standard out."""

    values = [0.0] * len(pathway_set)
    suppression_rate = [0.0] * len(pathway_set)

    for p in range(len(pathway_set)):
        values[p] = pathway_set[p].net_value / len(pathway_set[p].events)
        suppression_rate[p] = pathway_set[p].metadata["Suppression Rate"]


    if not SILENT:
        print("VALUE")
        print("  Mean: " + str(numpy.mean(values)))
        print("  Var:  " + str(numpy.var(values)))
        print("  STD:  " + str(numpy.std(values)))
        print("Suppression Rate: " + str(numpy.mean(suppression_rate)))

    summary = {}
    summary["Mean"] = numpy.mean(values)
    summary["Var"] = numpy.var(values)
    summary["STD"] = numpy.std(values)
    summary["Suppression Rate"] = numpy.mean(suppression_rate)

    return summary

def write_pathways(pathways, filename):
    f = open(filename,"w")

    f.write("SWMv2.1 Pathway Set\n")
    f.write("\n")

    #print metadata from the first pathway. It should be the same for all of them.
    f.write("MODEL PARAMETETS\n")
    f.write("Version: " + str(pathways[0].metadata["Version"]) + "\n")
    f.write("Timesteps: " + str(pathways[0].metadata["Timesteps"]) + "\n")
    f.write("Vulnerability Min: " + str(pathways[0].metadata["Vulnerability Min"]) + "\n")
    f.write("Vulnerability Max: " + str(pathways[0].metadata["Vulnerability Max"]) + "\n")
    f.write("Vulnerability Change After Suppression: " + str(pathways[0].metadata["Vulnerability Change After Suppression"]) + "\n")
    f.write("Vulnerability Change After Mild: " + str(pathways[0].metadata["Vulnerability Change After Mild"]) + "\n")
    f.write("Vulnerability Change After Severe: " + str(pathways[0].metadata["Vulnerability Change After Severe"]) + "\n")
    f.write("Timber Value Min: " + str(pathways[0].metadata["Timber Value Min"]) + "\n")
    f.write("Timber Value Max: " + str(pathways[0].metadata["Timber Value Max"]) + "\n")
    f.write("Timber Value Change After Suppression: " + str(pathways[0].metadata["Timber Value Change After Suppression"]) + "\n")
    f.write("Timber Value Change After Mild: " + str(pathways[0].metadata["Timber Value Change After Mild"]) + "\n")
    f.write("Timber Value Change After Severe: " + str(pathways[0].metadata["Timber Value Change After Severe"]) + "\n")
    f.write("Habitat Mild Interval: " + str(pathways[0].metadata["Habitat Mild Interval"]) + "\n")
    f.write("Habitat Severe Interval: " + str(pathways[0].metadata["Habitat Severe Interval"]) + "\n")
    f.write("Habitat Loss If No Severe Fire: " + str(pathways[0].metadata["Habitat Loss If No Severe Fire"]) + "\n")
    f.write("Habitat Loss If No Mild Fire: " + str(pathways[0].metadata["Habitat Loss If No Mild Fire"]) + "\n")
    f.write("Habitat Gain With Optimal Fire: " + str(pathways[0].metadata["Habitat Gain With Optimal Fire"]) + "\n")
    f.write("Suppression Cost - Mild: " + str(pathways[0].metadata["Suppression Cost - Mild"]) + "\n")
    f.write("Suppression Cost - Severe: " + str(pathways[0].metadata["Suppression Cost - Severe"]) + "\n")
    f.write("Severe Burn Cost: " + str(pathways[0].metadata["Severe Burn Cost"]) + "\n")
    f.write("\n")

    f.write("VALUE CONS HEAT MOISTURE TIMBER VULN HABITAT\n")
    for p in pathways:
        f.write(str(p.net_value / len(p.events)))
        for b in p.generation_policy_parameters:
            f.write(" " + str(b))
        f.write("\n")

    f.close()


def sweep_one_parameter(pathways, years, parameter_index, param_min, param_max, step=1, reference_policy=[0,0,0,0,0,0]):

    results = []

    pol = reference_policy

    if (parameter_index < 0) or (parameter_index >= 6):
        #check parameter index
        print("parameter index is out of range")
        return None

    else:
        #parameter index is fine

        for i in range(param_min, param_max+1, step):

            pol[parameter_index] = i

            pws = standard_MDP_set(pathways,years,pol,sampling_radius=1)

            #print("P" + str(parameter_index) + ": " + str(i))
            st = stats(pws, SILENT=True)
            st["Param"] = i
            results.append(st)


        #print results to a file
        f = open("sweep.txt",'w')
        f.write("SWMv2.1 Sweeping One Paramter\n")
        f.write("\n")
        f.write("Reference Policy: " + str(reference_policy) + "\n")
        f.write("Sweeping on Parameter " + str(parameter_index) + "\n")
        f.write("\n")
        f.write("PARAM MEAN STD SUPP_RATE\n")
        for result in results:
            f.write(str(result["Param"]) + " ")
            f.write(str(result["Mean"]) + " ")
            f.write(str(result["STD"]) + " ")
            f.write(str(result["Suppression Rate"]) + "\n") 


        f.close()

        return results


def classifier_stats(dataset, index_of_suppression_rate, index_of_SA_flag, index_of_LB_flag):
    """Prints classification percent-correct stats as a table

    ARGUEMENTS
    dataset: a 2D list or np.array in the format list[row][column] or np.array[row,column]

    index_of_suppression_rate: integer. The column index of the suppression rate value for a given row

    index_of_SA_flag: integer. The column index of the 0/1 flag for a row having been flagged as suppress-all

    index_of_LB_flag: integer. The column index of the 0/1 flag for a row having been flagged as let-burn

    RETURNS
    None (It just prints the table to standard out)

    """

    #shortening the names
    i_SA = index_of_SA_flag
    i_LB = index_of_LB_flag
    i_sup_rt = index_of_suppression_rate


    #get subsets
    SA_set = filter(lambda x: x[i_sup_rt]>=0.995, dataset)
    LB_set = filter(lambda x: x[i_sup_rt]<=0.005, dataset)
    MID_set = filter(lambda x: ((x[i_sup_rt]<0.995) and (x[i_sup_rt]>0.005)), dataset)

    high_mid_subset = filter(lambda x: ((x[i_sup_rt]<0.995) and (x[i_sup_rt]>0.95)), dataset)
    low_mid_subset = filter(lambda x: ((x[i_sup_rt]<0.05) and (x[i_sup_rt]>0.005)), dataset)


    #filter by occurences of correct and incorrect flags

    SA_and_flagged_SA =        filter(lambda x: x[i_SA],                           SA_set)
    SA_and_NOT_flagged_SA =    filter(lambda x: not x[i_SA],                       SA_set)
    LB_and_flagged_LB =        filter(lambda x: x[i_LB],                           LB_set)
    LB_and_NOT_flagged_LB =    filter(lambda x: not x[i_LB],                       LB_set)
    MID_and_not_flagged =      filter(lambda x: ((not x[i_SA]) and (not x[i_LB])), MID_set)
    MID_BUT_flagged =          filter(lambda x: (x[i_SA] or x[i_LB]),              MID_set)
    MID_high_and_not_flagged = filter(lambda x: ((not x[i_SA]) and (not x[i_LB])), high_mid_subset)
    MID_high_BUT_flagged =     filter(lambda x: (x[i_SA] or x[i_LB]),              high_mid_subset)
    MID_low_and_not_flagged =  filter(lambda x: ((not x[i_SA]) and (not x[i_LB])), low_mid_subset)
    MID_low_BUT_flagged =      filter(lambda x: (x[i_SA] or x[i_LB]),              low_mid_subset)

    #tabulating stats
    SA_c  =  len(SA_and_flagged_SA) 
    SA_ic =  len(SA_and_NOT_flagged_SA) 
    SA_pc = round(  float(len(SA_and_flagged_SA)) / float(len(SA_set)), 3)  
    SA_pa = round(  float(len(SA_and_flagged_SA)) / float(len(dataset)), 3)  
     
    LB_c  =  len(LB_and_flagged_LB) 
    LB_ic =  len(LB_and_NOT_flagged_LB) 
    LB_pc = round(  float(len(LB_and_flagged_LB)) / float(len(LB_set)), 3)  
    LB_pa = round(  float(len(LB_and_flagged_LB)) / float(len(dataset)), 3)  
     
    MID_c  =  len(MID_and_not_flagged) 
    MID_ic =  len(MID_BUT_flagged) 
    MID_pc = round(  float(len(MID_and_not_flagged)) / float(len(MID_set)), 3)  
    MID_pa = round(  float(len(MID_and_not_flagged)) / float(len(dataset)), 3)  
     
    MH_c  =  len(MID_high_and_not_flagged) 
    MH_ic =  len(MID_high_BUT_flagged) 
    MH_pc = round(  float(len(MID_high_and_not_flagged)) / float(len(high_mid_subset)), 3)  
    MH_pa = round(  float(len(MID_high_and_not_flagged)) / float(len(dataset)), 3)  
     
    ML_c  =  len(MID_low_and_not_flagged) 
    ML_ic =  len(MID_low_BUT_flagged) 
    ML_pc = round(  float(len(MID_low_and_not_flagged)) / float(len(low_mid_subset)), 3)  
    ML_pa = round(  float(len(MID_low_and_not_flagged)) / float(len(dataset)), 3)  

    total_correct = len(SA_and_flagged_SA) + len(LB_and_flagged_LB) + len(MID_and_not_flagged)
    TOTAL_perc = round( float( total_correct ) / float(len(dataset)) , 3  )
    total_incorrect = len(dataset) - total_correct

    print("\nClassification Statistics")
    print("-------------------------------------------------------------------------------")
    print("| Simulations               Classified:                 |  " +"%" + " Correct:         |")
    print("| Resulted in:              Correctly      Incorrectly  |  of group    of ALL |")
    print("|-----------------------------------------------------------------------------|")
    print("| SA                        {0:5.0f}            {1:5.0f}      |   {2:5.3f}       {3:5.3f} |").format(SA_c, SA_ic, SA_pc, SA_pa)
    print("| LB                        {0:5.0f}            {1:5.0f}      |   {2:5.3f}       {3:5.3f} |").format(LB_c, LB_ic, LB_pc, LB_pa)
    print("| Middle                    {0:5.0f}            {1:5.0f}      |   {2:5.3f}       {3:5.3f} |").format(MID_c, MID_ic, MID_pc, MID_pa)
    print("| TOTAL                     {0:5.0f}            {1:5.0f}      |               {2:.3f} |").format(total_correct, total_incorrect, TOTAL_perc)
    print("|-----------------------------------------------------------------------------|")
    print("| Mid-high                  {0:5.0f}            {1:5.0f}      |   {2:5.3f}       {3:5.3f} |").format(MH_c, MH_ic, MH_pc, MH_pa)
    print("| Suppression Rate (0.95 to 0.995)                      |                     |")
    print("| Mid-low                   {0:5.0f}            {1:5.0f}      |   {2:5.3f}       {3:5.3f} |").format(ML_c, ML_ic, ML_pc, ML_pa)
    print("| Suppression Rate (0.005 to 0.05)                      |                     |")
    print("-------------------------------------------------------------------------------")


#{0:.3f}.'.format(math.pi)

"""
Example Dataset for testing
ds = [[1.0,  1, 0], #In SA, correct
 [0.999,1, 0], #In SA, correct
 [0.999,1, 0], #In SA, correct
 [1.0,  0, 0], #In SA, incorrect
 [0.998,0 ,1], #In SA, incorrect
 [0.999,1 ,1], #In SA, incorrect (SORT OF...)
 [0.98, 0, 0], #In mid_high, correct
 [0.97, 0, 0], #In mid_high, correct
 [0.966,0, 0], #In mid_high, correct
 [0.97, 1, 0], #In mid_high, incorrect
 [0.96, 0, 1], #In mid_high, incorrect
 [0.96, 1, 1], #In mid_high, incorrect
 [0.94, 0, 0], #In mid, correct
 [0.43, 0, 0], #In mid, correct
 [0.29, 0, 0], #In mid, correct
 [0.83, 1, 0], #in mid, incorrect 
 [0.54, 0, 1], #in mid, incorrect
 [0.34, 1, 1], #in mid, incorrect
 [0.04, 0, 0], #In mid_low, correct
 [0.03, 0, 0], #In mid_low, correct
 [0.02, 0, 0], #In mid_low, correct
 [0.03, 0, 1], #in mid_low, incorrect 
 [0.02, 1, 0], #in mid_low, incorrect
 [0.01, 1, 1], #in mid_low, incorrect 
 [0.0,  0, 1], #in LB, correct
 [0.0,  0, 1], #in LB, correct
 [0.0,  0, 1], #in LB, correct
 [0.0,  1, 0], #in LB, incorrect
 [0.0,  0, 0], #in LB, incorrect
 [0.0,  1, 1] #in LB, incorrect (SORT OF...)
]
"""