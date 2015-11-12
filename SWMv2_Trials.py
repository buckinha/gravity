"""SWMv2 Trials"""

import MDP, MDP_opt, SWMv2, HKB_Heuristics, random, numpy, datetime
import os.path
from sphere_dist import sphere_dist

def standard_MDP_set(pathway_count, timesteps, policy, random_seed=0, VALUE_ON_HABITAT=False, percentage_habitat=0, sampling_radius=0.0):
    """
    Generates a set of SWM v2 pathways and returns them as a list of MDP pathway objects

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
    original_pol = SWMv2.sanitize_policy(policy)


    for i in range(pathway_count):

        #do policy perturbation
        pol = sphere_dist(center=original_pol, radius=sampling_radius, random_seed=random_seed+i)

        #run a simulation
        pw = SWMv2.simulate(timesteps, pol, random_seed=i+8500+random_seed, SILENT=True)

        #add the simulation to the list after converting it's form
        pathways[i] = SWMv2.convert_to_MDP_pathway(pw, VALUE_ON_HABITAT=VALUE_ON_HABITAT, percentage_habitat=percentage_habitat)

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
    
    original_pol = SWMv2.sanitize_policy(policy)

    i = 0
    for count in range(fail_at_count):

        #do policy perturbation
        pol = sphere_dist(center=original_pol, radius=sampling_radius, random_seed=random_seed+i)

        #run the pathway simulation
        pw = SWMv2.simulate(timesteps, pol, random_seed=i+8500+random_seed, SILENT=True)


        #check the suppression rate
        if (pw["Suppression Rate"] == 0) or (pw["Suppression Rate"] == 1):
            #this pathway needs to be rejected:
            #  keep i as it is, and continue the loop
            pass
        else:
            #this pathway can be accepted
            pathways[i] = SWMv2.convert_to_MDP_pathway(pw,VALUE_ON_HABITAT=VALUE_ON_HABITAT,percentage_habitat=percentage_habitat)
            i += 1

            #check for i being in range
            if i >= pathway_count:
                #we've filled the array, so quit
                break

    #we've exitted the loop. Check to see if the array was filled:
    if i < pathway_count:
        #we didn't fill the set, so... do what?
        print("WARNING: limited_MDP_set() could only find " + str(i) + " non-plateau policies...")
        pathways = pathways[:i]

    return pathways


