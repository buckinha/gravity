#sphere_dist()

import scipy.spatial.distance
import random

def sphere_dist(center,radius,random_seed=None):
    """Returns single points distributed uniformly from within a sphere 

    ARGUEMENTS
    center: a list containing the coordinates of the center of the sphere
    radius: the radius of the sphere. Absolute values are taken for negative numbers.
    random_seed: any value for re-seeding the random number generator. Use if you need
     replicablity, etc...  Default value is "None" which allows the system to seed the
     generator as it likes.

    RETURNS
    a float, drawn uniformly from within the sphere. Using a draw-discard method
    """

    #enforce strictly non-negative radii
    radius = abs(radius)

    x = [0.0] * len(center)

    while True:
        #draw a value from a sphere-inscribing cube
        for i in range(len(center)):
            x[i] = random.uniform(center[i] - radius, center[i]+radius)

        #check the radius. If it is within the sphere, break, and return x. If it isn't
        # continue the loop.
        if scipy.spatial.distance.euclidean(center, x) <= radius:
            break

    return x


