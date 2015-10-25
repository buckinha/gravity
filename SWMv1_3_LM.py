#SWM v1.3 Linear Model


def lm(x,CONS,P0,P1):
    """Expects x as a [k by M] numpy matrix"""
    return P0 * x[0] + P1 * x[1] + CONS