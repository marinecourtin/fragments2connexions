import numpy as np
import itertools

class Fragment:
    # span : list(int)
    # autonomy : float
    # rank : int
    def __init__(self, span, autonomy, rank):
        self.span = span
        self.autonomy = autonomy
        self.rank = rank


f = Fragment([0,1,2], 0.8, 0)
f2 = Fragment([1,2], 0.2, 3)

class Parametrization:
    def __init__(self, w_auto, w_rank):
        self.w_auto = w_auto
        self.w_rank = w_rank

p = Parametrization(1, 0.9)

def rank_score(rank, w_rank):
    if rank == 0:
        return 1
    else:
        return w_rank * rank_score(rank-1, w_rank)


def fragments2matrix(length, fragment_list, parameters):
    # length : int
    # fragment_list : list(Fragment)
    # parameters : Parametrization
    ## m : np.ndarray
    m = np.zeros((length, length))
    for f in fragment_list:
        for (i,j) in itertools.combinations(f.span, 2):
            # print(parameters.w_auto * f.autonomy + rank_score(f.rank, parameters.w_rank))
            m[i,j] += parameters.w_auto * f.autonomy + rank_score(f.rank, parameters.w_rank)
            m[j,i] += parameters.w_auto * f.autonomy + rank_score(f.rank, parameters.w_rank)
    return m

