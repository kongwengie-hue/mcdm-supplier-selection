import numpy as np

def entropy_weight(matrix):
    """Entropy method to calculate weights"""
    P = matrix / matrix.sum(axis=0)
    E = - (P * np.log(P + 1e-12)).sum(axis=0) / np.log(len(matrix))
    d = 1 - E
    w = d / d.sum()
    return w

def topsis(matrix, weights):
    """TOPSIS ranking method"""
    norm = matrix / np.sqrt((matrix**2).sum(axis=0))
    weighted = norm * weights

    ideal_best = weighted.max(axis=0)
    ideal_worst = weighted.min(axis=0)

    dist_best = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst)
    ranking = np.argsort(-score) + 1  # rank starts at 1
    return score, ranking
