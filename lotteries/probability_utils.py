import numpy as np


def claimant_probabilities(
    claimant_mat: np.array, group_probabilities: np.array
) -> np.array:
    """
    Simple matrix multiplication. The function just exists to make code more readable
    :param claimant_mat: matrix defining the rescue dilemma
    :param group_probabilities: probabilities for saving the individual groups
    :return: probabilities that the individual claimants are saved
    """
    return claimant_mat.dot(group_probabilities)
