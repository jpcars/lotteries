import random

import numpy as np
from scipy.special import comb
from itertools import combinations


def create_size_k_subsets(
    claimant_list: list, k: int, variant: str = "all", number_groups: int = 0
):
    number_claimants = len(claimant_list)
    if variant == "all":
        subsets = [list(i) for i in combinations(claimant_list, r=k)]
    elif variant == "random":
        if number_groups < comb(N=number_claimants, k=k, exact=True):
            subsets = random.sample(combinations(claimant_list, r=k), number_groups)
        else:
            raise ValueError(
                f"{number_groups=} is larger than the number of distinct combinations of length {k=}"
            )
    elif variant.endswith("ordered_disjoint_cover"):
        if number_claimants % k != 0:
            raise ValueError(
                "Disjoint cover doesn't exist, because number of claimants is not divisible by group size"
            )
        else:
            subsets = [
                [claimant_list[index] for index in range(i, i + k)]
                for i in range(0, number_claimants, k)
            ]
    return subsets


def small_large_example(claimants, small_group_size, large_group_size):
    if isinstance(claimants, int):
        claimants = list(range(claimants))
    elif isinstance(claimants, list):
        pass
    else:
        raise ValueError(f"claimants must be of type int or list.")
    if not small_group_size < large_group_size:
        raise ValueError(
            f"small_group_size has to be smaller than large_group_size, but received {small_group_size=} and {large_group_size=}"
        )
    result = create_size_k_subsets(
        claimants, small_group_size, variant="all"
    ) + create_size_k_subsets(
        claimants, large_group_size, variant="ordered_disjoint_cover"
    )
    return result


def array_has_no_identical_columns(arr: np.array):
    return not any([np.array_equal(arr[:, pair[0]], arr[:, pair[1]]) for pair in combinations(range(arr.shape[1]), 2)])


class RandomDilemma:
    def __init__(self, number_claimants, number_groups):
        self.number_claimants = number_claimants
        self.number_groups = number_groups
        self.claimant_mat = np.random.randint(2, size=(self.number_claimants, self.number_groups))
        self.check_valid_dilemma(self.claimant_mat)

    def add_random_claimant(self, n_groups=None):
        if n_groups:
            ones_array = np.ones(n_groups)
            zeros_array = np.zeros(self.number_groups - n_groups)
            new_row = np.concatenate([ones_array, zeros_array])
            np.random.shuffle(new_row)
        else:
            new_row = np.random.randint(2, size=self.number_groups)
        new_claimant_mat = np.vstack((self.claimant_mat, new_row))
        self.check_valid_dilemma(new_claimant_mat)
        return new_claimant_mat

    @staticmethod
    def check_valid_dilemma(claimant_mat):
        number_claimants, number_groups = claimant_mat.shape
        try:
            assert np.all(~np.isclose(claimant_mat.sum(axis=1),
                                      number_groups)), 'There is at least one claimant that is present in every group.'
            assert np.all(
                ~np.isclose(claimant_mat.sum(axis=1), 0)), 'There is at least one claimant that is not present in any group.'
            assert np.all(~np.isclose(claimant_mat.sum(axis=0),
                                      number_claimants)), 'There is at least one group that contains every claimant.'
            assert np.all(~np.isclose(claimant_mat.sum(axis=0), 0)), 'There is at least one empty group.'
            assert array_has_no_identical_columns(claimant_mat), 'There are at least two identical groups.'
        except AssertionError:
            print(claimant_mat)
            raise


if __name__ == "__main__":
    dilemma = RandomDilemma(number_claimants=10, number_groups=5)
    print(dilemma.claimant_mat)
    print(dilemma.add_random_claimant())
