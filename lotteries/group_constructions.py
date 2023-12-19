import random

import numpy as np
from scipy.special import comb
from itertools import combinations, product


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
    """
    This class creates random rescue dilemmas. It contains checks for making sure they are actually valid.
    """
    def __init__(self, number_claimants, number_groups, max_tries = 20):
        self.claimant_mat = None
        self.number_claimants = number_claimants
        self.number_groups = number_groups
        self.max_tries = max_tries
        self.size_error()
        self.create_claimant_mat()

    def size_error(self):
        """
        This class is not fit for large dilemmas. This method throws an error if either number_claimants of number_groups
        is too large.
        :return: None
        """
        max_size = 10
        if (self.number_claimants > max_size) or (self.number_groups > max_size):
            raise ValueError(f'Please only use values <={max_size} for number_claimants and number_groups.')

    def claimant_mat_candidate(self):
        """
        Create a candidate for a claimant matrix. This matrix does not necessarily fulfil the conditions that
        there are no claimants, which appear in each or in no group.

        Only use for moderate numbers of claimants and groups.

        :return: claimant_mat (numpy.array) - a claimant matrix of shape=(self.number_claimants, self.number_groups)
        """
        all_groups = [list(tup) for tup in product(range(2), repeat=self.number_claimants)][1:-1]  # [1:-1] gets rid of the entries with all zeros and all ones # noqa: E501
        all_groups = np.array(all_groups)
        rng = np.random.default_rng()
        claimant_mat = np.transpose(rng.choice(all_groups, size=self.number_groups, replace=False))
        return claimant_mat

    @staticmethod
    def check_valid_dilemma(claimant_mat: np.array):
        """
        Check whether claimant_mat fulfils all criteria for the associated rescue dilemma to be valid.
        :param claimant_mat: candidate for a claimant matrix defining a rescue dilemma
        :return: bool, True if claimant_mat is valid, else False
        """
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
            return True
        except AssertionError as e:
            return False

    def create_claimant_mat(self):
        """
        Try to create valid rescue dilemmas multiple times until it succeeds or the max number of tries is exceeded.
        Store claimant_mat in attribute self.claimant_mat
        :return: None
        """
        for i in range(self.max_tries):
            candidate = self.claimant_mat_candidate()
            if self.check_valid_dilemma(candidate):
                self.claimant_mat = self.claimant_mat_candidate()
                break

    def add_random_claimant(self, n_groups=None):
        """
        Given the rescue dilemma self.claimant_mat add a new claimant, who is a member of some of the groups.
        Which groups the claimant is added to is decided randomly.
        :param n_groups: number of groups the claimant will be added to
        :return: new claimant_mat
        """
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


if __name__ == "__main__":
    dilemma = RandomDilemma(number_claimants=10, number_groups=10)
    print(dilemma.claimant_mat)
