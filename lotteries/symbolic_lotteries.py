import itertools
from collections import defaultdict
from typing import Optional

import numpy as np
import pynauty as pn

from sympy.core.symbol import Symbol
from sympy import Matrix, zeros

from lotteries.read_write_utils import read_write


class Lottery:
    """
    Base class for the different lotteries
    """

    def __init__(self, claimant_mat: np.array, remove_subgroups=False):
        """

        :param claimant_mat: rows are claimants, columns are courses of actions, values are probabilities to be saved
        :param remove_subgroups: if true, all courses of action, which are entirely contained in others (i.e. every
                    claimant has at least the same probability to be saved) are deleted in the beginning
        """
        self.suborbits = None
        self.superorbits = None
        self.group_orbits = None
        self.claimant_orbits = None
        self.group_generators = None
        self.orbits = None
        self.claimant_generators = None
        self.lottery_name = None
        self.claimant_mat = claimant_mat
        self.number_claimants = self.claimant_mat.shape[0]
        self.number_groups = self.claimant_mat.shape[1]
        self.remove_subgroups = remove_subgroups
        self.unique_values = None
        # self.symbolic_matrix = None
        self.single_value_matrices = None
        self.canon_claimant_label = None
        self.canon_group_label = None
        self.inverse_canon_claimant_label = None
        self.inverse_canon_group_label = None
        self.nauty_certificate = None
        self.claims_mat = None
        self.group_probabilities = None
        self.nauty_graph = None
        self.supersets = defaultdict(set)
        self.subsets = defaultdict(set)

        self.compute_useful_matrices()
        self.construct_nauty_graph()
        self.compute_autgrp()
        self.compute_supersets()
        self.reduced_claimant_matrix = self.create_reduced_claimant_matrix()
        self.compute_canon_labels()

        self.base_claim = 1 / self.claimant_mat.shape[0]
        self.number_groups = (self.reduced_claimant_matrix.sum(axis=0) > 0).sum()

    # def probabilities(self) -> (pd.Series, pd.Series):
    #     """
    #     TODO: adjust
    #     Computes the probabilities that any particular group will win the lottery
    #     :return: series of group probabilities
    #     """
    #     group_probabilities = {}
    #     for group, properties in self.graph.edge_properties.items():
    #         group_probabilities[group] = properties.claim
    #     group_probabilities_series = pd.Series(
    #         data=group_probabilities,
    #         name=self.lottery_name
    #         + ("_subgroups_removed" if self.remove_subgroups else ""),
    #     )
    #     group_probabilities_series.sort_index(inplace=True)
    #     group_probabilities_series.index.name = "group"
    #     return group_probabilities_series

    def compute_useful_matrices(self):
        self.unique_values, inverse_indices = np.unique(
            self.claimant_mat, return_inverse=True
        )
        inverse_indices.shape = self.claimant_mat.shape
        single_value_matrices = []
        # symbolic_matrix = zeros(self.number_claimants, self.number_groups)
        for i in range(
            1, len(self.unique_values)
        ):  # start at 1 bc 0 also counts as unique value, but we don't want it
            single_value_matrices.append((inverse_indices == i) * 1)
        #     symbolic_matrix = symbolic_matrix + Matrix(
        #         (inverse_indices == i) * 1
        #     ) * Symbol(f"a{i}")
        self.single_value_matrices = single_value_matrices
        # self.symbolic_matrix = symbolic_matrix

    def construct_nauty_graph(self):
        nauty_incidence_dict = {}
        number_rows = self.number_claimants
        number_cols = self.number_groups
        vertex_count = number_rows + number_cols
        vertex_coloring = [
            {edge for edge in range(number_rows)},
            {edge for edge in range(number_rows, vertex_count)},
        ]
        for mat in self.single_value_matrices:
            color = set()
            row, col = np.where(mat == 1)
            for i, j in zip(row, col):
                nauty_incidence_dict[vertex_count] = [i, j + number_rows]
                color.add(vertex_count)
                vertex_count = vertex_count + 1
            vertex_coloring.append(color)
        self.nauty_graph = pn.Graph(
            number_of_vertices=vertex_count,
            adjacency_dict=nauty_incidence_dict,
            vertex_coloring=vertex_coloring,
        )

    def compute_canon_labels(self):
        self.nauty_certificate = pn.certificate(self.nauty_graph)
        canon_nauty_label = pn.canon_label(self.nauty_graph)
        self.canon_claimant_label = canon_nauty_label[: self.number_claimants]
        self.canon_group_label = [
            i - self.number_claimants
            for i in canon_nauty_label[
                self.number_claimants : self.number_claimants + self.number_groups
            ]
        ]
        self.inverse_canon_claimant_label = [
            self.canon_claimant_label.index(i)
            for i in range(len(self.canon_claimant_label))
        ]
        self.inverse_canon_group_label = [
            self.canon_group_label.index(i) for i in range(len(self.canon_group_label))
        ]

    def compute_generators(self, generators):
        if generators:
            generators = np.array(generators)
            self.claimant_generators = generators[:, : self.number_claimants]
            self.group_generators = (
                generators[
                    :,
                    self.number_claimants : self.number_claimants + self.number_groups,
                ]
                - self.number_claimants
            )

    @staticmethod
    def orbit_reps(orbits):
        orbit_dict = {}
        for representative in np.unique(orbits):
            orbit_dict[representative] = np.argwhere(orbits == representative).flatten()
        return orbit_dict

    def compute_orbits(self, orbits):
        orbits = np.array(orbits)
        claimant_orbits, group_orbits, _ = np.split(
            orbits, [self.number_claimants, self.number_claimants + self.number_groups]
        )
        group_orbits = group_orbits - self.number_claimants
        self.orbits = {
            "claimants": {
                "orbit_id": claimant_orbits,
                "members": self.orbit_reps(claimant_orbits),
            },
            "groups": {
                "orbit_id": group_orbits,  # array of all groups with their respective orbit ids
                "members": self.orbit_reps(group_orbits),
            },
        }

    def compute_autgrp(self):
        generators, _, _, orbits, _ = pn.autgrp(self.nauty_graph)
        self.compute_generators(generators)
        self.compute_orbits(orbits)

    def compute_supersets(self):
        A = self.claimant_mat
        if A.shape[1] == 1:
            self.subsets = {0: set()}
            self.supersets = {0: set()}
            self.suborbits = {0: set()}
            self.superorbits = {0: set()}
        else:

            def greater_equal_all(x, y):
                return np.greater_equal(x, y).all(axis=0)

            greater_equal_array = np.zeros((A.shape[1], A.shape[1]))
            permutations = np.array(list(itertools.permutations(range(A.shape[1]), 2)))

            greater_equal_array[
                permutations[:, 0], permutations[:, 1]
            ] = greater_equal_all(A[:, permutations[:, 0]], A[:, permutations[:, 1]])
            self.subsets = {
                key: set(np.argwhere(value == 1).flatten())
                for key, value in enumerate(greater_equal_array)
            }
            self.supersets = {
                key: set(np.argwhere(value == 1).flatten())
                for key, value in enumerate(greater_equal_array.T)
            }
            self.suborbits = {}
            for orbit_rep in self.orbits["groups"]["members"]:
                if self.subsets[orbit_rep]:
                    subsets_of_orbit_rep = np.array(list(self.subsets[orbit_rep]))
                    orbits_of_subsets_of_orbit_rep = self.orbits["groups"]["orbit_id"][
                        subsets_of_orbit_rep
                    ]
                    self.suborbits[orbit_rep] = set(orbits_of_subsets_of_orbit_rep)
                else:
                    self.suborbits[orbit_rep] = set()
            self.superorbits = {}
            for orbit_rep in self.orbits["groups"]["members"]:
                if self.supersets[orbit_rep]:
                    supersets_of_orbit_rep = np.array(list(self.supersets[orbit_rep]))
                    orbits_of_supersets_of_orbit_rep = self.orbits["groups"][
                        "orbit_id"
                    ][supersets_of_orbit_rep]
                    self.superorbits[orbit_rep] = set(orbits_of_supersets_of_orbit_rep)
                else:
                    self.superorbits[orbit_rep] = set()

    def create_reduced_claimant_matrix(self):
        """
        Computes reduced claimant matrix. For each group that is a subgroup of a different group, it replaces
        its entries with zeros, i.e. it pretends that all subgroups are empty.
        """
        reduced_claimant_matrix = self.claimant_mat.copy()
        if self.remove_subgroups:
            is_subgroup = []
            for group, supersets in self.supersets.items():
                if supersets:
                    is_subgroup.append(group)
            if is_subgroup:
                reduced_claimant_matrix[:, np.array(is_subgroup)] = 0
        return reduced_claimant_matrix


class GroupBasedLottery(Lottery):
    """
    Implements the general structure of an iterated group based lottery. Specific lotteries are implemented in
    subclasses.
    """

    def __init__(self, claimant_mat, remove_subgroups=False):
        super().__init__(claimant_mat, remove_subgroups)

    @read_write
    def compute(self, prob_dict=None) -> np.array:
        probabilities = self.claims_mat.sum(axis=0)
        if (self.number_groups == 1) | (self.number_claimants == 1):
            pass
        else:
            active_groups = set(range(self.number_groups))
            while len(active_groups) > 0:
                active_groups_copy = active_groups.copy()
                for group in active_groups_copy:
                    if (
                        len(self.subsets[group].intersection(active_groups)) > 0
                    ):  # don't compute groups, which still have active subgroups
                        pass
                    else:
                        # if group has supersets, iterate on the lottery only on the supersets
                        # if it doesn't: don't do anything
                        bigger_groups = sorted(list(self.supersets[group]))
                        if len(bigger_groups) > 0:
                            smaller_mat = self.claims_mat[:, bigger_groups]
                            smaller_mat = smaller_mat[
                                ~np.all(smaller_mat == 0, axis=1)
                            ]  # don't consider empty groups
                            next_lottery_iteration = self.__class__(
                                claimant_mat=smaller_mat
                            )
                            probs_from_next_iteration = next_lottery_iteration.compute(
                                prob_dict=prob_dict
                            )
                            next_lottery_iteration = None
                            probabilities[bigger_groups] = (
                                probabilities[bigger_groups]
                                + probabilities[group] * probs_from_next_iteration
                            )
                            probabilities[group] = 0
                        active_groups.remove(group)
        return probabilities

    @read_write
    def compute_on_orbits(self, prob_dict=None) -> np.array:
        probabilities = self.claims_mat.sum(axis=0)
        if len(np.unique(self.orbits["groups"]["orbit_id"])) == 1:
            pass
        else:
            active_orbit_reps = set(self.orbits["groups"]["members"].keys())
            while len(active_orbit_reps) > 0:
                active_orbits_copy = active_orbit_reps.copy()
                for orbit in active_orbits_copy:
                    if (
                        len(self.suborbits[orbit].intersection(active_orbit_reps)) > 0
                    ):  # don't compute orbits, which still have active suborbits
                        pass
                    else:
                        # if orbits has superorbits, iterate on the lottery only on the superorbits
                        # if it doesn't: don't do anything
                        bigger_groups = sorted(list(self.supersets[orbit]))
                        bigger_orbits = self.orbits["groups"]["orbit_id"][bigger_groups]
                        if len(bigger_groups) > 0:
                            smaller_mat = self.claims_mat[:, bigger_groups]
                            smaller_mat = smaller_mat[~np.all(smaller_mat == 0, axis=1)]
                            next_lottery_iteration = self.__class__(
                                claimant_mat=smaller_mat
                            )
                            probs_from_next_iteration = (
                                next_lottery_iteration.compute_on_orbits(
                                    prob_dict=prob_dict
                                )
                            )
                            next_lottery_iteration = None
                            orbit_prob = {}
                            for bigger_orbit, prob in zip(
                                bigger_orbits, probs_from_next_iteration
                            ):
                                if bigger_orbit not in orbit_prob:
                                    orbit_prob[bigger_orbit] = prob
                                else:
                                    orbit_prob[bigger_orbit] += prob
                            for bigger_orbit in orbit_prob:
                                for group in self.orbits["groups"]["members"][
                                    bigger_orbit
                                ]:
                                    probabilities[group] = probabilities[
                                        group
                                    ] + probabilities[orbit] * orbit_prob[
                                        bigger_orbit
                                    ] * len(
                                        self.orbits["groups"]["members"][orbit]
                                    ) / len(
                                        self.orbits["groups"]["members"][bigger_orbit]
                                    )
                            for group in self.orbits["groups"]["members"][orbit]:
                                probabilities[group] = 0
                        active_orbit_reps.remove(orbit)
        return probabilities


class EXCSLottery(GroupBasedLottery):
    """
    Implements the Exclusive Composition-sensitive lottery
    """

    def __init__(self, claimant_mat, remove_subgroups=False):
        super().__init__(claimant_mat, remove_subgroups)
        self.lottery_name = "EXCS"
        self.distributionally_relevant_in_group = None
        self.exclusivity_relations()
        self.claims()

    def exclusivity_relations(self):
        """
        computes matrix for distributional relevance. if entry (i,j) is 1, then claimant i
        should take claimant j into account in distributing their claim
        """
        A = self.reduced_claimant_matrix
        AT = A.transpose()
        divisor = np.column_stack([A.sum(axis=1) for i in range(A.shape[1])])
        condition = (~np.isclose(np.matmul(A / divisor, AT), 0)) & (
            ~np.isclose(np.matmul(A / divisor, AT), 1)
        )
        exclusivity = np.where(condition, 1, 0)
        self.distributionally_relevant_in_group = np.multiply(
            np.matmul(exclusivity, A), A
        )

    def claims(self):
        """
        Compute the non-iterated distributions of claims from the claimants to the groups
        """
        n_groups = self.reduced_claimant_matrix.sum(axis=1)
        total_distributionally_relevant = self.distributionally_relevant_in_group.sum(
            axis=1
        )
        self.claims_mat = (
            np.row_stack(
                [
                    (
                        self.reduced_claimant_matrix[claimant, :]
                        if total_exclusives == 0
                        else self.distributionally_relevant_in_group[claimant, :]
                        / total_exclusives
                    )
                    for claimant, (groups, total_exclusives) in enumerate(
                        zip(n_groups, total_distributionally_relevant)
                    )
                ]
            )
            * self.base_claim
        )


class EQCSLottery(GroupBasedLottery):
    """
    Implements the Equal Composition-Sensitive lottery
    """

    def __init__(self, claimant_mat, remove_subgroups=False):
        super().__init__(claimant_mat, remove_subgroups)
        self.lottery_name = "EQCS"
        self.claims()

    def claims(self):
        """
        Compute the non-iterated distributions of claims from the claimants to the groups
        """
        row_sums = self.reduced_claimant_matrix.sum(axis=1)
        self.claims_mat = (
            self.reduced_claimant_matrix / row_sums[:, np.newaxis] * self.base_claim
        )


class TaurekLottery(GroupBasedLottery):
    """
    Implements Taurek's coin toss.
    """

    def __init__(self, claimant_mat, remove_subgroups=False):
        super().__init__(claimant_mat, remove_subgroups)
        self.lottery_name = "TAUREK"
        self.claims()

    def claims(self):
        """
        Compute the non-iterated distributions of claims from the claimants to the groups
        There are many matrices that result in an equal chances coin toss for the outcome groups
        We choose the one that equally divides the group claim of 1/number_groups among all claimants, which are part
        of the group.
        """
        col_sums = self.reduced_claimant_matrix.sum(axis=0)
        divisor = np.transpose(col_sums[:, np.newaxis]) * self.number_groups
        divisor = np.where(divisor == 0, np.nan, divisor)
        self.claims_mat = np.where(
            np.transpose(col_sums[:, np.newaxis]) != 0,
            self.reduced_claimant_matrix / divisor,
            0,
        )


# class ClaimantBasedLottery(Lottery):
#     """
#     Implements the general structure of an iterated claimant based lottery. Specific lotteries are implemented in
#     subclasses.
#     """
#
#     def __init__(self, claimant_mat, remove_subgroups=False):
#         super().__init__(claimant_mat, remove_subgroups)
#
#     @read_write
#     def compute(self, prob_dict=None) -> np.array:
#         probabilities = self.claims_mat.sum(axis=0)
#         if (self.number_groups == 1) | (self.number_claimants == 1):
#             pass
#         else:
#             active_groups = set(range(self.number_groups))
#             while len(active_groups) > 0:
#                 active_groups_copy = active_groups.copy()
#                 for group in active_groups_copy:
#                     if (
#                         len(self.subsets[group].intersection(active_groups)) > 0
#                     ):  # don't compute groups, which still have active subgroups
#                         pass
#                     else:
#                         # if group has supersets, iterate on the lottery only on the supersets
#                         # if it doesn't: don't do anything
#                         bigger_groups = sorted(list(self.supersets[group]))
#                         if len(bigger_groups) > 0:
#                             smaller_mat = self.claims_mat[:, bigger_groups]
#                             smaller_mat = smaller_mat[
#                                 ~np.all(smaller_mat == 0, axis=1)
#                             ]  # don't consider empty groups
#                             next_lottery_iteration = self.__class__(
#                                 claimant_mat=smaller_mat
#                             )
#                             probs_from_next_iteration = next_lottery_iteration.compute(
#                                 prob_dict=prob_dict
#                             )
#                             next_lottery_iteration = None
#                             probabilities[bigger_groups] = (
#                                 probabilities[bigger_groups]
#                                 + probabilities[group] * probs_from_next_iteration
#                             )
#                             probabilities[group] = 0
#                         active_groups.remove(group)
#         return probabilities


def run_lottery():
    n_claimants = 6
    ones = [1 for i in range(int(n_claimants / 2))]
    zeroes = [0 for i in range(int(n_claimants / 2))]
    my_array = np.transpose(np.array([ones + zeroes, zeroes + ones]))
    for i in range(n_claimants):
        for j in range(i + 1, n_claimants):
            newcol = np.zeros(shape=(n_claimants, 1))
            newcol[i] = 1
            newcol[j] = 1
            my_array = np.hstack([my_array, newcol])

    # my_array = np.array([[1,0,1,1,1],[1,0,0,0,0],[0,1,1,1,1],[0,1,0,0,1],[0,0,0,1,0]])
    lottery1 = TaurekLottery(claimant_mat=my_array, remove_subgroups=False)
    print(lottery1.compute())
    # with np.printoptions(linewidth=np.inf):
    #     print(lottery1.claims_mat)
    #     print(lottery1.claims_mat.sum(axis=0).sum())
    # lottery2 = EXCSLottery(claimant_mat=my_array)
    # prob_dict1 = {}
    # prob_dict2 = {}
    # probs1 = lottery1.compute(prob_dict=prob_dict1)
    # probs2 = lottery2.compute_on_orbits(prob_dict=prob_dict2)
    # print(probs1)
    # print(probs2)
    # print(prob_dict1)
    # print(prob_dict2)

    my_dict = {}
    # lottery.compute(prob_dict=my_dict)


# def main():
#     import cProfile
#     import pstats
#
#     with cProfile.Profile() as pr:
#         run_lottery()
#
#     stats = pstats.Stats(pr)
#     stats.sort_stats(pstats.SortKey.TIME)
#     stats.print_stats()


if __name__ == "__main__":
    # import lotteries.lottery as other_lottery
    # groupie = [[1, 2], [3, 4], [1, 3], [1, 3, 5], [1, 3, 4]]
    # result_dict = {}
    # lottery = other_lottery.EXCSLottery(groupie, pruned=True)
    # lottery.compute()
    # for group in lottery.groups["inactive"].values():
    #     print(f"{group.name=}, {group.claim}\n")
    # claimant_mat = np.array([[1,0,1,1,1],[1,0,0,0,0],[0,1,1,1,1],[0,1,0,0,1],[0,0,0,1,0]])
    # lottery = EXCSLottery(claimant_mat=claimant_mat, remove_subgroups=True)
    # prob_dict = {}
    # print(lottery.compute(prob_dict=prob_dict))
    run_lottery()
