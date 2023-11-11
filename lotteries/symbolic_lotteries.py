from collections import defaultdict
from itertools import permutations

import pandas as pd
import numpy as np
import pynauty as pn

from sympy.core.symbol import Symbol
from sympy import Matrix, zeros


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
        self.lottery_name = None
        self.claimant_mat = claimant_mat
        self.number_claimants = self.claimant_mat.shape[0]
        self.number_groups = self.claimant_mat.shape[1]
        self.remove_subgroups = remove_subgroups
        self.binary_membership_matrix = None
        self.unique_values = None
        self.symbolic_matrix = None
        self.single_value_matrices = None
        self.canon_claimant_label = None
        self.canon_group_label = None
        self.nauty_certificate = None
        self.claims_mat = None
        self.supersets = defaultdict(set)
        self.subsets = defaultdict(set)

        self.compute_supersets()
        self.compute_useful_matrices()
        self.compute_canon_labels()

        self.base_claim = 1 / self.claimant_mat.shape[0]

    def probabilities(self) -> (pd.Series, pd.Series):
        """
        TODO: adjust
        Computes the probabilities that any particular group will win the lottery
        :return: series of group probabilities
        """
        group_probabilities = {}
        for group, properties in self.graph.edge_properties.items():
            group_probabilities[group] = properties.claim
        group_probabilities_series = pd.Series(
            data=group_probabilities,
            name=self.lottery_name
            + ("_subgroups_removed" if self.remove_subgroups else ""),
        )
        group_probabilities_series.sort_index(inplace=True)
        group_probabilities_series.index.name = "group"
        return group_probabilities_series

    def compute_useful_matrices(self):
        self.binary_membership_matrix = (~np.isclose(self.claimant_mat, 0)) * 1
        self.unique_values, inverse_indices = np.unique(
            self.claimant_mat, return_inverse=True
        )
        inverse_indices.shape = self.claimant_mat.shape
        single_value_matrices = []
        symbolic_matrix = zeros(self.number_claimants, self.number_groups)
        for i in range(len(self.unique_values)):
            single_value_matrices.append((inverse_indices == i) * 1)
            symbolic_matrix = symbolic_matrix + Matrix(
                (inverse_indices == i) * 1
            ) * Symbol(f"a{i}")
        self.single_value_matrices = single_value_matrices
        self.symbolic_matrix = symbolic_matrix

    def compute_canon_labels(self):
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
        nauty_graph = pn.Graph(
            number_of_vertices=vertex_count,
            adjacency_dict=nauty_incidence_dict,
            vertex_coloring=vertex_coloring,
        )
        self.nauty_certificate = pn.certificate(nauty_graph)
        canon_nauty_label = pn.canon_label(nauty_graph)
        self.canon_claimant_label = canon_nauty_label[:number_rows]
        self.canon_group_label = [
            i - number_rows
            for i in canon_nauty_label[number_rows : number_rows + number_cols]
        ]

    def store_values(self):
        """
        If the certificate is not yet present, write a new entry, else do nothing
        :return:
        """
        pass

    def retrieve_values(self, lottery_name=None):
        """
        Check whether there is an entry for the certificate. If so, retrieve results and do backwards translation
        :return:
        """
        return False

    def compute_supersets(self):
        A = self.claimant_mat
        supersets = {}
        subsets = {}
        for group1 in range(self.number_groups):
            supersets_of_group1 = set()
            subsets_of_group1 = set()
            for group2 in range(self.number_groups):
                if ((A[:, group1] <= A[:, group2]).all()) & (group1 != group2):
                    supersets_of_group1.add(group2)
                if ((A[:, group2] <= A[:, group1]).all()) & (group1 != group2):
                    subsets_of_group1.add(group2)
            supersets[group1] = supersets_of_group1
            subsets[group1] = subsets_of_group1
        self.supersets = supersets
        self.subsets = subsets


class GroupBasedLottery(Lottery):
    """
    Implements the general structure of an iterated group based lottery. Specific lotteries are implemented in
    subclasses.
    """

    def __init__(self, claimant_mat, remove_subgroups=False):
        super().__init__(claimant_mat, remove_subgroups)

    def compute(self):
        if self.retrieve_values(lottery_name=self.lottery_name):
            # TODO: implement
            pass
        else:
            active_groups = set(range(self.number_groups))
            if (self.number_groups == 1) | (self.number_claimants == 1):
                return self.claims_mat.sum(axis=0)
            else:
                probabilities = self.claims_mat.sum(axis=0)
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
                                ]
                                probs_from_next_iteration = self.__class__(
                                    claimant_mat=smaller_mat
                                ).compute()
                                probabilities[bigger_groups] = (
                                    probabilities[bigger_groups]
                                    + probabilities[group] * probs_from_next_iteration
                                )
                                probabilities[group] = 0
                            active_groups.remove(group)
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
        A = self.binary_membership_matrix
        AT = A.transpose()
        divisor = np.column_stack([A.sum(axis=1) for i in range(A.shape[1])])
        condition = (np.matmul(A / divisor, AT) > 0) & (np.matmul(A / divisor, AT) < 1)
        exclusivity = np.where(condition, 1, 0)
        self.distributionally_relevant_in_group = np.multiply(
            np.matmul(exclusivity, A), A
        )

    def claims(self):
        """
        Compute the non-iterated distributions of claims from the claimants to the groups
        """
        n_groups = self.binary_membership_matrix.sum(axis=1)
        total_distributionally_relevant = self.distributionally_relevant_in_group.sum(
            axis=1
        )
        self.claims_mat = (
            np.row_stack(
                [
                    (
                        self.binary_membership_matrix[claimant, :]
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
        row_sums = self.binary_membership_matrix.sum(axis=1)
        self.claims_mat = self.binary_membership_matrix / row_sums[:, np.newaxis] * self.base_claim


if __name__ == "__main__":
    claimant_mat = np.array(
        [
            [1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ]
    )
    lottery = EQCSLottery(claimant_mat=claimant_mat)
    probs = lottery.compute()
    print(probs)
