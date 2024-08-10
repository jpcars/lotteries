import itertools
from collections import defaultdict

import numpy as np
import pynauty as pn
from sympy import Matrix, Symbol, zeros

from lotteries.database.connect import execute_command
from lotteries.read_write_utils import read_write


class Lottery:
    """
    Base class for the different lotteries
    """

    def __init__(
        self,
        claimant_mat: np.array,
        remove_subgroups: bool = False,
        use_db_access: bool = False,
        has_uncertainty: bool = None,
    ):
        """

        :param claimant_mat: rows are claimants, columns are courses of actions, values are probabilities to be saved
        :param remove_subgroups: if true, all courses of action, which are entirely contained in others (i.e. every
            claimant has at least the same probability to be saved) are deleted in the beginning
        :param use_db_access: if set to true available, values are read from the database whenever possible and new
            values are written to it
        :param has_uncertainty: indicates whether the dilemma contains uncertainty. Normally this is determined
            automatically from the contents of claimant_mat. This parameter is only explicitly used in the recursive
            computation, because it might happen that a dilemma has uncertainty but not all its subdilemmas do.
            Setting this parameter makes sure that all subdilemmas are also treated as having uncertainty.


        """
        self.has_uncertainty = None
        self.use_db_access = use_db_access
        self.suborbits = None
        self.superorbits = None
        self.group_orbits = None
        self.claimant_orbits = None
        self.group_generators = None
        self.orbits = None
        self.claimant_generators = None
        self.lottery_code = None
        self.lottery_name = None
        self.claimant_mat = claimant_mat
        self.claimant_mat_identical_columns()
        self.number_claimants = self.claimant_mat.shape[0]
        self.number_groups = self.claimant_mat.shape[1]
        self.remove_subgroups = remove_subgroups
        self.unique_values = None
        self.symbolic_matrix = None
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

        self.compute_useful_matrices(force_uncertainty=has_uncertainty)
        self.construct_nauty_graph()
        self.compute_autgrp()
        self.compute_supersets()
        (
            self.reduced_claimant_matrix,
            self.reduced_number_groups,
        ) = self.create_reduced_claimant_matrix()
        self.compute_canon_labels()

        self.base_claim = 1 / self.number_claimants
        # self.number_groups = (self.reduced_claimant_matrix.sum(axis=0) > 0).sum()

    def register_lottery_in_db(self):
        """
        Checks whether lottery already exists in db and creates a record if it doesn't. It does not register a new
        lottery if either the lottery_code or the lottery_name (or both) already exist.
        TODO: Should this method be used to be able to update the name of a lottery?
        :return:
        """
        if (self.lottery_code is None) or (self.lottery_name is None):
            raise ValueError(
                "Both lottery_code and lottery_name must be set to register the lottery."
            )
        else:
            execute_command(
                command=f"""
                INSERT INTO dim_lotteries (lottery_code, lottery_name) VALUES (%s,%s) ON CONFLICT DO NOTHING;
                """,
                values=[self.lottery_code, self.lottery_name],
            )

    def claimant_mat_identical_columns(self):
        if any(
            [
                np.array_equal(
                    self.claimant_mat[:, pair[0]], self.claimant_mat[:, pair[1]]
                )
                for pair in itertools.combinations(range(self.claimant_mat.shape[1]), 2)
            ]
        ):
            raise ValueError(
                "Dilemmas with identical groups are currently not implemented."
            )

    def compute_useful_matrices(self, force_uncertainty):
        self.unique_values, inverse_indices = np.unique(
            self.claimant_mat, return_inverse=True
        )
        if force_uncertainty:
            self.has_uncertainty = True
        else:
            self.has_uncertainty = (
                (self.unique_values != 0) & (self.unique_values != 1)
            ).any()
        inverse_indices.shape = self.claimant_mat.shape
        single_value_matrices = []
        for i in range(
            1, len(self.unique_values)
        ):  # start at 1 bc 0 also counts as unique value, but we don't want it
            single_value_matrices.append((inverse_indices == i) * 1)
        self.single_value_matrices = single_value_matrices
        if len(self.single_value_matrices) > 1:
            self.symbolic_matrix = sum(
                (
                    Matrix(value) * Symbol(f"a{index}")
                    for index, value in enumerate(self.single_value_matrices)
                ),
                zeros(self.number_claimants, self.number_groups),
            )

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
        mat = self.claimant_mat
        if mat.shape[1] == 1:
            self.subsets = {0: set()}
            self.supersets = {0: set()}
            self.suborbits = {0: set()}
            self.superorbits = {0: set()}
        else:

            def greater_equal_all(x, y):
                return np.greater_equal(x, y).all(axis=0)

            greater_equal_array = np.zeros((mat.shape[1], mat.shape[1]))
            permutations = np.array(
                list(itertools.permutations(range(mat.shape[1]), 2))
            )

            greater_equal_array[
                permutations[:, 0], permutations[:, 1]
            ] = greater_equal_all(
                mat[:, permutations[:, 0]], mat[:, permutations[:, 1]]
            )
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
        number_groups_to_subtract = 0
        if self.remove_subgroups:
            is_subgroup = []
            for group, supersets in self.supersets.items():
                if supersets:
                    is_subgroup.append(group)
            number_groups_to_subtract = len(is_subgroup)
            if is_subgroup:
                reduced_claimant_matrix[:, np.array(is_subgroup)] = 0
        number_relevant_groups = self.number_groups - number_groups_to_subtract
        return reduced_claimant_matrix, number_relevant_groups


class GroupBasedLottery(Lottery):
    """
    Implements the general structure of an iterated group based lottery. Specific lotteries are implemented in
    subclasses.
    """

    def __init__(
        self,
        claimant_mat,
        remove_subgroups: bool = False,
        use_db_access: bool = True,
        has_uncertainty: bool = None,
    ):
        super().__init__(claimant_mat, remove_subgroups, use_db_access, has_uncertainty)

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
                            probabilities[bigger_groups] = (
                                probabilities[bigger_groups]
                                + probabilities[group] * probs_from_next_iteration
                            )
                            probabilities[group] = 0
                        active_groups.remove(group)
        return probabilities

    @read_write
    def compute_on_orbits(self) -> np.array:
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
                        # if orbit has superorbits, iterate the lottery only on the superorbits
                        # if it doesn't: don't do anything
                        bigger_groups = sorted(list(self.supersets[orbit]))
                        bigger_orbits = self.orbits["groups"]["orbit_id"][bigger_groups]
                        if len(bigger_groups) > 0:
                            # Note, we still use self.claimant_mat here instead of self.reduced_claimant_matrix
                            # we can do this because the exclusion of subgroups from consideration is handled by the
                            # claims_mat
                            smaller_mat = self.claimant_mat[:, bigger_groups]
                            smaller_mat = smaller_mat[~np.all(smaller_mat == 0, axis=1)]
                            next_lottery_iteration = self.__class__(
                                claimant_mat=smaller_mat,
                                remove_subgroups=self.remove_subgroups,
                                use_db_access=self.use_db_access,
                            )
                            probs_from_next_iteration = (
                                next_lottery_iteration.compute_on_orbits()
                            )
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

    @read_write
    def symbolic_compute_on_orbits(self) -> np.array:
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
                        # if orbit has superorbits, iterate the lottery only on the superorbits
                        # if it doesn't: don't do anything
                        bigger_groups = sorted(list(self.supersets[orbit]))
                        bigger_orbits = self.orbits["groups"]["orbit_id"][bigger_groups]
                        if len(bigger_groups) > 0:
                            # Note, we still use self.claimant_mat here instead of self.reduced_claimant_matrix
                            # we can do this because the exclusion of subgroups from consideration is handled by the
                            # claims_mat
                            smaller_mat = self.symbolic_matrix[:, bigger_groups]
                            smaller_mat = smaller_mat[~np.all(smaller_mat == 0, axis=1)]
                            next_lottery_iteration = self.__class__(
                                claimant_mat=smaller_mat,
                                remove_subgroups=self.remove_subgroups,
                                use_db_access=self.use_db_access,
                                has_uncertainty=self.has_uncertainty,
                            )
                            probs_from_next_iteration = (
                                next_lottery_iteration.symbolic_compute_on_orbits()
                            )
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

    def __init__(
        self,
        claimant_mat,
        remove_subgroups=False,
        use_db_access: bool = True,
        has_uncertainty: bool = None,
    ):
        super().__init__(claimant_mat, remove_subgroups, use_db_access, has_uncertainty)
        self.lottery_code = "EXCS"
        self.lottery_name = "Exclusive composition-sensitive lottery"
        self.register_lottery_in_db()
        self.distributionally_relevant_in_group = None
        self.exclusivity_relations()
        self.claims()

    def exclusivity_relations(self):
        """
        computes matrix for distributional relevance. if entry (i,j) is 1, then claimant i
        should take claimant j into account in distributing their claim
        """
        mat = self.reduced_claimant_matrix
        mat_transp = mat.transpose()
        divisor = np.column_stack([mat.sum(axis=1) for i in range(mat.shape[1])])
        condition = (~np.isclose(np.matmul(mat / divisor, mat_transp), 0)) & (
            ~np.isclose(np.matmul(mat / divisor, mat_transp), 1)
        )
        exclusivity = np.where(condition, 1, 0)
        self.distributionally_relevant_in_group = np.multiply(
            np.matmul(exclusivity, mat), mat
        )

    def claims(self):
        """
        Compute the non-iterated distributions of claims from the claimants to the groups
        """
        if self.has_uncertainty:
            raise NotImplementedError(
                f"Uncertainty is not yet implemented for {self.__name__}."
            )
        else:
            n_groups = self.reduced_claimant_matrix.sum(axis=1)
            total_distributionally_relevant = (
                self.distributionally_relevant_in_group.sum(axis=1)
            )
            self.claims_mat = (
                np.row_stack(
                    [
                        (
                            # If a claimant has no claimants that are distributionally relevant for them,
                            # they just divide their claim equally among the groups they are part of.
                            # Usually this means that they are alone in 1 group or that all the groups
                            # they are part of are identical.
                            self.reduced_claimant_matrix[claimant, :] / groups
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

    def __init__(
        self,
        claimant_mat,
        remove_subgroups=False,
        use_db_access: bool = True,
        has_uncertainty: bool = None,
    ):
        super().__init__(claimant_mat, remove_subgroups, use_db_access, has_uncertainty)
        self.lottery_code = "EQCS"
        self.lottery_name = "Equal composition-sensitive lottery"
        self.register_lottery_in_db()
        self.claims()

    def claims(self):
        """
        Compute the non-iterated distributions of claims from the claimants to the groups
        """
        if self.has_uncertainty:
            raise NotImplementedError(
                f"Uncertainty is not yet implemented for {self.__name__}."
            )
        else:
            row_sums = self.reduced_claimant_matrix.sum(axis=1)
            self.claims_mat = (
                self.reduced_claimant_matrix / row_sums[:, np.newaxis] * self.base_claim
            )


class TaurekLottery(GroupBasedLottery):
    """
    Implements Taurek's coin toss.
    """

    def __init__(
        self,
        claimant_mat,
        remove_subgroups=False,
        use_db_access: bool = True,
        has_uncertainty: bool = None,
    ):
        super().__init__(claimant_mat, remove_subgroups, use_db_access, has_uncertainty)
        self.lottery_code = "TAUR"
        self.lottery_name = "Taurek's coin toss"
        self.register_lottery_in_db()
        self.claims()

    def claims(self):
        """
        Compute the non-iterated distributions of claims from the claimants to the groups
        There are many matrices that result in an equal chances coin toss for the outcome groups
        We choose the one that equally divides the group claim of 1/number_groups among all claimants, which are part
        of the group.
        """
        if self.has_uncertainty:
            raise NotImplementedError(
                f"Uncertainty is not yet implemented for {self.__name__}."
            )
        else:
            binary_claimant_matrix = np.where(self.reduced_claimant_matrix > 0, 1, 0)
            col_sums = binary_claimant_matrix.sum(axis=0)
            divisor = np.transpose(col_sums[:, np.newaxis]) * self.reduced_number_groups
            divisor = np.where(divisor == 0, np.nan, divisor)
            self.claims_mat = np.where(
                np.transpose(col_sums[:, np.newaxis]) != 0,
                binary_claimant_matrix / divisor,
                0,
            )


class TILottery(Lottery):
    """
    Implements Timmermann's individualist lottery.
    """

    def __init__(
        self,
        claimant_mat,
        remove_subgroups=False,
        use_db_access: bool = True,
        has_uncertainty: bool = None,
    ):
        super().__init__(claimant_mat, remove_subgroups, use_db_access, has_uncertainty)
        self.lottery_code = "TI"
        self.lottery_name = "Timmermann's individualist lottery"
        self.register_lottery_in_db()
        if self.has_uncertainty:
            raise NotImplementedError(
                f"Uncertainty is not yet implemented for {self.__name__}."
            )

    def remaining_claimants_and_groups_after_next_pick(self, picked_claimant):
        groups_containing_picked_claimant = self.claimant_mat[picked_claimant]
        remaining_cols = np.flatnonzero(
            groups_containing_picked_claimant
            == np.max(groups_containing_picked_claimant)
        )
        remaining_rows = np.any(
            self.claimant_mat[:, remaining_cols] != 0, axis=1
        ).nonzero()[0]
        remaining_rows = np.delete(remaining_rows, remaining_rows == picked_claimant)
        return remaining_rows, remaining_cols

    @read_write
    def compute_on_orbits(self) -> np.array:
        probabilities = np.zeros(self.number_groups)
        for claimant_orbit_rep, all_claimants_in_orbit in self.orbits["claimants"][
            "members"
        ].items():
            (
                remaining_claimants,
                remaining_groups,
            ) = self.remaining_claimants_and_groups_after_next_pick(claimant_orbit_rep)
            if len(remaining_claimants) > 0:
                smaller_mat = self.claimant_mat[remaining_claimants][
                    :, remaining_groups
                ]
                next_lottery_iteration = self.__class__(
                    claimant_mat=smaller_mat,
                    remove_subgroups=self.remove_subgroups,
                    use_db_access=self.use_db_access,
                )
                probs_from_next_iteration = next_lottery_iteration.compute_on_orbits()
            else:
                probs_from_next_iteration = np.ones(len(remaining_groups)) / len(
                    remaining_groups
                )
            orbit_prob = defaultdict(int)
            # next loop can probably be optimized away
            for orbit_id, prob in zip(
                self.orbits["groups"]["orbit_id"][remaining_groups],
                probs_from_next_iteration,
            ):
                orbit_prob[orbit_id] += prob
            for orbit_id, prob in orbit_prob.items():
                group_prob = prob / len(self.orbits["groups"]["members"][orbit_id])
                for group in self.orbits["groups"]["members"][orbit_id]:
                    probabilities[group] = (
                        probabilities[group]
                        + len(all_claimants_in_orbit)
                        / self.number_claimants
                        * group_prob
                    )
        return probabilities


def main():
    # n_claimants = 6
    # ones = [1 for _ in range(int(n_claimants / 2))]
    # zeroes = [0 for _ in range(int(n_claimants / 2))]
    # my_array = np.transpose(np.array([ones + zeroes, zeroes + ones]))
    # for i in range(n_claimants):
    #     for j in range(i + 1, n_claimants):
    #         newcol = np.zeros(shape=(n_claimants, 1))
    #         newcol[i] = 1
    #         newcol[j] = 1
    #         my_array = np.hstack([my_array, newcol])

    my_array = np.array(
        [
            [1, 0, 0.5, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 1],
            [0, 0.4, 0, 1],
            [1, 1, 1, 1],
        ]
    )
    # for Lot in [EXCSLottery, EQCSLottery, TaurekLottery, TILottery]:
    #     lottery = Lot(claimant_mat=my_array, remove_subgroups=False)
    #     lottery.compute_on_orbits()
    lottery = TILottery(
        claimant_mat=my_array, remove_subgroups=False, use_db_access=True
    )
    # lottery.compute_on_orbits()
    x = Matrix([1, 1, 1, 1, 1])
    print(x.transpose() * lottery.symbolic_matrix)
    # print(np.array_equal(lottery.unique_values, np.array([0,0.5,1])))


if __name__ == "__main__":
    main()
