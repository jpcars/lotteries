import copy
from collections import defaultdict
from itertools import permutations

import pandas as pd


class Group:
    def __init__(self, name, claimants):
        self.name = name
        self.claimants = claimants
        self.is_subset_of = set()
        self.is_superset_of = set()
        self.claim = 0


class Claimant:
    def __init__(self, name):
        self.name = name
        self.others_exclusive_relative_to_this = defaultdict(dict)
        self.member_of = set()


class Lottery:
    """
    Base class for the different lotteries
    """

    def __init__(self, groups):
        self._groups = groups
        self.claimants = {}
        self.groups = {"active": {}, "inactive": {}}
        self.lottery_name = None
        self.initialized = False
        self.computed = False
        self.initialize()
        self.base_claim = 1 / len(self.claimants)

    def initialize(self):
        if not self.initialized:
            self.create_groups_and_claimants()
            self.supersets()
        self.initialized = True

    def create_groups_and_claimants(self):
        for index, group in (
            self._groups.items()
            if self.check_groups_type()
            else enumerate(self._groups)
        ):
            group_claimants = set()
            for claimant in group.claimants if self.check_groups_type() else group:
                if isinstance(claimant, Claimant):
                    claimant_name = claimant.name
                else:
                    claimant_name = str(claimant)
                if claimant_name not in self.claimants:
                    self.claimants[claimant_name] = Claimant(claimant_name)
                self.claimants[claimant_name].member_of.add(index)
                group_claimants.add(self.claimants[claimant_name])
            self.groups["active"][index] = Group(index, group_claimants)

    def check_groups_type(self):
        """Checks whether _groups is a dictionary of group instances"""
        if isinstance(self._groups, list):
            return False
        else:
            return all(isinstance(item, Group) for item in self._groups.values())

    def supersets(self):
        for group1, group2 in permutations(self.groups["active"].values(), 2):
            if group1.claimants <= group2.claimants:
                group1.is_subset_of.add(group2.name)
                group2.is_superset_of.add(group1.name)
            if group2.claimants <= group1.claimants:
                group2.is_subset_of.add(group1.name)
                group1.is_superset_of.add(group2.name)

    def deactivate_group(self, index):
        if index in self.groups["active"]:
            self.groups["inactive"][index] = self.groups["active"].pop(index)
        else:
            raise ValueError(
                f"Trying to deactivate group {index}, but not present in active groups"
            )

    def probabilities(self) -> (pd.Series, pd.Series):
        """
        Computes the probabilities that any particular group and any particular claimant will win the lottery
        :return: (series of group probabilities, series of claimant probabilities)
        """
        group_probabilities = {}
        claimant_probabilities = {}
        for group in self.groups["inactive"].values():
            group_probabilities[group.name] = group.claim
            if group.claim > 0:
                for claimant in group.claimants:
                    claimant_probabilities[claimant.name] = (
                        claimant_probabilities.get(claimant.name, 0) + group.claim
                    )
        group_probabilities_series = pd.Series(
            data=group_probabilities, name=self.lottery_name
        )
        group_probabilities_series.sort_index(inplace=True)
        group_probabilities_series.index.name = "group"
        claimant_probabilities_series = pd.Series(
            data=claimant_probabilities, name=self.lottery_name
        )
        claimant_probabilities_series.sort_index(inplace=True)
        claimant_probabilities_series.index.name = "claimant"
        return group_probabilities_series, claimant_probabilities_series


class EXCSLottery(Lottery):
    """
    Implements the Exclusive Composition-sensitive lottery
    """

    def __init__(self, groups):
        super().__init__(groups)
        self.lottery_name = "EXCS"
        self.exclusivity_relations()
        self.claims()

    def exclusivity_relations(self):
        """
        resets any exclusivity relationships that might previously exist and then goes through all pairs of claimants
        to check their exclusivity relationships
        """
        for claimant in self.claimants.values():
            claimant.others_exclusive_relative_to_this = defaultdict(set)
        for claimant1, claimant2 in permutations(self.claimants.values(), 2):
            intersection = claimant1.member_of.intersection(claimant2.member_of)
            for group_index in intersection:
                difference1 = claimant1.member_of.difference(claimant2.member_of)
                difference2 = claimant2.member_of.difference(claimant1.member_of)
                if len(difference1) > 0:
                    claimant1.others_exclusive_relative_to_this[group_index].add(
                        claimant2
                    )
                if len(difference2) > 0:
                    claimant2.others_exclusive_relative_to_this[group_index].add(
                        claimant1
                    )

    def claims(self):
        """
        Given the set of groups compute the non-iterated probabilities of receiving the benefits for
        each group
        :return:
        """
        for claimant in self.claimants.values():
            total_exclusives = sum(
                [
                    len(exclusive_claimants)
                    for exclusive_claimants in claimant.others_exclusive_relative_to_this.values()
                ]
            )
            number_groups = len(claimant.member_of)
            if number_groups == 0:
                raise ValueError(f"{claimant.name=} is not a member of any group.")
            elif total_exclusives == 0:
                for group_index in claimant.member_of:
                    group = self.groups["active"][group_index]
                    group.claim += self.base_claim / number_groups
            else:
                for group_index in claimant.member_of:
                    group = self.groups["active"][group_index]
                    if exclusive_in_group := claimant.others_exclusive_relative_to_this.get(
                        group_index
                    ):
                        group.claim += (
                            self.base_claim * len(exclusive_in_group) / total_exclusives
                        )

    def compute(self):
        if not self.computed:
            while len(self.groups["active"]) > 0:
                for index, group in self.groups["active"].copy().items():
                    if (
                        len(group.is_superset_of) > 0
                    ):  # don't compute groups, which still have active subgroups
                        pass
                    else:
                        if len(group.is_subset_of) > 0:
                            new_groups = {
                                i: copy.copy(self.groups["active"][i])
                                for i in group.is_subset_of
                            }
                            temp_lottery = EXCSLottery(new_groups)
                            for i, g in temp_lottery.groups["active"].items():
                                self.groups["active"][i].claim += group.claim * g.claim
                                self.groups["active"][i].is_superset_of.remove(index)
                            group.claim = 0
                        self.deactivate_group(index)
        self.computed = True


class EQCSLottery(Lottery):
    """
    Implements the Equal Composition-Sensitive lottery
    """

    def __init__(self, groups):
        super().__init__(groups)
        self.lottery_name = "EQCS"
        self.claims()

    def claims(self):
        """
        Given the set of groups compute the non-iterated probabilities of receiving the benefits for
        each group
        :return:
        """
        for claimant in self.claimants.values():
            number_groups = len(claimant.member_of)
            if number_groups == 0:
                raise ValueError(f"{claimant.name=} is not a member of any group.")
            for group_index in claimant.member_of:
                group = self.groups["active"][group_index]
                group.claim += self.base_claim / number_groups

    def compute(self):
        if not self.computed:
            while len(self.groups["active"]) > 0:
                for index, group in self.groups["active"].copy().items():
                    if (
                        len(group.is_superset_of) > 0
                    ):  # don't compute groups, which still have active subgroups
                        pass
                    else:
                        if len(group.is_subset_of) > 0:
                            new_groups = {
                                i: copy.copy(self.groups["active"][i])
                                for i in group.is_subset_of
                            }
                            temp_lottery = EQCSLottery(new_groups)
                            for i, g in temp_lottery.groups["active"].items():
                                self.groups["active"][i].claim += group.claim * g.claim
                                self.groups["active"][i].is_superset_of.remove(index)
                            group.claim = 0
                        self.deactivate_group(index)
        self.computed = True


if __name__ == "__main__":
    groupie = [[1, 2], [3, 4], [1, 3], [1, 3, 5], [1, 3, 4]]
    lottery = EQCSLottery(groupie)
    lottery.compute()
    for group in lottery.groups["inactive"].values():
        print(f"{group.name=}, {group.claim}\n")
