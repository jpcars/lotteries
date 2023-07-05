from collections import defaultdict
from itertools import permutations

import hypernetx as hnx
import pandas as pd
from typing import Any
import pynauty as pn


class Lottery:
    """
    Base class for the different lotteries
    """

    def __init__(self, claimant_groups: dict[Any, list[list[int, float]]], remove_subgroups=False):
        self.lottery_name = None
        self.canon_nauty_label = None
        self.nauty_certificate = None
        self.exclusivity = defaultdict(dict)

        self.remove_subgroups = remove_subgroups
        self.graph = hnx.Hypergraph(setsystem=claimant_groups, edge_properties={})
        if remove_subgroups:
            self.graph = self.graph.toplexes()

        self.base_claim = 1 / self.graph.shape[0]

    def probabilities(self) -> (pd.Series, pd.Series):
        """
        Computes the probabilities that any particular group will win the lottery
        :return: series of group probabilities
        """
        group_probabilities = {}
        for group, properties in self.graph.edge_properties.items():
            group_probabilities[group] = properties.claim
        group_probabilities_series = pd.Series(
            data=group_probabilities, name=self.lottery_name + ('_subgroups_removed' if self.remove_subgroups else '')
        )
        group_probabilities_series.sort_index(inplace=True)
        group_probabilities_series.index.name = "group"
        return group_probabilities_series

    def nauty_translator(self):
        edges = list(self.graph.edges())
        nodes = list(self.graph.nodes())
        translator_dict = {elem: idx for idx, elem in enumerate(edges + nodes)}
        self.to_nauty = translator_dict
        self.from_nauty = {value: key for key, value in translator_dict.items()}

    def nauty_certificate(self):
        nauty_incidence_dict = {self.to_nauty[edge]: [self.to_nauty[node] for node in nodes] for edge, nodes in self.graph.incidence_dict}
        nauty_graph = pn.Graph(number_of_vertices=len(nauty_incidence_dict),
                               adjacency_dict=nauty_incidence_dict,
                               vertex_coloring=[{self.to_nauty[edge] for edge in self.graph.edges()},
                                                 {self.to_nauty[node] for node in self.graph.nodes()}]
                               )
        self.canon_nauty_label = pn.canon_label(nauty_graph)
        self.nauty_certificate = pn.certificate(nauty_graph)

    def store_values(self):
        """
        If the certificate is not yet present, write a new entry, else do nothing
        :return:
        """
        pass

    def retrieve_values(self):
        """
        Check whether there is an entry for the certificate. If so, retrieve results and do backwards translation
        :return:
        """
        pass

    def exclusivity_relations(self):
        """
        resets any exclusivity relationships that might previously exist and then goes through all pairs of claimants
        to check their exclusivity relationships
        """
        self.exclusivity = defaultdict(dict)
        for claimant1, claimant2 in permutations(self.graph.nodes, 2):
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

class EXCSLottery(Lottery):
    """
    Implements the Exclusive Composition-sensitive lottery
    """

    def __init__(self, claimant_groups, remove_subgroups=False):
        super().__init__(claimant_groups, remove_subgroups)
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