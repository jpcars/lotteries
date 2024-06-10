from typing import Optional

import numpy as np
import functools

from lotteries.database.connect import execute_command


def read_write(method):
    """
    Wraps method in order to read or write its return values to/from a kwarg called prob_dict. If prob_dict is not
    passed, this decorator has no effect.
    Assumes that the instance method is called from has the following attributes:
        - self.lottery_name
        - self.nauty_certificate
        - self.inverse_canon_group_label
        - self.canon_group_label
        - self.canon_claimant_label
    :param method: method to be decorated
    :return: decorated method
    """

    @functools.wraps(method)
    def wrapper_decorator(self, *args, **kwargs):
        if self.use_db_access:
            probabilities, dilemma_key = read_values(
                translate_from_canon=self.inverse_canon_group_label,
                nauty_certificate=self.nauty_certificate,
                lottery_code=self.lottery_code,
                has_uncertainty=self.has_uncertainty,
            )
            if probabilities is not None:
                return probabilities
            else:
                probabilities = method(self, *args, **kwargs)
                if self.has_uncertainty:
                    canon_matrix_rep = self.reduced_claimant_matrix[
                        self.canon_claimant_label, :
                    ][:, self.canon_group_label].tolist()
                else:
                    canon_matrix_rep = self.reduced_claimant_matrix[
                        self.canon_claimant_label, :
                    ][:, self.canon_group_label].tolist()
                write_values(
                    group_probabilities=probabilities,
                    dilemma_key=dilemma_key,
                    translate_to_canon=self.canon_group_label,
                    nauty_certificate=self.nauty_certificate,
                    lottery_code=self.lottery_code,
                    canon_matrix_rep=canon_matrix_rep,
                    number_claimants=self.number_claimants,
                    number_groups=self.reduced_number_groups,
                    has_uncertainty=self.has_uncertainty,
                )
                return probabilities
        else:
            probabilities = method(self, *args, **kwargs)
            return probabilities

    return wrapper_decorator


def write_values(
    group_probabilities,
    dilemma_key,
    translate_to_canon,
    nauty_certificate,
    lottery_code,
    canon_matrix_rep,
    number_claimants,
    number_groups,
    has_uncertainty,
) -> None:
    """
    If the certificate is not yet present in dict, write a new entry, else do nothing
    :param group_probabilities: probabilities to store
    :param dilemma_key: if None, then dilemma doesn't exist in dilemmas table yet
    :param translate_to_canon: permute probability array to canonical labelling
    :param nauty_certificate: graph certificate to use
    :param lottery_code: name of the lottery
    :param canon_matrix_rep: the claimant_matrix defining the dilemma in canonical representation
    :param number_claimants: number of claimants
    :param number_groups: number of groups
    :param has_uncertainty: flag whether it is a dilemma with or without uncertainty, controls which table the values are written to
    """
    assert translate_to_canon, "self.canon_group_label should be set at this point"
    if dilemma_key is None:
        dilemma_key = insert_dilemma(
            canon_matrix_rep,
            has_uncertainty,
            nauty_certificate,
            number_claimants,
            number_groups,
        )
    if has_uncertainty:
        command = """INSERT INTO fact_probabilities_uncertainty (dilemma_key, lottery_code, group_probabilities) VALUES (%s,%s,%s);"""
        value_tuple = (
            dilemma_key,
            lottery_code,
            group_probabilities[translate_to_canon].tolist(),
        )
    else:
        command = """INSERT INTO fact_probabilities (dilemma_key, lottery_code, group_probabilities) VALUES (%s,%s,%s);"""
        value_tuple = (
            dilemma_key,
            lottery_code,
            group_probabilities[translate_to_canon].tolist(),
        )
    execute_command(
        command=command,
        values=value_tuple,
        has_return_value=False,
    )


def insert_dilemma(
    canon_matrix_rep,
    has_uncertainty,
    nauty_certificate,
    number_claimants,
    number_groups,
):
    if has_uncertainty:
        print("here1")
        dilemma_insertion_command = """INSERT INTO fact_dilemmas_uncertainty (graph_certificate, matrix_rep, number_claimants, number_groups) VALUES (%s,%s,%s,%s) RETURNING dilemma_key;"""
        dilemma_insertion_value_tuple = (
            nauty_certificate,
            canon_matrix_rep,
            number_claimants,
            number_groups,
        )
    else:
        dilemma_insertion_command = """INSERT INTO fact_dilemmas (graph_certificate, matrix_rep, number_claimants, number_groups) VALUES (%s,%s,%s,%s) RETURNING dilemma_key;"""
        dilemma_insertion_value_tuple = (
            nauty_certificate,
            canon_matrix_rep,
            number_claimants,
            number_groups,
        )
    dilemma_key = execute_command(
        command=dilemma_insertion_command,
        values=dilemma_insertion_value_tuple,
        has_return_value=True,
    )[0][0]
    return dilemma_key


def read_values(
    translate_from_canon, nauty_certificate, lottery_code, has_uncertainty
) -> Optional[np.array]:
    """
    Check whether there is an entry for the certificate and the lottery.
    If so, retrieve the group probabilities and do backwards translation
    :param translate_from_canon: relabelling to apply on the result
    :param nauty_certificate: graph certificate to use
    :param lottery_code: unique code to identify the lottery
    :param has_uncertainty: flag whether it is a dilemma with or without uncertainty, controls which table the values are written to
    :return: array of group probabilities translated to the labelling given by inverse_canon_group_label
    """
    assert (
        translate_from_canon
    ), "self.inverse_canon_group_label should be set at this point"
    if has_uncertainty:
        command = """SELECT
                        fact_dilemmas_uncertainty.dilemma_key,
                        fact_probabilities_uncertainty.group_probabilities
                    FROM fact_dilemmas_uncertainty
                    LEFT JOIN fact_probabilities_uncertainty
                        ON fact_probabilities_uncertainty.dilemma_key = fact_dilemmas_uncertainty.dilemma_key
                        AND fact_probabilities_uncertainty.lottery_code = %s
                    WHERE fact_dilemmas_uncertainty.graph_certificate = %s;
                    """
    else:
        command = """SELECT
                        fact_dilemmas.dilemma_key,
                        fact_probabilities.group_probabilities
                    FROM fact_dilemmas
                    LEFT JOIN fact_probabilities
                        ON fact_probabilities.dilemma_key = fact_dilemmas.dilemma_key
                        AND fact_probabilities.lottery_code = %s
                    WHERE fact_dilemmas.graph_certificate = %s;
                    """
    group_prob_return = execute_command(
        command=command,
        values=[lottery_code, nauty_certificate],
        has_return_value=True,
    )
    certificate_present = not (group_prob_return == [])
    probs = None
    dilemma_key = None
    if certificate_present:
        dilemma_key = group_prob_return[0][0]
        if (probabilities := group_prob_return[0][1]) is not None:
            probs = np.array(probabilities)[translate_from_canon]
    return probs, dilemma_key
