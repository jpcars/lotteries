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
        probabilities, dilemma_key = read_values(
            translate_from_canon=self.inverse_canon_group_label,
            nauty_certificate=self.nauty_certificate,
            lottery_code=self.lottery_code,
        )
        if probabilities is not None:
            return probabilities
        else:
            probabilities = method(self, *args, **kwargs)
            write_values(
                group_probabilities=probabilities,
                dilemma_key=dilemma_key,
                translate_to_canon=self.canon_group_label,
                nauty_certificate=self.nauty_certificate,
                lottery_code=self.lottery_code,
                canon_matrix_rep=self.reduced_claimant_matrix[
                    self.canon_claimant_label, :
                ][:, self.canon_group_label].tolist(),
                number_claimants=self.number_claimants,
                number_groups=self.reduced_number_groups,
            )
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
    """
    assert translate_to_canon, "self.canon_group_label should be set at this point"
    if dilemma_key is None:
        dilemma_key = execute_command(
            command="""
            INSERT INTO dilemmas (graph_certificate,matrix_rep,number_claimants,number_groups) VALUES (%s,%s,%s,%s) RETURNING dilemma_key;
            """,
            values=(
                nauty_certificate,
                canon_matrix_rep,
                number_claimants,
                number_groups,
            ),
            has_return_value=True,
        )[0][0]
    execute_command(
        command="""
                INSERT INTO probabilities (dilemma_key, lottery_code, group_probabilities) VALUES (%s,%s,%s);
                """,
        values=(
            dilemma_key,
            lottery_code,
            group_probabilities[translate_to_canon].tolist(),
        ),
        has_return_value=False,
    )


def read_values(
    translate_from_canon, nauty_certificate, lottery_code
) -> Optional[np.array]:
    """
    Check whether there is an entry for the certificate and the lottery.
    If so, retrieve the group probabilities and do backwards translation
    :param translate_from_canon: relabelling to apply on the result
    :param nauty_certificate: graph certificate to use
    :param lottery_code: unique code to identify the lottery
    :return: array of group probabilities translated to the labelling given by inverse_canon_group_label
    """
    assert (
        translate_from_canon
    ), "self.inverse_canon_group_label should be set at this point"
    group_prob_return = execute_command(
        command="""SELECT
                        dilemmas.dilemma_key,
                        probabilities.group_probabilities
                    FROM dilemmas
                    LEFT JOIN probabilities
                        ON probabilities.dilemma_key = dilemmas.dilemma_key
                        AND probabilities.lottery_code = %s
                    WHERE graph_certificate = %s;
                    """,
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
