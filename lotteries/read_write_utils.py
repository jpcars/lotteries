from typing import Optional

import numpy as np
import functools


def read_write_decorator(method):
    """
    Wraps method in order to read or write its return values to/from a kwarg called prob_dict. If prob_dict is not
    passed, this decorator has no effect.
    Assumes that the instance method is called from has the following attributes:
        - self.lottery_name
        - self.nauty_certificate
        - self.inverse_canon_group_label
        - self.canon_group_label
    :param method: method to be decorated
    :return: decorated method
    """

    @functools.wraps(method)
    def wrapper_decorator(self, *args, prob_dict=None, **kwargs):
        if prob_dict is not None:
            if (
                probabilities := read_values(
                    prob_dict=prob_dict,
                    translate_from_canon=self.inverse_canon_group_label,
                    nauty_certificate=self.nauty_certificate,
                    lottery_name=self.lottery_name,
                )
            ) is not None:
                return probabilities
            else:
                probabilities = method(self, *args, prob_dict=prob_dict, **kwargs)
                write_values(
                    group_probabilities=probabilities,
                    prob_dict=prob_dict,
                    translate_to_canon=self.canon_group_label,
                    nauty_certificate=self.nauty_certificate,
                    lottery_name=self.lottery_name,
                )
                return probabilities
        else:
            probabilities = method(self, *args, **kwargs)
            return probabilities

    return wrapper_decorator


def write_values(
    group_probabilities, prob_dict, translate_to_canon, nauty_certificate, lottery_name
) -> None:
    """
    If the certificate is not yet present in dict, write a new entry, else do nothing
    :param group_probabilities: probabilities to store
    :param prob_dict: dictionary to use
    :param translate_to_canon: permute probability array to canonical labelling
    :param nauty_certificate: graph certificate to use
    :param lottery_name: name of the lottery
    """
    assert translate_to_canon, "self.canon_group_label should be set at this point"
    if nauty_certificate not in prob_dict:
        prob_dict[nauty_certificate] = {
            lottery_name: group_probabilities[translate_to_canon]
        }
    elif lottery_name not in prob_dict[nauty_certificate]:
        prob_dict[nauty_certificate][lottery_name] = group_probabilities[
            translate_to_canon
        ]


def read_values(
    prob_dict, translate_from_canon, nauty_certificate, lottery_name
) -> Optional[np.array]:
    """
    Check whether there is an entry for the certificate and the lottery.
    If so, retrieve the group probabilities and do backwards translation
    :param prob_dict: dictionary to look up the result
    :param translate_from_canon: relabelling to apply on the result
    :param nauty_certificate: graph certificate to use
    :param lottery_name: name of the lottery
    :return: array of group probabilities translated to the labelling given by inverse_canon_group_label
    """
    assert (
        translate_from_canon
    ), "self.inverse_canon_group_label should be set at this point"
    if (graph_dict := prob_dict.get(nauty_certificate)) is not None:
        if (probabilities := graph_dict.get(lottery_name)) is not None:
            return probabilities[translate_from_canon]
