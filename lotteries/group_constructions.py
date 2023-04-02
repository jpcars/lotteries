import random

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
    ) + create_size_k_subsets(claimants, large_group_size, variant="ordered_disjoint_cover")
    return result


if __name__ == "__main__":
    print(small_large_example(claimants=9, small_group_size=2, large_group_size=3)
)