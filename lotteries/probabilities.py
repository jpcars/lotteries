from scipy.special import comb
import math


def probability(claimants, group_size_coarse, group_size_fine=2, lottery="EXCS"):
    if lottery == "EXCS":
        num = (
            group_size_coarse
            - 1
            + comb(group_size_coarse - 1, group_size_fine - 1, exact=True)
            * (group_size_fine - 1)
        )
        denom = (
            group_size_coarse
            - 1
            + comb(claimants - 1, group_size_fine - 1, exact=True)
            * (group_size_fine - 1)
        )
        result = num / denom
        return result
    elif lottery == "IL":
        return (
            math.factorial(group_size_coarse - 1)
            * math.factorial(claimants - group_size_fine)
            / (
                math.factorial(group_size_coarse - group_size_fine)
                * math.factorial(claimants - 1)
            )
        )


if __name__ == "__main__":
    print(probability(1000, 500))
