from scipy.special import comb
import math


def vong_1(claimants, group_size_coarse, group_size_fine=2, lottery="EXCS"):
    """Implements a generalized version of Vong's example on p.342ff."""
    if lottery == "EXCS":
        num = (
                1
                + comb(group_size_coarse - 1, group_size_fine - 1, exact=True)
                * (group_size_fine - 1) / (group_size_coarse - 1)
        )
        denom = (
                1
                + comb(claimants - 1, group_size_fine - 1, exact=True)
                * (group_size_fine - 1) / (group_size_coarse - 1)
        )
        result = num / denom
        return result
    elif lottery == "EQCS":
        num = (
                1
                + comb(group_size_coarse - 1, group_size_fine - 1, exact=True)
        )
        denom = (
                1
                + comb(claimants - 1, group_size_fine - 1, exact=True)
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


def vong_2(claimants, lottery="EXCS"):
    """Implements a generalized version of Vong's example on page ..."""
    if claimants % 2 != 0:
        raise ValueError('Number of claimants must be even.')
    if lottery == "EXCS":
        return (claimants - 2)**2 / claimants**2
    elif lottery == "EQCS":
        return (claimants - 2) / (2 * claimants)
    elif lottery == "IL":
        return 1 - 2 / comb(claimants - 1, claimants / 2, exact=True)


if __name__ == "__main__":
    print(vong_example_1(1000, 500))
