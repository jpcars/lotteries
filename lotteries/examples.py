from scipy.special import comb
import math
import numpy as np


def vong_1(claimants, size_1, size_2, lottery="EXCS"):
    """Implements a generalized version of Vong's example on p.342ff."""
    if size_1 > size_2:
        return 0
    else:
        if lottery == "EXCS":
                num = 1 + comb(size_2 - 1, size_1 - 1, exact=True) * (
                        size_1 - 1
                ) / (size_2 - 1)
                denom = 1 + comb(claimants - 1, size_1 - 1, exact=True) * (
                        size_1 - 1
                ) / (size_2 - 1)
                result = num / denom
                return result
        elif lottery == "EQCS":
            num = 1 + comb(size_2 - 1, size_1 - 1, exact=True)
            denom = 1 + comb(claimants - 1, size_1 - 1, exact=True)
            result = num / denom
            return result
        elif lottery == "TI":
            return (
                math.factorial(size_2 - 1)
                * math.factorial(claimants - size_1)
                / (
                    math.factorial(size_2 - size_1)
                    * math.factorial(claimants - 1)
                )
            )


def vong_2(claimants, lottery="EXCS"):
    """Implements a generalized version of Vong's example on page ..."""
    if claimants % 2 != 0:
        raise ValueError("Number of claimants must be even.")
    if lottery == "EXCS":
        return (claimants - 2) ** 2 / claimants**2
    elif lottery == "EQCS":
        return (claimants - 2) / (2 * claimants)
    elif lottery == "TI":
        summands = np.array(
            [
                1
                / (claimants - k)
                * np.prod([(int(claimants / 2) - k)/(claimants - k) for p in range(k)])
                for k in range(int(claimants / 2))
            ]
        )
        return 1 - 2 * np.sum(summands)


if __name__ == "__main__":
    print(vong_2(1000, lottery="IL"))
