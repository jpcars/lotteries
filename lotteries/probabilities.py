from scipy.special import comb


def probability(claimants, group_size_coarse, group_size_fine=2):
    num = (
        group_size_coarse
        - 1
        + comb(group_size_coarse - 1, group_size_fine - 1, exact=True)
    )
    denom = group_size_coarse - 1 + comb(claimants - 1, group_size_fine - 1, exact=True)
    result = num / denom
    return result


if __name__ == "__main__":
    print(probability(1000, 500))
