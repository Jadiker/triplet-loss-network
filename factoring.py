#!/usr/bin/env python3
import math
from typing import List, Tuple

def get_factors(num: int) -> List[int]:
    '''Gets a list of the factors of a number in order of size (low to high)'''
    # UPGRADE this is a slow implementation
    ans: List[int] = []
    for possible_factor in range(1, num + 1):
        if num % possible_factor == 0:
            ans.append(possible_factor)

    return ans

def get_middle_factors(num: int) -> Tuple[int, int]:
    '''Returns the two middle factors of a number'''
    factors = get_factors(num)
    number_of_factors = len(factors)
    midpoint = number_of_factors // 2
    if number_of_factors % 2 == 0:
        return (factors[midpoint - 1], factors[midpoint])
    else:
        return (factors[midpoint], factors[midpoint])

def get_square_factors(num: int) -> Tuple[int, int]:
    '''
    Determine the factors A and B for which an AxB rectangle has a ratio closest to a square,
    ...and can contain `num` amount of items without leaving a row or column open
    ...(if they are packed as closely as possible)
    '''

    def get_ratio_score(factors):
        # lower (closer to 0) is better
        return 1 - (factors[0] / factors[1])

    test_num = num
    best_factors = get_middle_factors(test_num)
    # smaller is better
    best_score = get_ratio_score(best_factors)
    while True:
        test_num += 1
        factors = get_middle_factors(test_num)
        small_factor, big_factor = factors
        # lines are the length of the smallest factor
        lines_filled = math.ceil(num / big_factor)
        if lines_filled == small_factor:
            score = get_ratio_score(factors)
            if score < best_score:
                best_factors = factors
                best_score = score
        else: # lines_filled < small_factor
            break

    return best_factors
