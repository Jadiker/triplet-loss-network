# pylint: disable = wildcard-import, unused-wildcard-import, missing-docstring
from factoring import *

def test_get_square_factors():
    assert get_square_factors(17) == (4, 5)
    assert get_square_factors(4) == (2, 2)
    assert get_square_factors(25) == (5, 5)
    assert get_square_factors(5) == (2, 3)

def test_get_middle_factors():
    assert get_middle_factors(9) == (3, 3)
    assert get_middle_factors(15) == (3, 5)

def test_get_factors():
    factors = set(get_factors(100))
    for val in [1, 2, 4, 5, 10, 20, 25, 50, 100]:
        factors.remove(val)
    assert len(factors) == 0 # pylint: disable = len-as-condition
