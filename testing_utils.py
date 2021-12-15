def fuzzy_compare(expected, tolerance, reality):
    '''
    Given three dictionaries with identical keys,
    check if the values are within the given tolerance.
    Tolerance is given as a tuple (below, above)
    '''
    all_passed = True
    for ((expected_key, expected_value), (tolerance_key, tolerance_value), (reality_key, reality_value)) in zip(expected.items(), tolerance.items(), reality.items()):
        assert expected_key == reality_key == tolerance_key # all keys should match 
        if reality_value >= expected_value - tolerance_value[0] and reality_value <= expected_value + tolerance_value[1]:
            print(f"passed\t\t{reality_key}:\t{reality_value} \tis within tolerance\t-{tolerance_value[0]} to {tolerance_value[1]}\tof {expected_value}") 
        else:
            print(f"===FAILED===\t{reality_key}:\t{reality_value} \tis NOT within tolerance\t-{tolerance_value[0]} to {tolerance_value[1]}\tof {expected_value}") 

            all_passed = False
    return all_passed

if __name__ == "__main__":

    example_expected = {
        'first key' : 1,
        'second key': 2,
        'third key': 3,
    }

    example_tolerance = {
        'first key' : (0,1),
        'second key': (0,10),
        'third key': (0,100),
    }

    example_reality = {
        'first key' : 1,
        'second key': 2,
        'third key': 300,
    }

    fuzzy_compare(example_expected, example_tolerance, example_reality)
