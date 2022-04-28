"""Examples from the paper."""

import shortlisting
import random


def running_example():
    scores = (10, 10, 9, 8, 6, 3, 3, 0)
    num_voters = 10
    results = {
        "av": [0, 1],
        "threshold-.5v": [0, 1, 2, 3, 4],
        "firstmajority": [0, 1, 2],
        "next-2": [0, 1, 2, 3, 4, 5, 6],
        "qNCSA-.5": [0, 1, 2, 3],
        "largestgap": [0, 1, 2, 3, 4],
        "firstkgap-2": [0, 1, 2, 3],
        "topsfirstkgap-3-2": [0, 1, 2],
    }
    for rule, expected in results.items():
        winning = shortlisting.compute(rule, scores, num_voters)
        assert winning == expected, f"{rule} : {winning} != {expected}"
        print(f"{rule} correct")


running_example()
