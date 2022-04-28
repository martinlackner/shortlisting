import numpy as np

SHORTLISTINGRULES = [
    "av",
    "majority",
    "firstmajority",
    "largestgap",
    "firstkgap",
    "sizepriority-5",
    "nextk",
    "2means",
    "2median",
    "qNCSA-0.5",
    "topsfirstkgap-5-.3v"
]


def get_parameter_values(parameter, max_score, num_voters=None):
    values = []
    parameters = parameter.split("-")[1:]
    assert len(parameters) >= 1, parameters
    for parameter in parameters:
        if parameter[-3:] == "max":  # relative to the max score
            values.append(max(int(float(parameter[:-3]) * max_score), 1))
        elif parameter[-1] == "v":  # relative to num_voters
            values.append(max(int(float(parameter[:-1]) * num_voters), 1))
        else:  # absolute
            try:
                values.append(int(parameter))
            except:
                raise ValueError(f"parameter {parameter} unclear")
    if len(values) == 1:
        return values[0]
    return values

def compute(name, scores, num_voters=None):
    """Returns the list of winning committees according to the named rule"""
    if name == "majority":
        return compute_majority(scores, num_voters)
    elif name == "av":
        return compute_av(scores)
    elif name == "firstmajority":
        return compute_firstmajority(scores)
    elif name == "largestgap":
        return compute_largestgap(scores)
    elif name == "2means":
        return compute_2means(scores)
    elif name == "2median":
        return compute_2median(scores)
    elif name[:9] == "firstkgap":
        parameter = get_parameter_values(name, max(scores), num_voters)
        return compute_firstkgap(scores, parameter)
    elif name.startswith("threshold-"):
        parameter = get_parameter_values(name, max(scores), num_voters)
        return compute_threshold(scores, parameter)
    elif name[:4] == "next":
        parameter = get_parameter_values(name, max(scores), num_voters)
        return compute_nextk(scores, parameter)
    elif name.startswith("sizepriority-"):
        parameter = get_parameter_values(name, max(scores), num_voters)
        return compute_sizepriority(scores, parameter)
    elif name.startswith("minsizeandkgap"):
        parameters = get_parameter_values(name, max(scores), num_voters)
        return compute_sizeandkgap(scores, parameters[0], parameters[1], minimumsize=True)
    elif name.startswith("topsfirstkgap"):
        parameters = get_parameter_values(name, max(scores), num_voters)
        return compute_sizeandkgap(scores, parameters[0], parameters[1], minimumsize=False)
    elif name.startswith("qNCSA-"):
        paramq = float(name[len("qNCSA-"):])
        return compute_qNCSA(scores, paramq, num_voters)
    else:
        raise NotImplementedError(
            "Shortlisting rule " + str(name) + " not known"
        )


def fullname(name, parameter=None):
    """Returns the full name of the voting rule"""
    if name == "majority":
        return "0.5n-Threshold"
    elif name == "av":
        return "Approval Voting"
    elif name == "firstmajority":
        return "First Majority"
    elif name == "largestgap":
        return "Largest Gap"
    elif name == "2means":
        return "2-Means Clustering"
    elif name == "2median":
        return "2-Median Clustering"
    elif name[:9] == "firstkgap":
        if len(name) > 10:
            parameter = name[10:]
        if parameter.endswith("v"):
            parameter = parameter[:-1] + "n"
        if parameter is None:
            parameter = "?"
        return "First " + str(parameter) + "-Gap"
    elif name[:4] == "next":
        if name == "nextk" or name == "next":
            parameter = 2
        else:
            parameter = int(name[4:])
        return "Next-" + str(parameter)
    elif name[:12] == "sizepriority":
        if len(name) > 12:
            assert name[12] == "-"
            parameter = name[13:]
        if parameter is None:
            return "Size Priority"
        return f"Inc. Size Priority-{parameter}"
    elif name.startswith("topsfirstkgap"):
        parameters = name.split("-")
        return f"Top-{parameters[1]}-First-{parameters[2]}-Gap"
    elif name.startswith("qNCSA"):
        parameters = name.split("-")
        return f"{parameters[1]}-NCSA"
    else:
        return name


########################################################################


def approval_scores(profile):
    scores = [0] * profile.num_cand
    for v in profile:
        for cand in v.approved:
            scores[cand] += v.weight
    return scores


def compute_av(scores):
    """Returns the list of winning committees according to Approval Voting"""
    committee = [c for c in range(len(scores)) if scores[c] == max(scores)]
    return committee


def compute_threshold(scores, cap):
    """Returns the list of winning committees according to y Threshold rule"""
    committee = [c for c in range(len(scores)) if scores[c] > cap]
    return committee


def compute_majority(scores, num_voters):
    return compute_threshold(scores, cap=num_voters / 2.)


def compute_firstmajority(scores):
    """Returns the list of winning committees
    according to the First Majority rule"""
    ordered_cands = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    firstsum = 0
    committee = []
    for c in ordered_cands:
        committee.append(c)
        firstsum += scores[c]
        if firstsum > sum(scores) // 2:
            break

    return sorted(committee)


def compute_largestgap(scores):
    """Returns the list of winning committees
    according to the Largest Gap rule"""
    scores_sorted = sorted(scores, reverse=True)
    gaps = [scores_sorted[i] - scores_sorted[i + 1] for i in range(len(scores) - 1)]
    gapvalue = max(
        [scores_sorted[i] for i in range(len(scores) - 1) if gaps[i] == max(gaps)]
    )
    committee = [c for c in range(len(scores)) if scores[c] >= gapvalue]

    return committee


def compute_sizeandkgap(scores, size, gap, minimumsize=True):
    """..."""
    scores_sorted = sorted(scores, reverse=True)
    kgaps = [
        i
        for i in range(len(scores) - 1)
        if scores_sorted[i] - scores_sorted[i + 1] >= gap
    ]
    if not kgaps:
        committee = list(range(len(scores)))
    else:
        firstkgap = min(kgaps)
        threshold = scores_sorted[firstkgap]
        committee = [c for c in range(len(scores)) if scores[c] >= threshold]
    if minimumsize and len(committee) < size:
        return compute_sizepriority(scores, size)
    if not minimumsize and len(committee) > size:
        return compute_sizepriority(scores, size)
    return committee


def compute_firstkgap(scores, gap):
    """Returns the list of winning committees
    according to the First k-Gap rule"""
    return compute_sizeandkgap(scores, size=0, gap=gap)


def compute_sizepriority(scores, size):
    """Returns the list of winning committees
    according to the Size Priority rule"""
    scores_sorted = sorted(scores, reverse=True)
    if isinstance(size, list):
        priority = size
    else:
        priority = sorted(list(range(size, len(scores) + 1)))
    for psize in priority:
        threshold = scores_sorted[psize - 1]
        committee = [c for c in range(len(scores)) if scores[c] >= threshold]
        if len(committee) == psize:
            return committee

    raise AssertionError(
        "This should not happen. \
                         Size Priority found no committee."
    )


def compute_nextk(scores, nextk):
    """Returns the list of winning committees
    according to the Next-k rule"""
    ordered_cands = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    committee = []
    for i in range(len(scores)):
        c = ordered_cands[i]
        committee.append(c)
        if len(committee) == len(scores):
            continue
        # non-tiebreaking
        if scores[c] == scores[ordered_cands[i + 1]]:
            continue
        score_nextk = sum(
            [
                scores[ordered_cands[j]]
                for j in range(i + 1, min(i + nextk + 1, len(scores)))
            ]
        )
        if score_nextk < scores[c]:
            break

    return sorted(committee)


def compute_2means(scores):
    """Returns the list of winning committees
    according to the 2-Means Clustering rule"""
    return mindistance_clustering(scores, "means")


def compute_2median(scores):
    """Returns the list of winning committees
    according to the 2-Median Clustering rule"""
    return mindistance_clustering(scores, "median")


def mindistance_clustering(scores, objective):
    def twomeans(cluster1, cluster2):
        m1 = np.mean(cluster1) if cluster1 else 0
        m2 = np.mean(cluster2) if cluster2 else 0
        return sum([np.abs(x - m1) ** 2 for x in cluster1]) + sum(
            [np.abs(x - m2) ** 2 for x in cluster2]
        )

    def twomedian(cluster1, cluster2):
        m1 = np.median(cluster1) if cluster1 else 0
        m2 = np.median(cluster2) if cluster2 else 0
        return sum([np.abs(x - m1) for x in cluster1]) + sum(
            [np.abs(x - m2) for x in cluster2]
        )

    if objective == "means":
        objectivefct = twomeans
    elif objective == "median":
        objectivefct = twomedian
    else:
        raise NotImplemented(objective + " is not an implemented objective function.")
    ordered_cands = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    objfct_scores = []
    for i in range(len(scores) + 1):
        cluster1 = [scores[c] for c in ordered_cands[:i]]
        cluster2 = [scores[c] for c in ordered_cands[i:]]
        objfct_scores.append(objectivefct(cluster1, cluster2))
    cut = objfct_scores.index(min(objfct_scores))

    return sorted(ordered_cands[:cut])


def compute_qNCSA(scores, paramq, num_voters):
    ncsa_scores = [0] * (len(scores) + 1)
    ncsa_scores[0] = -len(scores) * num_voters  # size 0 is not allowed
    scores_sorted = sorted(scores, reverse=True)
    for size in range(1, len(scores) + 1):
        ncsa_scores[size] = ((2 * sum(scores_sorted[: size]) - size * num_voters) / (size ** paramq))
    opt_size = max([size for size, val in enumerate(ncsa_scores) if val == max(ncsa_scores)])
    cutoff_score = scores_sorted[opt_size-1]
    # ordered_cands = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
    return sorted(cand for cand, score in enumerate(scores) if score >= cutoff_score)
