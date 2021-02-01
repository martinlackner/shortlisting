import numpy as np


VARRULES = ["av", "majority", "firstmajority",
            "largestgap", "firstkgap", "sizepriority",
            "nextk", "2means", "2median"]


def compute_rule(name, profile):
    """Returns the list of winning committees according to the named rule"""
    if name == "majority":
        return compute_majority(profile)
    elif name == "av":
        return compute_av(profile)
    elif name == "firstmajority":
        return compute_firstmajority(profile)
    elif name == "largestgap":
        return compute_largestgap(profile)
    elif name == "2means":
        return compute_2means(profile)
    elif name == "2median":
        return compute_2median(profile)
    elif name[:9] == "firstkgap":
        if len(name) > 9:
            parameter = int(name[9:])
        else:
            parameter = max(len(profile.preferences) / 20, 1)
        return compute_firstkgap(profile, parameter)
    elif name[:4] == "next":
        if name == "nextk" or name == "next":
            parameter = 2
        else:
            parameter = int(name[4:])
        return compute_nextk(profile, parameter)
    elif name[:12] == "sizepriority":
        if len(name) > 12:
            param = name[12:]
            size = int(param)
        else:
            size = profile.num_cand / 3.
        return compute_sizepriority(profile, size)
    else:
        raise NotImplementedError("Variable multi-winner rule " +
                                  str(name) + " not known")


def fullname(name, parameter=None):
    """Returns the full name of the voting rule"""
    if name == "majority":
        return "0.5|V|-Threshold"
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
        if len(name) > 9:
            parameter = int(name[9:])
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
            parameter = "opt=" + name[12:]
        if parameter is None:
            return "Size Priority"
        return "Size Priority"  # ("+str(parameter)+")"
    else:
        raise NotImplementedError("Variable multi-winner rule " +
                                  str(name) + " not known")


########################################################################


def approval_scores(profile):
    appr_score = [0] * profile.num_cand
    for pref in profile.preferences:
        for cand in pref.approved:
            appr_score[cand] += pref.weight
    return appr_score


def compute_av(profile):
    """Returns the list of winning committees according to Approval Voting"""
    appr_score = approval_scores(profile)
    committee = [c for c in range(profile.num_cand)
                 if appr_score[c] == max(appr_score)]
    return committee


def compute_majority(profile):
    """Returns the list of winning committees according to the Majority rule"""
    appr_score = approval_scores(profile)
    committee = [c for c in range(profile.num_cand)
                 if appr_score[c] >= profile.totalweight() / 2.]
    return committee


def compute_firstmajority(profile):
    """Returns the list of winning committees
    according to the First Majority rule"""
    appr_score = approval_scores(profile)
    ordered_cands = sorted(range(profile.num_cand),
                           key=lambda x: appr_score[x], reverse=True)
    firstsum = 0
    committee = []
    for c in ordered_cands:
        committee.append(c)
        firstsum += appr_score[c]
        if firstsum > sum(appr_score) // 2:
            break

    return sorted(committee)


def compute_largestgap(profile):
    """Returns the list of winning committees
    according to the Largest Gap rule"""
    appr_score = approval_scores(profile)
    appr_score_sorted = sorted(appr_score, reverse=True)
    gaps = [appr_score_sorted[i] - appr_score_sorted[i + 1]
            for i in range(profile.num_cand - 1)]
    gapvalue = max([appr_score_sorted[i] for i in range(profile.num_cand - 1)
                    if gaps[i] == max(gaps)])
    committee = [c for c in range(profile.num_cand)
                 if appr_score[c] >= gapvalue]

    return committee


def compute_firstkgap(profile, paramk):
    """Returns the list of winning committees
    according to the First k-Gap rule"""
    appr_score = approval_scores(profile)
    appr_score_sorted = sorted(appr_score, reverse=True)
    kgaps = [i for i in range(profile.num_cand - 1)
             if appr_score_sorted[i] - appr_score_sorted[i + 1] >= paramk]
    if not kgaps:
        committee = list(range(profile.num_cand))
    else:
        firstkgap = min(kgaps)
        threshold = appr_score_sorted[firstkgap]
        committee = [c for c in range(profile.num_cand)
                     if appr_score[c] >= threshold]
    return committee


def compute_sizepriority(profile, size):
    """Returns the list of winning committees
    according to the Size Priority rule"""
    appr_score = approval_scores(profile)
    appr_score_sorted = sorted(appr_score, reverse=True)
    if isinstance(size, list):
        priority = size
    else:
        # priority = sorted(list(range(profile.num_cand + 1)),
        #                   key=lambda x: abs(size - x))
        priority = sorted(list(range(size, profile.num_cand + 1)))
    for psize in priority:
        threshold = appr_score_sorted[psize - 1]
        committee = [c for c in range(profile.num_cand)
                     if appr_score[c] >= threshold]
        if len(committee) == psize:
            return committee

    raise AssertionError("This should not happen. \
                         Size Priority found no committee.")


def compute_nextk(profile, nextk):
    """Returns the list of winning committees
    according to the Next-k rule"""
    appr_score = approval_scores(profile)
    ordered_cands = sorted(range(profile.num_cand),
                           key=lambda x: appr_score[x], reverse=True)
    committee = []
    for i in range(profile.num_cand):
        c = ordered_cands[i]
        committee.append(c)
        if len(committee) == profile.num_cand:
            continue
        # non-tiebreaking
        if appr_score[c] == appr_score[ordered_cands[i + 1]]:
            continue
        score_nextk = sum([appr_score[ordered_cands[j]]
                           for j in range(i + 1,
                                          min(i + nextk + 1, profile.num_cand))
                           ])
        if score_nextk < appr_score[c]:
            break

    return sorted(committee)


def compute_2means(profile):
    """Returns the list of winning committees
    according to the 2-Means Clustering rule"""
    return mindistance_clustering(profile, "means")


def compute_2median(profile):
    """Returns the list of winning committees
    according to the 2-Median Clustering rule"""
    return mindistance_clustering(profile, "median")


def mindistance_clustering(profile, objective):
    def twomeans(cluster1, cluster2):
        m1 = np.mean(cluster1) if cluster1 else 0
        m2 = np.mean(cluster2) if cluster2 else 0
        return (sum([np.abs(x - m1)**2 for x in cluster1]) +
                sum([np.abs(x - m2)**2 for x in cluster2]))

    def twomedian(cluster1, cluster2):
        m1 = np.median(cluster1) if cluster1 else 0
        m2 = np.median(cluster2) if cluster2 else 0
        return (sum([np.abs(x - m1) for x in cluster1]) +
                sum([np.abs(x - m2) for x in cluster2]))

    if objective == "means":
        objectivefct = twomeans
    elif objective == "median":
        objectivefct = twomedian
    else:
        raise NotImplemented(objective +
                             " is not an implemented objective function.")
    appr_score = approval_scores(profile)
    ordered_cands = sorted(range(profile.num_cand),
                           key=lambda x: appr_score[x], reverse=True)
    objfct_scores = []
    for i in range(profile.num_cand + 1):
        cluster1 = [appr_score[c] for c in ordered_cands[:i]]
        cluster2 = [appr_score[c] for c in ordered_cands[i:]]
        objfct_scores.append(objectivefct(cluster1, cluster2))
    cut = objfct_scores.index(min(objfct_scores))

    return sorted(ordered_cands[:cut])
