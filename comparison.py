import shortlisting as var
import random
from preferences import Profile
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


def run_single_experiment(varrules, num_voter,
                          quality, perceived_quality,
                          num_objective_voters=0):

    num_cand = len(quality)

    bestcands = [c for c in range(len(quality))
                 if quality[c] == max(quality)]

    averqual = dict.fromkeys(varrules, 0)
    likelibest = dict.fromkeys(varrules, 0)
    commeffective = dict.fromkeys(varrules, 0)
    aversize = dict.fromkeys(varrules, 0)
    varsize = {rule: [] for rule in varrules}

    # generate votes
    appr_sets = []
    for i in range(0, num_voter):
        appr = []
        for c in range(num_cand):
            if i < num_objective_voters:
                # voter i is objective (uses quality)
                if random.random() < quality[c]:
                    appr.append(c)
            else:
                # voter i is objective (uses perceived_quality)
                if random.random() < perceived_quality[c]:
                    appr.append(c)
        appr_sets.append(appr)
    profile = Profile(num_cand)
    profile.add_preferences(appr_sets)

    # compute rules
    for rule in varrules:
        comm = var.compute_rule(rule, profile)
        if comm:  # comm not empty
            averqual[rule] += np.average([quality[c] for c in comm])
        if set(bestcands).issubset(comm):
            likelibest[rule] += 1
            if len(comm) <= num_cand / 3:
                commeffective[rule] += 1
            # commeffective[rule] += np.average([quality[c] for c in comm])
        aversize[rule] += len(comm)
        varsize[rule].append(len(comm))

    return averqual, likelibest, commeffective, aversize, varsize


def run_experiments(num_voters, num_cand, xvals, varrules,
                    averqual_series, likelibest_series,
                    commeffective_series, aversize_series,
                    varsize_series, distribution, model):
    for xval in xvals:
        for rule in varrules:
            averqual_series[rule].append(0)
            likelibest_series[rule].append(0)
            commeffective_series[rule].append(0)
            aversize_series[rule].append(0)
            varsize_series[rule].append([])

        for _ in range(iterations):
            # quality = [0.9, 0.88, 0.7, 0.65, 0.4, 0.2]
            if distribution == "uniform":
                quality = [random.random() for _ in range(num_cand)]
            elif distribution == "normal":
                lower, upper = 0, 1
                mu, sigma = .5, 0.2
                quality = stats.truncnorm.rvs((lower - mu) / sigma,
                                              (upper - mu) / sigma,
                                              loc=mu, scale=sigma,
                                              size=num_cand)
            else:
                raise Exception("Distribution " + distribution + " not known.")

            if model == "noise model":
                # Compute the noisy version of the candidate quality
                noisy_quality = [(1 - xval) * quality[c] + xval * 0.5
                                 for c in range(num_cand)]
                averqual, likelibest, commeffective, aversize, varsize =\
                    run_single_experiment(varrules, num_voters,
                                          quality, noisy_quality)
            elif model == "bias model":
                # Compute the biased version of the candidate quality
                # the (negative) bias applies to half of the cands
                biased_quality = ([quality[c] * .5
                                   for c in range(num_cand // 2)]
                                  + [quality[c]
                                     for c in range(num_cand // 2, num_cand)])
                # always bias against objectively best cand.
                for c in range(num_cand):
                    if quality[c] == max(quality):
                        biased_quality[c] = quality[c] * .5
                averqual, likelibest, commeffective, aversize, varsize =\
                    run_single_experiment(
                        varrules, num_voters, quality, biased_quality,
                        num_objective_voters=int((1 - xval) * num_voters))
            else:
                raise Exception("Model " + model + " not known.")

            for rule in varrules:
                averqual_series[rule][-1] += averqual[rule] * 1. / iterations
                likelibest_series[rule][-1] += \
                    likelibest[rule] * 1. / iterations
                commeffective_series[rule][-1] += \
                    commeffective[rule] * 1. / iterations
                aversize_series[rule][-1] += aversize[rule] * 1. / iterations
                varsize_series[rule][-1].append(varsize[rule])

        for rule in varrules:
            varsize_series[rule][-1] = np.var(varsize_series[rule][-1])


# EXPERIMENTS

distributions = ["normal", "uniform"]
models = ["noise model", "bias model"]

varrules = ["av", "majority", "firstmajority",
            "largestgap", "firstkgap5", "sizepriority4"]
# "next2" ,"2means", "2median"]

num_cand = 30
num_voters = 100

for model in models:
    for dist in distributions:

        averqual_series = {rule: [] for rule in varrules}
        likelibest_series = {rule: [] for rule in varrules}
        commeffective_series = {rule: [] for rule in varrules}
        aversize_series = {rule: [] for rule in varrules}
        varsize_series = {rule: [] for rule in varrules}
        xvals = np.linspace(0, 1, 21)
        iterations = 1000

        run_experiments(num_voters, num_cand, xvals, varrules,
                        averqual_series, likelibest_series,
                        commeffective_series, aversize_series,
                        varsize_series, dist, model)

        # PLOTTING

        colors = {
            "av": "-b",
            "majority": "-r",
            "firstmajority": ":g",
            "largestgap": ":c",
            "firstkgap5": "--m",
            "sizepriority4": "-.k",
            "next2": "--g",
            "2means": "--r",
            "2median": "--b",
        }

        plt.figure(figsize=(12, 12))
        plt.subplot(4, 1, 1)
        title = ('Shortlisting experiments - ' + dist
                 + "ly distributed quality, " + model)
        # plt.title(title)
        plt.ylabel('average quality')
        plt.ylim(0.6, 1.0)
        for rule in varrules:
            plt.plot(xvals, averqual_series[rule], colors[rule],
                     label=var.fullname(rule))
        # plt.xticks(xvals, [])

        plt.subplot(4, 1, 2)
        for rule in varrules:
            plt.plot(xvals, likelibest_series[rule], colors[rule],
                     label=var.fullname(rule))
        plt.ylabel('best cand. included')
        plt.ylim(0.45, 1.01)
        # plt.xticks(xvals, [])

        plt.subplot(4, 1, 3)
        for rule in varrules:
            plt.plot(xvals, aversize_series[rule], colors[rule],
                     label=var.fullname(rule))
        plt.ylabel('average size')
        # plt.xticks(xvals, [])

        plt.subplot(4, 1, 4)
        for rule in varrules:
            plt.plot(xvals,
                     commeffective_series[rule],
                     colors[rule],
                     label=var.fullname(rule))
        plt.ylabel('sensible winner sets')
        plt.ylim(0, 1.01)

#         plt.subplot(5, 1, 5)
#         for rule in varrules:
#             plt.plot(xvals, varsize_series[rule], colors[rule],
#                      label=var.fullname(rule))
#         plt.ylabel('variance of size')
#         # plt.xticks(xvals, [])

        if model == "noise model":
            plt.xlabel('noise (lambda)')
        if model == "bias model":
            plt.xlabel('fraction of biased voters (gamma)')

        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(title.replace(" ", "-"), bbox_inches='tight', pad_inches=0)
        plt.close()

        ###############################

        plt.figure(figsize=(7, 4))
        title = ('Shortlisting experiments - average quality -' + dist
                 + "ly distributed quality, " + model)
        # plt.title(title)
        plt.ylabel('average quality')
        if model == "noise model":
            plt.ylim(0.5, 0.9)
        if model == "bias model":
            plt.ylim(0.5, 0.9)
        for rule in varrules:
            plt.plot(xvals, averqual_series[rule], colors[rule],
                     label=var.fullname(rule))

        if model == "noise model":
            plt.xlabel('noise (lambda)')
        if model == "bias model":
            plt.xlabel('fraction of biased voters (gamma)')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                   fancybox=True, shadow=False, ncol=3)

        plt.tight_layout()
        plt.savefig(title.replace(" ", "-"), pad_inches=0)
        plt.close()

        plt.figure(figsize=(7, 3.2))
        title = ('Shortlisting experiments - sensible winner sets -' + dist
                 + "ly distributed quality, " + model)
        # plt.title(title)
        for rule in varrules:
            plt.plot(xvals,
                     commeffective_series[rule],
                     colors[rule],
                     label=var.fullname(rule))
        plt.ylabel('sensible winner sets')
        plt.ylim(0, 1.01)

        if model == "noise model":
            plt.xlabel('noise (lambda)')
        if model == "bias model":
            plt.xlabel('fraction of biased voters (gamma)')

        # plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(title.replace(" ", "-"), pad_inches=0)
        plt.close()

        plt.figure(figsize=(7, 3.2))
        title = ('Shortlisting experiments - average size -' + dist
                 + "ly distributed quality, " + model)
        # plt.title(title)
        for rule in varrules:
            plt.plot(xvals, aversize_series[rule], colors[rule],
                     label=var.fullname(rule))
        plt.ylabel('average size')

        if model == "noise model":
            plt.xlabel('noise (lambda)')
        if model == "bias model":
            plt.xlabel('fraction of biased voters (gamma)')

        # plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(title.replace(" ", "-"), pad_inches=0)
        plt.close()

        plt.figure(figsize=(7, 3.2))
        title = ('Shortlisting experiments - best cand included -' + dist
                 + "ly distributed quality, " + model)
        # plt.title(title)
        for rule in varrules:
            plt.plot(xvals, likelibest_series[rule], colors[rule],
                     label=var.fullname(rule))
        plt.ylabel('best cand. included')
        plt.ylim(0.45, 1.01)

        if model == "noise model":
            plt.xlabel('noise (lambda)')
        if model == "bias model":
            plt.xlabel('fraction of biased voters (gamma)')

        # plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(title.replace(" ", "-"), pad_inches=0)
        plt.close()
