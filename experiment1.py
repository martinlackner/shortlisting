import shortlisting
import random
from abcvoting.preferences import Profile
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import pickle
import os
import hashlib

font = {"size": 14}
plt.rc("font", **font)


def run_single_experiment(
    varrules, num_voters, quality, perceived_quality, num_objective_voters=0
):
    num_cand = len(quality)

    bestcands = [c for c in range(len(quality)) if quality[c] == max(quality)]
    assert len(bestcands) == 1

    averqual = {}
    likelibest = {}
    aversize = {}
    varsize = {}

    # generate votes
    appr_sets = []
    for i in range(0, num_voters):
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
    profile.add_voters(appr_sets)
    scores = shortlisting.approval_scores(profile)
    maxapprovalscore = max(scores) / len(profile)

    # compute rules
    for rule in varrules:
        comm = shortlisting.compute(rule, scores, num_voters=len(profile))
        if comm:  # comm not empty
            averqual[rule] = np.average([quality[c] for c in comm])
        else:
            averqual[rule] = 0
        if set(bestcands).issubset(comm):
            likelibest[rule] = 1
        else:
            likelibest[rule] = 0
        aversize[rule] = len(comm)
        varsize[rule] = [len(comm)]

    return averqual, likelibest, aversize, varsize, maxapprovalscore


def run_experiments(
    num_voters, num_cand, xvals, varrules, distribution, model, iterations
):
    args = "/".join(
        [
            str(x)
            for x in (
                num_voters,
                num_cand,
                xvals,
                varrules,
                distribution,
                model,
                iterations,
            )
        ]
    )
    id = hashlib.md5(args.encode("utf-8")).hexdigest()
    picklefile = f"instances-{id}.pickle"
    if os.path.exists(picklefile):
        print("loading results from " + picklefile)
        with open(picklefile, "rb") as f:
            data = pickle.load(f)
            assert data[0] == args
            return data[1:]

    averqual_series = {rule: [] for rule in varrules}
    likelibest_series = {rule: [] for rule in varrules}
    aversize_series = {rule: [] for rule in varrules}
    varsize_series = {rule: [] for rule in varrules}
    maxapprovalscore_series = []

    for xval in xvals:
        for rule in varrules:
            averqual_series[rule].append(0)
            likelibest_series[rule].append(0)
            aversize_series[rule].append(0)
            varsize_series[rule].append([])

        for _ in range(iterations):
            # quality = [0.9, 0.88, 0.7, 0.65, 0.4, 0.2]
            if distribution == "uniform":
                quality = [random.random() for _ in range(num_cand)]
            elif distribution == "normal":
                lower, upper = 0, 1
                mu, sigma = 0.75, 0.2
                quality = stats.truncnorm.rvs(
                    (lower - mu) / sigma,
                    (upper - mu) / sigma,
                    loc=mu,
                    scale=sigma,
                    size=num_cand,
                )
                # plt.hist(quality, 10, density=True, facecolor='g', alpha=0.75)
                # plt.show()
                # plt.close()
            else:
                raise Exception("Distribution " + distribution + " not known.")

            if model == "noise model":
                # Compute the noisy version of the candidate quality
                noisy_quality = [
                    (1 - xval) * quality[c] + xval * 0.5 for c in range(num_cand)
                ]
                (
                    averqual,
                    likelibest,
                    aversize,
                    varsize,
                    maxapprovalscore,
                ) = run_single_experiment(varrules, num_voters, quality, noisy_quality)
            elif model == "bias model":
                # Compute the biased version of the candidate quality
                # the (negative) bias applies to half of the cands
                biased_quality = [quality[c] * 0.5 for c in range(num_cand // 2)] + [
                    quality[c] for c in range(num_cand // 2, num_cand)
                ]
                # always bias against objectively best cand.
                for c in range(num_cand):
                    if quality[c] == max(quality):
                        biased_quality[c] = quality[c] * 0.5
                kgap_param = 5 + int(20 * xval)

                (
                    averqual,
                    likelibest,
                    aversize,
                    varsize,
                    maxapprovalscore,
                ) = run_single_experiment(
                    varrules,
                    num_voters,
                    quality,
                    biased_quality,
                    num_objective_voters=int((1 - xval) * num_voters),
                )
            else:
                raise Exception("Model " + model + " not known.")

            assert len(aversize) == len(varrules)
            assert len(likelibest) == len(varrules)

            maxapprovalscore_series.append(maxapprovalscore)

            for rule in varrules:
                try:
                    averqual_series[rule][-1] += averqual[rule] * 1.0 / iterations
                except KeyError as error:
                    print(averqual_series)
                    print(averqual)
                    print(averqual_series.keys())
                    print(averqual.keys())
                    raise KeyError from error
                likelibest_series[rule][-1] += likelibest[rule] * 1.0 / iterations
                aversize_series[rule][-1] += aversize[rule] * 1.0 / iterations
                varsize_series[rule][-1].append(varsize[rule])

        for rule in varrules:
            varsize_series[rule][-1] = np.var(varsize_series[rule][-1])

    return_data = [
        averqual_series,
        likelibest_series,
        aversize_series,
        varsize_series,
        maxapprovalscore_series,
    ]

    print("writing instances to", picklefile)
    with open(picklefile, "wb") as f:
        pickle.dump([args] + return_data, f)

    return return_data


# EXPERIMENTS
if __name__ == "__main__":

    distributions = ["normal", "uniform"]
    models = ["noise model", "bias model"]

    # "next2" ,"2means", "2median"]

    num_cand = 30
    num_voters = 100

    for model in models:
        colors = {
            "av": "-m",
            "majority": "orange",
            "firstmajority": ":k",
            "largestgap": ":c",
            "firstkgap-5": "--b",
            "sizepriority-4": "-.r",
            # "next-2": "--g",
            # "2means": "--r",
            # "2median": "--b",
            # "threshold-.5max": "-.g",
            "topsfirstkgap-10-5": "-.g",
            "qNCSA-0.5": "-k",
        }
        hightlight_rules = list(colors.keys())

        for dist in distributions:
            print(model, dist)

            xvals = np.linspace(0, 1, 21)

            (
                averqual_series,
                likelibest_series,
                aversize_series,
                varsize_series,
                maxapprovalscore_series,
            ) = run_experiments(
                num_voters,
                num_cand,
                xvals,
                hightlight_rules,
                dist,
                model,
                iterations=1000,
            )

            # PLOTTING
            plt.figure(figsize=(12, 12))
            plt.subplot(4, 1, 1)
            title = (
                "Shortlisting experiments - "
                + dist
                + "ly distributed quality, "
                + model
            )
            # plt.title(title)
            plt.ylabel("average quality")
            plt.ylim(-0.01, 1.01)
            for rule in hightlight_rules:
                plt.plot(
                    xvals,
                    averqual_series[rule],
                    colors[rule],
                    label=shortlisting.fullname(rule),
                )
            # plt.xticks(xvals, [])

            plt.subplot(4, 1, 2)
            for rule in hightlight_rules:
                plt.plot(
                    xvals,
                    likelibest_series[rule],
                    colors[rule],
                    label=shortlisting.fullname(rule),
                )
            plt.ylabel("precision")
            plt.ylim(-0.01, 1.01)
            # plt.xticks(xvals, [])

            plt.subplot(4, 1, 3)
            for rule in hightlight_rules:
                plt.plot(
                    xvals,
                    aversize_series[rule],
                    colors[rule],
                    label=shortlisting.fullname(rule),
                )
            plt.ylabel("average size")
            # plt.xticks(xvals, [])

            if model == "noise model":
                plt.xlabel("noise (γ)")
            if model == "bias model":
                plt.xlabel("fraction of biased voters (γ)")

            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(title.replace(" ", "-"), bbox_inches="tight", pad_inches=0)
            plt.close()

            ###############################

            plt.figure(figsize=(7, 5.5))
            title = (
                "Shortlisting experiments - average quality -"
                + dist
                + "ly distributed quality, "
                + model
            )
            # plt.title(title)
            plt.ylabel("average quality")
            if model == "noise model":
                plt.ylim(0.5, 1)
            if model == "bias model":
                plt.ylim(0.5, 1)
            for rule in hightlight_rules:
                plt.plot(
                    xvals,
                    averqual_series[rule],
                    colors[rule],
                    label=shortlisting.fullname(rule),
                )

            if model == "noise model":
                plt.xlabel("noise (γ)")
            if model == "bias model":
                plt.xlabel("fraction of biased voters (γ)")

            plt.tight_layout()
            plt.savefig(title.replace(" ", "-"), pad_inches=0)
            plt.close()

            plt.figure(figsize=(7, 5.5))
            title = (
                "Shortlisting experiments - average size -"
                + dist
                + "ly distributed quality, "
                + model
            )
            # plt.title(title)
            for rule in hightlight_rules:
                plt.plot(
                    xvals,
                    aversize_series[rule],
                    colors[rule],
                    label=shortlisting.fullname(rule),
                )
            plt.ylabel("average size")

            if model == "noise model":
                plt.xlabel("noise (γ)")
            if model == "bias model":
                plt.xlabel("fraction of biased voters (γ)")

            # plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(title.replace(" ", "-"), pad_inches=0)
            plt.close()

            plt.figure(figsize=(7, 5.5))
            title = (
                "Shortlisting experiments - best cand included -"
                + dist
                + "ly distributed quality, "
                + model
            )
            # plt.title(title)
            for rule in hightlight_rules:
                plt.plot(
                    xvals,
                    likelibest_series[rule],
                    colors[rule],
                    label=shortlisting.fullname(rule),
                )
            plt.ylabel("precision")
            plt.ylim(-0.01, 1.03)

            plt.legend(
                fancybox=True,
                shadow=False,
                ncol=2,
                # bbox_to_anchor=(0, 1.1, 1, 0.2),
                bbox_to_anchor=(0, 1.03, 1, 0.2),
                loc="lower left",
                mode="expand",
                borderaxespad=0,
            )

            if model == "noise model":
                plt.xlabel("noise (γ)")
            if model == "bias model":
                plt.xlabel("fraction of biased voters (γ)")

            # plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(title.replace(" ", "-"), pad_inches=0)
            plt.grid(True)
            plt.close()
