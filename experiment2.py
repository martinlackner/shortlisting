from os import path, walk
import shortlisting
import matplotlib.pyplot as plt
import numpy as np
import experiment1
import random

font = {"size": 11}
plt.rc("font", **font)


def dominate(aver_size1, aver_size2, likelibest1, likelibest2):
    """Does rule1 (aver_size1, likelibest1) dominate rule2 (aver_size2, likelibest2)?"""
    return (aver_size1 < aver_size2 and likelibest1 >= likelibest2) or (
        aver_size1 <= aver_size2 and likelibest1 > likelibest2
    )


def read_hugo_data(dir):
    hugo_files = []
    for (dirpath, dirnames, filenames) in walk(dir):
        hugo_files.extend([(f, path.join(dirpath, f)) for f in filenames])

    print(f"{len(hugo_files)} files")

    average_length = 0
    winner_ranks = []
    minimal_shortlist_lengths = []
    hugo_data = []

    for filename, fullpath in hugo_files:
        assert int(filename[:4]) in [2017, 2018, 2019, 2020, 2021]
        with open(fullpath, "r") as f:
            lines = f.readlines()

        year = int(lines[0])
        assert year in [2017, 2018, 2019, 2020, 2021]
        category = lines[1].strip()
        try:
            num_votes = int(lines[2].strip())
        except ValueError as error:
            raise ValueError(
                f"error in {filename}: {lines[2].strip()} is not an integer."
            ) from error
        winner = lines[3].strip().lower()
        assert lines[4].strip() == "", lines[4]

        data = [line.split(";") for line in lines[5:] if line.strip()]
        data = [d for d in data if d[-1].strip() != "WITHDRAWN"]

        scores = []
        for d in data:
            d[0] = d[0].strip().lower()
            try:
                scores.append(int(d[-1]))
            except ValueError as error:
                raise ValueError(
                    f"error in {filename}: {d[-1]} is not an integer."
                ) from error

        assert all(sc <= num_votes for sc in scores)
        assert any(d[0] == winner for d in data), winner

        winner_position = [d[0] for d in data].index(winner)
        winner_rank = len([sc for sc in scores if sc >= scores[winner_position]])
        minimal_shortlist_lengths.append(winner_rank)
        winner_ranks.append(winner_rank)

        average_length += len(data)

        hugo_data.append((year, category, scores, winner_position, num_votes))

    average_length = average_length / len(hugo_files)
    print(f"average list length = {average_length:.2f}")
    print(f"average winner position: {sum(winner_ranks) / len(winner_ranks) + 1:.2f}")
    print(
        "average optimal length of non-tiebreaking shortlists: "
        f"{sum(minimal_shortlist_lengths) / len(minimal_shortlist_lengths):.2f}"
    )

    positions = list(range(1, 8))
    num_winner_positions = [
        len([p for p in winner_ranks if p == position]) for position in positions
    ]
    assert all(1 <= p <= 7 for p in winner_ranks), winner_ranks
    assert sum(num_winner_positions) == len(hugo_files)
    plt.bar(positions, num_winner_positions, width=0.9, color="green")
    plt.xticks(positions, positions)  # labels get centered
    plt.xlabel("Position of the winning candidate in the shortlist")
    plt.ylabel("Number of instances")
    plt.tight_layout()
    plt.savefig("hugo-winning-position")
    plt.close()

    aver_ideal_length = sum(minimal_shortlist_lengths) / len(minimal_shortlist_lengths)
    return hugo_data, aver_ideal_length


def run_single_experiment(varrules, scores, winner, num_voters=None):
    contains_best = {}
    size = {}

    # compute rules
    for rule in varrules:
        comm = shortlisting.compute(rule, scores, num_voters)
        if winner in comm:
            contains_best[rule] = 1
        else:
            contains_best[rule] = 0
        size[rule] = len(comm)

    return contains_best, size


def experiment(name):
    print(f"Starting {name}:\n\n")

    if name == "hugo":
        sensible_sizes = [2, 3, 4, 5, 6, 7]
        max_interesting_size = 8
        min_interesting_size = 0.95
    elif name == "noise model":
        sensible_sizes = list(range(2, 15))
        max_interesting_size = 15
        min_interesting_size = 0.95
    elif name == "bias model":
        sensible_sizes = list(range(2, 15))
        max_interesting_size = 15
        min_interesting_size = 3.95
    else:
        raise ValueError
    sp_rules = [f"sizepriority-{size}" for size in sensible_sizes]
    highlight_rules = sp_rules + [
        "av",
        "firstmajority",
        "largestgap",
        # "firstkgap-0.05v",
    ]  # , "firstkgap-0.05max"]
    rules = [
        "av",
        "firstmajority",
        "largestgap",
        "next-2",
        "next-3",
        "2means",
        "2median",
        "qNCSA-0",
    ] + sp_rules
    for param in range(1, 100):
        rules.append(f"firstkgap-{param*5}")
        param = f"0.{param:02d}"
        rules.append(f"qNCSA-{param}")
        rules.append(f"firstkgap-{param}max")
        rules.append(f"firstkgap-{param}v")
        rules.append(f"threshold-{param}max")
        rules.append(f"threshold-{param}v")
        for size in sensible_sizes:
            rules.append(f"topsfirstkgap-{size}-{param}max")
            rules.append(f"topsfirstkgap-{size}-{param}v")
    rules.append("qNCSA-1")

    likelibest = dict.fromkeys(rules, 0)
    aver_size = dict.fromkeys(rules, 0)
    sizes = {rule: list() for rule in rules}

    if name == "hugo":
        hugodir = "/home/martin/Documents/work/repos/jansvn/journal/hugo/data"
        hugodata, aver_ideal_length = read_hugo_data(hugodir)
        maxapprovalscore = []
        for year, category, scores, winner, num_votes in hugodata:
            contains_best, size = run_single_experiment(
                rules, scores, winner, num_votes
            )
            maxapprovalscore.append(max(scores) / num_votes)
            for rule in rules:
                aver_size[rule] += size[rule]
                sizes[rule].append(size[rule])
                likelibest[rule] += contains_best[rule]
        for rule in rules:
            aver_size[rule] /= len(hugodata)
            likelibest[rule] /= len(hugodata)
    elif name in ["noise model", "bias model"]:
        random.seed(123)
        num_cand = 30
        num_voters = 100

        xvals = np.linspace(0, 1, 21)
        xvals = [x for x in xvals if 0.0 <= x <= 0.5]

        (
            _,
            likelibest,
            aver_size,
            sizes,
            maxapprovalscore,
        ) = experiment1.run_experiments(
            num_voters,
            num_cand,
            xvals,
            rules,
            distribution="normal",
            model=name,
            iterations=1000,
        )
        for rule in rules:
            likelibest[rule] = np.average(likelibest[rule])
            aver_size[rule] = np.average(aver_size[rule])
    else:
        raise ValueError

    sorted_rules = sorted(aver_size, key=aver_size.get)

    pareto_frontier_rules = []
    for rule in sorted_rules:
        if not any(
            dominate(
                aver_size[otherrule],
                aver_size[rule],
                likelibest[otherrule],
                likelibest[rule],
            )
            for otherrule in rules
        ):
            pareto_frontier_rules.append(rule)

    print("highlight rules:")
    for rule in highlight_rules:
        print(f"  {rule:30s} {aver_size[rule]:12.3f} {likelibest[rule]:12.3f}")

    if name == "hugo":
        for sp_rule in sp_rules:
            print(f"\nbetter than {sp_rule}:")
            print(
                f"  {sp_rule:30s} {aver_size[sp_rule]:12.3f} {likelibest[sp_rule]:12.3f}"
            )
            print(f"------------------------------------")
            for rule in sorted_rules:
                if (
                    aver_size[rule] < aver_size[sp_rule]
                    and likelibest[rule] >= likelibest[sp_rule]
                ):
                    print(
                        f"  {rule:30s} & {aver_size[rule]:12.3f} & {likelibest[rule]:12.3f}"
                    )

    fig, ax = plt.subplots(figsize=(8, 5))

    pareto_x = [aver_size[rule] for rule in pareto_frontier_rules if rule in rules] + [
        100
    ]
    pareto_y = [likelibest[rule] for rule in pareto_frontier_rules if rule in rules]
    pareto_y.append(pareto_y[-1])
    plt.fill_between(
        x=pareto_x,
        y1=pareto_y,
        # where=(-1 < t) & (t < 1),
        color="black",
        alpha=0.2,
    )

    # ax.plot(
    #     [aver_size[rule] for rule in pareto_frontier_rules if rule in rules],
    #     [likelibest[rule] for rule in pareto_frontier_rules if rule in rules],
    #     color="red",
    #     marker=".",
    #     label="Pareto Frontier",
    # )
    ax.scatter(
        [aver_size[rule] for rule in sp_rules if rule in rules],
        [likelibest[rule] for rule in sp_rules if rule in rules],
        color="red",
        marker="D",
        zorder=5,
        label="Increasing Size Priority",
    )
    ax.plot(
        [
            aver_size[rule]
            for rule in rules
            if "threshold" in rule and rule.endswith("v")
        ],
        [
            likelibest[rule]
            for rule in rules
            if "threshold" in rule and rule.endswith("v")
        ],
        color="orange",
        marker=".",
        label="Threshold (% voters)",
    )
    ax.plot(
        [
            aver_size[rule]
            for rule in rules
            if "threshold" in rule and rule.endswith("x")
        ],
        [
            likelibest[rule]
            for rule in rules
            if "threshold" in rule and rule.endswith("x")
        ],
        "--",
        color="orange",
        marker=".",
        label="Max-Score Threshold (% max. score)",
    )
    if name != "hugo":
        ax.plot(
            [aver_size[rule] for rule in rules if "qNCSA" in rule],
            [likelibest[rule] for rule in rules if "qNCSA" in rule],
            color="black",
            marker=".",
            label="q-NCSA",
        )
    ax.plot(
        [
            aver_size[rule]
            for rule in rules
            if rule.startswith("firstkgap") and rule.endswith("x")
        ],
        [
            likelibest[rule]
            for rule in rules
            if rule.startswith("firstkgap") and rule.endswith("x")
        ],
        "b--",
        marker=".",
        label="First k-Gap (% max. score)",
    )
    ax.plot(
        [
            aver_size[rule]
            for rule in rules
            if rule.startswith("firstkgap") and rule.endswith("v")
        ],
        [
            likelibest[rule]
            for rule in rules
            if rule.startswith("firstkgap") and rule.endswith("v")
        ],
        "b",
        marker=".",
        label="First k-Gap (% voters)",
    )
    maxlikeli_y = max(pareto_y)
    maxlikeli_x = min(
        [
            aver_size[rule]
            for rule in pareto_frontier_rules
            if likelibest[rule] == maxlikeli_y
        ]
    )
    ax.scatter([maxlikeli_x], [maxlikeli_y], color="black", marker="o", zorder=10)

    if name == "hugo":
        ax.annotate(
            "optimal rules",
            (maxlikeli_x, maxlikeli_y),
            ha="right",
            va="bottom",
            xytext=(70, -70),
            textcoords="offset pixels",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
            # bbox=dict(boxstyle="round", fc="w")
        )

    ax.scatter(
        [aver_size[rule] for rule in highlight_rules if rule not in sp_rules],
        [likelibest[rule] for rule in highlight_rules if rule not in sp_rules],
        color="black",
        marker="o",
        zorder=10,
    )

    #
    # ax.annotate(
    #     "theoretical optimum",
    #     (aver_ideal_length, 1),
    #     ha="right",
    #     va="bottom",
    #     xytext=(70, -70),
    #     textcoords="offset pixels",
    #     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
    #     # bbox=dict(boxstyle="round", fc="w")
    # )

    plt.legend(loc="lower right")

    count = 1
    pareto_interesting_rules_id = []
    for i, rule in enumerate(pareto_frontier_rules):
        pareto_interesting_rules_id.append(count)
        if (
            rule in rules
            and rule not in highlight_rules
            and all(
                aver_size[rule] != aver_size[otherrule]
                or likelibest[rule] != likelibest[otherrule]
                for otherrule in pareto_frontier_rules[:i]
            )
        ):
            # ax.annotate(
            #     count,
            #     (aver_size[rule], likelibest[rule]),
            #     ha="right",
            #     xytext=(-5, 5),
            #     textcoords="offset pixels",
            # )
            count += 1
    for rule in highlight_rules:
        yoff = -9
        arrowprops = None
        ha = "left"
        va = "top"
        xoff = 4
        if name == "hugo":
            if rule == "largestgap":
                va = "bottom"
                yoff = 0
            elif rule in ["sizepriority-2", "sizepriority-3"]:
                ha = "center"
            elif rule in ["sizepriority-7"]:
                va = "bottom"
                yoff = 0
            elif "sizepriority" in rule and rule not in [
                "sizepriority-2",
                "sizepriority-3",
                "sizepriority-7",
            ]:
                ha = "right"
                va = "bottom"
                yoff = 0
                xoff = -4
            elif rule == "firstkgap-0.05v":
                ha = "right"
                va = "bottom"
                yoff = 0
                xoff = -4
                # ha = "right"
                # xoff = 70
                # yoff = -40
                # arrowprops = dict(arrowstyle="->", connectionstyle="arc3")
            elif aver_size[rule] > 6.2:
                ...
                # ha = "right"
                # xoff = -10
        else:
            if "sizepriority" in rule:
                ha = "right"
                va = "bottom"
                yoff = 0
                xoff = -4
            elif rule == "firstmajority":
                ha = "right"
                xoff = 70
                if name == "noise model":
                    yoff = -30
                else:
                    yoff = -50
                arrowprops = dict(arrowstyle="->", connectionstyle="arc3")
            if name == "bias model" and rule in ["sizepriority-4"]:
                yoff = 0
                arrowprops = None
                ha = "left"
                va = "bottom"
                xoff = 4
        try:
            text = shortlisting.fullname(rule)
        except NotImplementedError:
            text = rule
        if rule.startswith("sizepriority"):
            text = f'ISP-{rule.split("-")[-1]}'
        ax.annotate(
            text,
            (aver_size[rule], likelibest[rule]),
            ha=ha,
            va=va,
            xytext=(xoff, yoff),
            textcoords="offset pixels",
            arrowprops=arrowprops,
            # bbox=dict(boxstyle="round", fc="w")
        )

    plt.xlabel("Average size")
    plt.ylabel("Likelihood to contain the actual winner")
    ax.set_xlim(min_interesting_size, max_interesting_size)
    ax.set_ylim(0.4, 1.04)

    print("\nPareto frontier:")
    for rule, id in zip(pareto_frontier_rules, pareto_interesting_rules_id):
        if rule in rules:
            print(
                f"  ({id}) {rule:30s} {aver_size[rule]:12.3f} {likelibest[rule]:12.3f}"
            )

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{name}-data", pad_inches=0)
    plt.close()

    plt.hist(maxapprovalscore, 10, density=True, facecolor="g", alpha=0.75)
    plt.xlabel("Relative maximum approval score")
    # plt.ylabel('Probability')
    plt.savefig(f"histogram-maxapprscore-{name}")
    plt.close()

    print(f"{name} done.\n\n")


experiment("hugo")
experiment("bias model")
experiment("noise model")
