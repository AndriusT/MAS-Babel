"""Draw the MAS-Babel topology overview shown in the README (assets/architecture_diagram.png).

    python assets/make_topology_diagram.py                    # PNG for the README
    python assets/make_topology_diagram.py --svg figure.svg   # + vector copy (e.g. for a paper)

One panel per implemented topology, using the library's names; colours mark the
role of each agent.  Fonts are sized for the README, where the figure is shown
~900 px wide.  Re-run after adding or changing a topology.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402

OUT = Path(__file__).resolve().parent / "architecture_diagram.png"

# role -> (fill, edge)
COLORS = {
    "task": ("#ECECEC", "#8A8A8A"),
    "generator": ("#DCE9F9", "#3F6FAE"),
    "critic": ("#FBE1D7", "#C2513A"),
    "aggregator": ("#FFF1C9", "#B8921C"),
    "final": ("#DDF1E1", "#3E8E55"),
}
INK = "#333333"
EXCHANGE = "#7A5BA6"   # debate: answers shared between debaters
NOTE = "#555555"

# font sizes (points)
FS_BOX = 16
FS_TITLE = 21
FS_KEY = 14.5
FS_LABEL = 14
FS_NOTE = 13.5
FS_LEGEND = 16
FS_FOOT = 13.5


# ---------------------------------------------------------------------------
# drawing helpers (panel coordinates: x 0..10, y about -1.6..13)
# ---------------------------------------------------------------------------

class Box:
    def __init__(self, ax, x, y, w, h, text, role, fs=FS_BOX):
        self.x, self.y, self.w, self.h = x, y, w, h
        fc, ec = COLORS[role]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.3",
                                    fc=fc, ec=ec, lw=1.8, zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK, zorder=3,
                linespacing=1.2)

    def top(self, dx=0.0):
        return (self.x + dx, self.y + self.h / 2)

    def bottom(self, dx=0.0):
        return (self.x + dx, self.y - self.h / 2)

    def left(self, dy=0.0):
        return (self.x - self.w / 2, self.y + dy)

    def right(self, dy=0.0):
        return (self.x + self.w / 2, self.y + dy)


def arrow(ax, p0, p1, rad=0.0, color=INK, style="-|>", ls="-", lw=1.7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=17, color=color,
                                 lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2, zorder=1))


def label(ax, x, y, text, fs=FS_LABEL, color=NOTE, style="italic", **kw):
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=color,
            style=style, zorder=4, linespacing=1.25, **kw)


def panel(ax, number, title, key):
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.7, 13.0)
    ax.axis("off")
    ax.text(5, 12.25, f"{number}. {title}", ha="center", va="bottom", fontsize=FS_TITLE,
            weight="bold", color="#1F2D4D")
    ax.text(5, 11.65, key, ha="center", va="bottom", fontsize=FS_KEY, family="monospace",
            color=NOTE)


def task_and_final(ax, rule):
    """Task box, Final answer box, and the panel's rule as a note below it."""
    task = Box(ax, 5, 10.55, 5.0, 1.15, "Task", "task")
    final = Box(ax, 5, -0.05, 5.0, 1.15, "Final answer", "final")
    label(ax, 5, -1.75, rule, fs=FS_NOTE, style="normal")
    return task, final


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def single(ax):
    panel(ax, 1, "Single agent", "single")
    task, final = task_and_final(ax, "one model call\n(baseline)")
    agent = Box(ax, 5, 5.8, 5.0, 1.6, "Agent", "generator")
    arrow(ax, task.bottom(), agent.top())
    arrow(ax, agent.bottom(), final.top())


def loop_panel(ax, number, title, key, gen_name, critic_name, out_label, back_label, rule):
    """Generator <-> critic loop inside a dashed box (self-reflection, peer review, adversarial)."""
    panel(ax, number, title, key)
    task, final = task_and_final(ax, rule)
    # dashed box enclosing the loop; straight arrows in from the task and out to the answer
    x0, x1, y0, y1 = 0.25, 9.75, 3.95, 8.55
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0.02,rounding_size=0.45",
                                fc="#FAFAFA", ec="#8C8C8C", lw=1.6, linestyle=(0, (5, 4)),
                                zorder=0))
    arrow(ax, task.bottom(), (5, y1))
    arrow(ax, (5, y0), final.top())
    gen = Box(ax, 2.3, 6.25, 3.3, 2.0, gen_name, "generator")
    crit = Box(ax, 7.7, 6.25, 3.3, 2.0, critic_name, "critic")
    arrow(ax, gen.right(0.35), crit.left(0.35), rad=-0.5)
    arrow(ax, crit.left(-0.35), gen.right(-0.35), rad=-0.5)
    label(ax, 5, 7.9, out_label)
    label(ax, 5, 4.6, back_label)


def self_reflection(ax):
    loop_panel(ax, 2, "Self-reflection", "self_reflection",
               "Agent", "Same agent\nreflects", "proposal", "revise",
               "stops on  VERDICT: APPROVE\nor max_rounds")


def peer_review(ax):
    loop_panel(ax, 3, "Peer review", "peer_review",
               "Architect", "Reviewer", "proposal", "feedback",
               "stops on the Reviewer's\nVERDICT: APPROVE  or max_rounds")


def adversarial(ax):
    loop_panel(ax, 4, "Adversarial", "adversarial",
               "Architect", "Adversary", "proposal", "challenge",
               "stops on the Adversary's\nVERDICT: APPROVE  or max_rounds")


def three_agents(ax, y, name, role="generator"):
    xs = (1.65, 5.0, 8.35)
    names = (f"{name} 1", f"{name} 2", f"{name} N")
    return [Box(ax, x, y, 2.85, 1.3, n, role, fs=FS_BOX - 1.5) for x, n in zip(xs, names)]


def parallel_ensemble(ax):
    panel(ax, 5, "Parallel ensemble", "parallel_ensemble")
    task, final = task_and_final(ax, "experts answer\nindependently, in parallel")
    experts = three_agents(ax, 6.6, "Expert")   # centred between Task and Judge
    for e, rad in zip(experts, (0.12, 0.0, -0.12)):
        arrow(ax, task.bottom(), e.top(), rad=rad)
    judge = Box(ax, 5, 2.5, 5.0, 1.5, "Judge\nsynthesises", "aggregator")
    for e, rad in zip(experts, (0.12, 0.0, -0.12)):
        arrow(ax, e.bottom(), judge.top(), rad=rad)
    arrow(ax, judge.bottom(), final.top())


def debate(ax):
    panel(ax, 6, "Debate", "debate")
    task, final = task_and_final(ax, "stops early once all debaters\nagree, or after max_rounds")
    d = three_agents(ax, 8.15, "Debater")
    for b, rad in zip(d, (0.12, 0.0, -0.12)):
        arrow(ax, task.bottom(), b.top(), rad=rad)
    # rounds 2..R: every debater sees the others' latest answers (arcs under the row)
    kw = dict(color=EXCHANGE, style="<|-|>", lw=1.7)
    arrow(ax, d[0].bottom(0.5), d[1].bottom(-0.5), rad=0.55, **kw)
    arrow(ax, d[1].bottom(0.5), d[2].bottom(-0.5), rad=0.55, **kw)
    arrow(ax, d[0].bottom(-0.35), d[2].bottom(0.35), rad=0.3, **kw)
    label(ax, 5, 5.25, "rounds 2..R: each sees\nthe others' answers", color=EXCHANGE)
    vote = Box(ax, 5, 2.5, 5.0, 1.5, "Majority vote\n(or Judge)", "aggregator")
    arrow(ax, (5, 4.65), vote.top())
    arrow(ax, vote.bottom(), final.top())


# ---------------------------------------------------------------------------

def main(svg=None):
    fig, axes = plt.subplots(2, 3, figsize=(15, 13.4))
    for draw, ax in zip((single, self_reflection, peer_review, adversarial,
                         parallel_ensemble, debate), axes.flat):
        draw(ax)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.115, wspace=0.06, hspace=0.12)

    handles = [Patch(fc=COLORS[r][0], ec=COLORS[r][1], label=lbl) for r, lbl in (
        ("task", "Task"), ("generator", "Generator agent"), ("critic", "Critic role"),
        ("aggregator", "Aggregator"), ("final", "Final answer"))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=FS_LEGEND,
               bbox_to_anchor=(0.5, 0.045), handlelength=1.6, handleheight=1.2,
               columnspacing=1.8)
    fig.text(0.5, 0.008,
             "All topologies: the final answer is the text after 'FINAL ANSWER:'.\n"
             "With execute_code=True, a CodeRunner runs the generators' code and the outputs "
             "are fed back.",
             ha="center", va="bottom", fontsize=FS_FOOT, color=NOTE, linespacing=1.4)

    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"wrote {OUT}")
    if svg:
        fig.savefig(svg, facecolor="white")
        print(f"wrote {svg}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--svg", help="also save a vector copy to this path")
    main(ap.parse_args().svg)
