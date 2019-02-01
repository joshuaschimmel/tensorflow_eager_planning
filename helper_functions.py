import csv
import numpy as np
import matplotlib.pyplot as plt
import pendulum as pend


def min_max_norm(v : float,
                 v_min: float, v_max: float,
                 n_min: float = 0, n_max: float = 1
                 ) -> float:
    """Does min/max normalization.

    :param v: value
    :param v_min: min value
    :param v_max: max value
    :param n_min: new min value default 0
    :param n_max: new max value default 1
    :return: normalised value
    """
    new_v = np.multiply(np.divide((v - v_min), (v_max - v_min)),
                        (n_max - n_min)
                        ) + n_min
    return new_v


def plot_graphs(title: str, plot_list: list) -> None:
    """Plots the elements in plot dicts.

    Acces every plot plot_list's dict and uses "value" and "label"
    keywords for the plotting.

    :param plot_dicts: list of dicts with "values" and "label" kwds.
    :return: returns nothing
    """
    plt.figure()
    plt.suptitle(title)
    for plot_data in plot_list:
        plt.plot(plot_data["values"],
                 plot_data["format"],
                 label=plot_data["label"]
                 )

    plt.ylim(-10, 10)
    plt.grid(b=True, alpha=0.25, linestyle="--")
    plt.tick_params(axis="both", which="major", direction="out")
    plt.legend()
    plt.show()