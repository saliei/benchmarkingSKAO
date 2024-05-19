#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

original_version = [r"\texttt{opening\_dataset}", r"\texttt{gridding}", r"\texttt{fourier_transform}"]
original_times = [0.0196, 232.0669, 0.3949]
original_colors = ["darkgray", "dimgray", "gray"]

#versions_no_dist = ["v0", "v1", "v2", "v3", "v4", "v5"]
#times_no_dist = []

def original():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(original_version, original_times, width=0.2, edgecolor="black", linewidth=1, color=original_colors)
    ax.set_title(r"\rmfamily{Original Version}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.set_yscale("log")
    ax.text(-0.1, max(original_times)-100, r'\texttt{opening\_dataset:} 0.02', ha='left', va='center', fontsize=9.5, color='black')
    ax.text(-0.1, max(original_times)-165, r'\texttt{gridding:} 232.06', ha='left', va='center', fontsize=9.5, color='black')
    ax.text(-0.1, max(original_times)-200, r'\texttt{fourier\_transform:} 0.40', ha='left', va='center', fontsize=9.5, color='black')
    plt.show()

def get_plot(func, *args):
    funcs_dict = {"original": original}
    plot = funcs_dict.get(func)(*args)
    return plot

if __name__ == "__main__":
    plot_name = sys.argv[1]
    get_plot(plot_name)

