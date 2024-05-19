#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})


version_original = [r"\texttt{opening\_dataset}", r"\texttt{gridding}", r"\texttt{fourier_transform}"]
times_original = [0.0196, 232.0669, 0.3949]
colors_original = ["dimgray" for _ in range(len(times_original))]

versions_no_dist = [r"\texttt{v1}", r"\texttt{v2}", r"\texttt{v3}", r"\texttt{v4}", r"\texttt{v5}"]
times_no_dist = [66.40, 1.75, 23.63, 26.77, 2.09]
colors_no_dist = ["dimgray" for _ in range(len(times_no_dist))]

v6_nworkers = ["1", "2", "4", "8", "16"]
times_v6 = [2.16, 1.68, 1.61, 1.99, 3.34]
colors_v6 = ["dimgray" for _ in range(len(times_v6))]


def original():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(version_original, times_original, width=0.2, edgecolor="black", linewidth=1, color=colors_original)
    ax.set_title(r"\rmfamily{Original Version}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.set_yscale("log")
    ax.text(-0.1, max(times_original)-100, r'\texttt{opening\_dataset:} 0.02', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.1, max(times_original)-168, r'\texttt{gridding:} 232.06', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.1, max(times_original)-200, r'\texttt{fourier\_transform:} 0.40', ha='left', va='center', fontsize=10, color='black')
    plt.savefig("v0_original.png")

def v1tov5():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(versions_no_dist, times_no_dist, width=0.4, edgecolor="black", linewidth=1, color=colors_no_dist)
    ax.set_title(r"\rmfamily{Versions 1 to 5}")
    ax.set_xlabel(r"\rmfamily{gridding function version}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.text(0.4, max(times_no_dist)-1, r'\texttt{v1_index_cal_jitted:} 66.40', ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-6, r'\texttt{v2_gridding_jitted:} 1.75', ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-11, r'\texttt{v3_single_timestep_vectorized:} 23.63', ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-16, r'\texttt{v4_single_timestep_vectorized_jitted:} 26.77', ha='left', va='center', fontsize=10, color='black')
    ax.text(0.4, max(times_no_dist)-21, r'\texttt{v5_gridding_vectorized:} 2.09', ha='left', va='center', fontsize=10, color='black')
    plt.savefig("v1tov5.png")


def v6():
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(v6_nworkers, times_v6, width=0.4, edgecolor="black", linewidth=1, color="dimgray")
    ax.set_title(r"\rmfamily{Version 6}")
    ax.set_xlabel(r"\rmfamily{number of threads}")
    ax.set_ylabel(r"\rmfamily{Time(s)}")
    ax.text(-0.21, max(times_v6)-0.1, r'1\ \ \ thread: 2.16', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.3, r'2\ \ threads: 1.68', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.5, r'4\ \ threads: 1.61', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.7, r'8\ \ threads: 1.99', ha='left', va='center', fontsize=10, color='black')
    ax.text(-0.21, max(times_v6)-0.9, r'16 threads: 3.34', ha='left', va='center', fontsize=10, color='black')
    plt.savefig("v6.png")



def get_plot(func, *args):
    plot = funcs_dict.get(func)(*args)
    return plot

if __name__ == "__main__":
    funcs_dict = {"original": original,
                  "v1tov5": v1tov5,
                  "v6": v6}
    funcs_list = list(funcs_dict.keys())

    plot_name = sys.argv[1]
    if plot_name not in funcs_list:
        print("plot name should be one of: {}".format(', '.join(funcs_list)))
        sys.exit(1)

    get_plot(plot_name)

