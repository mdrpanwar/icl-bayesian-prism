from collections import OrderedDict
import re
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names
from samplers import get_data_sampler
from tasks import get_task_sampler

import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, SGDRegressor
import numpy as np
import cvxpy
from cvxpy import Variable, Minimize, Problem
from cvxpy import norm as cvxnorm
from cvxpy import vec as cvxvec

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['text.usetex'] = True

matplotlib.rcParams.update({
    'axes.titlesize': 8,
  'figure.titlesize': 10,
  'legend.fontsize': 10,
  'xtick.labelsize': 6,
  'ytick.labelsize': 6,
})
run_dir = "../models"

seed=42
torch.manual_seed(seed)

SPINE_COLOR = 'gray'

def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)
    return ax

def get_df_from_pred_array(pred_arr, n_points, offset = 0):
    # pred_arr --> b x pts-1
    batch_size=pred_arr.shape[0]
    flattened_arr = pred_arr.ravel()
    points = np.array(list(range(offset, n_points)) * batch_size)
    df = pd.DataFrame({'y': flattened_arr, 'x': points})
    return df

def lineplot_with_ci(pred_or_err_arr, n_points, offset, label, ax, seed, linewidth=1):
    sns.lineplot(data=get_df_from_pred_array(pred_or_err_arr, n_points=n_points, offset = offset), 
                y="y", x="x",
                label=label, 
                ax=ax, n_boot=1000, 
                seed=seed, 
                ci=90,
                linewidth=linewidth,
    )

def do_pickle_plot_core(filename, dict_to_pickle):
    with open(f'{filename}.pkl', 'wb') as handle:
        pickle.dump(dict_to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)

def get_pickle_plot_core(filename):
    b=0
    with open(f'{filename}.pkl', 'rb') as handle:
        b = pickle.load(handle)
    return b

def sns_plot_heatmap_paper(w_plot, cmap, text_dict, xticklabels='auto', yticklabels='auto', annot=False, round_to_digits=3, is_numpy=False, ax=None, annot_size=4, **kwargs):
    # w_plot shd be (a, b) shape
    # a is plotted along x-axis
    # b is plotted along y-axis
    if is_numpy:
        sns.heatmap(w_plot.round(round_to_digits).transpose(),linewidth=.3, annot=annot, annot_kws={"size":annot_size}, cmap=cmap, ax=ax, yticklabels=yticklabels, xticklabels=xticklabels, **kwargs)
    else:
        sns.heatmap(w_plot.numpy().round(round_to_digits).transpose(),linewidth=.3, annot=annot, annot_kws={"size":annot_size}, cmap=cmap, ax=ax, yticklabels=yticklabels, xticklabels=xticklabels, **kwargs)
    ax.set_title(text_dict["title"])
    ax.set_ylabel(text_dict["ylabel"])
    ax.set_xlabel(text_dict["xlabel"])

def sns_lineplot_paper(xs, n_dims, plot_tsr, plot_suffix, save_filename, dict_to_pickle, axes=2, figsize=(10, 5), font_scale=1.5, ylim_ax=1.5, legend_pos=(1.15, 0.95)):
    sns.set(style = "whitegrid", font_scale=font_scale)
    n_points=xs.shape[1]

    fig, ax1, ax2, ax3 = 0,0,0,0
    if axes == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=figsize, constrained_layout=True)
    elif axes == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=figsize, constrained_layout=True)
    
    if axes >= 1:
        for tsr_tup in plot_tsr["ax1"]:
            lineplot_with_ci(tsr_tup[0] /n_dims , n_points, offset = 0, label=tsr_tup[1], ax=ax1, seed=seed, linewidth=1.75)

        ax1.set_xlabel("$k$\n(\# in-context examples)")
        ax1.set_ylabel("$\\texttt{loss@}k$")
        ax1.set_title("Evaluation on $T_1$ prompts ($\mathbf{\mathit{w}} \sim \mathcal{N}_{d}(\mu_1, \Sigma_1)$)")
        format_axes(ax1)

    if axes >= 2:
        for tsr_tup in plot_tsr["ax2"]:
            lineplot_with_ci(tsr_tup[0] /n_dims , n_points, offset = 0, label=tsr_tup[1], ax=ax2, seed=seed, linewidth=1.75)

        ax2.set_xlabel("$k$\n(\# in-context examples)")
        ax2.set_ylabel("$\\texttt{loss@}k$")
        ax2.set_title("Evaluation on $T_2$ prompts ($\mathbf{\mathit{w}} \sim \mathcal{N}_{d}(\mu_2, \Sigma_2)$)")
        format_axes(ax2)

    ax1.set_ylim([-0.1, ylim_ax])
    ax2.set_ylim([-0.1, ylim_ax])

    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=legend_pos)
    for line in leg.get_lines():
        line.set_linewidth(5)
        
    
    plt.savefig(f"final_plots/{save_filename}_{plot_suffix}.pdf", dpi = 300, bbox_inches = "tight")
    
    do_pickle_plot_core(os.path.join("final_plot_pickles", f"{save_filename}_{plot_suffix}"), dict_to_pickle)
    plt.show()


def sns_lineplot_paper_evolution(xs, n_dims, plot_tsr, plot_suffix, save_filename, dict_to_pickle, axes=2, figsize=(10, 5), font_scale=1.5, ylim_ax=(-0.1, 1.5), legend_pos=(1.15, 0.95), ylabel=""):
    sns.set(style = "whitegrid", font_scale=font_scale)
    n_points=xs.shape[1]

    fig, ax1, ax2, ax3 = 0,0,0,0
    if axes == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2,figsize=figsize, constrained_layout=True)
    elif axes == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=figsize, constrained_layout=True)
    
    if axes >= 1:
        for tsr_tup in plot_tsr["ax1"]:
            lineplot_with_ci(tsr_tup[0] , n_points, offset = 0, label=tsr_tup[1], ax=ax1, seed=seed, linewidth=1.75)

        ax1.set_xlabel("$k$\n(\# in-context examples)")
        ax1.set_ylabel(ylabel)
        ax1.set_title("On $T_1$ prompts")
        format_axes(ax1)

    if axes >= 2:
        for tsr_tup in plot_tsr["ax2"]:
            lineplot_with_ci(tsr_tup[0] , n_points, offset = 0, label=tsr_tup[1], ax=ax2, seed=seed, linewidth=1.75)

        ax2.set_xlabel("$k$\n(\# in-context examples)")
        ax2.set_ylabel(ylabel)
        ax2.set_title("On $T_2$ prompts")
        format_axes(ax2)

    ax1.set_ylim([ylim_ax[0], ylim_ax[1]])
    ax2.set_ylim([ylim_ax[0], ylim_ax[1]])

    ax1.legend().set_visible(False)
    ax2.legend().set_visible(False)

    handles, labels = ax1.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=legend_pos)
    for line in leg.get_lines():
        line.set_linewidth(5)
        
    
    plt.savefig(f"final_plots/{save_filename}_{plot_suffix}.pdf", dpi = 300, bbox_inches = "tight")
    
    do_pickle_plot_core(os.path.join("final_plot_pickles", f"{save_filename}_{plot_suffix}"), dict_to_pickle)
    plt.show()



def sns_plot_evolution_heatmap(xs, cbar_lims, plot_tsr_1, plot_tsr_2, title_plot_1, title_plot_2, plot_suffix, save_filename, dict_to_pickle, font_scale=1.5, is_numpy=[True, True], annot=[False, False], annot_sizes=[8,8]):
    sns.set(style = "whitegrid", font_scale=font_scale)
    n_points=xs.shape[1]
    num_funcs=10
    cbar_ll, cbar_ul = cbar_lims[0], cbar_lims[1]
    cmap="icefire"


    fig, (ax1, ax2, axcb) = plt.subplots(1, 3,figsize=(10, 5), constrained_layout=True, gridspec_kw={'width_ratios':[1,1,0.08]})

    ax1.get_shared_y_axes().join(ax2)
    sns_plot_heatmap_paper(plot_tsr_1[:num_funcs, :], cmap=cmap, text_dict={
            "title": title_plot_1,
            "ylabel": "$k$ (\# in-context examples)",
            "xlabel": "Samples of $w$"}, 
        annot=annot[0], annot_size=annot_sizes[0], is_numpy=is_numpy[0], ax=ax1, xticklabels=range(1, num_funcs+1), cbar=False, vmin=cbar_ll, vmax=cbar_ul)

    format_axes(ax1)

    sns_plot_heatmap_paper(plot_tsr_2[:num_funcs, :], cmap=cmap, text_dict={
            "title": title_plot_2,
            "ylabel": "$k$ (\# in-context examples)",
            "xlabel": "Samples of $w$"}, 
        annot=annot[1], annot_size=annot_sizes[1], is_numpy=is_numpy[1], ax=ax2, xticklabels=range(1, num_funcs+1), cbar_ax=axcb, vmin=cbar_ll, vmax=cbar_ul)

    ax2.set_ylabel("")

    format_axes(ax2)
        
    plt.savefig(f"final_plots/{save_filename}_{plot_suffix}.pdf", dpi = 300, bbox_inches = "tight")
    
    do_pickle_plot_core(os.path.join("final_plot_pickles", f"{save_filename}_{plot_suffix}"), dict_to_pickle)


def matplotlib_fontsize_set():
    sns.set(font_scale = 2.5)
    matplotlib.rcParams.update({
        'axes.titlesize': 25,
        'figure.titlesize': 17,
        'legend.fontsize': 15,
        'xtick.labelsize': 25,
        'ytick.labelsize': 25,
    })
