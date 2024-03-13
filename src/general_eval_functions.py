from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np
import matplotlib as mpl
import copy 
from torch.distributions.multivariate_normal import MultivariateNormal

import cvxpy
from cvxpy import Variable, Minimize, Problem
from cvxpy import norm as cvxnorm
from cvxpy import vec as cvxvec

from eval import get_run_metrics, read_run_dir, get_model_from_run
from models import GDModel
from base_models import TwoLayerNeuralNetwork, ThreeLayerNeuralNetwork
from samplers import get_data_sampler
from tasks import get_task_sampler
from sklearn.linear_model import LinearRegression, Lasso, LassoCV


sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')
mpl.rcParams['figure.dpi'] = 300

run_dir = "../models"

# Ensure that all reference errors return a numpy array of shape (n_points-1,)
def get_prompts_from_task(eval_prompts_task):
    prompts_from_task = None
    if eval_prompts_task == "relu_2nn_regression" or eval_prompts_task == "relu_3nn_regression":
        prompts_from_task = "neural_network" 
    elif eval_prompts_task == "linear_regression":
        prompts_from_task = "linear_function"
    return prompts_from_task

def get_nn_layer_from_task(eval_prompts_task):
    nn_layer = None
    if eval_prompts_task == "relu_2nn_regression":
        nn_layer = 2
    elif eval_prompts_task == "relu_3nn_regression":
        nn_layer = 3
    else:
        raise NotImplementedError
    return nn_layer

def get_learning_rate_for_GD(prompts_from_task):
    lr = None
    if prompts_from_task == "neural_network":
        lr = 5e-3 
    elif prompts_from_task == "linear_function":
        lr = 5e-2
    return lr

def get_gradient_descent_reference_preds(xs, ys, args, train_steps=5000):
    nn_model=0
    if args["nn_layer_num"] == 2:
        nn_model = TwoLayerNeuralNetwork
    elif args["nn_layer_num"] == 3:
        nn_model = ThreeLayerNeuralNetwork
    else:
        raise NotImplementedError

    gd_model = GDModel(nn_model, {'in_size': args["n_dims"], 'hidden_size': args["hidden_size"], 'out_size' :1}, opt_alg = 'adam', batch_size = 10, lr = args["learning_rate"], num_steps = train_steps)
    return gd_model(xs, ys, verbose=True)

def get_gradient_descent_preds_and_reference_errors(xs, ys, args, train_steps=5000):
    gd_preds = get_gradient_descent_reference_preds(xs, ys, args, train_steps=train_steps).detach().cpu()
    return gd_preds[:,1:], (ys[:,1:]-gd_preds[:,1:]).square().mean(axis = 0).numpy()


def get_linear_regression_rerefence_errors(xs, ys): # xs -> b, p, d; ys -> b, p
    # Least Squares Optimization
    lsq_errors = []
    for i in tqdm(range(1, xs.shape[1])):
        preds = []
        for batch_id in range(xs.shape[0]):
            preds.append(
            # fit n_points -1 regressors for each entry in batch
            LinearRegression(fit_intercept = False).fit(xs[batch_id,:i], ys[batch_id,:i])\
                .predict(xs[batch_id,i:i+1])[0]
            )
        preds = np.array(preds).squeeze()
        lsq_errors.append(((ys[:,i] - preds)**2).mean(axis = 0).numpy())
    return np.array(lsq_errors) # p-1

def get_linear_regression_rerefence_errors_no_mean(xs, ys): # xs -> b, p, d; ys -> b, p
    # Least Squares Optimization
    lsq_errors = []
    lsq_preds = []
    for i in tqdm(range(1, xs.shape[1])):
        preds = []
        for batch_id in range(xs.shape[0]):
            preds.append(
            # fit n_points -1 regressors for each entry in batch
            LinearRegression(fit_intercept = False).fit(xs[batch_id,:i], ys[batch_id,:i])\
                .predict(xs[batch_id,i:i+1])[0]
            )
        preds = np.array(preds).squeeze()
        lsq_errors.append(((ys[:,i] - preds)**2).numpy())
        lsq_preds.append(preds)
    return np.array(lsq_errors).T, np.array(lsq_preds).T # p-1


def get_sparse_regression_reference_errors_no_mean(xs, ys):
    lasso_errors = []
    lasso_preds = []
    for i in tqdm(range(1, xs.shape[1])):
        preds = []
        for batch_id in range(xs.shape[0]):
            if i < 5:
                preds.append(
                Lasso(fit_intercept = False).fit(xs[batch_id,:i], ys[batch_id,:i])\
                    .predict(xs[batch_id,i:i+1])[0]
                )
            else:
                preds.append(
                LassoCV(fit_intercept = False).fit(xs[batch_id,:i], ys[batch_id,:i])\
                    .predict(xs[batch_id,i:i+1])[0]
                )
        preds = np.array(preds).squeeze()
        lasso_errors.append(((ys[:,i] - preds)**2).numpy())
        lasso_preds.append(preds)
    return np.array(lasso_errors).T, np.array(lasso_preds).T

def get_sparse_regression_reference_errors(xs, ys):
    lasso_errors = []
    for i in tqdm(range(1, xs.shape[1])):
        preds = []
        for batch_id in range(xs.shape[0]):
            if i < 5:
                preds.append(
                Lasso(fit_intercept = False).fit(xs[batch_id,:i], ys[batch_id,:i])\
                    .predict(xs[batch_id,i:i+1])[0]
                )
            else:
                preds.append(
                LassoCV(fit_intercept = False).fit(xs[batch_id,:i], ys[batch_id,:i])\
                    .predict(xs[batch_id,i:i+1])[0]
                )
        preds = np.array(preds).squeeze()
        lasso_errors.append(((ys[:,i] - preds)**2).mean(axis = 0).numpy())
    return np.array(lasso_errors)

def get_sign_vec_cs_reference_errors(xs, ys, n_dims):
    # Inf Norm Optimization
    mat_dim = int(np.sqrt(xs.shape[2]))
    baseline_errors_batch = []
    for b in tqdm(range(xs.shape[0])):
        errors = []
        for t in range(xs.shape[1] - 1):
            w_star = Variable([n_dims, 1])
            obj = Minimize(cvxnorm(w_star, 'inf'))
            constraints = [ys[b,:t+1].numpy()[:,np.newaxis] == (xs[b,:t+1].numpy() @ w_star)]
            prob = Problem(obj, constraints)
            result = prob.solve()#verbose=True)
            if prob.status == cvxpy.OPTIMAL:
                pred = w_star.value[:,0] @ xs[b,t+1].numpy()
                errors.append((pred - ys[b,t+1].numpy())**2)
            else:
                errors.append(prob.value)
        baseline_errors_batch.append(errors)
    return np.array(baseline_errors_batch).mean(0)

def getMSE(a, b):
    return ((a-b)**2).mean(axis=0)

def get_plot_label_MSE_wrt_GT(basline_name):
    return f"|MSE(TF_pred, GT) - MSE({basline_name}, GT)|"

def plot_prediction_diff_wrt_ground_truth(
    transformer_loss: list, 
    task_losses: dict,
    task_kwargs: dict = {}
):
    # |MSE(TF_pred, GT) - MSE(Baseline_pred, GT)|
    x_axis_items = np.arange(1, transformer_loss.shape[0]+1)
    # plt.plot(x_axis_items, transformer_preds, lw=2, label="Transformer")
    if "sign_vec_cs" in task_losses:
        pred_diff_wrt_GT = np.abs(transformer_loss - task_losses["sign_vec_cs"])
        plt.plot(x_axis_items, pred_diff_wrt_GT, label = get_plot_label_MSE_wrt_GT("Inf Norm Minimization"))
        plt.scatter(task_kwargs["sign_vec_cs"]["bound"] + 1,0, color="red", label="Bound")
    if "linear_regression" in task_losses:
        pred_diff_wrt_GT = np.abs(transformer_loss - task_losses["linear_regression"])
        plt.plot(x_axis_items, pred_diff_wrt_GT, lw=2, label = get_plot_label_MSE_wrt_GT("Least Squares"))
    if "sparse_regression" in task_losses:
        pred_diff_wrt_GT = np.abs(transformer_loss - task_losses["sparse_regression"])
        plt.plot(x_axis_items, pred_diff_wrt_GT, lw=2, label = get_plot_label_MSE_wrt_GT("Lasso"))
    if "relu_2nn_regression" in task_losses:
        pred_diff_wrt_GT = np.abs(transformer_loss - task_losses["relu_2nn_regression"])
        plt.plot(x_axis_items, pred_diff_wrt_GT, lw=2, label = get_plot_label_MSE_wrt_GT("2-layer NN (GD)"))
    if "relu_3nn_regression" in task_losses:
        pred_diff_wrt_GT = np.abs(transformer_loss - task_losses["relu_3nn_regression"])
        plt.plot(x_axis_items, pred_diff_wrt_GT, lw=2, label = get_plot_label_MSE_wrt_GT("3-layer NN (GD)"))

    # plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples")
    plt.ylabel("|MSE(TF_pred, GT) - MSE(Baseline_pred, GT)|")
    plt.legend()
    plt.show()

def plot_prediction_diff(
    transformer_preds: list, # b, n_points-1
    task_preds: dict, # task_preds["<task name>"] is of shape (b, n_points-1)
    task_kwargs: dict = {}
):
    # MSE(TF_pred, TaskReference_pred)
    x_axis_items = np.arange(1, transformer_preds.shape[1]+1)
    # plt.plot(x_axis_items, transformer_preds, lw=2, label="Transformer")
    if "sign_vec_cs" in task_preds:
        pred_diff = getMSE(transformer_preds, task_preds["sign_vec_cs"])
        plt.plot(x_axis_items, pred_diff, label = "MSE(TF_preds, Inf Norm Minimization preds")
        plt.scatter(task_kwargs["sign_vec_cs"]["bound"] + 1,0, color="red", label="Bound")
    if "linear_regression" in task_preds:
        pred_diff = getMSE(transformer_preds, task_preds["linear_regression"])
        plt.plot(x_axis_items, pred_diff, lw=2, label = "MSE(TF_preds, Least Squares preds)")
    if "sparse_regression" in task_preds:
        pred_diff = getMSE(transformer_preds, task_preds["sparse_regression"])
        plt.plot(x_axis_items, pred_diff, lw=2, label = "MSE(TF_preds, Lasso preds")
    if "relu_2nn_regression" in task_preds:
        pred_diff = getMSE(transformer_preds, task_preds["relu_2nn_regression"])
        plt.plot(x_axis_items, pred_diff, lw=2, label = "MSE(TF_preds, 2-layer NN (GD) preds)")
    if "relu_3nn_regression" in task_preds:
        pred_diff = getMSE(transformer_preds, task_preds["relu_3nn_regression"])
        plt.plot(x_axis_items, pred_diff, lw=2, label = "MSE(TF_preds, 3-layer NN (GD) preds)")

    # plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples")
    plt.ylabel("MSE(TF_pred, Baseline_pred)")
    plt.legend()
    plt.show()

def get_df_from_pred_array(pred_arr):
    # pred_arr --> b x pts-1
    b=pred_arr.shape[0]
    p=pred_arr.shape[1]
    flattened_arr = pred_arr.ravel()
    points = np.array(list(range(1,p+1))*b)
    df = pd.DataFrame({'y': flattened_arr, 'x': points})
    return df

def plot_results_with_sd_bars(
    transformer_loss: list, 
    task_losses: dict,
    task_kwargs: dict = {}
):
    # all the loss arrays passed to this method must be of shape batch_size x n_points-1
    fig = plt.figure()
    ax = plt.gca()
    
    sns.lineplot(data=get_df_from_pred_array(transformer_loss), y="y",x="x",ci='sd', lw=2, label="Transformer", ax=ax)
    if "sign_vec_cs" in task_losses:
        sns.lineplot(data=get_df_from_pred_array(task_losses["sign_vec_cs"]), y="y",x="x",ci='sd', lw=2, label="Inf Norm Minimization", ax=ax)
        plt.scatter(task_kwargs["sign_vec_cs"]["bound"] + 1,0, color="red", label="Bound", ax=ax)
    if "linear_regression" in task_losses:
        sns.lineplot(data=get_df_from_pred_array(task_losses["linear_regression"]), y="y",x="x",ci='sd', lw=2, label="Least Squares", ax=ax)
    if "sparse_regression" in task_losses:
        sns.lineplot(data=get_df_from_pred_array(task_losses["sparse_regression"]), y="y",x="x",ci='sd', lw=2, label="Lasso", ax=ax)
    if "relu_2nn_regression" in task_losses:
        sns.lineplot(data=get_df_from_pred_array(task_losses["relu_2nn_regression"]), y="y",x="x",ci='sd', lw=2, label="2-layer NN, GD", ax=ax)
    if "relu_3nn_regression" in task_losses:
        sns.lineplot(data=get_df_from_pred_array(task_losses["relu_3nn_regression"]), y="y",x="x",ci='sd', lw=2, label="3-layer NN, GD", ax=ax)

    # plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples")
    plt.ylabel("squared error")
    plt.legend()
    plt.show()

def plot_results(
        plot_title,
        points_to_plot,
    transformer_loss: list, 
    task_losses: dict,
    task_kwargs: dict = {}, **kwargs
):
    # x_axis_items = np.arange(1, transformer_loss.shape[0]+1)
    plot_from_index_x_axis = kwargs["plot_from_index_x_axis"] if "plot_from_index_x_axis" in kwargs else 1
    x_axis_items = np.arange(plot_from_index_x_axis, points_to_plot)

    plt.plot(x_axis_items, transformer_loss[:points_to_plot], lw=2, label="Transformer")
    if "sign_vec_cs" in task_losses:
        plt.plot(x_axis_items, task_losses["sign_vec_cs"][:points_to_plot], label = "Inf Norm Minimization")
        plt.scatter(task_kwargs["sign_vec_cs"]["bound"] + 1,0, color="red", label="Bound")
    if "linear_regression" in task_losses:
        plt.plot(x_axis_items, task_losses["linear_regression"][:points_to_plot], lw=2, label = "Least Squares")
    if "sparse_regression" in task_losses:
        plt.plot(x_axis_items, task_losses["sparse_regression"][:points_to_plot], lw=2, label = "Lasso")
    if "relu_2nn_regression" in task_losses:
        plt.plot(x_axis_items, task_losses["relu_2nn_regression"][:points_to_plot], lw=2, label = "2-layer NN, GD")
    if "relu_3nn_regression" in task_losses:
        plt.plot(x_axis_items, task_losses["relu_3nn_regression"][:points_to_plot], lw=2, label = "3-layer NN, GD")
    if "PME T1" in task_losses:
        plt.plot(x_axis_items, task_losses["PME T1"][:points_to_plot], lw=2, label = "PME T1")
    if "PME T2" in task_losses:
        plt.plot(x_axis_items, task_losses["PME T2"][:points_to_plot], lw=2, label = "PME T2")
    if "PME Mix." in task_losses:
        plt.plot(x_axis_items, task_losses["PME Mix."][:points_to_plot], lw=2, label = "PME Mix.")


    # plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    if "xticklabels" in kwargs:
        plt.xticks(kwargs["xticklabels"])
    plt.xlabel("# in-context examples")
    plt.ylabel("squared error")
    plt.title(plot_title)
    plt.legend()
    plt.show()

def plot_results_multiple_LR(
    transformer_losses: list, # list of many transformer losses say k
    lr_losses: list, # list of k reference LR losses
    suffixes = ["-ID", "-OOD"]
):
    x_axis_items = np.arange(1, transformer_losses[0].shape[0]+1)
    for tloss, sfx in zip(transformer_losses, suffixes):
        plt.plot(x_axis_items, tloss, lw=2, label="Transformer"+sfx)
    for lsq_loss, sfx in zip(lr_losses, suffixes):
        plt.plot(x_axis_items, lsq_loss, lw=2, label = "Least Squares"+sfx)

    # plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
    plt.xlabel("# in-context examples")
    plt.ylabel("squared error")
    plt.legend()
    plt.show()

def compute_average_loss_difference(loss1, loss2, printString):
    assert loss1.shape == loss2.shape, "loss1 and loss2 must have same length"
    print(f"Average difference between losses {printString}:", np.abs(loss1 - loss2).mean())

def average_loss_from_index(ind, loss, modelName):
    print(f"Average loss for {modelName} from index {ind} is:", loss[ind:].mean())


def load_GMM_model_with_run_id(run_id):
    task = "gaussian_mixture_linear_regression"
    run_path = os.path.join(run_dir, task, run_id)
    recompute_metrics = False

    if recompute_metrics:
        get_run_metrics(run_path)  # these are normally precomputed at the end of training

    model, conf = get_model_from_run(run_path)

    n_dims = conf.model.n_dims

    mean = torch.zeros(size=(n_dims,))
    try:
        mean_1st_dim = conf.training.task_kwargs["gaussian_centre_abs"]
        print(f"Means of loaded model: {mean_1st_dim}, {-mean_1st_dim}")
        mean[0] = mean_1st_dim
    except KeyError:
        mean[0] = 1
        print("Assuming gaussian_centre_abs = 1")
    cov = torch.eye(n_dims)
    cov[0,0] = 1e-8
    distrib1 = MultivariateNormal(loc=mean, covariance_matrix=cov)
    distrib2 = MultivariateNormal(loc=-mean, covariance_matrix=cov)
    conf.training.task_kwargs["distrib1"] = distrib1
    conf.training.task_kwargs["distrib2"] = distrib2
    
    return model, conf, n_dims, conf.training.task_kwargs["mixing_ratio"], distrib1, distrib2, mean

def sample_xy(n_points, batch_size, conf, n_dims):
    task_kwargs_copy = copy.deepcopy(conf.training.task_kwargs)

    data_sampler = get_data_sampler(conf.training.data, n_dims)
    xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)
    ys_t1, ys_t2, task_t1, task_t2, w_b_t1, w_b_t2 = None, None, None, None, None, None
    print("xs shape:", xs.shape)
    print("model task:", conf.training.task)
    for prompts_from_task in [0,1]:
        task_kwargs_copy["mixing_ratio"] = 1-prompts_from_task
        task_sampler = get_task_sampler(
            conf.training.task,
            n_dims,
            batch_size,
            **task_kwargs_copy
        )
        task = task_sampler()
        print("prompts sampled from Gaussian component:", prompts_from_task)
        if prompts_from_task == 0:
            ys_t1 = task.evaluate(xs)
            task_t1 = task
            w_b_t1 = task.w_b
            print("ys_t1 shape:", ys_t1.shape)
        else:
            ys_t2 = task.evaluate(xs)
            task_t2 = task
            w_b_t2 = task.w_b
            print("ys_t2 shape:", ys_t2.shape)

    # sample xs, ys
    return xs, ys_t1, ys_t2, task_t1, task_t2, w_b_t1, w_b_t2


def get_TF_and_LSQ(model, xs, ys, task):
# get TF and baseline preds
    model.eval()
    with torch.no_grad():
        transformer_pred = model(xs, ys)

    lsq_errors, lsq_preds = get_linear_regression_rerefence_errors_no_mean(xs, ys)

    # plot with error bars
    metric = task.get_metric()
    transformer_loss_no_mean = metric(transformer_pred, ys).numpy()[:,1:]

    return lsq_errors, lsq_preds, transformer_pred, transformer_loss_no_mean

def get_TF_preds(model, xs, ys, task, from_index=1):
    model.eval()
    with torch.no_grad():
        transformer_pred = model(xs, ys)
    metric = task.get_metric()
    transformer_loss_no_mean = metric(transformer_pred, ys).numpy()[:,from_index:]
    return transformer_pred, transformer_loss_no_mean

def plot_all(lsq_errors, lsq_preds, transformer_pred, transformer_loss_no_mean, 
        pme_t1, t1_errors_batch_no_mean, t1_preds_batch_no_mean,
        pme_t2, t2_errors_batch_no_mean, t2_preds_batch_no_mean,
        pme_mix, pmix_errors_batch_no_mean, pmix_preds_batch_no_mean,
        points_to_plot,
        plot_title,
        no_LSQ=False,
        to_plot=["errorbar", "usual-lineplot", "MSE_diff_wrt_GT", "MSE_diff_direct"], **kwargs):
    if "errorbar" in to_plot:
        plot_results_with_sd_bars(
            transformer_loss_no_mean, 
            task_losses={
                # eval_prompts_task: gradient_descent_errors}
                        "linear_regression": lsq_errors, }
                        # "sparse_regression": lasso_errors,}
            #             "sign_vec_cs": baseline_errors_batch},
            # task_kwargs={"sign_vec_cs": {"bound": n_dims//2}}
        )


    # plot without error bars
    transformer_loss_mean = transformer_loss_no_mean.mean(axis=0)

    if "usual-lineplot" in to_plot:
    # MSE(TF/Baseline pred, Ground Truth)
        plot_results(
            plot_title,
            points_to_plot,
            transformer_loss_mean, 
            task_losses={
                # eval_prompts_task: gradient_descent_errors}
                        "linear_regression": lsq_errors.mean(axis=0), 
                        "PME T1": t1_errors_batch_no_mean.mean(axis=0), 
                        "PME T2": t2_errors_batch_no_mean.mean(axis=0), 
                        "PME Mix.": pmix_errors_batch_no_mean.mean(axis=0),
                        } if not no_LSQ else {
                # eval_prompts_task: gradient_descent_errors}
                        # "linear_regression": lsq_errors.mean(axis=0), 
                        "PME T1": t1_errors_batch_no_mean.mean(axis=0), 
                        "PME T2": t2_errors_batch_no_mean.mean(axis=0), 
                        "PME Mix.": pmix_errors_batch_no_mean.mean(axis=0),
                        }, **kwargs 
                        # "sparse_regression": lasso_errors,}
            #             "sign_vec_cs": baseline_errors_batch},
            # task_kwargs={"sign_vec_cs": {"bound": n_dims//2}}
        )

    # print(np.sum(gradient_descent_errors == transformer_loss))

    if "MSE_diff_wrt_GT" in to_plot:
    # |MSE(TF_pred, GT) - MSE(Baseline_pred, GT)|
        plot_prediction_diff_wrt_ground_truth(
            transformer_loss_mean, # 64, 100
            task_losses={"linear_regression": lsq_errors.mean(axis=0)} # 64, 100
                        # "linear_regression": lsq_errors, 
                        # "sparse_regression": lasso_errors,}
            #             "sign_vec_cs": baseline_errors_batch},
            # task_kwargs={"sign_vec_cs": {"bound": n_dims//2}}
        )

    # print(np.sum(gradient_descent_errors == transformer_loss))

    if "MSE_diff_direct" in to_plot:
    # MSE(TF_pred, Baseline_pred)
        plot_prediction_diff(
            transformer_pred[:,1:], # 64, 100
            task_preds={"linear_regression": lsq_preds} # 64, 100
                        # "linear_regression": lsq_errors, 
                        # "sparse_regression": lasso_errors,}
            #             "sign_vec_cs": baseline_errors_batch},
            # task_kwargs={"sign_vec_cs": {"bound": n_dims//2}}
        )

    # print(np.sum(gradient_descent_errors == transformer_loss))


def sns_plot_heatmap(w_plot, figsize, cmap, text_dict, yticklabels='auto', annot=False, round_to_digits=3, is_numpy=False):
    # w_plot shd be (a, b) shape
    # a is plotted along x-axis
    # b is plotted along y-axis
    fig, ax = plt.subplots(figsize=figsize)
    # fig.suptitle("Recovered Weight Vectors - First Coordinate")
    # fig.supylabel("Prompt length")
    # fig.supxlabel("Batch")
    if is_numpy:
        sns.heatmap(w_plot.round(round_to_digits).transpose(),linewidth=.3, annot=annot, annot_kws={"size":4}, cmap=cmap, ax=ax, yticklabels=yticklabels)
    else:
        sns.heatmap(w_plot.numpy().round(round_to_digits).transpose(),linewidth=.3, annot=annot, annot_kws={"size":4}, cmap=cmap, ax=ax, yticklabels=yticklabels)
    ax.set_title(text_dict["title"])
    ax.set_ylabel(text_dict["ylabel"])
    ax.set_xlabel(text_dict["xlabel"])