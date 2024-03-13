from collections import OrderedDict
import re
import os

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


from  torch.distributions import multivariate_normal

global model, n_dims, distrib1, distrib2, mean, multivariate_standard_normal_dists

def set_all_pme_comp_globals(mdl, nd, dist1, dist2, mn):
    global model, n_dims, distrib1, distrib2, mean, multivariate_standard_normal_dists
    model = mdl
    n_dims = nd
    distrib1 = dist1
    distrib2 = dist2
    mean = mn
    multivariate_standard_normal_dists = [multivariate_normal.MultivariateNormal(loc=torch.zeros(size=(k,)), covariance_matrix=torch.eye(k)) for k in range(1,n_dims+1)]

def print_all_globals():
    global model, n_dims, distrib1, distrib2, mean, multivariate_standard_normal_dists
    print(f"n_dims: {n_dims}")
    print(f"distrib1: mean:{[int(x) for x in distrib1.loc.tolist()]}, cov_diag:{[int(x) for x in torch.diagonal(distrib1.covariance_matrix).tolist()]}")
    print(f"distrib2: mean:{[int(x) for x in distrib2.loc.tolist()]}, cov_diag:{[int(x) for x in torch.diagonal(distrib2.covariance_matrix).tolist()]}")
    print(f"mean: {[int(x) for x in mean.tolist()]}")
    print(f"model: {model}")


def evaluate_standard_multivariate_normal(d, x):
    assert x.shape[0] == d
    assert d>0 and d<=n_dims
    return torch.exp(multivariate_standard_normal_dists[d-1].log_prob(x)).detach().cpu().numpy()
    
def evaluate_normal_given_mean_covar(component, x):
    if component == "mean_plus_one":
        return torch.exp(distrib1.log_prob(x)).detach().cpu().numpy()
    else:
        return torch.exp(distrib2.log_prob(x)).detach().cpu().numpy()

def extend_last(w_batch_arr, remaining_pts):
    return w_batch_arr[:,-1][:,np.newaxis].repeat(remaining_pts,axis=1)
    # copies the last vector in second dim remaining_pts times. Returns an array of shape (w_batch_arr.shape[0], remaining_pts)


### INDIVIDUAL PME COMPUTATION ###
def compute_PME_and_pTP(computing_for_component, xs, ys, prompts_from_component): # xs can at most has n_dims + 1 points; We use a max of n_dims points to solve LPP and make predictions for the next point (until the n_dims+1-th point)
    if computing_for_component == "mean_plus_one":
        center_of_gaussian = mean.detach().cpu().numpy()
    else:
        center_of_gaussian = (-mean).detach().cpu().numpy()
    w_batch = []
    preds_batch = []
    errors_batch = []
    
    ptp_batch = []
    n_dims = xs.shape[2]
    pts_input = xs.shape[1]
    batch_size = xs.shape[0]
    for b in tqdm(range(batch_size)):
        ptps = []
        ws = []
        preds_points = []
        error_points = []

        ws.append(center_of_gaussian)
        pred = ws[0] @ xs[b,0].numpy()
        preds_points.append(pred)
        error_points.append((pred - ys[b,0].numpy())**2)
        ptp = 1e-5 # very small probability
        ptps.append(ptp)
        for plen in range(1, n_dims + 1):
            w_star = Variable([n_dims, 1])
            obj = Minimize(cvxnorm(w_star-center_of_gaussian[:,np.newaxis], 2))
            constraints = [w_star[0] == center_of_gaussian[0], ys[b,:plen].numpy()[:,np.newaxis] == (xs[b,:plen].numpy() @ w_star)]

            prob = Problem(obj, constraints)
            result = prob.solve()

            if w_star.value is not None:
                pred = w_star.value[:,0] @ xs[b,plen].numpy()
                preds_points.append(pred)
                error_points.append((pred - ys[b,plen].numpy())**2)
                ws.append(w_star.value[:,0])
                
                if plen < n_dims-1:
                    w = w_star.value[1:,0] # (n_dim-1,)
                    num_constraints = plen
                    subspace_dim = n_dims-num_constraints-1
                    w_norm = np.linalg.norm(w)
                    x = torch.zeros(subspace_dim); x[0] = w_norm
                    ptp = evaluate_standard_multivariate_normal(subspace_dim, x)
                elif plen == n_dims-1:
                    w = w_star.value[:,0]
                    ptp = evaluate_normal_given_mean_covar(computing_for_component, torch.from_numpy(w))
                    
                elif plen == n_dims:
                    ptp = 1. if computing_for_component == prompts_from_component else 0.
                ptps.append(ptp)
                
            else:
                ws.append(ws[-1])
                preds_points.append(preds_points[-1])
                error_points.append(error_points[-1])
                ptps.append(ptps[-1])
        ptp_batch.append(ptps)
        w_batch.append(ws)
        errors_batch.append(error_points)
        preds_batch.append(preds_points)
        
        
    w_batch_arr = np.array(w_batch)
    errors_batch_arr = np.array(errors_batch)
    preds_batch_arr = np.array(preds_batch)
    ptp_batch_arr = np.array(ptp_batch)
    
    ones_or_zero = 0.
    if computing_for_component == prompts_from_component:
        ones_or_zero = 1.
    remaining_pts = pts_input - ptp_batch_arr.shape[1] # one less because if input pts are 1,...p then predictions are made for points 2,...p looking at the previous points. So, we get p-1 and not p predictions
    remaining_probs = np.full((batch_size, remaining_pts), ones_or_zero)
    ptp_batch_arr = np.concatenate((ptp_batch_arr, remaining_probs), axis=1)
    
    w_batch_arr = np.concatenate((w_batch_arr, extend_last(w_batch_arr, remaining_pts)), axis=1)
    errors_batch_arr = np.concatenate((errors_batch_arr, extend_last(errors_batch_arr, remaining_pts)), axis=1)
    preds_batch_arr = np.concatenate((preds_batch_arr, extend_last(preds_batch_arr, remaining_pts)), axis=1)
    return  w_batch_arr, errors_batch_arr, preds_batch_arr, ptp_batch_arr # b, n_dims, n_dims -- PME w for each batch entry, at each prompt length


def compute_PME_and_pTP_non_T_prompt(computing_for_component, xs, ys, prompts_from_component="special"): # xs can at most has n_dims + 1 points; We use a max of n_dims points to solve LPP and make predictions for the next point (until the n_dims+1-th point)
    if computing_for_component == "mean_plus_one":
        center_of_gaussian = mean.detach().cpu().numpy()
    else:
        center_of_gaussian = (-mean).detach().cpu().numpy()
    w_batch = []
    preds_batch = []
    errors_batch = []
    
    ptp_batch = []
    n_dims = xs.shape[2]
    pts_input = xs.shape[1]
    batch_size = xs.shape[0]
    for b in tqdm(range(batch_size)):
        ptps = []
        ws = []
        preds_points = []
        error_points = []

        ws.append(center_of_gaussian)
        pred = ws[0] @ xs[b,0].numpy()
        preds_points.append(pred)
        error_points.append((pred - ys[b,0].numpy())**2)
        ptp = 1e-5 # very small probability
        ptps.append(ptp)
        for plen in range(1, n_dims + 1):
            w_star = Variable([n_dims, 1])
            obj = Minimize(cvxnorm(w_star-center_of_gaussian[:,np.newaxis], 2))
            constraints = [w_star[0] == center_of_gaussian[0], ys[b,:plen].numpy()[:,np.newaxis] == (xs[b,:plen].numpy() @ w_star)]

            prob = Problem(obj, constraints)
            result = prob.solve()
            
            if w_star.value is not None:
                pred = w_star.value[:,0] @ xs[b,plen].numpy()
                preds_points.append(pred)
                error_points.append((pred - ys[b,plen].numpy())**2)
                ws.append(w_star.value[:,0])
                
                if plen < n_dims-1:
                    w = w_star.value[1:,0] # (n_dim-1,)
                    num_constraints = plen
                    subspace_dim = n_dims-num_constraints-1
                    w_norm = np.linalg.norm(w)
                    x = torch.zeros(subspace_dim); x[0] = w_norm
                    ptp = evaluate_standard_multivariate_normal(subspace_dim, x)
                elif plen == n_dims-1:
                    w = w_star.value[:,0]
                    ptp = evaluate_normal_given_mean_covar(computing_for_component, torch.from_numpy(w))
                    
                elif plen == n_dims:
                    if prompts_from_component not in ["mean_plus_one", "mean_minus_one"]:
                        ptp = ptps[-1]
                    elif computing_for_component == prompts_from_component:
                        ptp = 1.
                    else:
                        ptp = 0.
                ptps.append(ptp)
                
            else:
                ws.append(ws[-1])
                preds_points.append(preds_points[-1])
                error_points.append(error_points[-1])
                ptps.append(ptps[-1])
        ptp_batch.append(ptps)
        w_batch.append(ws)
        errors_batch.append(error_points)
        preds_batch.append(preds_points)
        
        
    w_batch_arr = np.array(w_batch)
    errors_batch_arr = np.array(errors_batch)
    preds_batch_arr = np.array(preds_batch)
    ptp_batch_arr = np.array(ptp_batch)
    
    remaining_pts = pts_input - ptp_batch_arr.shape[1] # one less because if input pts are 1,...p then predictions are made for points 2,...p looking at the previous points. So, we get p-1 and not p predictions
    ptp_batch_arr = np.concatenate((ptp_batch_arr, extend_last(ptp_batch_arr, remaining_pts)), axis=1)
    
    w_batch_arr = np.concatenate((w_batch_arr, extend_last(w_batch_arr, remaining_pts)), axis=1)
    errors_batch_arr = np.concatenate((errors_batch_arr, extend_last(errors_batch_arr, remaining_pts)), axis=1)
    preds_batch_arr = np.concatenate((preds_batch_arr, extend_last(preds_batch_arr, remaining_pts)), axis=1)
    return  w_batch_arr, errors_batch_arr, preds_batch_arr, ptp_batch_arr # b, n_dims, n_dims -- PME w for each batch entry, at each prompt length

### MIX PME COMPUTATION ###
def get_betas_and_PME_mix(pT1P, pT2P, pme_t1, pme_t2, alpha1, alpha2):
    denominator = alpha1*pT1P + alpha2*pT2P
    beta1 = alpha1*pT1P / denominator
    beta2 = alpha2*pT2P / denominator

    pts_to_slice = pT1P.shape[1]

    pme_mix = beta1[:,:,np.newaxis]*pme_t1[:,:pts_to_slice,:] + beta2[:,:,np.newaxis]*pme_t2[:,:pts_to_slice,:]
    return beta1, beta2, pme_mix

### Errors and predictions for PME Mix ###
def get_pme_mix_preds_and_errors(pme_mix, xs, ys): # xs can at most has n_dims + 1 points; We use a max of n_dims points to solve LPP and make predictions for the next point (until the n_dims+1-th point)
    n_dims = xs.shape[2]
    preds_batch = []
    errors_batch = []
    for b in tqdm(range(xs.shape[0])):
        ws = []
        preds_points = []
        error_points = []
        for t in range(pme_mix.shape[1]): # t should take values in [0, n_dims-1]
            pred = pme_mix[b,t] @ xs[b,t].numpy()
            preds_points.append(pred)
            error_points.append((pred - ys[b,t].numpy())**2)
        errors_batch.append(error_points)
        preds_batch.append(preds_points)
    return np.array(preds_batch), np.array(errors_batch) # b, n_dims, n_dims -- PME w for each batch entry, at each prompt length

# Recover weights

def get_xyw(conf, n_points, mixing_ratio_for_dist_selection, batch_size):
    n_dims = conf.model.n_dims
    conf.training.task_kwargs["mixing_ratio"]=mixing_ratio_for_dist_selection
    data_sampler = get_data_sampler(conf.training.data, n_dims)
    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        batch_size,
        **conf.training.task_kwargs
    )

    task = task_sampler()
    xs = data_sampler.sample_xs(b_size=batch_size, n_points=n_points)
    ys = task.evaluate(xs)
    w_b = task.w_b

    return xs, ys, w_b


def get_probed_weights_errors_cos_sim(conf, n_points, mixing_ratio_for_dist_selection, update_xyw_method=None, is_get_xyw=False, batch_size=64, is_sample_new_xyw=True, use_xyw=None):
    if is_sample_new_xyw:
        xs, ys, w_b = get_xyw(conf, n_points, mixing_ratio_for_dist_selection, batch_size)
    else:
        xs, ys, w_b = use_xyw

    w_probed_prompt_len = []
    error_prompt_len = []
    cos_sim_prompt_len = []

    if update_xyw_method is not None:
        xs, ys, w_b = update_xyw_method(xs, ys, w_b)

    for prompt_len in tqdm(range(0, n_points)):
        if prompt_len == 0:
            w_b, w_probed, error, cos_sim = recover_weights_no_prompt_v1(model, w_b)
        else:
            w_b, w_probed, error, cos_sim = recover_weights_v1(model, xs[:,:prompt_len,:], ys[:,:prompt_len], w_b)
        w_probed_prompt_len.append(w_probed)
        error_prompt_len.append(error)
        cos_sim_prompt_len.append(cos_sim)

    w_probed_tsr=torch.stack(w_probed_prompt_len, dim=-1) 
    
    if is_get_xyw:
        return xs, ys, w_b, w_probed_tsr, error_prompt_len, cos_sim_prompt_len
    else: 
        return w_b, w_probed_tsr, error_prompt_len, cos_sim_prompt_len
    # shapes: (batch, n_dims);  (batch, n_dim, prompt len);  (batch, prompt len);  (batch, prompt len)


def get_probed_weights_errors_cos_sim_v2(conf, n_points, mixing_ratio_for_dist_selection, update_xyw_method=None, is_get_xyw=False, batch_size=64, is_sample_new_xyw=True, use_xyw=None):
    if is_sample_new_xyw:
        xs, ys, w_b = get_xyw(conf, n_points, mixing_ratio_for_dist_selection, batch_size)
    else:
        xs, ys, w_b = use_xyw

    data_sampler = get_data_sampler(conf.training.data, n_dims)
    w_probed_prompt_len = []
    error_prompt_len = []
    cos_sim_prompt_len = []

    if update_xyw_method is not None:
        xs, ys, w_b = update_xyw_method(xs, ys, w_b)

    for prompt_len in tqdm(range(0, n_points)):
        if prompt_len == 0:
            w_b, w_probed, error, cos_sim = recover_weights_no_prompt_v2(model, xs, w_b, data_sampler)
        else:
            w_b, w_probed, error, cos_sim = recover_weights_v2(model, xs[:,:prompt_len,:], ys[:,:prompt_len], w_b, data_sampler)
        w_probed_prompt_len.append(w_probed)
        error_prompt_len.append(error)
        cos_sim_prompt_len.append(cos_sim)

    w_probed_tsr=torch.stack(w_probed_prompt_len, dim=-1)

    if is_get_xyw:
        return xs, ys, w_b, w_probed_tsr, error_prompt_len, cos_sim_prompt_len
    else:
        return w_b, w_probed_tsr, error_prompt_len, cos_sim_prompt_len
    # shapes: (batch, n_dims);  (batch, n_dim, prompt len);  (batch, prompt len);  (batch, prompt len)

def recover_weights_no_prompt_v2(model, xs, w_b, data_sampler):
    model.to("cuda:0")
    n_dims = w_b.size(1)

    x_probes = data_sampler.sample_xs(b_size = xs.shape[0], n_points = 2 * xs.shape[-1] + 1)
    y_probes = []
    for i in range( 2 * xs.shape[-1] + 1):
        x_prompt = x_probes[:,i:i+1,:]
        y_prompt = torch.zeros(xs.shape[0], 1)
        with torch.no_grad():
            pred = model(x_prompt.to("cuda:0"), y_prompt.to("cuda:0")).cpu()
        y_probes.append(pred[:,-1:])

    y_probes = torch.cat(y_probes, axis = 1)
    w_probed = []

    for batch in range(len(x_probes)):
        x, y = x_probes[batch], y_probes[batch]
        probe_model = LinearRegression(fit_intercept = False)
        probe_model.fit(x, y)
        w_probed.append(torch.tensor(probe_model.coef_[np.newaxis]).float())

    w_probed = torch.cat(w_probed, axis = 0)
    error = ((w_probed - w_b[:,:,0])**2).mean(axis = 1).mean()
    cos_sim = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)(w_probed, w_b[:,:,0]).mean()
    model.to("cpu")
    return w_b, w_probed, error, cos_sim

def recover_weights_v2(model, xs, ys, w_b,  data_sampler):
    model.to("cuda:0")
    batch_size = xs.size(0)
    n_dims = w_b.size(1)
    
    x_probes = data_sampler.sample_xs(b_size = xs.shape[0], n_points = 2 * xs.shape[-1] + 1)
    y_probes = []
    for i in range( 2 * xs.shape[-1] + 1):
        x_prompt = torch.concat([xs, x_probes[:,i:i+1,:]], axis = 1)
        y_prompt = torch.concat([ys, torch.zeros(xs.shape[0], 1)], axis = 1)
        with torch.no_grad():
            pred = model(x_prompt.to("cuda:0"), y_prompt.to("cuda:0")).cpu()
        y_probes.append(pred[:,-1:])
    
    y_probes = torch.cat(y_probes, axis = 1)
    w_probed = []
        
    for batch in range(len(x_probes)):
        x, y = x_probes[batch], y_probes[batch]
        probe_model = LinearRegression(fit_intercept = False)
        probe_model.fit(x, y)
        w_probed.append(torch.tensor(probe_model.coef_[np.newaxis]).float())
    
    w_probed = torch.cat(w_probed, axis = 0)
    error = ((w_probed - w_b[:,:,0])**2).mean(axis = 1).mean()
    cos_sim = torch.nn.CosineSimilarity(dim = 1, eps = 1e-6)(w_probed, w_b[:,:,0]).mean()
    model.to("cpu")
    return w_b, w_probed, error, cos_sim