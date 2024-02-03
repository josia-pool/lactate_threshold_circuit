# -*- coding: utf-8 -*-
"""
Created on Wed May 25 10:42:17 2022

This file was made to process the results from lhs sampling.
And to further optimize a selection of the best scores.
@author: josia
"""
#Import needed modules
import argparse
import os
import sys

import numpy as np
import scipy.optimize as opt
import pandas as pd
import matplotlib.pyplot as plt
import time
import multiprocessing
from tqdm import tqdm



#Import specific models
sys.path.append("models/")
from ODEModel_v2 import dCas_model

from Lactatemodel_succinate_p3 import Lactate_model_succinate_p3

def dCas9_further_optimization(results):
    """
    This function optimizes the dCas9 model
    Takes a selection of lhs results as input.
    Tries to optimize with +-10% bounds to get a better fit assuming we are close to the optimal solution.

    Args:
        results (np.array): Array with lhs results
    
    Returns:
        optimized_results (np.array): Array with optimized results
    """
    model = dCas_model()
    parameter_sets = results[:,:len(results[0])-1]
    optimized_sets = np.zeros((len(parameter_sets),len(parameter_sets[1])))
    optimized_scores = np.zeros((len(parameter_sets)))
    
    print("Starting optimization of parameter sets")
    for i in tqdm(range(len(parameter_sets))):
        #Dont take the first parameter, which is the induction concentration
        bounds = opt.Bounds(parameter_sets[i,1:]*0.9,parameter_sets[i,1:]*1.1)
        opt_score = opt.minimize(model.lhs_result_opt, parameter_sets[i,1:], bounds=bounds, method='L-BFGS-B', options={'disp':False,'maxiter':100, 'gtol': 1e-05, 'ftol': 1e-05})
        #Add the first parameter back to the optimized parameter set
        optimized_sets[i] = np.concatenate(([0],opt_score.x))
    optimized_results = np.column_stack((optimized_sets,optimized_scores))
    
    return optimized_results

def p3_further_optimization(results):
    """
    This function optimizes the Lactate model
    Takes a selection of lhs results as input.
    Tries to optimize with +-10% bounds to get a better fit.

    Args:
        results (np.array): Array with lhs results
    
    Returns:
        optimized_results (np.array): Array with optimized results
    """
    model = Lactate_model_succinate_p3()
    parameter_sets = results[:,:len(results[0])-1]
    optimized_sets = np.zeros((len(parameter_sets),len(parameter_sets[1])))
    optimized_scores = np.zeros((len(parameter_sets)))

    print("Starting optimization of parameter sets")
    for i in tqdm(range(len(parameter_sets))):
        bounds = opt.Bounds(parameter_sets[i]*0.9,parameter_sets[i]*1.1)
        opt_score = opt.minimize(model.lhs_result_opt, parameter_sets[i], bounds=bounds, method='L-BFGS-B', options={'disp':False,'maxiter':100, 'gtol': 1e-05, 'ftol': 1e-05})
        optimized_sets[i] = opt_score.x
        optimized_scores[i] = opt_score.fun
    optimized_results = np.column_stack((optimized_sets,optimized_scores))

    return optimized_results

if __name__ == "__main__":
    #Argument parsing
    parser = argparse.ArgumentParser(description="Josia's script(s) for modeling - Optimization wrapper")
    parser.add_argument("--file", help="Specify amount of samples.")
    parser.add_argument("--method", help="Specify the sampling method used.")
    parser.add_argument("--model", help="Specify the model to use.")
    parser.add_argument("--cores",  default=8 , help="Specify the number of cores to use.", type=int)

    args = parser.parse_args()
    
    #Get a list of files in the processed folder.
    filelist = os.listdir("samples/simulated")

    #Check model type and call the right function.
    start_time = time.perf_counter()
    if args.model == "dCas9":
        model = dCas_model()
    elif args.model == "p3model":
        model = Lactate_model_succinate_p3()
    else:
        print("No model found using those arguments!")
        sys.exit()

    #Load results based on our arguments.
    results = np.empty((0,len(model.params)+1))
    for file in filelist:
        if args.file in file and args.method in file:
            file_data = pd.read_csv(f"samples/simulated/{file}",header= 0)
            results= np.concatenate((results,file_data))
    results = np.array(results) #Useless?

    #Score should be the last column.
    scores = results[:,-1]
    #Sort results based on score
    index = scores.argsort()
    results = results[index[::]]
    
    #Make a selection of the best scores and save.
    selection = results[:1000]
    
    #Create header for the file
    header_list = [x for x in model.param_labels]
    header_list.append("score")

    np.savetxt(f"lhs_results/best_score_{args.file}_{args.method}.csv", selection, delimiter=',', header=','.join(header_list), comments='')
    
    #Report results of the 1000 best lhs parameters
    model.report_results(results[:,-1], results[:,:len(model.params)], folder='lhs_results')

    #Multiprocessing
    used_cores = args.cores
    pool = multiprocessing.Pool(used_cores)
    
    #Test time spent
    start_time = time.perf_counter()
    
    #Specify amount of runs/batches of model optimization we want to run.
    runs = args.cores
    sample_list = np.array_split(selection, runs)
    #Results array dimensions need to match the amount of parameters+score
    opt_results = np.empty((0,len(model.params)+1))
    #Select and run optimization function based on model type.
    if args.model == "dCas9":   
        all_results = pool.map(dCas9_further_optimization, sample_list)
        #Combine results from all processes
        for result in all_results:
            opt_results = np.concatenate((opt_results,result))

        #Round off hill coefficients to nearest integer
        opt_results[:,26:28] = np.around(opt_results[:,26:28])
    elif args.model == "p3model":
        all_results = pool.map(p3_further_optimization, sample_list)
        #Combine results from all processes
        for result in all_results:
            opt_results = np.concatenate((opt_results,result))
            
    #Close multiprocessing pool
    pool.close()

    #Sort opt_results based on score
    scores = opt_results[:,-1]
    index = scores.argsort()
    opt_results = opt_results[index[::]]
    
    #Report and save results
    model.report_results(opt_results[:,-1], opt_results[:,:len(model.params)], folder='minimization_results')
    np.savetxt(f"minimization_results/minimized_results__{args.file}_{args.method}.csv", opt_results, delimiter=',', header=','.join(header_list), comments='')