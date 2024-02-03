# -*- coding: utf-8 -*-
"""
Created on Mon May 23 13:45:47 2022

This file was made to run the seperate sample lists on the ssb server
Can also be done on your PC but the more you run the slower all of them will get.
Did not check if this is faster/slower than just running 1mil at once on your PC
@author: josia
"""

import time
import argparse
import os
import numpy as np
import re
import sys
from tqdm import tqdm

sys.path.append("models/")

from ODEModel_v2 import dCas_model #Original CrisprI model - Fit to paper data

from Lactatemodel_succinate_p3 import Lactate_model_succinate_p3 #Keller parameters - Fit to ALP3 data from Bas

def optimize_dCasmodel():
    """
    This function optimizes the dCas9 model

    Args:
        None
    
    Returns:
        model: The model object
        scores: The scores of the parameter sets
    """
    model = dCas_model()
    model.process_sample(f"samples/{file}")

    #Round the hill coefficients
    model.sampled_space[:,26:28] = np.around(model.sampled_space[:,26:28])
    
    #Simulate each sample in the list.
    scores = np.zeros(len(model.sampled_space))
    for i in tqdm(range(len(model.sampled_space))):
        # print(f"Starting iteration number {i+1}.\n")
        
        #Simulate induction at 7 and 9 hours and calculate the score of each.
        _time, _output = model.sim_induced(model.sampled_space[i,:],7)
        scores[i] = model.weighted_sum(model.time, _time,_output[:,-1], model.induction_data[:,0], model.induction_errors[:,0], model.sampled_space[i,-1])
        
        _time, _output = model.sim_induced(model.sampled_space[i,:],9)
        scores[i] = scores[i] + model.weighted_sum(model.time, _time,_output[:,-1], model.induction_data[:,1], model.induction_errors[:,1], model.sampled_space[i,-1])
    return model, scores

def optimize_p3model():
    """
    Args:
        None

    Returns:
        model: The model object
        scores: The scores of the parameter sets
    """
    model = Lactate_model_succinate_p3()
    model.process_sample(f"samples/{file}")
    
    scores = np.zeros(len(model.sampled_space))

    print("Simulating and fitting p3 model...")
    for i in tqdm(range(len(model.sampled_space))):
        #Simulate the first model
        FLUOD_outputs, simtime = model.simulate_concentrations(model.sampled_space[i], model.concentrations_data)
        #Score the model
        scores[i] =  model.score_model(model.time, simtime, FLUOD_outputs, model.FLUOD_data_p3, model.FLUOD_error_p3, 1)

    return model, scores

if __name__ == "__main__":
    #Calculate the time script started
    start_time = time.perf_counter()

    #Argument parsing
    parser = argparse.ArgumentParser(description="Josia's script(s) for modeling - Optimization wrapper")
    parser.add_argument("--file", help="Specify the lhs sample number to use.")
    parser.add_argument("--method", help="Specify the sampling method used.")
    parser.add_argument("--model", help="Specify the model to use.")

    args = parser.parse_args()
    
    #Pick the right file based on our arguments.
    filelist = os.listdir('samples/')
    r = re.compile(f"\S+{args.method}\S+{args.file}.csv")
    filelist = list(filter(r.match, filelist))
    
    if filelist == []:
        print("No file found using those arguments!")
        sys.exit()
        
    file = filelist[0]
    #File should be good, now optimization can begin.
    #Pick the right model based on our arguments.
    if args.model == "dCas9":
        model, scores = optimize_dCasmodel()
    elif args.model == "p3model":
        model, scores = optimize_p3model()
    else:
        print("No model found using those arguments!")
        sys.exit()
        
    #Calculate the time script ended
    end_time = time.perf_counter()
    print(f"Model optimization completed in {end_time-start_time} seconds.\n")
    print("Writing results to the /samples/simulated folder!")
    
    #When done, remove the sampled file and write a textfile of the results
    os.remove(f"samples/{file}")
    results = np.column_stack((model.sampled_space,scores))

    #Create header for the file
    header_list = [x for x in model.param_labels]
    header_list.append("score")
    np.savetxt(f"samples/simulated/results_{file}", results, delimiter=',', header=','.join(header_list), comments='')
    print("All done!")