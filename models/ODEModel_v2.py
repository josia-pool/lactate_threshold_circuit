# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 10:34:06 2022
@author: Josia Pool
"""
#Needed imports
#import logging #Logging > print
import multiprocessing #Enables parralel processes
import os #File handling
import time #Time
import argparse #For adding file arguments
from distutils.util import strtobool

import numpy as np #Maths and stuff
from scipy.integrate import odeint #Differential equations
import scipy.interpolate as interpolate #Simulation interpolation
import matplotlib.pyplot as plt #Plotting
import pandas as pd #Dataframes etc.
    
from pyDOE import lhs #Latin hypercube sampling
import scipy.optimize as opt #Optimization

#Check if the required directories exist, if not make them
#Logs is for potential logs, only with old optimize_induction
#Figures is only used for unspecified reports.
dirlist = ['logs/', 'samples/', 'samples/simulated', 'figures/', 'lhs_results', 'minimization_results']
for directory in dirlist:
    if not os.path.exists(directory):
        os.makedirs(directory)

#Generic model class - What should every model have?
class ODEModel():
    def __init__(self):
        print("Model initiated!")
        self.params = [] #Parameters
        self.param_labels = [] #Labels for the parameters
        self.initial_conditions = [] #Initial conditions
        self.labels = [] #Labels for the initial conditions

        #Check if the required directories exist, if not make them
        #Logs is for potential logs, only with old optimize_induction
        #Figures is only used for unspecified reports.
        dirlist = ['logs/', 'samples/', 'samples/simulated', 'figures/', 'lhs_results', 'minimization_results']
        for directory in dirlist:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def simulateODE(self, parameters, x_init, t_end = 18, Nsteps=200):
        """
        Simulates the ODE's with the given parameters and initial conditions.

        Args:
            parameters: Array/List with the parameters.
            x_init: Array/List with the initial conditions.
            t_end: End time of the simulation.
            Nsteps: Number of steps in the simulation.
        
        Returns:
            timepoints: Timepoints of the simulation.
            x_t: Array with the simulated data for each state variable.
        """
        # Make a numpy array with the time steps for which you want output
        timepoints = np.linspace(0, t_end, Nsteps)
        # Calculate state variables at requested time points
        x_t = odeint(self.deriv, x_init, timepoints, args = (parameters,))
        
        # Return time points and state variables at those time points
        # This technically combines both variables into a tuple which can be unpacked after calling the function
        return timepoints, x_t
    
    def weighted_sum(self, xaxis, sim_xaxis, sim, data, error, scale=1):
        """
        Scoring function for a single data-series.
        Calculates the weighted sum of the squared error between the data and the simulation.
        Option for scaling simulation with the data.

        Args:
            xaxis (numpy array): x-axis of the data.
            sim_xaxis (numpy array): x-axis of the simulation.
            sim (numpy array): Simulation data.
            data (numpy array): Data.
            error (numpy array): Error of the data.
            scale (float): Scale factor for the simulation. (Sometimes the scaling is applied before passing the simulation to this function)
        
        Returns:
            score: Weighted sum of the squared error between the data and the simulation.
        """
        #Check if there are any zero's or almost zero values and replace by 1
        replace_index = np.where(error < 10e-6)[0]
        error[replace_index] = 1
        #Change the shape of the simulated data to the same amount of datapoints
        if (len(sim) != len(data)):
            sim_interp = interpolate.interp1d(sim_xaxis, sim)
            sim = sim_interp(xaxis.tolist())

        difference = (sim*scale-data)**2
        variance = error**2
        scores = difference/variance

        # for i in range(len(scores)):
        #     print(f"Timepoint: {time[i]} - Simulation: {sim[i]} - Data: {data[i]} - Variance: {error[i]} - Score: {scores[i]}")
        score = np.sum(scores)
        
        return score
    
    def sample(self, nr):
        """
        Uniformly sample the parameter space using Latin Hypercube sampling.
        Based on the normal_upper and normal_lower bounds to be defined in __init__.

        Args:
            nr (int): Number of samples to be taken.
        
        Returns:
            self.sampled_space (numpy array): Array with the sampled parameters.
        """
        #Sample each parameter
        self.sampled_space= lhs(len(self.params), samples =nr)
        
        # Now, for each row/parameter in the sample matrix, we need to re-scale these using the bounds
        for i in range(0, len(self.normal_upper)):
            self.sampled_space[:,i] = self.normal_lower[i] + (self.normal_upper[i]-self.normal_lower[i])*self.sampled_space[:,i]
        
        #Should you like a 2D example of the parameter space   
        #plt.scatter(self.sampled_space[:,1],self.sampled_space[:,2])
        #plt.title("Normal LHS sampling")
        #plt.xlabel("k_b_dcas")
        #plt.ylabel("k_b_sgrna")
        #plt.show() #Dont use this in a terminal
                
        return self.sampled_space
    
    def sample_log10(self, nr):
        """
        Sample the parameter space using Latin Hypercube sampling.
        However this is done in logarithmic space and then transformed.
        Use to get more parameters sets with lower values.

        Args:
            nr (int): Number of samples to be taken.
        
        Returns:
            self.sampled_space (numpy array): Array with the logarithmically sampled parameters.
        """
        #Sample each parameter
        self.sampled_space= lhs(len(self.params), samples =nr)
        
        # Now, for each row/parameter in the sample matrix, we need to re-scale these using the bounds
        for i in range(0, len(self.log_upper)):
            self.sampled_space[:,i] = self.log_lower[i] + (self.log_upper[i]-self.log_lower[i])*self.sampled_space[:,i]
        
        self.sampled_space = 10**self.sampled_space
        
        #Should you like a 2D example of the parameter space          
        #plt.title("Log LHS sampling")
        #plt.scatter(self.sampled_space[:,1],self.sampled_space[:,2])
        #plt.xlabel("k_b_dcas")
        #plt.ylabel("k_b_sgrna")
        #plt.show() #Dont use this in a terminal
        
        return self.sampled_space
    
    def save_samples(self, nr, method):
        """
        Splits the sampled space into 10 chunks to make it more accessible instead of having one big .csv file.
        Saves each chunk in a separate file.

        Args:
            nr (int): Number of samples in total.
            method (str): Method used to sample the parameter space. Either "normal" or "log", or whatever string to identify your sampled space.
        """
        split = np.array_split(self.sampled_space, 10)

        #Create header for the file
        header_list = [x for x in self.param_labels]

        for i in range(10):
            np.savetxt(f"samples/{nr}_{method}_samples_part_{i+1}.csv",split[i], delimiter=",", header = ",".join(header_list), comments = "")
    
    def process_sample(self, file):
        """
        Loads a sample file from .csv and returns a numpy array.
        Used in optim_wrapper.py

        Args:
            file (str): Path to the sample file.
        
        Returns:
            self.sampled_space (numpy array): Array with the sampled parameters.
        """
        file = pd.read_csv(file, header = 0)
        samples = file.to_numpy()
        self.sampled_space = samples

        return self.sampled_space
    
    def random_sample(self, num_p):
        """
        Randomly sample a parameter set from 0 to 1.

        Args:
            num_p (int): Number of parameters to be sampled.
        
        Returns:
            sample (list): list with sampled parameters from 0 to 1.
        """
        sample = []
        for i in range(0,num_p):
            sample.append(np.random.random_sample())
        
        return sample     
    
    #Will plot a solution with all components in the system. (unscaled)
    def plot(self, xaxis, states, labels):
        """
        Plots solution states and labels according to the labels list.
        self.header, self.x_label, self.y_label is to be defined in the model that inherits this class.
        Does not scale in the case of plotting Fluorescence, for example.

        Args:
            xaxis: x-axis values of the plot.
            states: Solution states to be plotted.
            labels: Labels of the solution states.
        """
        for i in range(len(states[0,:])):   
            plt.plot(xaxis,states[:,i], label=labels[i])
        plt.title(self.header)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()
        
#Specific model class - What is specific to only this (type of) model?
class dCas_model(ODEModel):
    def __init__(self):
        """
        Initialize the model.
        Define all needed variables such as parameters, bounds, etc.
        Also should load any data to be fitted.
        """
        super().__init__()
        #Parameter definition
        as_input = 0 #Inducer concentration - Unit: mM
        k_b_dcas = 1 #Constitutive production of dCas - Unit: mM/hr
        k_b_sgrna = 1 #Constitutive production of sgRNA - Unit: mM/hr
        
        k_pt_dcas = 1 #Speed at which dCas_mRNA is translated - Unit: 1/hr
        k_pt_gfp = 1 #Speed at which GFP_mRNA is translated - Unit: 1/hr
        
        k_f_cascomplex_fw = 1 #Formation rate of dCas-sgRNA complex - Unit: mM^-1/hr
        k_f_cascomplex_rv = 1 #Dissociation rate of CasComplex - Unit: 1/hr
        
        k_f_ascomplex_fw = 1 #Formation rate of asRNA-sgRNA complex - Unit: mM^-1/hr
        k_f_ascomplex_rv = 1 #Dissociation rate of asRNA-sgRNA complex - Unit: 1/hr
        
        k_f_ascascomplex1_fw = 1 #Formation rate of asCasComplex - Unit: mM^-1/hr
        k_f_ascascomplex1_rv = 1 #Dissociation rate of asCascomplex - Unit: 1/hr

        k_f_ascascomplex2_fw = 1 #Formation rate of asCasComplex but dCas comes last - Unit: mM^-1/hr
        k_f_ascascomplex2_rv = 1 #Dissociation rate of asCasComplex but dCas comes last - Unit: 1/hr

        k_p_asrna = 1 #Max production rate of asRNA - Unit: mM/hr
        k_p_gfpmrna = 1 #Max production rate of GFP_mRNA - Unit: mM/hr
        
        k_d_dcasmrna = 1 #dCas_mRNA degradation - Unit: 1/hr
        k_d_dcas = 1 #dCas degradation - Unit: 1/hr
        k_d_sgrna = 1 #sgRNA degradation - Unit: 1/hr
        k_d_cascomplex = 1 #casComplex degradation - Unit: 1/hr
        k_d_asrna = 1 #asRNA degradation - Unit: 1/hr
        k_d_ascomplex = 1 #asRNA-sgRNA degradation - Unit: 1/hr
        k_d_ascascomplex = 1 #asCasComplex degradation - Unit: 1/hr
        k_d_gfpmrna = 1 #GFP_mRNA degradation - Unit: 1/hr
        k_d_gfp = 1 #GFP degradation - Unit: 1/hr
        
        kd_asrna = 1 #Dissociation/ Michealis menten constant of asRNA inducer - Unit: mM
        kd_dcas = 1 #Dissociation/ Michealis menten constant of casComplex repression - Unit: mM
        
        n_inducer = 2 #Hill coefficient inducer - Unit: Dimensionless
        n_cascomplex = 2 #Hill coefficient dCas repression - Unit: Dimensionless
        
        scale = 100 #Scaling factor amount of GFP to fluorescence - Unit: A.U./mM
        
        #Pack parameters
        self.params_standard = [as_input, k_b_dcas, k_b_sgrna, k_pt_dcas, k_pt_gfp, 
                                k_f_cascomplex_fw, k_f_cascomplex_rv, k_f_ascomplex_fw, k_f_ascomplex_rv,
                                k_f_ascascomplex1_fw, k_f_ascascomplex1_rv, k_f_ascascomplex2_fw, k_f_ascascomplex2_rv,  
                                k_p_asrna, k_p_gfpmrna, k_d_dcasmrna, k_d_dcas, k_d_sgrna, k_d_cascomplex, k_d_asrna, k_d_ascomplex,
                                k_d_ascascomplex, k_d_gfpmrna, k_d_gfp, kd_asrna, kd_dcas, n_inducer, n_cascomplex, scale]
        #Normal sampling upper bounds definition
        as_input_upper = 0 #Inducer concentration - Unit: mM
        k_b_dcas_upper = 10 #Constitutive production of dCas - Unit: mM/hr
        k_b_sgrna_upper = 10 #Constitutive production of sgRNA - Unit: mM/hr
        
        k_pt_dcas_upper = 10 #Speed at which dCas_mRNA is translated - Unit: 1/hr
        k_pt_gfp_upper = 10 #Speed at which GFP_mRNA is translated - Unit: 1/hr
        
        k_f_cascomplex_fw_upper = 10 #Formation rate of dCas-sgRNA complex - Unit: mM^-1/hr
        k_f_cascomplex_rv_upper = 10 #Dissociation rate of CasComplex - Unit: 1/hr
        
        k_f_ascomplex_fw_upper = 10 #Formation rate of asRNA-sgRNA complex - Unit: mM^-1/hr
        k_f_ascomplex_rv_upper = 10 #Dissociation rate of asRNA-sgRNA complex - Unit: 1/hr
        
        k_f_ascascomplex1_fw_upper = 10 #Formation rate of asCasComplex - Unit: mM^-1/hr
        k_f_ascascomplex1_rv_upper = 10 #Dissociation rate of asCascomplex - Unit: 1/hr

        k_f_ascascomplex2_fw_upper = 10 #Formation rate of asCasComplex but dCas comes last - Unit: mM^-1/hr
        k_f_ascascomplex2_rv_upper = 10 #Dissociation rate of asCasComplex but dCas comes last - Unit: 1/hr

        k_p_asrna_upper = 10 #Max production rate of asRNA - Unit: mM/hr
        k_p_gfpmrna_upper = 10 #Max production rate of GFP_mRNA - Unit: mM/hr
        
        k_d_dcasmrna_upper = 10 #dCas_mRNA degradation - Unit: 1/hr
        k_d_dcas_upper = 10 #dCas degradation - Unit: 1/hr
        k_d_sgrna_upper = 10 #sgRNA degradation - Unit: 1/hr
        k_d_cascomplex_upper = 10 #casComplex degradation - Unit: 1/hr
        k_d_asrna_upper = 10 #asRNA degradation - Unit: 1/hr
        k_d_ascomplex_upper = 10 #asRNA-sgRNA degradation - Unit: 1/hr
        k_d_ascascomplex_upper = 10 #asCasComplex degradation - Unit: 1/hr
        k_d_gfpmrna_upper = 10 #GFP_mRNA degradation - Unit: 1/hr
        k_d_gfp_upper = 0.05 #GFP degradation - Unit: 1/hr
        
        kd_asrna_upper = 10 #Dissociation/ Michealis menten constant of asRNA inducer - Unit: mM
        kd_dcas_upper =  10 #Dissociation/ Michealis menten constant of casComplex repression - Unit: mM
        
        n_inducer_upper = 5 #Hill coefficient inducer - Unit: Dimensionless
        n_cascomplex_upper = 5#Hill coefficient dCas repression - Unit: Dimensionless
        
        scale_upper = 1000 #Scaling factor amount of GFP to fluorescence - Unit: A.U./mM
        
        #Normal sampling lower bounds definition
        as_input_lower = 0 #Inducer concentration - Unit: mM
        k_b_dcas_lower = 0 #Constitutive production of dCas - Unit: mM/hr
        k_b_sgrna_lower = 0 #Constitutive production of sgRNA - Unit: mM/hr
        
        k_pt_dcas_lower = 0 #Speed at which dCas_mRNA is translated - Unit: 1/hr
        k_pt_gfp_lower = 0 #Speed at which GFP_mRNA is translated - Unit: 1/hr
        
        k_f_cascomplex_fw_lower = 0 #Formation rate of dCas-sgRNA complex - Unit: mM^-1/hr
        k_f_cascomplex_rv_lower = 0 #Dissociation rate of CasComplex - Unit: 1/hr
        
        k_f_ascomplex_fw_lower = 0 #Formation rate of asRNA-sgRNA complex - Unit: mM^-1/hr
        k_f_ascomplex_rv_lower = 0 #Dissociation rate of asRNA-sgRNA complex - Unit: 1/hr
        
        k_f_ascascomplex1_fw_lower = 0 #Formation rate of asCasComplex - Unit: mM^-1/hr
        k_f_ascascomplex1_rv_lower = 0 #Dissociation rate of asCascomplex - Unit: 1/hr

        k_f_ascascomplex2_fw_lower = 0 #Formation rate of asCasComplex but dCas comes last - Unit: mM^-1/hr
        k_f_ascascomplex2_rv_lower = 0 #Dissociation rate of asCasComplex but dCas comes last - Unit: 1/hr

        k_p_asrna_lower = 0 #Max production rate of asRNA - Unit: mM/hr
        k_p_gfpmrna_lower = 0 #Max production rate of GFP_mRNA - Unit: mM/hr
        
        k_d_dcasmrna_lower = 0 #dCas_mRNA degradation - Unit: 1/hr
        k_d_dcas_lower = 0 #dCas degradation - Unit: 1/hr
        k_d_sgrna_lower = 0 #sgRNA degradation - Unit: 1/hr
        k_d_cascomplex_lower = 0 #casComplex degradation - Unit: 1/hr
        k_d_asrna_lower = 0 #asRNA degradation - Unit: 1/hr
        k_d_ascomplex_lower = 0 #asRNA-sgRNA degradation - Unit: 1/hr
        k_d_ascascomplex_lower = 0 #asCasComplex degradation - Unit: 1/hr
        k_d_gfpmrna_lower = 0 #GFP_mRNA degradation - Unit: 1/hr
        k_d_gfp_lower = 0 #GFP (degradation/)dilution - Unit: 1/hr
        
        kd_asrna_lower = 0 #Dissociation/ Michealis menten constant of asRNA inducer - Unit: mM
        kd_dcas_lower = 0 #Dissociation/ Michealis menten constant of casComplex repression - Unit: mM
        
        n_inducer_lower = 1 #Hill coefficient inducer - Unit: Dimensionless
        n_cascomplex_lower = 1 #Hill coefficient dCas repression - Unit: Dimensionless
        
        scale_lower = 0 #Scaling factor amount of GFP to fluorescence - Unit: A.U./mM
        
        self.normal_upper = [as_input_upper, k_b_dcas_upper, k_b_sgrna_upper, k_pt_dcas_upper, k_pt_gfp_upper, 
                                k_f_cascomplex_fw_upper, k_f_cascomplex_rv_upper, k_f_ascomplex_fw_upper, k_f_ascomplex_rv_upper,
                                k_f_ascascomplex1_fw_upper, k_f_ascascomplex1_rv_upper, k_f_ascascomplex2_fw_upper, k_f_ascascomplex2_rv_upper,  
                                k_p_asrna_upper, k_p_gfpmrna_upper, k_d_dcasmrna_upper, k_d_dcas_upper, k_d_sgrna_upper, k_d_cascomplex_upper, k_d_asrna_upper, k_d_ascomplex_upper,
                                k_d_ascascomplex_upper, k_d_gfpmrna_upper, k_d_gfp_upper, kd_asrna_upper, kd_dcas_upper, n_inducer_upper, n_cascomplex_upper, scale_upper]
        self.normal_lower = [as_input_lower, k_b_dcas_lower, k_b_sgrna_lower, k_pt_dcas_lower, k_pt_gfp_lower, 
                                k_f_cascomplex_fw_lower, k_f_cascomplex_rv_lower, k_f_ascomplex_fw_lower, k_f_ascomplex_rv_lower,
                                k_f_ascascomplex1_fw_lower, k_f_ascascomplex1_rv_lower, k_f_ascascomplex2_fw_lower, k_f_ascascomplex2_rv_lower,  
                                k_p_asrna_lower, k_p_gfpmrna_lower, k_d_dcasmrna_lower, k_d_dcas_lower, k_d_sgrna_lower, k_d_cascomplex_lower, k_d_asrna_lower, k_d_ascomplex_lower,
                                k_d_ascascomplex_lower, k_d_gfpmrna_lower, k_d_gfp_lower, kd_asrna_lower, kd_dcas_lower, n_inducer_lower, n_cascomplex_lower, scale_lower]
        #Log sampling bounds definition
        as_input_upper_log = 0 #Inducer concentration
        k_b_dcas_upper_log = 1 #Const. production of dCas
        k_b_sgrna_upper_log = 1 #Const. production of sgRNA
        
        k_pt_dcas_upper_log = 1 #Max production asRNA
        k_pt_gfp_upper_log = 1 #Max production GFP
        
        k_f_cascomplex_fw_upper_log = 1 #Formation rate of dCas-sgRNA complex
        k_f_cascomplex_rv_upper_log = 1 #Dissociation rate of CasComplex
        
        k_f_ascomplex_fw_upper_log = 1 #Formation rate of asRNA-sgRNA complex
        k_f_ascomplex_rv_upper_log = 1 #Dissociation rate of asRNA-sgRNA complex
        
        k_f_ascascomplex1_fw_upper_log = 1 #Formation rate of asCasComplex
        k_f_ascascomplex1_rv_upper_log = 1 #Dissociation rate of asCascomplex

        k_f_ascascomplex2_fw_upper_log = 1 #Formation rate of asCasComplex but dCas comes last
        k_f_ascascomplex2_rv_upper_log = 1 #Dissociation rate of asCasComplex but dCas comes last

        k_p_asrna_upper_log = 1
        k_p_gfpmrna_upper_log = 1
        
        k_d_dcasmrna_upper_log = 1 #dCas_mRNA degradation
        k_d_dcas_upper_log = 1 #dCas degradation
        k_d_sgrna_upper_log = 1 #sgRNA degradation
        k_d_cascomplex_upper_log = 1 #casComplex degradation
        k_d_asrna_upper_log = 1 #asRNA degradation
        k_d_ascomplex_upper_log = 1 #asRNA-sgRNA degradation
        k_d_ascascomplex_upper_log = 1 #asCasComplex degradation
        k_d_gfpmrna_upper_log = 1 #GFP_mRNA degradation
        k_d_gfp_upper_log = 1 #GFP degradation
        
        kd_asrna_upper_log = 1
        kd_dcas_upper_log =  1
        
        n_inducer_upper_log = 0.7 #hill coefficient inducer
        n_cascomplex_upper_log = 0.7#hill coefficient dCas repression
        
        scale_upper_log = 2.7
        
        #Log sampling lower bounds definition
        as_input_lower_log = 0 #Inducer concentration
        k_b_dcas_lower_log = -3 #Const. production of dCas
        k_b_sgrna_lower_log = -3 #Const. production of sgRNA
        
        k_pt_dcas_lower_log = -3 #Max production asRNA
        k_pt_gfp_lower_log = -3 #Max production GFP
        
        k_f_cascomplex_fw_lower_log = -3 #Formation rate of dCas-sgRNA complex
        k_f_cascomplex_rv_lower_log = -3 #Dissociation rate of CasComplex
        
        k_f_ascomplex_fw_lower_log = -3 #Formation rate of asRNA-sgRNA complex
        k_f_ascomplex_rv_lower_log = -3 #Dissociation rate of asRNA-sgRNA complex
        
        k_f_ascascomplex1_fw_lower_log = -3 #Formation rate of asCasComplex
        k_f_ascascomplex1_rv_lower_log = -3 #Dissociation rate of asCascomplex

        k_f_ascascomplex2_fw_lower_log = -3 #Formation rate of asCasComplex but dCas comes last
        k_f_ascascomplex2_rv_lower_log = -3 #Dissociation rate of asCasComplex but dCas comes last

        k_p_asrna_lower_log = -3
        k_p_gfpmrna_lower_log = -3
        
        k_d_dcasmrna_lower_log = -3 #dCas_mRNA degradation
        k_d_dcas_lower_log = -3 #dCas degradation
        k_d_sgrna_lower_log = -3 #sgRNA degradation
        k_d_cascomplex_lower_log = -3 #casComplex degradation
        k_d_asrna_lower_log = -3 #asRNA degradation
        k_d_ascomplex_lower_log = -3 #asRNA-sgRNA degradation
        k_d_ascascomplex_lower_log = -3 #asCasComplex degradation
        k_d_gfpmrna_lower_log = -3 #GFP_mRNA degradation
        k_d_gfp_lower_log = -3 #GFP degradation
        
        kd_asrna_lower_log = -3
        kd_dcas_lower_log = -3
        
        n_inducer_lower_log = 0 #hill coefficient inducer
        n_cascomplex_lower_log = 0#hill coefficient dCas repression
        
        scale_lower_log = 2
        
        self.log_upper = [as_input_upper_log, k_b_dcas_upper_log, k_b_sgrna_upper_log, k_pt_dcas_upper_log, k_pt_gfp_upper_log, 
                                k_f_cascomplex_fw_upper_log, k_f_cascomplex_rv_upper_log, k_f_ascomplex_fw_upper_log, k_f_ascomplex_rv_upper_log,
                                k_f_ascascomplex1_fw_upper_log, k_f_ascascomplex1_rv_upper_log, k_f_ascascomplex2_fw_upper_log, k_f_ascascomplex2_rv_upper_log,  
                                k_p_asrna_upper_log, k_p_gfpmrna_upper_log, k_d_dcasmrna_upper_log, k_d_dcas_upper_log, k_d_sgrna_upper_log, k_d_cascomplex_upper_log, k_d_asrna_upper_log, k_d_ascomplex_upper_log,
                                k_d_ascascomplex_upper_log, k_d_gfpmrna_upper_log, k_d_gfp_upper_log, kd_asrna_upper_log, kd_dcas_upper_log, n_inducer_upper_log, n_cascomplex_upper_log, scale_upper_log]
        self.log_lower = [as_input_lower_log, k_b_dcas_lower_log, k_b_sgrna_lower_log, k_pt_dcas_lower_log, k_pt_gfp_lower_log, 
                                k_f_cascomplex_fw_lower_log, k_f_cascomplex_rv_lower_log, k_f_ascomplex_fw_lower_log, k_f_ascomplex_rv_lower_log,
                                k_f_ascascomplex1_fw_lower_log, k_f_ascascomplex1_rv_lower_log, k_f_ascascomplex2_fw_lower_log, k_f_ascascomplex2_rv_lower_log,  
                                k_p_asrna_lower_log, k_p_gfpmrna_lower_log, k_d_dcasmrna_lower_log, k_d_dcas_lower_log, k_d_sgrna_lower_log, k_d_cascomplex_lower_log, k_d_asrna_lower_log, k_d_ascomplex_lower_log,
                                k_d_ascascomplex_lower_log, k_d_gfpmrna_lower_log, k_d_gfp_lower_log, kd_asrna_lower_log, kd_dcas_lower_log, n_inducer_lower_log, n_cascomplex_lower_log, scale_lower_log]
        self.sampled_space = np.empty(0)
        
        self.params = self.params_standard
        self.param_labels = ["as_input","k_b_dcas","k_b_sgrna","k_pt_dcas","k_pt_gfp",
                             "k_f_cascomplex_fw","k_f_cascomplex_rv","k_f_ascomplex_fw","k_f_ascomplex_rv",
                             "k_f_ascascomplex1_fw","k_f_ascascomplex1_rv","k_f_ascascomplex2_fw","k_f_ascascomplex2_rv",
                             "k_p_asrna","k_p_gfpmrna","k_d_dcasmrna","k_d_dcas","k_d_sgrna","k_d_cascomplex","k_d_asrna",
                             "k_d_ascomplex","k_d_ascascomplex","k_d_gfpmrna","k_d_gfp","kd_asrna","kd_dcas","n_inducer","n_cascomplex",
                             "scale"]
        for i in range(len(self.params)):
                       print(f"index {i}: {self.param_labels[i]}")
        #Initial state variables.
        dCas_mRNA = 0
        dCas = 0
        sgRNA = 0
        casComplex = 0
        asRNA = 0
        asComplex = 0
        asCasComplex = 0
        GFP_mRNA = 0
        GFP = 1
        
        #Pack initial conditions
        self.init_standard = [dCas_mRNA, dCas, sgRNA, casComplex, asRNA, 
                              asComplex, asCasComplex, GFP_mRNA, GFP]
        self.init = self.init_standard
        
        #Load the data
        datasheet = pd.read_excel("Data/dCas9Antisense data.xlsx", 
                                  sheet_name="CleanedData", header= None)
        
        self.induction_data = datasheet.iloc[2:len(datasheet), 1:4]
        self.control_data = datasheet.iloc[2:len(datasheet), 4:7]
        self.induction_errors = datasheet.iloc[2:len(datasheet), 8:11]
        self.control_errors = datasheet.iloc[2:len(datasheet), 11:14]
        self.time = datasheet.iloc[2:len(datasheet), 0]
        
        self.induction_data = self.induction_data.to_numpy(dtype='float64')
        self.control_data = self.control_data.to_numpy(dtype='float64')
        self.induction_errors = self.induction_errors.to_numpy(dtype='float64')
        self.control_errors = self.control_errors.to_numpy(dtype='float64')
        self.time = self.time.to_numpy(dtype='float64')
        
        #'Standard' output plot variables
        self.header = 'dCas-Antisense simulation'
        self.x_label = 'Time in hours'
        self.y_label = 'Output'
        self.labels = ["dCas_mRNA","dCas","sgRNA","casComplex","asRNA","asComplex","asCasComplex","GFP_mRNA","GFP fluorescence"]
    
    def deriv(self, y, t, p):
        """
        Function which calculates the derivatives and returns them.
        Every ODEModel needs a deriv function - The 'heart' of the model.

        Args:
            y : Array of concentrations of all species in the model.
            t : Array of time points.
            p : Array of parameters.
        
        Returns:
            dydt : Array of derivatives of all species in the model.
        """
        #Get state variable information from y
        dCas_mRNA, dCas , sgRNA, CasComplex, asRNA, asComplex, asCasComplex, GFP_mRNA, GFP = y[0:9]
        
        #Get parameters
        as_input, k_b_dcas, k_b_sgrna, k_pt_dcas, k_pt_gfp, k_f_cascomplex_fw, k_f_cascomplex_rv, k_f_ascomplex_fw, k_f_ascomplex_rv, k_f_ascascomplex1_fw, k_f_ascascomplex1_rv, k_f_ascascomplex2_fw, k_f_ascascomplex2_rv,  k_p_asrna, k_p_gfpmrna, k_d_dcasmrna, k_d_dcas, k_d_sgrna, k_d_cascomplex, k_d_asrna, k_d_ascomplex, k_d_ascascomplex, k_d_gfpmrna, k_d_gfp, kd_asrna, kd_dcas, n_inducera, n_cascomplexa = p[0:28]
        
        #Make sure to round hill coefficients.
        n_inducer = np.round(n_inducera)
        n_cascomplex = np.round(n_cascomplexa)
        
        #Calculate derivatives
        d_dCas_mRNA_dt  = k_b_dcas - dCas_mRNA*(1.2+k_d_dcasmrna) 
        d_dCas_dt       = k_pt_dcas*dCas_mRNA - dCas*(1.2+k_d_dcas) + k_f_cascomplex_rv*CasComplex - k_f_cascomplex_fw*sgRNA*dCas + k_f_ascascomplex2_rv*asCasComplex - k_f_ascascomplex2_fw*dCas*asComplex
        d_sgRNA_dt      = k_b_sgrna - sgRNA*(1.2+k_d_sgrna) + k_f_cascomplex_rv*CasComplex - k_f_cascomplex_fw*sgRNA*dCas - k_f_ascomplex_fw*asRNA*sgRNA + k_f_ascomplex_rv*asComplex
        d_CasComplex_dt = k_f_cascomplex_fw*sgRNA*dCas - k_f_cascomplex_rv*CasComplex - k_f_ascascomplex1_fw*asRNA*CasComplex  + k_f_ascascomplex1_rv*asCasComplex -CasComplex*(1.2+k_d_cascomplex) 
        d_asRNA_dt      = as_input**n_inducer/(kd_asrna**n_inducer+as_input**n_inducer)*k_p_asrna+ k_f_ascomplex_rv*asComplex - k_f_ascomplex_fw*asRNA*sgRNA+ k_f_ascascomplex1_rv*asCasComplex - k_f_ascascomplex1_fw*asRNA*CasComplex - asRNA*(1.2+k_d_asrna)
        d_asComplex_dt  = k_f_ascomplex_fw*asRNA*sgRNA - k_f_ascomplex_rv*asComplex - asComplex*(1.2+k_d_ascomplex) + k_f_ascascomplex2_rv*asCasComplex - k_f_ascascomplex2_fw*dCas*asComplex
        d_asCasComplex_dt = k_f_ascascomplex1_fw*asRNA*CasComplex- k_f_ascascomplex1_rv*asCasComplex - k_f_ascascomplex2_rv*asCasComplex + k_f_ascascomplex2_fw*dCas*asComplex - asCasComplex*(1.2+k_d_ascascomplex)
        d_GFP_mRNA_dt   = k_p_gfpmrna*(kd_dcas**n_cascomplex/(kd_dcas**n_cascomplex + CasComplex**n_cascomplex)) - GFP_mRNA*(1.2+k_d_gfpmrna)
        d_GFP_dt        = k_pt_gfp*GFP_mRNA-GFP*(1.2+k_d_gfp)
        
        dydt = [d_dCas_mRNA_dt, d_dCas_dt, d_sgRNA_dt, d_CasComplex_dt, d_asRNA_dt, d_asComplex_dt, d_asCasComplex_dt, d_GFP_mRNA_dt, d_GFP_dt]
        
        return dydt
    
    def sim_induced(self, p, induce_time):
        """
        Simulate the induction of asRNA at a given time.

        Args:
            p : Array of parameters.
            induce_time : Time at which asRNA is induced.
        
        Returns:
            sim_time : Array of time points.
            sim_output : Array of concentrations of all species in the model.
        """
        #Calculate initial conditions for GFP_mRNA and GFP
        #Assuming no CasComplex and steady state prior to experiment
        #Also assuming constant dilution.
        self.init[7] = p[14]/(p[22]+1.2)
        self.init[8] = (p[4]*p[14])/((p[22]+1.2)*(p[23]+1.2))
        
        timepoints, output = self.simulateODE(p, self.init, induce_time, induce_time*3600)
        #Change inducer concentration/strength.
        p[0] = 5

        timepoints2, output2 = self.simulateODE(p, output[-1], 18-induce_time, (18-induce_time)*3600)
        #Change it back, since for some reason once changed it carries over.
        p[0] = 0

        #Concatenate outputs
        sim_time = np.concatenate((timepoints,timepoints2[1:]+induce_time))
        sim_output = np.concatenate((output,output2[1:]))
        
        return sim_time, sim_output

    def optimize_induction(self, iterations, induction_time, data, error, samplemethod = "normal"):
        """
        Function to optimize the model.
        However, this function is not used in the main program.
        Since this doesnt really work using multiple processes, it is not used anymore.

        Args:
            iterations : Number of samples to run.
            induction_time : Time at which asRNA is induced.
            data : Array of experimental data.
            error : Array of error in experimental data.
            samplemethod : Method to sample parameters. Default is normal latin hypercube. choices are "normal" and "log".

        Returns:
            sorted_scores : Array of scores of all samples.
            sorted_sets : Array of parameters of all samples.
        """
        #Calculate the time optimization started
        start_time = time.perf_counter()
        
        #Array of empty scores
        scores = np.zeros(iterations,)
        
        #Determine sampling method and sample parameter space accordingly
        if samplemethod != "log":
            self.sampled_space = self.sample(iterations)
        else:
            self.sampled_space = self.sample_log10(iterations)
        
        #Round the hill coefficients
        self.sampled_space[:,26:28] = np.around(self.sampled_space[:,26:28])
        
        #Simulate all parameter sets
        for i in range(0,iterations):
            print(f"Starting iteration number {i}.")            
            #Simulate induction at 7 and 9 hours and calculate the score of each.
            _time, _output = self.sim_induced(self.sampled_space[i,:],7)
            scores[i] = self.weighted_sum(self.time,_time,_output[:,-1], self.induction_data[:,0], self.induction_errors[:,0], self.sampled_space[i,-1])
            
            _time, _output = self.sim_induced(self.sampled_space[i,:],induction_time)
            scores[i] = scores[i] + self.weighted_sum(self.time, _time,_output[:,-1], data, error, self.sampled_space[i,-1])
            print(f"Iteration number {i} gave a score of {scores[i]}\n")
        #Calculate the time it took to sample, simulate and score
        end_time = time.perf_counter()
        print(f"Model optimization completed in {end_time-start_time} seconds.")
        #Sort sampled spaces and scores based on score
        sort_index = scores.argsort()
        sorted_scores = scores[sort_index[::]]
        sorted_sets = self.sampled_space[sort_index[::]]
        
        #Write a report of the optimization.
        logtime = time.strftime("%d%b%y_%H_%M_%S", time.gmtime())
        with open(f"logs/model_optimization_{iterations}_{logtime}.txt","a+") as file:
            file.write(f"Model optimization performed on {logtime}\n")
            file.write(f"Model optimization completed in {end_time-start_time} seconds.\n")
            file.write(f"Sampling method used: {samplemethod}\n\n")
            
            #Induction time.
            file.write(f"Antisense RNA induced at T={induction_time}.\n\n")
            if samplemethod == "normal":
                #Initial conditions. Not neccesary if made dependable on parameter set.
                file.write("Initial conditions used:\n")
                for label, conc in zip(self.labels, self.init_standard):
                    file.write(f"{label}: {conc} \n")
                #Standard parameters and upper bounds factor.
                file.write("Lower bounds are 0 for all parameters.\n")
                file.write("Upperbounds are a factor 10 of standard values.\n\n")
                file.write("Model starting parameters:\n")
                for label, value in zip(self.param_labels, self.params_standard):  
                    file.write(f"{label}: {value}\n")
            else:
                pass #Write reports about log sampling 
            #Simulated parameter sets and their scores.
            file.write("\n Optimization results:\n")
            for score, sets in zip(sorted_scores, sorted_sets):
                file.write(f"{sets} gave a score of {score}\n")

        return sorted_scores, sorted_sets
    
    def lhs_result_opt(self, p):
        """
        Function to be passed in process_optim_results.py
        Used to further optimize latin hypercube sampling results.
        Basically the same as simulate_concentrations, but the function for opt.minimize() can only return one value.

        Args:
            p : Array of parameters.
        
        Returns:
            score : Score of the parameter set.
        """
        #Add inducer parameter
        p = np.concatenate(([0],p))
        #Simulate induction at 7 and 9 hours and calculate the score of each.
        _time, _output = self.sim_induced(p,7)
        score = self.weighted_sum(self.time, _time,_output[:,-1], self.induction_data[:,0], self.induction_errors[:,0], p[-1])
        
        _time, _output = self.sim_induced(p,9)
        score = score + self.weighted_sum(self.time, _time,_output[:,-1], self.induction_data[:,1], self.induction_errors[:,1], p[-1])
       
        return score
    
    def report_results(self, scores, sets, folder='figures'):
        """
        Function to report the results of the model fitting.
        Select the top 1000 from sorted scores,sets
        Save them and make a histogram of score and parameter distribution

        Args:
            scores : Array of scores of all samples.
            sets : Array of parameters of all samples.
            folder : Folder to save the figures in. Default is "figures".
        
        Returns:
            None, but saves figures and parameter sets.
        """
        #Sort the scores and sets just in case they are not sorted.
        sort_index = scores.argsort()
        scores = scores[sort_index[::]]
        sets = sets[sort_index[::]]
        
        #Select X(now 1000) of top scores
        selection = sets[:1000]
        scores = scores[:1000]
        
        #Make a file with top sets and scores
        top_results = np.column_stack((selection,scores))
        #Create header for the file
        header_list = [x for x in self.param_labels]
        header_list.append("score")
        np.savetxt(f"{folder}/best_score_{len(selection)}_outof_{len(sets)}.csv", top_results, delimiter=',', header=','.join(header_list), comments='')
        
        #Make histograms of the score distribution
        median = np.round(np.median(scores),2)
        plt.figure()
        plt.hist(scores, bins=100)
        plt.title(f"Distribution of scores in the top {len(selection)} scores out of {len(sets)}.")
        plt.axvline(x=median,ymin = 0, ymax=1, label=f"Median: {median}", color="red", linestyle="--", linewidth=2)
        plt.xlabel("Score value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(f"{folder}/score_dist_{len(selection)}_{len(sets)}.png", dpi= 300)
        plt.close()
        
        #Simulate the best parameter set
        #Select the parameter set with the best score
        best_score = np.argmin(scores)
        best_params = selection[best_score]
        print(f"The best parameters were: {best_params} and gave a score of: {scores[best_score]}")
        
        #Compare induction at 7h data with simulation - In scoring function
        simtime , output = self.sim_induced(best_params, 7)
        
        plt.figure()
        plt.errorbar(self.time, self.induction_data[:,0], yerr=self.induction_errors[:,0], xerr=None, label="Data")
        plt.plot(simtime, output[:,-1]*best_params[-1], label="Simulated best fit")
        plt.xlabel("Time in hours") 
        plt.ylabel("GFP fluorescence output")
        plt.title("Comparing simulation to data - asRNA induced at 7h")
        plt.legend()
        plt.savefig(f"{folder}/data_simulation_comparison_{len(selection)}_{len(sets)}_7h.png", dpi= 300)
        plt.close()
        
        #Compare induction at 9h data with simulation - In scoring function
        simtime , output = self.sim_induced(best_params, 9)
        
        plt.figure()
        plt.errorbar(self.time, self.induction_data[:,1], yerr=self.induction_errors[:,1], xerr=None, label="Data")
        plt.plot(simtime, output[:,-1]*best_params[-1], label="Simulated best fit")
        plt.xlabel("Time in hours") 
        plt.ylabel("GFP fluorescence output")
        plt.title("Comparing simulation to data - asRNA induced at 9h")
        plt.legend()
        plt.savefig(f"{folder}/data_simulation_comparison_{len(selection)}_{len(sets)}_9h.png", dpi= 300)
        plt.close()

        #Compare induction at 11h data with simulation - Used as validation
        simtime , output = self.sim_induced(best_params, 11)
        
        plt.figure()
        plt.errorbar(self.time, self.induction_data[:,2], yerr=self.induction_errors[:,2], xerr=None, label="Data")
        plt.plot(simtime, output[:,-1]*best_params[-1], label="Simulated best fit")
        plt.xlabel("Time in hours") 
        plt.ylabel("GFP fluorescence output")
        plt.title("Comparing simulation to data - asRNA induced at 11h")
        plt.legend()
        plt.savefig(f"{folder}/data_simulation_comparison_{len(selection)}_{len(sets)}_11h.png", dpi= 300)
        plt.close()
        
        #Make histograms of the individual parameters
        #Also calculate the median.
        median_params = []
        for i in range(len(self.param_labels)):
            values = []
            for j in range(len(selection)):
                values.append(selection[j][i])
            median = np.round(np.median(values),2)
            median_params.append(median)
            plt.figure()
            plt.hist(values, bins=100)
            plt.axvline(x=np.median(values),ymin =0, ymax=1, label=f"Median: {median}", color="red", linestyle="--", linewidth=2)
            plt.title(f"Distribution of {self.param_labels[i]} in the top {len(selection)} scores out of {len(sets)}.")
            plt.xlabel("Parameter value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(f"{folder}/{self.param_labels[i]}_{len(selection)}_{len(sets)}.png", dpi= 300)
            plt.close()
        
        #Simulate the median parameter set induced at 11h.
        simtime , output = self.sim_induced(median_params, 11)
        plt.figure()
        plt.errorbar(self.time, self.induction_data[:,2], yerr=self.induction_errors[:,2], xerr=None, label="Data")
        plt.plot(simtime, output[:,-1]*median_params[-1], label="Simulated fit")
        plt.xlabel("Time in hours") 
        plt.ylabel("GFP fluorescence output")
        plt.title("Comparing simulation with median params to data - asRNA induced at 11h")
        plt.legend()
        plt.savefig(f"{folder}/median_parameters_{len(selection)}_{len(sets)}_11h.png", dpi= 300)
        plt.close()
        
        #Simulate best 100? scores, and report average + stdev etc.
        #Need for sorted list.
        sub_selection = selection[:100]
        outputs = np.zeros((len(sub_selection+1),len(output[:,-1])))
        errors = np.zeros(len(sub_selection))
        for i in range(len(sub_selection)):
            _time, output = self.sim_induced(selection[i], 11)
            outputs[i] = output[:,-1]*selection[i,-1]
        
        for i in range(len(output[:,-1])):
            outputs[-1,i] = np.average(outputs[:,i])
            errors = np.std(outputs[:,i])
        plt.figure()
        plt.errorbar(self.time, self.induction_data[:,2], yerr=self.induction_errors[:,2], xerr=None, label="Data")
        plt.errorbar(simtime, outputs[-1], yerr=errors, xerr=None, label="Simulated average", ecolor='#FBA649')
        plt.xlabel("Time in hours") 
        plt.ylabel("GFP fluorescence output")
        plt.title("100 best simulations params vs data - asRNA induced at 11h")
        plt.legend()
        plt.savefig(f"{folder}/100_averaged_simulations_{len(selection)}_{len(sets)}_11h.png", dpi= 300)
        plt.close()


def optimize(samples):
    """
    Function to be passed to multiprocessing.Pool.map.

    Args:
        samples (list): List of samples to be evaluated.
    
    Returns:
        samples (list): List of samples to be evaluated.
        scores (list): List of scores for each sample.
    """
    model = dCas_model()
    scores = np.zeros(len(samples))
    
    for i in range(len(samples)):
        #Simulate induction at 7 and 9 hours and calculate the score of each.
        _time, _output = model.sim_induced(samples[i],7)
        score = model.weighted_sum(model.time, _time,_output[:,-1], model.induction_data[:,0], model.induction_errors[:,0], samples[i,-1])

        _time, _output = model.sim_induced(samples[i],9)
        score = score + model.weighted_sum(model.time, _time,_output[:,-1], model.induction_data[:,1], model.induction_errors[:,1], samples[i,-1])
        scores[i] = score
        
    return samples, scores

#What to run if this is the main file
if __name__ == "__main__":
    #Argument parsing
    parser = argparse.ArgumentParser(description="Josia's script(s) for modeling")
    parser.add_argument("--multiprocessing", help="Specify if using multiprocessing optimization.", default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--samplesize", help="Total sample size", default="120")
    args = parser.parse_args()
    
    #Multiprocessing script which: 
    #1.Samples a by argument specified sample size
    #2.Divide parameter space into sections
    #3.Start process that simulates and scores each section
    #4.Report
    if args.multiprocessing == True:
        #Samples for optim_wrapper
        model = dCas_model()
        model.sample(int(args.samplesize))
        #Round the hill coefficients
        model.sampled_space[:,26:28] = np.around(model.sampled_space[:,26:28])
        #Save samples
        model.save_samples(int(args.samplesize), "test")
        
        #Multiprocessing
        used_cores = 8
        pool = multiprocessing.Pool(used_cores)
        
        #Test time spent
        start_time = time.perf_counter()
        
        #Specify amount of runs/batches of model optimization we want to run.
        runs = 8
        sample_list = np.array_split(model.sampled_space, runs)      
        all_results = pool.map(optimize, sample_list)

        #Create empty arrays which will hold all scores and parameter sets.
        scores = np.empty(0)
        #Parameter array dimensions need to match the amount of parameters
        params = np.empty((0,len(all_results[0][0][0])))
        
        #Combine results from all processes
        for result in all_results:
            scores = np.concatenate((scores,result[1]))
            params = np.concatenate((params,result[0]))
            
        #Sort sampled spaces and scores based on score
        sort_index = scores.argsort()
        sorted_scores = scores[sort_index[::]]
        sorted_sets = params[sort_index[::]]
        
        #Combine scores and parameter sets and save the result.
        combined = np.column_stack((sorted_sets,sorted_scores))
        #Create a header
        header_list = [x for x in model.param_labels]
        header_list.append("score")
        
        np.savetxt("multiprocessing_results.csv", combined, delimiter=',', header=','.join(header_list), comments='')
        
        #Report time spent
        end_time = time.perf_counter()
        print(f"Optimization took {end_time-start_time} seconds")
        
        #Select the parameter set with the best score
        best_score = np.argmin(sorted_scores)
        best_params = sorted_sets[best_score]
        
        #Parralel processing is done, close
        pool.close()
        
        #Report results of the 1000 best lhs parameters
        #model = dCas_model()
        model.report_results(sorted_scores, sorted_sets, folder='figures')
    else:
        #Samples for optim_wrapper
        model = dCas_model()
        model.sample(1000000)
        #Round the hill coefficients
        model.sampled_space[:,26:28] = np.around(model.sampled_space[:,26:28])
        #Save samples
        model.save_samples(int(args.samplesize), "normal")
    #Simulates the model with standard params and plots all components
    #model = dCas_model()
    #model.init = model.init_standard
    #simtime , output = model.sim_induced(model.params, 9)
    #model.plot(simtime,output,model.labels)