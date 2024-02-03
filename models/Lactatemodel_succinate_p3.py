# -*- coding: utf-8 -*-
"""
@author: Josia Pool

This is the model that I used for modeling the lactate dynamics
"""
#Import required modules
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.integrate import odeint
import scipy.interpolate as interpolate 

import multiprocess
import argparse
import time
from tqdm import tqdm
from distutils.util import strtobool
import os
import re

from ODEModel_v2 import ODEModel

#Standard plot stuff
dashline = (1, (3, 3))
colors = {'Orange': '#fba649', "Turquoise": '#45c4af', "Red": '#ec6c5f', "Green": '#4D8B31', "Purple": '#641877'}
color_list = np.array([np.array([251,166,73], dtype='int'),np.array([69,196,175], dtype='int'),np.array([236,108,95], dtype='int'),
                       np.array([76,139,48], dtype='int'),np.array([100,25,120], dtype='int'),np.array([221,204,119], dtype='int'),
                       np.array([204,121,167], dtype='int'),np.array([254, 226, 195], dtype='int'),
                       np.array([196, 217, 187], dtype='int'),np.array([194, 236, 229], dtype='int')])/255
from cycler import cycler
custom_cycler = (cycler(color=color_list, linewidth=[2,2,2,2,2,2,2,2,2,2]))
plt.rc('axes', prop_cycle=custom_cycler)

#Helper functions
def sorted_nicely( l ): 
    """ Sort the given iterable in the way that humans expect.""" 
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def get_label_rotation(angle, offset):
    """
    Used to calculate the rotation of a label for a polar bar plot.
    """
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment

def add_labels(angles, value, labels, offset, ax, size):
    """
    Adds labels to a polar bar plot
    """
    # This is the space between the end of the bar and the label
    padding = 0.1

    # Iterate over angles, values, and labels, to add all of them.
    for angle, label, in zip(angles, labels):
        angle = angle
        
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)

        # And finally add the text
        ax.text(
            x=angle, 
            y=value+ padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor",
            size=size
        )

def cm_to_inch(cm):
    """
    Convert centimeters to inches.
    """
    return cm / 2.54

class Lactate_model_succinate_p3(ODEModel):
    def __init__(self):
        """
        Initialize the model.
        Define all needed variables such as parameters, bounds, etc.
        Also should load any data to be fitted.

        From single model optimization 22SEP22:
        k_b_p3 = 9.8467
        k_b_p11 = 6.3658
        k_op_succ = 8.5275
        k_pt_lldr = 5.1846
        k_pt_gfp = 0.5225
        k_f_lldrcomplex = 0.1092
        k_d_lldrmRNA = 7.4873
        k_d_lldr = 1.6778
        k_d_lldrcomplex = 8.0627
        k_d_gfpmRNA = 0.1117
        k_d_gfp = 0.0129
        alpaga_basal = 4.8407
        alpaga_act1 = 2.9463
        alpaga_act2 = 8.1949
        alpaga_k3 = 51.0053
        alpaga_k_act1 = 89.0561
        alpaga_k_act2 = 78.1187
        copy_nr = 5.1704
        mumax = 0.4691
        tc = 10.8084
        cap = 0.3003
        sigma = 4.0988
        scale = 301.5452


        """
        super().__init__()
        #Common parameter definitions
        #Parameters for various lldr promoters
        
        # k_b_p3 = 2.9555 #Basal production of p3 (strong promoter)- Unit: mM/hr
        # k_b_p11 = 5.7280 #Basal production of p11 (weak promoter) - Unit: mM/hr

        # k_op_succ = 3.9457 #Background production of lldr from the active operon in succinate conditions
        # #k_op_glc = 1 #Background production of lldr from the active operon in glucose, aerobic conditions

        # #Translation rates
        # k_pt_lldr = 7.3001 #Speed at which lldR_mRNA is translated - Unit: 1/hr
        # k_pt_gfp = 9.1463 #Speed at which GFP_mRNA is translated - Unit: 1/hr

        # #Reaction rates
        # k_f_lldrcomplex = 0.8324 #Reaction rate of lldR complex formation- Unit: mM^-2/hr

        # #Degradation parameters
        # k_d_lldrmRNA = 1.0113 #Degradation rate of lldR_mRNA - Unit: 1/hr
        # k_d_lldr = 6.2103 #Degradation rate of lldR_mRNA - Unit: 1/hr
        # k_d_lldrcomplex = 8.2988 #Degradation rate of lldR_complex - Unit: 1/hr
        # k_d_gfpmRNA = 0.027 #Degradation rate of GFP_mRNA - Unit: 1/hr
        # k_d_gfp = 0.0104 #Degradation rate of GFP - Unit: 1/hr

        # #Keller paper parameters
        # alpaga_basal = 6.0722 #Activator independent basal production of alpaga promoter - Unit: mM/hr?
        # alpaga_act1 = 1.3808 #Activator 1 dependent production of alpaga promoter - Unit: mM/hr?
        # alpaga_act2 = 6.8816 #Activator 2 dependent production of alpaga promoter - Unit: mM/hr?
        # alpaga_k3 = 39.0097 #cr*kr1*kr2 Repressor binding constant
        # alpaga_k_act1 = 82.9971 #Activator 1(lldr) binding constant - Unit: mM
        # alpaga_k_act2 = 25.0894 #Activator 2(lldr_complex) binding constant - Unit: mM

        # #Plasmid parameters
        # copy_nr = 5.2369 #Number of copies of the plasmid
        # #Growth parameters
        # mumax = 0.4691 #Maximum growth rate - Unit: 1/hr
        # tc = 11.1720 #Time where exponential growth is reached - Unit: hr
        # cap = 0.3076 #Maximum capacity - Unit: OD600
        # sigma = 4.3772 #spread of the sigmoidal growth curve - Unit: hr

        # #Scale factor GFP to Fluorescence
        # scale = 12.6486 # Unit: A.U./mM

        #Best fit parameters for single model optimization 08OCT22
        #Rescaled after correcting for only cell volume dilution
        gamma = 4.462
        k_b_p3 = 38.422
        k_b_p11 = 2.275 #Unused
        k_op_succ = 2.784
        k_pt_lldr = 8.91
        k_pt_gfp = 0.705
        k_f_lldrcomplex = 0.234
        k_d_lldrmrna = 7.939
        k_d_lldr = 3.758
        k_d_lldrcomplex = 0.768
        k_d_gfpmrna = 0.067
        k_d_gfp = 0.003
        alpaga_basal = 7.8
        alpaga_act1 = 3.904
        alpaga_act2 = 22.172
        alpaga_k3 = 1.933
        alpaga_k_act1 = 42.532
        alpaga_k_act2 = 67.022
        copy_nr = 5.199
        mumax = 0.405
        tc = 10.423
        cap = 0.234
        sigma = 4.292
        scale = 59.633

        #Original growth parameters from fitting to growth curve
        #mumax = 0.5212435 #Maximum growth rate - Unit: 1/hr
        #tc = 10.41064018 #Time where exponential growth is reached - Unit: hr
        #cap = 0.33368693 #Maximum capacity - Unit: OD600
        #sigma = 4.26537504 #spread of the sigmoidal growth curve - Unit: hr

        self.parameters_standard = [k_b_p3, k_b_p11, k_op_succ ,k_pt_lldr, k_pt_gfp, k_f_lldrcomplex, k_d_lldrmrna, k_d_lldr, k_d_lldrcomplex, k_d_gfpmrna, k_d_gfp, alpaga_basal, alpaga_act1, alpaga_act2, alpaga_k3, alpaga_k_act1, alpaga_k_act2, copy_nr, mumax, tc, cap, sigma, gamma, scale]
        self.params = [elem for elem in self.parameters_standard]
        self.param_labels = ['k_b_p3', 'k_b_p11', 'k_op_succ','k_pt_lldr', 'k_pt_gfp', 'k_f_lldrcomplex', 'k_d_lldrmRNA', 'k_d_lldr', 'k_d_lldrcomplex', 'k_d_gfpmRNA', 'k_d_gfp', 'alpaga_basal', 'alpaga_act1', 'alpaga_act2', 'alpaga_k3', 'alpaga_k_act1', 'alpaga_k_act2', 'copy_nr', 'mumax', 'tc', 'cap', 'sigma', 'gamma', 'scale']

        # for i in range(len(self.params)):
        #         print(f"index {i}: {self.param_labels[i]}")

        #Initial conditions
        Lactate = 0
        lldr_mRNA = 0
        lldr = 0
        lldr_complex = 0
        GFP_mRNA = 0
        GFP = 0
        Cx = 0.05 #Initial concentration of biomass - Unit: OD

        #Pack the initial conditions
        self.initial_conditions = [Lactate, lldr_mRNA, lldr, lldr_complex, GFP_mRNA, GFP, Cx]
        self.init = self.initial_conditions

        #Load data to be fitted
        self.FLU_data = pd.read_csv("Data/15mMSuccinate_aerobic_FLU_processed.csv", header=0)#Data from paper: pd.read_excel('Data/Lactate_data.xlsx', header=0, engine='openpyxl')
        self.OD_data = pd.read_csv("Data/15mMSuccinate_aerobic_OD_processed.csv", header=0)
        self.FLUOD_data = pd.read_csv("Data/15mMSuccinate_aerobic_FLUOD_processed.csv", header=0)

        #Load error (st.dev) for data to be fitted
        self.FLU_error = pd.read_csv("Data/15mMSuccinate_aerobic_FLU_std_processed.csv", header=0)
        self.OD_error = pd.read_csv("Data/15mMSuccinate_aerobic_OD_std_processed.csv", header=0)
        self.FLUOD_error = pd.read_csv("Data/15mMSuccinate_aerobic_FLUOD_std_processed.csv", header=0)
        
        #self.concentrations_data = np.array([0.1, 0.5, 1.0, 5, 10, 50])
        self.concentrations_data = np.array([0.05,0.2,0.5,2,5,20])
        self.concentrations = self.concentrations_data #np.logspace(-3, 1.69897000434, num=50)

        #Process data to be fitted
        self.time = self.FLU_data.iloc[:,0]

        self.FLU_data = self.FLU_data.iloc[:,1:]
        self.OD_data = self.OD_data.iloc[:,1:]
        self.FLUOD_data = self.FLUOD_data.iloc[:,1:]

        self.FLU_error = self.FLU_error.iloc[:,1:]
        self.OD_error = self.OD_error.iloc[:,1:]
        self.FLUOD_error = self.FLUOD_error.iloc[:,1:]

        #Not done!!! - Implemented similar thing in score function
        #Replace any 0 values with average error value in that column
        # for i in range(len(self.FLU_error.columns)):
        #     self.FLU_error.iloc[:,i] = self.FLU_error.iloc[:,i].replace(0, self.FLU_error.iloc[:,i].mean())
        #     self.OD_error.iloc[:,i] = self.OD_error.iloc[:,i].replace(0, self.OD_error.iloc[:,i].mean())
        #     self.FLUOD_error.iloc[:,i] = self.FLUOD_error.iloc[:,i].replace(0, self.FLUOD_error.iloc[:,i].mean())

        #Select data to be fitted - here for P3 promoter
        #Disregard 0mM concentration
        self.FLU_data_p3 = self.FLU_data.iloc[:,1:7].to_numpy()
        self.OD_data_p3 = self.OD_data.iloc[:,1:7].to_numpy()
        self.FLUOD_data_p3 = self.FLUOD_data.iloc[:,1:7].to_numpy()

        self.FLU_error_p3 = self.FLU_error.iloc[:,1:7].to_numpy()
        self.OD_error_p3 = self.OD_error.iloc[:,1:7].to_numpy()
        self.FLUOD_error_p3 = self.FLUOD_error.iloc[:,1:7].to_numpy()

        self.dose_response = self.FLUOD_data.iloc[-1,1:7].to_numpy()
        self.dose_response_error = self.FLUOD_error.iloc[-1,1:7].to_numpy()

        #'Standard' output plot variables
        self.header = 'Lactate biosensor simulations - 15mM succinate'
        self.x_label = 'Time in hours'
        self.y_label = 'Output'
        self.labels = ['Lactate', 'Lldr_mRNA', 'lldr', 'lldr_complex', 'GFP_mRNA', 'GFP', 'Cx']
        #Unique marker for each variable
        self.markers = ['o', '*', '^', 's', 'D', 'v', 'p', 'h', 'H', '<', '>', '8', 'p', 'h', 'H', '<', '>']
        #Normal sampling upper and lower bounds
        k_b_p3_upper = 10
        k_b_p11_upper = 10

        k_op_succ_upper = 10
        #k_op_glc_upper = 10

        k_pt_lldr_upper = 10
        k_pt_gfp_upper = 10

        k_f_lldrcomplex_upper = 10

        k_d_lldrmRNA_upper = 10
        k_d_lldrcomplex_upper = 10
        k_d_lldr_upper = 10
        k_d_gfpmRNA_upper = 10
        k_d_gfp_upper = 0.05 #Assuming that GFP half life is very long

        alpaga_basal_upper = 10 
        alpaga_act1_upper = 10
        alpaga_act2_upper = 10
        alpaga_k3_upper = 100
        alpaga_k_act1_upper = 100
        alpaga_k_act2_upper = 100

        copy_nr_upper = 5

        mumax_upper = mumax
        tc_upper = tc
        cap_upper = cap
        sigma_upper = sigma

        scale_upper = 1000

        k_b_p3_lower = 0
        k_b_p11_lower = 0
        k_op_succ_lower = 0
        
        k_pt_lldr_lower = 0
        k_pt_gfp_lower = 0

        k_f_lldrcomplex_lower = 0

        k_d_lldrmRNA_lower = 0
        k_d_lldrcomplex_lower = 0
        k_d_lldr_lower = 0
        k_d_gfpmRNA_lower = 0
        k_d_gfp_lower = 0

        alpaga_basal_lower = 0 
        alpaga_act1_lower = 0 
        alpaga_act2_lower = 0
        alpaga_k3_lower = 0 
        alpaga_k_act1_lower = 0
        alpaga_k_act2_lower = 0

        copy_nr_lower = 5

        mumax_lower = mumax
        tc_lower = tc
        cap_lower = cap
        sigma_lower = sigma

        scale_lower = 1

        #Pack the upper and lower bounds for normal sampling, same order as parameters
        self.normal_upper = [k_b_p3_upper, k_b_p11_upper, k_op_succ_upper, k_pt_lldr_upper, k_pt_gfp_upper, k_f_lldrcomplex_upper, k_d_lldrmRNA_upper, k_d_lldr_upper, k_d_lldrcomplex_upper, k_d_gfpmRNA_upper, k_d_gfp_upper, alpaga_basal_upper, alpaga_act1_upper, alpaga_act2_upper, alpaga_k3_upper, alpaga_k_act1_upper, alpaga_k_act2_upper, copy_nr_upper, mumax_upper, tc_upper, cap_upper, sigma_upper, scale_upper]
        self.normal_lower = [k_b_p3_lower, k_b_p11_lower, k_op_succ_lower, k_pt_lldr_lower, k_pt_gfp_lower, k_f_lldrcomplex_lower, k_d_lldrmRNA_lower, k_d_lldr_lower, k_d_lldrcomplex_lower, k_d_gfpmRNA_lower, k_d_gfp_lower, alpaga_basal_lower, alpaga_act1_lower, alpaga_act2_lower, alpaga_k3_lower, alpaga_k_act1_lower, alpaga_k_act2_lower, copy_nr_lower, mumax_lower, tc_lower, cap_lower, sigma_lower, scale_lower]
        
        #Parameter bounds for logaritmic sampling
        k_b_p3_upper_log = 2
        k_b_p11_upper_log = 2

        k_op_succ_upper_log = 2
        #k_op_glc_upper = 10

        k_pt_lldr_upper_log = 2
        k_pt_gfp_upper_log = 2

        k_f_lldrcomplex_upper_log = 2

        k_d_lldrmRNA_upper_log = 2
        k_d_lldrcomplex_upper_log = 2
        k_d_lldr_upper_log = 2
        k_d_gfpmRNA_upper_log = 2
        k_d_gfp_upper_log = -1.3 #Assuming that GFP half life is very long

        alpaga_basal_upper_log = 2 
        alpaga_act1_upper_log = 2
        alpaga_act2_upper_log = 2
        alpaga_k3_upper_log = 2
        alpaga_k_act1_upper_log = 2
        alpaga_k_act2_upper_log = 2

        copy_nr_upper_log = np.log10(5) #Keep these constant

        mumax_upper_log = np.log10(mumax) #Keep these constant
        tc_upper_log = np.log10(tc) #Keep these constant
        cap_upper_log = np.log10(cap) #Keep these constant
        sigma_upper_log = np.log10(sigma) #Keep these constant

        scale_upper_log = 3

        k_b_p3_lower_log = -2
        k_b_p11_lower_log = -2
        k_op_succ_lower_log = -2
        
        k_pt_lldr_lower_log = -2
        k_pt_gfp_lower_log = -2

        k_f_lldrcomplex_lower_log = -2

        k_d_lldrmRNA_lower_log = -2
        k_d_lldrcomplex_lower_log = -2
        k_d_lldr_lower_log = -2
        k_d_gfpmRNA_lower_log = -2
        k_d_gfp_lower_log = -3 #Assuming that GFP half life is very long

        alpaga_basal_lower_log = -2
        alpaga_act1_lower_log = -2
        alpaga_act2_lower_log = -2
        alpaga_k3_lower_log = -2 
        alpaga_k_act1_lower_log = -2
        alpaga_k_act2_lower_log = -2

        copy_nr_lower_log = np.log10(5)

        mumax_lower_log = np.log10(mumax)
        tc_lower_log = np.log10(tc)
        cap_lower_log = np.log10(cap)
        sigma_lower_log = np.log10(sigma)

        scale_lower_log = 0

        #Pack the upper and lower bounds for logaritmic sampling, same order as parameters
        self.log_upper = [k_b_p3_upper_log, k_b_p11_upper_log, k_op_succ_upper_log, k_pt_lldr_upper_log, k_pt_gfp_upper_log, k_f_lldrcomplex_upper_log, k_d_lldrmRNA_upper_log, k_d_lldr_upper_log, k_d_lldrcomplex_upper_log, k_d_gfpmRNA_upper_log, k_d_gfp_upper_log, alpaga_basal_upper_log, alpaga_act1_upper_log, alpaga_act2_upper_log, alpaga_k3_upper_log, alpaga_k_act1_upper_log, alpaga_k_act2_upper_log, copy_nr_upper_log, mumax_upper_log, tc_upper_log, cap_upper_log, sigma_upper_log, scale_upper_log]
        self.log_lower = [k_b_p3_lower_log, k_b_p11_lower_log, k_op_succ_lower_log, k_pt_lldr_lower_log, k_pt_gfp_lower_log, k_f_lldrcomplex_lower_log, k_d_lldrmRNA_lower_log, k_d_lldr_lower_log, k_d_lldrcomplex_lower_log, k_d_gfpmRNA_lower_log, k_d_gfp_lower_log, alpaga_basal_lower_log, alpaga_act1_lower_log, alpaga_act2_lower_log, alpaga_k3_lower_log, alpaga_k_act1_lower_log, alpaga_k_act2_lower_log, copy_nr_lower_log, mumax_lower_log, tc_lower_log, cap_lower_log, sigma_lower_log, scale_lower_log]

        self.sampled_space = np.empty(0)

    def __eq__(self, other: object) -> bool:
        """
        This method is used to compare two objects of the same class.
        Made to check if the compared models can be trained with the same parameter set.
        Args:
            other (object): The object to compare to.
        
        Returns:
            bool: True if the objects are equal, False otherwise.
        """
        print("Checking model compatibility")
        #Check if parameters are the same
        if self.params == other.params and self.param_labels == other.param_labels:
            print("Parameters are the same.")
            #Check if initial conditions are the same
            if self.initial_conditions == other.initial_conditions and self.labels == other.labels:
                print("Initial conditions are the same.")
                if self.normal_lower == other.normal_lower and self.normal_upper == other.normal_upper:
                    print("Sampling bounds are the same.")
                    print("Model is compatible!")
                    return True

        print("Model is not compatible!")
        return False

    def __str__(self):
        info = "This model is used to simulate the dynamics of lactate-dependent GFP production."
        return info
    
    def __repr__(self):
        return self.__str__()

    def deriv(self, y, t, p):
        """
        Function which calculates the derivatives and returns them..
        Every ODEModel needs a deriv function - The 'heart' of the model.

        Args:
            y : Array of concentrations of all species in the model.
            t : Array of time points.
            p : Array of parameters.
        
        Returns:
            dydt : Array of derivatives of all species in the model.
        """
        #Unpack the state vector
        Lactate, lldr_mRNA, lldr, lldr_complex, GFP_mRNA, GFP, Cx = y
        #Unpack the parameter vector
        k_b_p3, k_b_p11, k_op_succ,k_pt_lldr, k_pt_gfp, k_f_lldrcomplex, k_d_lldrmRNA, k_d_lldr, k_d_lldrcomplex, k_d_gfpmRNA, k_d_gfp, alpaga_basal, alpaga_act1, alpaga_act2, alpaga_k3, alpaga_k_act1, alpaga_k_act2, copy_nr, mumax, tc, cap, sigma, gamma, scale = p
        
        #Keller paper synthesis rate
        #AlpaGA promoter
        top_a = alpaga_basal + (alpaga_act1 + alpaga_basal)*(alpaga_k_act1*lldr) + (alpaga_act2 + alpaga_basal)*(alpaga_k_act2*lldr_complex) #(epsilon_r/2)*rep_p*alpaga_basal*alpaga_k3*lldr**2
        bottom_a = gamma + alpaga_k_act1*lldr + alpaga_k_act2*lldr_complex + alpaga_k3*lldr**2
        
        #Calculate growth rate
        mu = mumax*np.exp(-0.5*((t-tc)/sigma)**2)*(1-Cx/cap)

        #Calculate the derivatives
        d_Lactate_dt = 0 #We assume that the lactate concentration is constant
        d_Lldr_mRNA_dt = mu/mumax*(k_op_succ + copy_nr*k_b_p3) -lldr_mRNA*k_d_lldrmRNA -lldr_mRNA*mu
        d_lldr_dt =  mu*(k_pt_lldr*lldr_mRNA) -k_f_lldrcomplex*lldr*Lactate**2 -lldr*k_d_lldr -lldr*mu
        d_lldr_complex_dt =  k_f_lldrcomplex*lldr*Lactate**2 - lldr_complex*k_d_lldrcomplex - lldr_complex*mu
        d_GFP_mRNA_dt = mu/mumax*(copy_nr*top_a/bottom_a) - GFP_mRNA*k_d_gfpmRNA - GFP_mRNA*mu
        d_GFP_dt =  mu*(k_pt_gfp*GFP_mRNA) - GFP*k_d_gfp-GFP*mu
        d_Cx_dt = Cx*mu
        
        #Pack the derivative vector
        dydt = [d_Lactate_dt, d_Lldr_mRNA_dt, d_lldr_dt, d_lldr_complex_dt, d_GFP_mRNA_dt, d_GFP_dt, d_Cx_dt]

        return dydt

    def score_model(self, time, simtime, outputs, data, error, scale=1):
        """
        Score the model for a given simulation output.

        Args:
            time (np.array): Time vector of the simulations.
            simtime (np.array): Time vector of the data.
            outputs (np.array): Simulations output.
            data (np.array): Data to compare the simulations output to.
            error (np.array): Error of the data.
            scale : scaling simulation output to match data
        """
        score = 0
        # print(f"Length of concentrations: {len(self.concentrations_data)}") #7
        # print(f"Length of outputs: {len(outputs)}") #75600
        # print(f"Length of data: {len(data)}") #251
        # print(f"Length of error: {len(error)}") #251

        # print(f"Shape of concentrations: {np.shape(self.concentrations_data)}") 
        # print(f"Shape of outputs: {np.shape(outputs)}")
        # print(f"Shape of data: {np.shape(data)}")
        # print(f"Shape of error: {np.shape(error)}")

        #Score the model for each concentration time series
        for i, _concentration in enumerate(self.concentrations_data):
            #Calculate the weighted sum of squared error for each output
            score += self.weighted_sum(time, simtime, outputs[:,i], data[:,i], error[:,i], scale=scale)
            #print(f"Score for {_concentration} is {self.weighted_sum(time, simtime, outputs[:,i], data[:,i], error[:,i], scale=scale)}")
            #print(f"Concentration:{concentration} - Score:{score-prev_score}")
            #print(f"Simulation concentration: {output[0]}")
            #print(f"Data header: {series[0]}")

        #Score the growth equation
        #Assuming that growth is unaffected by lactate (fitting to the od gained at 0.1 mM lactate)
        #No need to scale the growth score
        #prev_score = score
        #score += self.weighted_sum(time, simtime, outputs[0][:,-1], self.OD_data_p3[:,0], self.OD_error_p3[:,0], 1)

        #print(f"Growth Score:{score-prev_score}")

        #Score the dose-response curve
        #prev_score = score
        #score += self.weighted_sum(self.concentrations_data, concentrations, responses, self.dose_response, self.dose_response_error, scale)
        #print(f"Dose-response Score:{score-prev_score}")
        #print(f"Total Score:{score}")

        return score

    # def simulate_experiment(self, p, init, concentration1, concentration2):
    #     """
    #     Function that will try to simulate the experiment that Bas will do in the lab.

    #     Args:
    #         p : Parameter vector
    #         init : Initial conditions
    #         concentration1 : Concentration of lactate in the overnight experiment
    #         concentration2 : Concentration of lactate in the plate-reader experiment
        
    #     Returns:
    #         sim_time1/2 : Time vector of the simulation
    #         sim_output1/2 : Output of the simulation
    #     """
    #     #Simulate overnight growth (Before plate reader)
    #     init[0] = concentration1
    #     sim_time, sim_output = self.simulateODE(p, init, 24, 24*3600)

    #     #Dilute back to original OD
    #     #Get the dilution factor
    #     dil_factor = sim_output[-1,-1]/sim_output[0,-1]

    #     #Calculate new initial conditions
    #     init_new = sim_output[-1]/dil_factor
        
    #     #Change lactate concentration
    #     init_new[0] = concentration2

    #     # print(init_new)
    #     #Simulate the (Plate-reader) experiment 
    #     sim_time2, sim_output2 = self.simulateODE(p, init_new, 21, 21*3600)

    #     #Dont concatenate since we usually only want the second part of the experiment
    #     return sim_time, sim_output, sim_time2, sim_output2

    def simulate_experiment(self, p, init, concentration1, concentration2):
        """
        Function that will try to simulate the experiment that Bas will do in the lab.
        """
        #Simulate overnight growth (Before plate reader)
        init[0] = concentration1
        sim_time, sim_output = self.simulateODE(p, init, 24, 24*3600)

        #Dilute the cell volume
        init_new = sim_output[-1]
        init_new[-1] = sim_output[0,-1]
   
        #Change lactate concentration
        init_new[0] = concentration2
        #Simulate the (Plate-reader) experiment 
        sim_time2, sim_output2 = self.simulateODE(p, init_new, 21, 24*3600)

        #Dont concatenate since we usually only want the second part of the experiment
        return sim_time, sim_output, sim_time2, sim_output2

    def simulate_concentrations(self, p, concentrations):
        """
        Simulate the model for each concentration in the data.
        Returns a list of time series for each concentration.

        GFP is divided by OD and scaled by the scale factor.(p[-1])

        Args:
            p : Parameter vector
            concentrations : List/array of concentrations to simulate

        Returns:
            gfp_outputs : List of fluorescense time series for each concentration
            sim_time2 : Time vector of the simulation
        """
        gfp_outputs = []
        for i in concentrations:
            sim_time, sim_output, sim_time2, sim_output2 = self.simulate_experiment(p, self.init, 0, i)
            #Get the GFP output and divide by the OD and scale.
            gfp_output = sim_output2[:,-2]*p[-1]
            #Append the FLUOD simulation output
            gfp_outputs.append(gfp_output)

        #Which format do we want to return the data in?
        gfp_outputs = np.array(gfp_outputs).T
        #Return the outputs and time as FLUOD
        return gfp_outputs, sim_time2
    
    def calculate_ec(self, concentrations, response, ec_percent):
        """
        Used to calculate the desired EC value from a dose-response curve.

        Args:
            concentrations : List/array of concentrations
            response : List/array of responses
            ec_percent : Percentage of the maximum response to calculate the EC for

        Returns:
            ec_conc : The ECx concentration
            ec_response : The ECx response
        """
        ec_response = (max(response)-min(response))*ec_percent+min(response)
        response_reduced = response - ec_response
        freduced = interpolate.UnivariateSpline(concentrations, response_reduced, s=0)
        #Sometimes returns multiple values for some reason...
        try:
            ec_conc = freduced.roots()[0]
        except:
            ec_conc = 0
        #print(f"EC{ec_percent} concentration:{ec_conc}")
        #print(f"EC{ec_percent} response:{ec_response}")
        return ec_conc, ec_response

    def evaluate_dose_response(self, p, concentrations):
        """
        Simulate and score the model for each concentration in the data.

        Args:
            p : Parameter vector
            concentrations : List/array of concentrations to simulate

        Returns:
            ec : list of EC5, EC50, EC95 values
            ec_response : list of EC5, EC50, EC95 responses
            responses : dose-response curve
            simulations : time series simulations of the dose-response curve (Not used since it took too much memory)
        """
        responses = []
        simulations = []
        for i in range(len(concentrations)):
            timepoints, output, time2, output2 = self.simulate_experiment(p, self.init, 0, concentrations[i])
            responses.append(output2[-1][-2]*p[-1])
        
        #Interpolate the responses
        ec5, ec5_response = self.calculate_ec(concentrations, responses, 0.05)
        ec50, ec50_response = self.calculate_ec(concentrations, responses, 0.5)
        ec95, ec95_response = self.calculate_ec(concentrations, responses, 0.95)
        #Store ec and responses
        ec = [ec5, ec50, ec95]
        ec_response = [ec5_response, ec50_response, ec95_response]
        #Reshape the simulations
        simulations = np.array(simulations).T
        
        return ec, ec_response, responses, simulations

    def lhs_result_opt(self, p):
        """
        Function to be passed in process_optim_results.py
        Used to further optimize latin hypercube sampling results.
        Basically the same as simulate_concentrations, but the function for opt.minimize can only return one value.

        Args:
            p : Parameter vector
        
        Returns:
            score : Score of the model for the given parameter vector
        """
        gfp_outputs, sim_time = self.simulate_concentrations(p, self.concentrations_data)
        
        #Score the model
        score = self.score_model(self.time, sim_time, gfp_outputs, self.FLUOD_data_p3, self.FLUOD_error_p3, 1)

        return score

    def oaat_analysis_mp(self, _num = 9, folder='figures', plot_data = False):
        """
        This function will perform one at a time analysis of the model.
        Reports EC values and dose-response curves.
        Also make plots and bar charts of the results.
        Uses multiprocessing to speed up the process.

        Args:
            _num : Number of samples to use for one parameter
            folder : Folder to save the figures in
            plot_data : Whether to plot the data in the figures or not
        
        Returns:
           nothing, but saves figures of the results
        """
        #Check if the folder exists
        if not os.path.exists(folder):
            os.makedirs(folder)
        #Check if _num is an uneven integer
        if _num%2 == 0 and _num<2:
            print('Number of perturbations must be an uneven integer!')
            raise ValueError

        #Empty array to store oaat parameter sets
        samplelist = np.empty((0,len(self.params)))
        #Sample 9 different perturbations of the parameters
        log = np.linspace(-1,1, num=_num)
        #Create a list of parameter sets with the perturbations
        #Last six parameters are left as is (wouldnt make sense to change them)
        for i, parameter in enumerate(self.params[:len(self.params)-1]):
            print(f"Sampling parameter {self.param_labels[i]}")
            sample = (10**log)*parameter
            for value in sample:
                params = self.params.copy()
                params[i] = value
                samplelist = np.row_stack((samplelist,params))
        #Log or linspace?
        concentrations = np.logspace(-1,1.7,50)
        #concentrations = np.linspace(0,40,50)

        #Make some lists/arrays to store info and results.
        oat_dose_curves = np.empty((0,len(concentrations)))
        sim_list = []
        calculated_ec = np.zeros((0,3))
        ec_responses = np.zeros((0,3))

        def multiprocess_wrapper(sample_batch):
            """
            Wrapper function for multiprocessing one-at-a-time analysis.

            Args:
                sample_batch : List of parameter sets to simulate
            
            Returns:
                dose_curves : List of dose-response curves
                calculated_ec : List of EC values
                ec_responses : List of EC responses
            """
            model = Lactate_model_succinate_p3()
            
            concentrations = np.logspace(-1,1.7,50)
            dose_curves = np.zeros((len(sample_batch),len(concentrations)))
            calculated_ec = np.zeros((len(sample_batch),3))
            ec_responses = np.zeros((len(sample_batch),3))

            for i, sample in tqdm(enumerate(sample_batch), total=len(sample_batch)):
                #Simulate a dose-response curve
                ec, ec_response, responses, _simulations = model.evaluate_dose_response(sample, concentrations)
                #Store the results
                dose_curves[i] = responses
                calculated_ec[i] = ec
                ec_responses[i] = ec_response
            
            return dose_curves, calculated_ec, ec_responses
        
        #Split the samplelist into batches
        sample_batches = np.array_split(samplelist, 8)
        #Create a pool of workers
        with multiprocess.Pool(8) as pool:
            results = pool.map(multiprocess_wrapper, sample_batches)
        #Process the results
        for result in results:
            oat_dose_curves = np.row_stack((oat_dose_curves, result[0]))
            calculated_ec = np.row_stack((calculated_ec, result[1]))
            ec_responses = np.row_stack((ec_responses, result[2]))
        #Close the pool
        pool.close()

        #Lists to store max and min EC5 values - Used for the barplot.
        min_ec = []
        max_ec = []

        original_ec = []
        original_background = []

        factor_10_increase = []
        factor_10_decrease = []

        factor_10_increase_background = []
        factor_10_decrease_background = []

        #Go through the results for each parameter, make a figure of the dose-response curve and save it.
        #Also calculate the max and min EC5 values for each parameter.
        print("Processing results for each parameter...")
        for i in tqdm(range(int(len(oat_dose_curves)/_num))):
            #Calculate max, min and add original.
            #Convert to float so it doesnt end up as a list of np.arrays.
            #min_ec.append(float(min(calculated_ec[i*_num:i*_num+_num,0])))
            #max_ec.append(float(max(calculated_ec[i*_num:i*_num+_num,0])))
            original_ec.append(float(calculated_ec[i*_num+int(_num/2),0]))
            original_background.append(float(oat_dose_curves[i*_num+int(_num/2),0]))

            #Calculate factor 10 increase and decrease
            increase_difference = calculated_ec[i*_num+_num-1,0]-original_ec[i]
            decrease_difference = calculated_ec[i*_num,0]-original_ec[i]

            factor_10_increase.append(float(increase_difference))
            factor_10_decrease.append(float(decrease_difference))

            factor_10_increase_background.append(float(oat_dose_curves[i*_num+_num-1,0]-original_background[i]))
            factor_10_decrease_background.append(float(oat_dose_curves[i*_num,0]-original_background[i]))

            fig, ax = plt.subplots(figsize=(16, 9), tight_layout = True)
            for j in range(_num):
                if 10**log[j] != 1:
                    plt.plot(concentrations, oat_dose_curves[i*_num+j], label=f'factor {np.round(10**log[j],3)}', color=color_list[j])
                else:
                    plt.plot(concentrations, oat_dose_curves[i*_num+j], label=f'Original', color=color_list[j])
                plt.plot(calculated_ec[i*_num+j], ec_responses[i*_num+j], 'or')
            if plot_data:
                plt.errorbar(self.concentrations, self.dose_response, yerr=self.dose_response_error, label='Data', marker='o')
            ##plt.title(f"Oaat analysis of factor 0.1-10 change in parameter {self.param_labels[i]}")
            plt.ylabel('Fluorescence/OD600 (a.u.)')
            plt.xlabel('Lactate concentration (mM)')
            plt.xscale('log')
            plt.axvspan(1.5,3, color=colors['Green'], alpha=0.3, label='Healthy')
            plt.axvspan(10,25, color=colors['Red'], alpha=0.3, label='Cancer')
            plt.axvline(1.5,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2)
            plt.axvline(3,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2)
            plt.axvline(10,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2)
            plt.axvline(25,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2)
            plt.legend()
            plt.savefig(f"{folder}/{self.param_labels[i]}_oaat_result.svg")
            plt.close()

        #Make a normal and polar plot of the EC values.
        labels = self.param_labels[:len(self.params)-1]
        #Remove alpaga_act1 from labels and factor_10_increase and factor_10_decrease
        labels = labels[:12]+labels[13:]
        factor_10_increase = factor_10_increase[:12]+factor_10_increase[13:]
        factor_10_decrease = factor_10_decrease[:12]+factor_10_decrease[13:]

        factor_10_increase_background = factor_10_increase_background[:12]+factor_10_increase_background[13:]
        factor_10_decrease_background = factor_10_decrease_background[:12]+factor_10_decrease_background[13:]

        #First the normal plot
        fig, ax = plt.subplots(figsize=(16, 9), tight_layout = True)
        ax.bar(labels, factor_10_increase, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
        #ax.bar(labels, original_ec, width=1, color="#45c4af", edgecolor='black', label="EC5 - Original", bottom=0)
        ax.bar(labels, factor_10_decrease, width=1, color="#ec6c5f",edgecolor='black', label="Factor 10 decrease", bottom=0)
        plt.xticks(rotation=90)
        plt.legend(loc='upper center')
        plt.ylabel('Change in EC5 concentration (mM)')
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_barchart.svg", dpi= 300)
        plt.close()

        #Then the polar plot
        ANGLES = []
        XTICK_angles = []
        N = len(labels)

        for i in range(N):
            ANGLES.append(np.radians(360/N*i))
            XTICK_angles.append(np.radians(360/N*i)+np.pi/N)
            
        WIDTH = (2*np.pi)/N
        OFFSET = np.pi / 2

        fig, ax = plt.subplots(figsize=(16, 9), tight_layout = True, subplot_kw={"projection": "polar"})

        ax.bar(ANGLES, factor_10_increase, width=WIDTH, color="#4d8b31", edgecolor='black', label="EC5 - Max", bottom=0)
        #ax.bar(ANGLES, original_ec, width=WIDTH, color="#45c4af", edgecolor='black', label="EC5 - Original", bottom=0)
        ax.bar(ANGLES, factor_10_decrease, width=WIDTH, color="#ec6c5f",edgecolor='black', label="EC5 - Min", bottom=0)

        ax.set_xticks(XTICK_angles)
        ax.set_xticklabels([])
        ax.set_theta_offset(OFFSET)
        ax.set_rlim(0,max(factor_10_decrease + factor_10_increase))
        ax.set_rorigin(-1)
        add_labels(ANGLES, max(factor_10_decrease + factor_10_increase), labels, OFFSET, ax, 10)
        plt.legend(loc='center')
        plt.savefig(f"{folder}/oaat_polar_barchart.svg", dpi= 300)
        plt.close()

        #Make a horizontal bar chart of the EC values
        fig, ax = plt.subplots(figsize=(12, 12), tight_layout = True)
        ax.barh(labels, factor_10_increase, height=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", left=0)
        #ax.barh(labels, original_ec, height=1, color="#45c4af", edgecolor='black', label="EC5 - Original", left=0)
        ax.barh(labels, factor_10_decrease, height=1, color="#ec6c5f",edgecolor='black', label="Factor 10 decrease", left=0)
        plt.legend(loc='upper center')
        plt.xlabel('Change in EC5 concentration (mM)')
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_horizontal_barchart.svg", dpi= 300)
        plt.close()

        #Make a barplot of the factor 10 increase and decrease.
        fig, ax = plt.subplots(figsize=(16, 9), tight_layout = True)
        ax.bar(labels, factor_10_increase, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
        ax.bar(labels, factor_10_decrease, width=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", bottom=0)
        plt.xticks(rotation=90)
        plt.legend(loc='upper center')
        plt.ylabel('Change in EC5 (mM)')
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_barchart_factor_10.svg", dpi= 300)
        plt.close()

        #Make a bar chart of the background values
        fig, ax = plt.subplots(figsize=(16, 9), tight_layout = True)
        ax.bar(labels, factor_10_increase_background, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
        ax.bar(labels, factor_10_decrease_background, width=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", bottom=0)
        plt.xticks(rotation=90)
        plt.legend(loc='upper center')
        plt.ylabel('Change in background fluorescence (a.u.)')
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_barchart_factor_10_background.svg", dpi= 300)
        plt.close()

        #Make a horizontal bar chart of the factor 10 background increase and decrease.
        fig, ax = plt.subplots(figsize=(12, 12), tight_layout = True)
        ax.barh(labels, factor_10_increase_background, height=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", left=0)
        ax.barh(labels, factor_10_decrease_background, height=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", left=0)
        plt.legend(loc='upper center')
        plt.xlabel('Change in background fluorescence (a.u.)')
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_horizontal_barchart_factor_10_background.svg", dpi= 300)
        plt.close()

        #Write the function that we used for our model.
        import inspect
        with open(f"{folder}/deriv.txt", "w") as f:
            f.write("This model was used for these results:\n")
            f.write("______________________________________\n")
            f.write(inspect.getsource(self.deriv))

    def report_results(self, scores, sets, folder='figures'):
        """
        Function to report the results of the model fitting.
        Select the top 1000 from sorted scores,sets
        Save them and make a histogram of score and parameter distribution

        Args:
            scores: list of scores
            sets: list of parameter sets
            folder: folder to save the results in

        Returns:
            None, saves the results to the folder
        """
        #Check if the folder exists, if not make it
        if not os.path.exists(folder):
            os.makedirs(folder)

        #Select X(now 1000) of top scores
        selection = sets[:1000]
        scores = scores[:1000]

        if len(scores) > 1:
            #Sort scores just in case they are not
            index = scores.argsort()
            sets = sets[index[::]]
            scores = scores[index[::]]
        
            #Make a file with top sets and scores
            top_results = np.column_stack((selection,scores))
            #Create header for the file
            header_list = [x for x in self.param_labels]
            header_list.append("score")
            np.savetxt(f"{folder}/best_score_{len(selection)}_outof_{len(sets)}.csv", top_results, delimiter=',' ,header=','.join(header_list), comments='')
        
            #Make histograms of the score distribution
            median = np.round(np.median(scores),2)
            plt.figure(figsize=(14,9))
            plt.hist(scores, bins=100)
            #plt.title(f"Distribution of scores in the top {len(selection)} scores out of {len(sets)}.")
            plt.axvline(x=median,ymin = 0, ymax=1, label=f"Median: {median}", color="red", linestyle="--", linewidth=2)
            plt.xlabel("Score value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(f"{folder}/score_dist_{len(selection)}_{len(sets)}.svg", dpi= 200)
            plt.close()

        #Simulate the best parameter set
        #Select the parameter set with the best score
        best_score = np.argmin(scores)
        #If we have multiple parameter sets passed, perform median simulation, 100 average simulation and make histograms of parameter distribution
        #Otherwise just simulate the parameter set
        if len(scores) > 1:
            best_params = selection[best_score]

            #Make histograms of the individual parameters
            #Also calculate the median.
            median_params = []
            for i in range(len(self.param_labels)):
                values = []
                for j in range(len(selection)):
                    values.append(selection[j][i])
                median = np.round(np.median(values),2)
                median_params.append(median)
                plt.figure(figsize=(14,9))
                plt.hist(values, bins=100)
                plt.axvline(x=np.median(values),ymin =0, ymax=1, label=f"Median: {median}", color="red", linestyle="--", linewidth=2)
                #plt.title(f"Distribution of {self.param_labels[i]} in the top {len(selection)} scores out of {len(sets)}.")
                plt.xlabel("Parameter value")
                plt.ylabel("Frequency")
                plt.legend()
                plt.savefig(f"{folder}/{self.param_labels[i]}_{len(selection)}_{len(sets)}.svg", dpi= 200)
                plt.close()
            
            #Simulate the median parameter set for concentrations.
            FLUOD_outputs, simtime = self.simulate_concentrations(median_params, self.concentrations)
            plt.figure(figsize=(14,9))
            plt.errorbar(self.concentrations_data, self.dose_response, self.dose_response_error, label="Data", color=colors['Purple'])
            plt.plot(self.concentrations, FLUOD_outputs[-1,:], label="Simulated fit", color=colors['Green'])
            plt.xlabel("Lactate concentration (mM)")
            plt.xscale("log") 
            plt.ylabel("Fluorescence/OD600 (a.u.)")
            #plt.title("Comparing simulation with median params to data")
            plt.legend()
            plt.savefig(f"{folder}/median_parameters_{len(selection)}_{len(sets)}.svg", dpi= 200)
            plt.close()
            
            #Simulate best 100? scores, and report average dose-response + stdev etc.
            #Need for sorted list.
            sub_selection = selection[:100]
            outputs = np.zeros((len(sub_selection+1),len(FLUOD_outputs[0,:])))
            errors = np.zeros(len(sub_selection))

            for i in range(len(sub_selection)):
                FLUOD_outputs, simtime = self.simulate_concentrations(sub_selection[i], self.concentrations)
                outputs[i] = FLUOD_outputs[-1,:]
            
            for i in range(len(outputs[0])):
                outputs[-1,i] = np.average(outputs[:,i])
                errors = np.std(outputs[:,i])
            plt.figure(figsize=(14,9))
            plt.errorbar(self.concentrations_data, self.dose_response, self.dose_response_error, label="Data", color=colors['Purple'])
            plt.errorbar(self.concentrations, outputs[-1], errors, label="Simulation", color=colors['Orange'])
            plt.xlabel("Lactate concentration (mM)")
            plt.xscale("log") 
            plt.ylabel("Fluorescence/OD600 (a.u.)")
            plt.legend()
            plt.savefig(f"{folder}/100_averaged_simulations_{len(selection)}_{len(sets)}.svg", dpi= 200)
            plt.close()
        else:
            best_params = selection
        print(f"The best parameters were: {best_params} and gave a score of: {scores[best_score]}")
            
        #Compare simulation vs data for the best parameter set
        FLUOD_outputs, simtime = self.simulate_concentrations(best_params, self.concentrations)
        #Plot the simulation
        plt.figure(figsize=(14,9))
        plt.errorbar(self.concentrations_data, self.dose_response, self.dose_response_error, label="Data", color=colors['Purple'], capsize=5)
        plt.plot(self.concentrations_data, FLUOD_outputs[-1,:], label="Simulation", color=colors['Orange'])
        plt.axvline(1.5,ymin =0, ymax=1, label="Healthy", color=colors['Green'], linestyle="--", linewidth=2)
        plt.axvline(3,ymin =0, ymax=1, color=colors['Green'], linestyle="--", linewidth=2)
        plt.axvline(10,ymin =0, ymax=1, label="Cancer", color=colors['Red'], linestyle="--", linewidth=2)
        plt.axvline(25,ymin =0, ymax=1, color=colors['Red'], linestyle="--", linewidth=2)
        plt.xlabel("Lactate concentration (mM)")
        plt.xscale("log")
        plt.ylabel("Fluorescence/OD600 (a.u.)")
        plt.legend()
        plt.savefig(f"{folder}/simulation_vs_data_{len(selection)}_{len(sets)}.svg", dpi= 200)
        plt.close()


        #For the best simulation, make a time-series plot for each concentration
        for i in range(len(self.concentrations)):
            sim_time, sim_output, sim_time2, sim_output2 = self.simulate_experiment(best_params, self.init, 0, self.concentrations[i])
            plt.figure(figsize=(14,9))
            for j in range(len(sim_output2[0])):
                #if j > 0:
                plt.plot(sim_time2, sim_output2[:,j], label=self.labels[j], marker=self.markers[j], markevery=3600)
            plt.legend(loc="upper left")
            plt.xlabel("Time (h)")
            plt.ylabel("Concentration (mM)")
            #plt.title(f"Time series for lactate concentration {self.concentrations[i]} mM")
            plt.savefig(f"{folder}/time_series_{self.concentrations[i]}mM_{len(selection)}_{len(sets)}.svg", dpi= 200)
            plt.close()

        #Also make a series for the OD
        sim_time, sim_output, sim_time2, sim_output2 = self.simulate_experiment(best_params, self.init, 0, 0)
        plt.figure(figsize=(14,9))
        plt.plot(sim_time2, sim_output2[:,-1], label="Simulated OD", marker="o", markevery=3600, color=colors['Orange'])
        plt.errorbar(self.time, self.OD_data_p3[:,4], self.OD_error_p3[:,4], label="Data", color=colors['Purple'])
        plt.legend(loc="upper left")
        plt.xlabel("Time (h)")
        plt.ylabel("Optical density at 600nm")
        #plt.title("Time series for OD")
        plt.savefig(f"{folder}/time_series_OD_{len(selection)}_{len(sets)}.svg", dpi= 200)
        plt.close()

        #Make a figure for the "overnight" simulation
        plt.figure(figsize=(14,9))
        for j in range(len(sim_output[0])):
            #if j > 0:
            plt.plot(sim_time, sim_output[:,j], label=self.labels[j], marker=self.markers[j], markevery=3600)
        plt.legend(loc="upper left")
        plt.xlabel("Time (h)")
        plt.ylabel("Concentration (mM)")
        #plt.title(f"Time series for overnight growth (0mM Lactate)")
        plt.savefig(f"{folder}/time_series_overnight_{len(selection)}_{len(sets)}.svg", dpi= 200)
        plt.close()

        #Also make a series for the GFP
        for i in range(len(self.concentrations)):
            sim_time, sim_output, sim_time2, sim_output2 = self.simulate_experiment(best_params, self.init, 0, self.concentrations[i])
            plt.figure(figsize=(14,9))
            plt.plot(sim_time2, sim_output2[:,-2]*best_params[-1], label="Simulated", marker="o", markevery=3600, color=colors['Green'])#/sim_output2[:,-1]*best_params[-1]
            plt.errorbar(self.time, self.FLUOD_data_p3[:,i], self.FLUOD_error_p3[:,i], label="Data", color=colors['Purple'])
            plt.legend(loc="upper left")
            plt.xlabel("Time (h)")
            plt.ylabel("Fluorescence/OD600 (a.u.)")
            #plt.title(f"Time series for GFP at {self.concentrations[i]} mM Lactate")
            plt.savefig(f"{folder}/time_series_GFP_{self.concentrations[i]}mM_{len(selection)}_{len(sets)}.svg", dpi= 200)
            plt.close()
        
        #Write the function that we used for our model.
        import inspect
        with open(f"{folder}/deriv.txt", "w") as f:
            f.write("This model was used for these results:\n")
            f.write("______________________________________\n")
            f.write(inspect.getsource(self.deriv))

def optimize(samples):
    """
    Optimization function for multiprocessing.

    Args:
        samples (list): List of samples to be optimized.
    
    Returns:
        samples (list): List of samples to be optimized.
        scores (list): List of scores for each sample.
    """
    model = Lactate_model_succinate_p3()
    scores = np.zeros(len(samples))

    for i in tqdm(range(len(samples))):
        FLUOD_outputs, simtime = model.simulate_concentrations(samples[i], model.concentrations_data)
        #Score the model
        scores[i] =  model.score_model(model.time, simtime, FLUOD_outputs, model.FLUOD_data_p3, model.FLUOD_error_p3, 1)
    
    return samples, scores

if __name__ == '__main__':
    #Argument parsing
    parser = argparse.ArgumentParser(description="Josia's script(s) for modeling")
    parser.add_argument("--multiprocessing", help="Specify if using multiprocessing optimization.", default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument("--samplesize", help="Total sample size", default="1000000")
    args = parser.parse_args()
    
    #Multiprocessing script which: 
    #1.Samples a by argument specified sample size
    #2.Divide parameter space into sections
    #3.Start process that simulates and scores each section
    #4.Report
    if args.multiprocessing == True:
        #Samples for optim_wrapper
        model = Lactate_model_succinate_p3()
        model.sample(int(args.samplesize))
        
        #Round the hill coefficients (Not needed since we fixed the hill coefficients)
        #model.sampled_space[:,26:28] = np.around(model.sampled_space[:,26:28])
        
        #Save samples
        model.save_samples(int(args.samplesize), "lactatemodel")
        
        #Multiprocessing
        used_cores = 8
        pool = multiprocess.Pool(used_cores)
        
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
        
        #Report time spent
        end_time = time.perf_counter()
        print(f"Optimization took {end_time-start_time} seconds")
        
        #Select the parameter set with the best score
        best_score = np.argmin(sorted_scores)
        best_params = sorted_sets[best_score]
        
        #Parralel processing is done, close
        pool.close()
        
        #Report results of the 1000 best lhs parameters
        model = Lactate_model_succinate_p3()
        folder = 'figures/mudependent_10000samples_likekhammash_test'
        model.report_results(sorted_scores, sorted_sets, folder=folder)
        np.savetxt(f"{folder}/multiprocessing_results.csv", combined, delimiter=',')
    else:
        model = Lactate_model_succinate_p3()
        # #Samples for optim_wrapper
        #model.sample(1000000)
        #Save samples
        #model.save_samples(int(args.samplesize), "14sep_correction")
            #     plt.show()
        SMALL_SIZE = 18
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=12)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)
        #Perform one at a time analysis
        #model.oaat_analysis_mp(folder="figures/final_sensitivity_lactatemodel_plotted", plot_data=True)
        #model.oaat_analysis_mp(folder="figures/final_sensitivity_lactatemodel_notplotted", plot_data=False)
                #Also make a series for the GFP

        responses = []
        #Test GFP time series
        for i in range(len(model.concentrations)):
            sim_time, sim_output, sim_time2, sim_output2 = model.simulate_experiment(model.params, model.init, 0, model.concentrations[i])
            #Append final gfp value
            responses.append(sim_output2[-1,-2])
            #Plot the time series
            plt.figure(figsize=(14,9))
            plt.plot(sim_time2, sim_output2[:,-2]*model.params[-1], label="GFP", marker="o", markevery=3600)
            plt.errorbar(model.time, model.FLUOD_data_p3[:,i], model.FLUOD_error_p3[:,i], label="Data")
            plt.legend(loc="upper left")
            plt.xlabel("Time (h)")
            plt.ylabel("Fluorescence/OD600")
            #plt.title(f"Time series for GFP at {model.concentrations[i]} mM Lactate")
            plt.savefig(f"figures/modeltest/time_series_GFP_{model.concentrations[i]}mM.svg", dpi= 200)
            plt.close()

        #Plot the responses
        plt.figure(figsize=(14,9))
        plt.plot(model.concentrations, np.array(responses)*model.params[-1], label="GFP", marker="o")
        plt.errorbar(model.concentrations, model.dose_response, model.dose_response_error, label="Data")

        plt.legend(loc="upper left")
        plt.xlabel("Lactate concentration (mM)")
        plt.ylabel("Fluorescence/OD600")
        plt.xscale("log")
        #plt.title(f"Time series for GFP at {model.concentrations[i]} mM Lactate")
        plt.savefig(f"figures/modeltest/dose_response_GFP.svg", dpi= 200)
        plt.close()

        # #Also make a figure for the OD600
        # plt.figure(figsize=(14,9))
        # plt.plot(sim_time2, sim_output2[:,-1], label="OD600", marker="o", markevery=3600)
        # plt.errorbar(model.time, model.OD_data_p3[:,5], model.OD_error_p3[:,5], label="Data") #Index 5 is 50 mM lactate
        # plt.legend(loc="upper left")
        # plt.xlabel("Time (h)")
        # plt.ylabel("OD600")
        # #plt.title(f"Time series for OD600")
        # plt.savefig(f"figures/modeltest/time_series_OD600_{model.concentrations[i]}mM.svg", dpi= 200)
        # plt.close()