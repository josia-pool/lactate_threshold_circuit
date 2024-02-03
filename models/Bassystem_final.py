# -*- coding: utf-8 -*-
"""
@author: Josia Pool

This is the combined model of the lactate dynamics and the crispri-asrna submodels
Added growth and transport to the model
"""

#Import required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import odeint
from scipy.optimize import leastsq
import scipy.interpolate as interpolate 

import multiprocess#ing this library accepts functions which are not defined globally
import argparse
import time
from distutils.util import strtobool
import os
from ODEModel_v2 import ODEModel
from tqdm import tqdm

#Standard plot stuff
dashline = (1, (3, 3))
colors = {'Orange': '#fba649', "Turquoise": '#45c4af', "Red": '#ec6c5f', "Green": '#4D8B31', "Purple": '#641877'}
color_list = np.array([np.array([251,166,73], dtype='int'),np.array([69,196,175], dtype='int'),np.array([236,108,95], dtype='int'),
                       np.array([76,139,48], dtype='int'),np.array([100,25,120], dtype='int'),np.array([221,204,119], dtype='int'),
                       np.array([204,121,167], dtype='int'),np.array([254, 226, 195], dtype='int'),
                       np.array([136,34,85], dtype='int'),np.array([194, 236, 229], dtype='int')])/255
from cycler import cycler
custom_cycler = (cycler(color=color_list, linewidth=[2,2,2,2,2,2,2,2,2,2]))
plt.rc('axes', prop_cycle=custom_cycler)

#Igem standard bar plot
#Bar plot parameters
BAR_WIDTH = 0.30
SPACING = 0.05
CAP_SIZE = 5
SCALE = 2
def standard_hbarchart(labels, data_labels, x_data, error, filename, x_label= "", y_label= "", _figwidth=16, _figheight=9, _nozeroerror = False):
    bars_nr = len(data_labels)
    SCALE = bars_nr-1
    
    fig, ax= plt.subplots(figsize=(_figwidth,_figheight), tight_layout=True)
    ax.set_prop_cycle(custom_cycler)
    nr = np.arange(len(labels))*SCALE

    for i, data in enumerate(x_data):
        colors = []
        for j in range(bars_nr):
            colors.append(color_list[i])
        
        if _nozeroerror:
            if sum(error[i]) != 0:
                plt.barh(nr, data, xerr=error[i], label=data_labels[i], capsize=CAP_SIZE, color=colors, edgecolor=np.array(colors)*0.8, linewidth=2, height=BAR_WIDTH)
            else:
                plt.barh(nr, data, label=data_labels[i], color=colors, edgecolor=np.array(colors)*0.8, linewidth=2, height=BAR_WIDTH)
        else:
            plt.barh(nr, data, xerr=error[i], label=data_labels[i], capsize=CAP_SIZE, color=colors, edgecolor=np.array(colors)*0.8, linewidth=2, height=BAR_WIDTH)
        nr = [x+SPACING+BAR_WIDTH for x in nr]
        
    plt.yticks([r*SCALE + (BAR_WIDTH+SPACING)*(bars_nr-1)/2 for r in range(len(labels))], labels)
    if x_label !="":
        plt.xlabel(x_label)
    if y_label !="":
        plt.ylabel(y_label)
    plt.legend(loc="best")
    plt.savefig(f"{filename}.svg", dpi=300)
    plt.show()
#Helper functions
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

class Bassystem(ODEModel):
    def __init__(self):
        """
        Initialize the model.
        Define all needed variables such as parameters, bounds, etc.
        Also should load any data to be fitted. - Not available (yet)
        """
        super().__init__()
        #Define the parameters
        #Parameters based on best scoring models
        #asRNa and lactate parameters from week 25-29JUL22
        gamma = 4.462 #Dilution factor
        #Parameter definition
        k_b_dcas = 33.117 #Constitutive production of dCas - Unit: mM/hr
        k_b_sgrna = 20.034 #Constitutive production of sgRNA - Unit: mM/hr
        k_d_ascascomplex = 5.331  #asCasComplex degradation + dilution - Unit: 1/hr
        k_d_ascomplex = 10.793 #asRNA-sgRNA degradation + dilution - Unit: 1/hr
        k_d_asrna = 5.751 #asRNA degradation + dilution - Unit: 1/hr
        k_d_cascomplex = 1.151 #casComplex degradation + dilution - Unit: 1/hr
        k_d_dcas = 5.076 #dCas degradation + dilution - Unit: 1/hr
        k_d_dcasmrna = 2.389 #dCas_mRNA degradation + dilution - Unit: 1/hr
        k_d_gfp = 0.002371 #GFP degradation + dilution - Unit: 1/hr - 0.0288 may be better if half-life is 24hrs (But requires refitting of the model)
        k_d_lldr = 0.007198 #Degradation rate of lldR - Unit: 1/hr
        k_d_lldrcomplex = 3.332 #Degradation rate of lldR_complex - Unit: 1/hr
        k_d_lldrmrna = 6.263 #Degradation rate of lldR_mRNA - Unit: 1/hr
        k_d_sgrna = 1.389 #sgRNA degradation + dilution - Unit: 1/hr
        k_f_ascascomplex1_fw = 1.348 #Formation rate of asCasComplex - Unit: mM^-1/hr
        k_f_ascascomplex1_rv = 1.100 #Dissociation rate of asCascomplex - Unit: 1/hr
        k_f_ascascomplex2_fw = 1.44 #Formation rate of asCasComplex but dCas comes last - Unit: mM^-1/hr
        k_f_ascascomplex2_rv = 1.760 #Dissociation rate of asCasComplex but dCas comes last - Unit: 1/hr
        k_f_ascomplex_fw = 0.735 #Formation rate of asRNA-sgRNA complex - Unit: mM^-1/hr
        k_f_ascomplex_rv = 7.799 #Dissociation rate of asRNA-sgRNA complex - Unit: 1/hr
        k_f_cascomplex_fw = 2.158 #Formation rate of dCas-sgRNA complex - Unit: mM^-1/hr
        k_f_cascomplex_rv = 0.855 #Dissociation rate of CasComplex - Unit: 1/hr
        k_f_lldrcomplex = 10.113 #Forward reaction rate of lldR complex formation- Unit: mM^-2/hr
        k_p_gfpmrna = 3.007 #Max production rate of GFP_mRNA - Unit: mM/hr
        k_pt_dcas = 6.854 #Speed at which dCas_mRNA is translated - Unit: 1/hr
        k_pt_gfp = 6.605 #Speed at which GFP_mRNA is translated - Unit: 1/hr
        k_pt_lldr = 9.042 #Speed at which lldR_mRNA is translated - Unit: 1/hr
        kd_dcas = 5.096 #Dissociation/ Michealis menten constant of casComplex repression - Unit: mM/dimensionless? Cancels out either way
        n_alpaga =  2 #Hill coefficient - Unit: Dimensionless
        n_cascomplex = 4 #Hill coefficient dCas repression - Unit: Dimensionless
        mumax = 0.4284 #Maximal growth rate - Unit: 1/hr
        tc = 5 #Time at which exponential growth rate is reached - Unit: hr
        cap = 0.8 #Carrying capacity - Unit: OD
        scale = 166.727 #Scaling factor amount of GFP to fluorescence - Unit: A.U./mM

        k_b_p3 = 38.422
        k_b_p11 = 2.275
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

        #Pack the parameters
        self.parameters_standard = [k_b_dcas,k_op_succ, k_b_p3, k_b_sgrna,k_d_ascascomplex,k_d_ascomplex,k_d_asrna,k_d_cascomplex,k_d_dcas,k_d_dcasmrna,k_d_gfp,k_d_gfpmrna,k_d_lldr,k_d_lldrcomplex,k_d_lldrmrna,k_d_sgrna,k_f_ascascomplex1_fw,k_f_ascascomplex1_rv,k_f_ascascomplex2_fw,k_f_ascascomplex2_rv,k_f_ascomplex_fw,k_f_ascomplex_rv,k_f_cascomplex_fw,k_f_cascomplex_rv,k_f_lldrcomplex,alpaga_basal,alpaga_act1,alpaga_act2,alpaga_k_act1,alpaga_k_act2,alpaga_k3,k_p_gfpmrna,k_pt_dcas,k_pt_gfp,k_pt_lldr,kd_dcas,n_cascomplex,copy_nr,mumax,tc,cap,sigma,gamma,scale]

        self.params = self.parameters_standard
        
        self.param_labels = ['k_b_dcas','k_op_succ', 'k_b_p3', 'k_b_sgrna','k_d_ascascomplex','k_d_ascomplex','k_d_asrna','k_d_cascomplex','k_d_dcas','k_d_dcasmrna','k_d_gfp','k_d_gfpmrna','k_d_lldr','k_d_lldrcomplex','k_d_lldrmrna','k_d_sgrna','k_f_ascascomplex1_fw','k_f_ascascomplex1_rv','k_f_ascascomplex2_fw','k_f_ascascomplex2_rv','k_f_ascomplex_fw','k_f_ascomplex_rv','k_f_cascomplex_fw','k_f_cascomplex_rv','k_f_lldrcomplex','alpaga_basal','alpaga_act1','alpaga_act2','alpaga_k_act1','alpaga_k_act2','alpaga_k3','k_p_gfpmrna','k_pt_dcas','k_pt_gfp','k_pt_lldr','kd_dcas','n_cascomplex','copy_nr','mumax','tc','cap','sigma','gamma','scale']
        self.plot_param_labels = ['$k\_b_{dcas}$','$k\_op_{succ}$', '$k\_b_{p3}$', '$k\_b_{sgrna}$','$k\_d_{ascascomplex}$','$k\_d_{ascomplex}$','$k\_d_{asrna}$','$k\_d_{cascomplex}$','$k\_d_{dcas}$','$k\_d_{dcasmrna}$','$k\_d_{gfp}$','$k\_d_{gfpmrna}$','$k\_d_{lldr}$','$k\_d_{lldrcomplex}$','$k\_d_{lldrmrna}$','$k\_d_{sgrna}$','$k\_f_{3_{fw}}$','$k\_f_{3_{rv}}$','$k\_f_{4_{fw}}$','$k\_f_{4_{rv}}$','$k\_f_{2_{fw}}$','$k\_f_{2_{rv}}$','$k\_f_{1_{fw}}$','$k\_f_{1_{rv}}$','$k\_f_{lldrcomplex}$','$k\_b_{alp}$','$alp_{act1}$','$alp_{act2}$','$alp\_k_{act1}$','$alp\_k_{act2}$','$alp_{k3}$','$k\_p_{gfpmrna}$','$k\_pt_{dcas}$','$k\_pt_{gfp}$','$k\_pt_{lldr}$','$kd_{dcas}$','$n_{cascomplex}$','$copy\_nr$','$\mu_{max}$','$t_c$','$cap$','$\sigma$','$\gamma$','$scale$']
        
        # for i in range(len(self.params)):
        #     print(f"index {i}: {self.param_labels[i]}, value={round(self.params[i],3)}")
        
        #Initial state variables
        dCas_mRNA = 0
        dCas = 0
        sgRNA = 0
        CasComplex = 0
        asRNA = 0
        asComplex = 0
        asCasComplex = 0
        Lactate = 0
        Lldr_mRna = 0
        lldr = 0
        lldr_complex = 0
        Cx = 0.05
        GFP_mRNA = 0 #k_p_gfpmrna/k_d_gfpmrna
        GFP = 0 #(k_pt_gfp*k_p_gfpmrna)/(k_d_gfpmrna*k_d_gfp)

        #print(f"GFP_mRNA: {GFP_mRNA}")
        #print(f"GFP: {GFP}")

        #Pack the initial conditions
        self.initial_conditions = [dCas_mRNA, dCas, sgRNA, CasComplex, asRNA, asComplex, asCasComplex, Lactate, Lldr_mRna, lldr, lldr_complex, GFP_mRNA, GFP, Cx]
        self.init = self.initial_conditions

        #'Standard' output labels
        self.header = 'Lactate responsive CRISPRi-asRNA model'
        self.x_label = 'Time in hours'
        self.y_label = 'Concentration in mM'
        self.labels = ['dCas_mRNA', 'dCas', 'sgRNA', 'casComplex', 'asRNA', 'asComplex', 'asCasComplex', 'Lactate', 'Lldr_mRna', 'lldr', 'lldr_complex', 'GFP_mRna', 'GFP', 'Biomass']

        #Load data to be fitted
        self.FLU_data = pd.read_csv("Data/15mMSuccinate_aerobic_FLU_processed.csv", header=0)#Data from paper: pd.read_excel('Data/Lactate_data.xlsx', header=0, engine='openpyxl')
        self.OD_data = pd.read_csv("Data/15mMSuccinate_aerobic_OD_processed.csv", header=0)
        self.FLUOD_data = pd.read_csv("Data/15mMSuccinate_aerobic_FLUOD_processed.csv", header=0)

        #Load error (st.dev) for data to be fitted
        self.FLU_error = pd.read_csv("Data/15mMSuccinate_aerobic_FLU_std_processed.csv", header=0)
        self.OD_error = pd.read_csv("Data/15mMSuccinate_aerobic_OD_std_processed.csv", header=0)
        self.FLUOD_error = pd.read_csv("Data/15mMSuccinate_aerobic_FLUOD_std_processed.csv", header=0)
        
        #self.concentrations_data = [0.1, 0.5, 1.0, 5, 10, 50]
        self.concentrations_data = np.array([0,0.05,0.2,0.5,5,2,20])
        self.concentrations = self.concentrations_data #np.logspace(-3, 1.69897000434, num=50)

        #Process data to be fitted
        self.time = self.FLU_data.iloc[:,0]

        self.FLU_data = self.FLU_data.iloc[:,1:]
        self.OD_data = self.OD_data.iloc[:,1:]
        self.FLUOD_data = self.FLUOD_data.iloc[:,1:]

        self.FLU_error = self.FLU_error.iloc[:,1:]
        self.OD_error = self.OD_error.iloc[:,1:]
        self.FLUOD_error = self.FLUOD_error.iloc[:,1:]

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

    def deriv(self, y, t, p):
        """
        Function which calculates the derivatives and returns them.
        Every ODEModel needs a deriv function - The 'heart' of the model.
        """
        #Unpack the state vector
        dCas_mRNA, dCas, sgRNA, CasComplex, asRNA, asComplex, asCasComplex, Lactate, Lldr_mRna, lldr, lldr_complex, GFP_mRNA, GFP, Cx= y
        #Unpack the parameter vector
        k_b_dcas,k_op_succ, k_b_p3, k_b_sgrna,k_d_ascascomplex,k_d_ascomplex,k_d_asrna,k_d_cascomplex,k_d_dcas,k_d_dcasmrna,k_d_gfp,k_d_gfpmrna,k_d_lldr,k_d_lldrcomplex,k_d_lldrmrna,k_d_sgrna,k_f_ascascomplex1_fw,k_f_ascascomplex1_rv,k_f_ascascomplex2_fw,k_f_ascascomplex2_rv,k_f_ascomplex_fw,k_f_ascomplex_rv,k_f_cascomplex_fw,k_f_cascomplex_rv,k_f_lldrcomplex,alpaga_basal,alpaga_act1,alpaga_act2,alpaga_k_act1,alpaga_k_act2,alpaga_k3,k_p_gfpmrna,k_pt_dcas,k_pt_gfp,k_pt_lldr,kd_dcas,n_cascomplex,copy_nr,mumax,tc,cap,sigma,gamma,scale = p
        
        #AlpaGA promoter
        top_a = alpaga_basal + (alpaga_act1 + alpaga_basal)*(alpaga_k_act1*lldr) + (alpaga_act2+ alpaga_basal)*(alpaga_k_act2*lldr_complex) #(epsilon_r/2)*rep_p*alpaga_basal*alpaga_k3*lldr**2
        bottom_a = gamma+ alpaga_k_act1*lldr + alpaga_k_act2*lldr_complex + alpaga_k3*(lldr**2)
        
        #Calculate growth rate
        mu = mumax*np.exp(-0.5*((t-tc)/sigma)**2)*(1-Cx/cap)
        
        #Calculate derivatives
        d_dCas_mRNA_dt  =  mu/mumax*(k_b_dcas*copy_nr) - dCas_mRNA*(k_d_dcasmrna+mu)
        d_dCas_dt       = mu*(k_pt_dcas*dCas_mRNA) + k_f_cascomplex_rv*CasComplex - k_f_cascomplex_fw*sgRNA*dCas + k_f_ascascomplex2_rv*asCasComplex - k_f_ascascomplex2_fw*dCas*asComplex - dCas*(k_d_dcas+mu)
        d_sgRNA_dt      = mu/mumax*(k_b_sgrna*copy_nr) + k_f_cascomplex_rv*CasComplex - k_f_cascomplex_fw*sgRNA*dCas - k_f_ascomplex_fw*asRNA*sgRNA + k_f_ascomplex_rv*asComplex- sgRNA*(k_d_sgrna+mu)
        d_CasComplex_dt = k_f_cascomplex_fw*sgRNA*dCas - k_f_cascomplex_rv*CasComplex - k_f_ascascomplex1_fw*asRNA*CasComplex  + k_f_ascascomplex1_rv*asCasComplex -CasComplex*(k_d_cascomplex+mu) 
        d_asRNA_dt      = mu/mumax*(copy_nr*top_a/bottom_a)+ k_f_ascomplex_rv*asComplex - k_f_ascomplex_fw*asRNA*sgRNA+ k_f_ascascomplex1_rv*asCasComplex - k_f_ascascomplex1_fw*asRNA*CasComplex - asRNA*(k_d_asrna+mu)
        d_asComplex_dt  = k_f_ascomplex_fw*asRNA*sgRNA - k_f_ascomplex_rv*asComplex + k_f_ascascomplex2_rv*asCasComplex - k_f_ascascomplex2_fw*dCas*asComplex - asComplex*(k_d_ascomplex+mu)
        d_asCasComplex_dt = k_f_ascascomplex1_fw*asRNA*CasComplex- k_f_ascascomplex1_rv*asCasComplex - k_f_ascascomplex2_rv*asCasComplex + k_f_ascascomplex2_fw*dCas*asComplex - asCasComplex*(k_d_ascascomplex+mu)
        d_Lactate_dt = 0
        d_Lldr_mRna_dt = mu/mumax*(k_op_succ + copy_nr*k_b_p3)   -Lldr_mRna*(k_d_lldrmrna+mu)
        d_lldr_dt = mu*(k_pt_lldr*Lldr_mRna)     -k_f_lldrcomplex*lldr*(Lactate**2) -lldr*(k_d_lldr+mu)
        d_lldr_complex_dt = k_f_lldrcomplex*lldr*(Lactate**2)- lldr_complex*(k_d_lldrcomplex+mu)
        d_GFP_mRNA_dt   = mu/mumax*(copy_nr*k_p_gfpmrna*(kd_dcas**n_cascomplex/(kd_dcas**n_cascomplex + CasComplex**n_cascomplex)))- GFP_mRNA*(k_d_gfpmrna+mu)
        d_GFP_dt        = mu*k_pt_gfp*GFP_mRNA-GFP*(k_d_gfp+mu)
        d_Cx_dt = Cx*mu
        #Pack the derivatives
        dydt = [d_dCas_mRNA_dt, d_dCas_dt, d_sgRNA_dt, d_CasComplex_dt, d_asRNA_dt, d_asComplex_dt, d_asCasComplex_dt, d_Lactate_dt, d_Lldr_mRna_dt, d_lldr_dt, d_lldr_complex_dt, d_GFP_mRNA_dt, d_GFP_dt, d_Cx_dt]
        return dydt

    def sim_induced(self, p, total_time, induce_time, induce_concentration):
        """
        Function which simulates the model for a given set of parameters and at given time adds Lactate.
        """
        timepoints, output = self.simulateODE(p, self.init, induce_time, induce_time*3600)
        #Change Lactate concentration.
        output[-1,7] = induce_concentration
        timepoints2, output2 = self.simulateODE(p, output[-1], total_time-induce_time, (total_time-induce_time)*3600)

        #Concatenate outputs
        sim_time = np.concatenate((timepoints,timepoints2[1:]+induce_time))
        sim_output = np.concatenate((output,output2[1:]))
        
        return sim_time, sim_output

    def simulate_experiment(self, p, init, concentration1, concentration2):
        """
        Function that will try to simulate the experiment that Bas will do in the lab.
        """
        #Round off hill coefficients to nonzero integers
        p[36] = 1 if p[36] < 0.5 else round(p[36])
        #p[37] = 1 if p[37] < 0.5 else round(p[37]) If n_alpaga was used.

        #Simulate overnight growth (Before plate reader)
        init[7] = concentration1
        sim_time, sim_output = self.simulateODE(p, init, 24, 24*3600)

        #Dilute the cell volume
        init_new = sim_output[-1]
        init_new[-1] = sim_output[0,-1]
   
        #Change lactate concentration
        init_new[7] = concentration2
        #Simulate the (Plate-reader) experiment 
        sim_time2, sim_output2 = self.simulateODE(p, init_new, 21, 24*3600)

        #Dont concatenate since we usually only want the second part of the experiment
        return sim_time, sim_output, sim_time2, sim_output2
   
    def calculate_ec(self, concentrations, response, ec_percent):
        """
        Used to calculate the desired EC value from a dose-response curve.
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
     
    # def oaat_analysis(self, _num = 9, folder='figures'):
    #     """
    #     This function will perform one at a time analysis of the model.
    #     Reports EC values and dose-response curves.
    #     Also make plots and bar charts of the results.
    #     """
    #     #Check if the folder exists
    #     if not os.path.exists(folder):
    #         os.makedirs(folder)

    #     #Check if _num is an uneven integer
    #     if _num%2 == 0 and _num<2:
    #         print('Number of perturbations must be an uneven integer!')
    #         raise ValueError

    #     #Empty array to store oaat parameter sets
    #     samplelist = np.empty((0,len(self.params)))

    #     #Sample 9 different perturbations of the parameters
    #     log = np.linspace(-1,1, num=_num)

    #     #Create a list of parameter sets with the perturbations
    #     #Last six parameters are left as is (wouldnt make sense to change them)
    #     for i, parameter in enumerate(self.params[:len(self.params)-6]):
    #         print(f"Sampling parameter {self.param_labels[i]}")
    #         sample = (10**log)*parameter
    #         for value in sample:
    #             params = self.params.copy()
    #             params[i] = value
    #             samplelist = np.row_stack((samplelist,params))

    #     #Log or linspace?
    #     concentrations = np.logspace(-1,1.7,50)
    #     #concentrations = np.linspace(0,40,50)

    #     #Make some lists/arrays to store info and results.
    #     oat_dose_curves = np.zeros((len(samplelist),len(concentrations)))
    #     sim_list = []
    #     calculated_ec = np.zeros((len(samplelist),3))
    #     ec_responses = np.zeros((len(samplelist),3))

    #     print("Running OAAT analysis...")
    #     #Simulate a dose-response curve for each parameter set
    #     for i, sample in tqdm(enumerate(samplelist), total=len(samplelist)):
    #         ec, ec_response, responses, simulations = self.evaluate_dose_response(sample, self.init, concentrations)
    #         #Add simulations to the list
    #         #sim_list.append(simulations)
    #         #Add EC values and EC responses to the list
    #         calculated_ec[i] = ec
    #         ec_responses[i] = ec_response
    #         #Add responses to the list
    #         oat_dose_curves[i] = np.array(responses)

    #     #Lists to store max and min EC5 values - Used for the barplot.
    #     min_ec = []
    #     max_ec = []
    #     original_ec = []

    #     factor_10_increase = []
    #     factor_10_decrease = []
    #     #Go through the results for each parameter, make a figure of the dose-response curve and save it.
    #     #Also calculate the max and min EC5 values for each parameter.
    #     print("Processing results for each parameter...")
    #     for i in tqdm(range(int(len(oat_dose_curves)/_num))):
    #         #Calculate max, min and add original.
    #         #Convert to float so it doesnt end up as a list of np.arrays.
    #         #min_ec.append(float(min(calculated_ec[i*_num:i*_num+_num,0])))
    #         #max_ec.append(float(max(calculated_ec[i*_num:i*_num+_num,0])))
    #         original_ec.append(float(calculated_ec[i*_num+int(_num/2),0]))

    #         #Calculate factor 10 increase and decrease
    #         increase_difference = calculated_ec[i*_num+_num-1,0]-original_ec[i]
    #         decrease_difference = calculated_ec[i*_num,0]-original_ec[i]
    #         factor_10_increase.append(float(increase_difference))
    #         factor_10_decrease.append(float(decrease_difference))

    #         fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
    #         for j in range(_num):
    #             plt.plot(calculated_ec[i*_num+j], ec_responses[i*_num+j], 'or')
    #             if 10**log[j] != 1:
    #                 plt.plot(concentrations, oat_dose_curves[i*_num+j], label=f'factor {np.round(10**log[j],3)}')
    #             else:
    #                 plt.plot(concentrations, oat_dose_curves[i*_num+j], label=f'Original')
    #         #plt.title(f"Oaat analysis of factor 0.1-10 change in parameter {self.param_labels[i]}")
    #         plt.ylabel('Response')
    #         plt.xlabel('Lactate concentration (mM)')
    #         plt.xscale('log')
    #         plt.axvspan(1.5,3, color=colors['Green'], alpha=0.3, label='Healthy', 1)
    #         plt.axvspan(10,25, color=colors['Red'], alpha=0.3, label='Cancer', 1)
    #         plt.axvline(1.5,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2, 1)
    #         plt.axvline(3,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2, 1)
    #         plt.axvline(10,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2, 1)
    #         plt.axvline(25,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2, 1)
    #         plt.legend()
    #         plt.savefig(f"{folder}/{self.param_labels[i]}_oaat_result.svg")
    #         plt.close()

    #     #Make a normal and polar plot of the EC values.
    #     labels = self.param_labels[:len(self.params)-6]
    #     #Omit alpaga_act1 because it is interpolated incorrectly.
    #     labels = labels[:26]+labels[27:]
    #     factor_10_decrease = factor_10_decrease[:26]+factor_10_decrease[27:]
    #     factor_10_increase = factor_10_increase[:26]+factor_10_increase[27:]

    #     #First the normal plot
    #     fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
    #     ax.bar(labels, factor_10_increase, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
    #     #ax.bar(labels, original_ec, width=1, color="#45c4af", edgecolor='black', label="EC5 - Original", bottom=0)
    #     ax.bar(labels, factor_10_decrease, width=1, color="#ec6c5f",edgecolor='black', label="Factor 10 decrease", bottom=0)
    #     plt.xticks(rotation=90)
    #     plt.legend(loc='upper center')
    #     plt.ylabel('Lactate concentration(mM)')
    #     ##plt.title('Effect of single parameter changes on EC5')
    #     plt.savefig(f"{folder}/oaat_barchart.svg", dpi= 300)
    #     plt.close()

    #     #Then the polar plot
    #     ANGLES = []
    #     XTICK_angles = []
    #     N = len(labels)

    #     for i in range(N):
    #         ANGLES.append(np.radians(360/N*i))
    #         XTICK_angles.append(np.radians(360/N*i)+np.pi/N)
            
    #     WIDTH = (2*np.pi)/N
    #     OFFSET = np.pi / 2

    #     fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True, subplot_kw={"projection": "polar"})

    #     ax.bar(ANGLES, factor_10_increase, width=WIDTH, color="#4d8b31", edgecolor='black', label="EC5 - Max", bottom=0)
    #     #ax.bar(ANGLES, original_ec, width=WIDTH, color="#45c4af", edgecolor='black', label="EC5 - Original", bottom=0)
    #     ax.bar(ANGLES, factor_10_decrease, width=WIDTH, color="#ec6c5f",edgecolor='black', label="EC5 - Min", bottom=0)

    #     ax.set_xticks(XTICK_angles)
    #     ax.set_xticklabels([])
    #     ax.set_theta_offset(OFFSET)
    #     ax.set_rlim(0,max(factor_10_decrease + factor_10_increase))
    #     ax.set_rorigin(-1)
    #     add_labels(ANGLES, max(factor_10_decrease + factor_10_increase), labels, OFFSET, ax, 10)
    #     plt.legend(loc='center')
    #     plt.savefig(f"{folder}/oaat_polar_barchart.svg", dpi= 300)
    #     plt.close()

    #     #Make a barplot of the factor 10 increase and decrease.
    #     fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
    #     ax.bar(labels, factor_10_increase, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
    #     ax.bar(labels, factor_10_decrease, width=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", bottom=0)
    #     plt.xticks(rotation=90)
    #     plt.legend(loc='upper center')
    #     plt.ylabel('Change in EC5 (mM)')
    #     ##plt.title('Effect of single parameter changes on EC5')
    #     plt.savefig(f"{folder}/oaat_barchart_factor_10.svg", dpi= 300)
    #     plt.close()

    def oaat_analysis_mp(self, _num = 9, folder='figures', plot_data = False):
        """
        This function will perform one at a time analysis of the model.
        Reports EC values and dose-response curves.
        Also make plots and bar charts of the results.
        Uses multiprocessing to speed up the process.
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
            model = Bassystem()
            
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

            fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
            for j in range(_num):
                if 10**log[j] != 1:
                    plt.plot(concentrations, oat_dose_curves[i*_num+j], label=f'Factor {np.round(10**log[j],2)}', color=color_list[j])
                else:
                    plt.plot(concentrations, oat_dose_curves[i*_num+j], label=f'Original', color=color_list[j])
                plt.plot(calculated_ec[i*_num+j], ec_responses[i*_num+j], 'or')
            if plot_data:
                plt.errorbar(self.concentrations, self.dose_response, yerr=self.dose_response_error, label='Data', marker='o')
            ##plt.title(f"Oaat analysis of factor 0.1-10 change in parameter {self.param_labels[i]}")
            plt.ylabel('Fluorescence/OD600 (a.u.)')
            plt.xlabel('L-lactate concentration (mM)')
            plt.xscale('log')
            plt.axvspan(1.5,3, color=colors['Green'], alpha=0.3, label='Healthy')
            plt.axvspan(10,25, color=colors['Red'], alpha=0.3, label='Cancer')
            plt.axvline(1.5,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2)
            plt.axvline(3,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2)
            plt.axvline(10,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2)
            plt.axvline(25,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2)
            plt.legend(fontsize=11)
            plt.ylim(ymin=0)
            plt.savefig(f"{folder}/{self.param_labels[i]}_oaat_result.svg")
            plt.close()

        #Make a normal and polar plot of the EC values.
        labels = self.plot_param_labels[:len(self.params)-1]
        #Remove alpaga_act1 and act2 from labels and factor_10_increase and factor_10_decrease
        labels = labels[:26]+labels[28:]

        for i, label in enumerate(labels):
            print(f"Index {i} is {label}")

        factor_10_increase = factor_10_increase[:26]+factor_10_increase[28:]
        factor_10_decrease = factor_10_decrease[:26]+factor_10_decrease[28:]

        factor_10_increase_background = factor_10_increase_background[:26]+factor_10_increase_background[28:]
        factor_10_decrease_background = factor_10_decrease_background[:26]+factor_10_decrease_background[28:]

        #First the normal plot
        fig, ax = plt.subplots(figsize=(9, 5.0625), tight_layout = True)
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

        fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True, subplot_kw={"projection": "polar"})

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
        #plt.legend(loc='upper center')
        plt.xlabel('Change in EC5 concentration (mM)')
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_horizontal_barchart.svg", dpi= 300)

        #Make a barplot of the factor 10 increase and decrease.
        fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
        ax.bar(labels, factor_10_increase, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
        ax.bar(labels, factor_10_decrease, width=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", bottom=0)
        plt.xticks(rotation=90)
        plt.legend(loc='upper center')
        plt.ylabel('Change in EC5 (mM)')
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_barchart_factor_10.svg", dpi= 300)
        plt.close()

        #Make a bar chart of the background values
        fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
        x = np.arange(len(labels))
        x = [2*i for i in x]

        ax.bar(x, factor_10_increase_background, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
        ax.bar(x, factor_10_decrease_background, width=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", bottom=0)

        plt.xticks(x, labels, rotation=90)
        plt.legend(loc='upper center')
        plt.ylabel('Change in background fluorescence (a.u.)')
        plt.ylim(np.min(np.array([factor_10_increase_background, factor_10_decrease_background])*1.1),500)
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_barchart_factor_10_background.svg", dpi= 300)
        plt.close()

        #Make a horizontal bar chart of the factor 10 background increase and decrease.
        fig, ax = plt.subplots(figsize=(12, 12), tight_layout = True)
        ax.barh(labels, factor_10_increase_background, height=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", left=0)
        ax.barh(labels, factor_10_decrease_background, height=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", left=0)
        plt.legend(loc='upper center')
        plt.xlabel('Change in background fluorescence (a.u.)')
        plt.xlim(np.min(np.array([factor_10_increase_background, factor_10_decrease_background])*1.1),500)
        ##plt.title('Effect of single parameter changes on EC5')
        plt.savefig(f"{folder}/oaat_horizontal_barchart_factor_10_background.svg", dpi= 300)
        plt.close()

        #Figure that only shows the EC5 values for asRNA parameters
        asRNA_labels = labels[0:1] + labels[3:10] + labels[15:24] + labels[30:31] + labels[33:35]
        asRNA_factor_10_increase = factor_10_increase[0:1] + factor_10_increase[3:10] + factor_10_increase[15:24] + factor_10_increase[30:31] + factor_10_increase[33:35]
        asRNA_factor_10_decrease = factor_10_decrease[0:1] + factor_10_decrease[3:10] + factor_10_decrease[15:24] + factor_10_decrease[30:31] + factor_10_decrease[33:35]

        asRNA_factor_10_decrease_background = factor_10_decrease_background[0:1] + factor_10_decrease_background[3:10] + factor_10_decrease_background[15:24] + factor_10_decrease_background[30:31] + factor_10_decrease_background[33:35]
        asRNA_factor_10_increase_background = factor_10_increase_background[0:1] + factor_10_increase_background[3:10] + factor_10_increase_background[15:24] + factor_10_increase_background[30:31] + factor_10_increase_background[33:35]

        #EC5 changes
        fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
        ax.bar(asRNA_labels, asRNA_factor_10_increase, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
        ax.bar(asRNA_labels, asRNA_factor_10_decrease, width=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", bottom=0)
        plt.xticks(rotation=90)
        plt.legend(loc='upper center')
        plt.ylabel('Change in EC5 (mM)')
        plt.savefig(f"{folder}/oaat_barchart_factor_10_asRNA.svg", dpi= 300)
        plt.close()

        #Background changes
        fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
        ax.bar(asRNA_labels, asRNA_factor_10_increase_background, width=1, color="#4d8b31", edgecolor='black', label="Factor 10 increase", bottom=0)
        ax.bar(asRNA_labels, asRNA_factor_10_decrease_background, width=1, color="#45c4af", edgecolor='black', label="Factor 10 decrease", bottom=0)
        plt.xticks(rotation=90)
        plt.legend(loc='upper center')
        plt.ylabel('Change in background fluorescence (a.u.)')
        plt.ylim(np.min(np.array([asRNA_factor_10_increase_background, asRNA_factor_10_decrease_background])*1.1),500)
        plt.savefig(f"{folder}/oaat_barchart_factor_10_background_asRNA.svg", dpi= 300)
        plt.close()

        #Calculate original characteristics
        ec1, ec_response1, responses1, _simulations1 = self.evaluate_dose_response(self.params, concentrations)
        org_fold_change = np.round((max(responses1)-min(responses1))/min(responses1),4)
        org_background = min(responses1)/max(responses1)
        #Save results to a csv file
        with open(f"{folder}/oaat_results.csv", "w") as f:
            print("Writing results to csv file...")
            f.write("Parameter, Perturbation, EC5, EC50, EC95, Fold change, Background fluorescence \n")
            f.write(f"Original, 0, {np.round(ec1[0],2)}, {np.round(ec1[1],2)}, {np.round(ec1[2],2)}, {np.round(org_fold_change,2)}, {np.round(org_background,2)} \n")
            for i in tqdm(range(int(len(oat_dose_curves)/_num))):
                for j in range(_num):
                    if 10**log[j] != 1:
                        f.write(f"{self.param_labels[i]}, {np.round(10**log[j],2)}, {np.round(calculated_ec[i*_num+j][0],2)}, {np.round(calculated_ec[i*_num+j][1],2)}, {np.round(calculated_ec[i*_num+j][2],2)}, {np.round((max(oat_dose_curves[i*_num+j])-min(oat_dose_curves[i*_num+j]))/min(oat_dose_curves[i*_num+j]),2)}, {np.round(min(oat_dose_curves[i*_num+j])/max(oat_dose_curves[i*_num+j]),2)} \n")
                    else:
                        pass

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
        """
        #Check if the folder exists, if not make it
        if not os.path.exists(folder):
            os.makedirs(folder)

        #Select X(now 1000) of top scores


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
            fig, ax = plt.subplots(figsize=(14,9))
            plt.hist(scores, bins=100)
            #plt.title(f"Distribution of scores in the top {len(selection)} scores out of {len(sets)}.")
            plt.axvline(x=median,ymin = 0, ymax=1, label=f"Median: {median}", color="red", linestyle="--", linewidth=2)
            plt.xlabel("Score value")
            plt.ylabel("Frequency")
            plt.legend()
            plt.savefig(f"{folder}/score_dist_{len(selection)}_{len(sets)}.svg", dpi= 300)
            plt.close()
        
        selection = sets[:1000]
        scores = scores[:1000]

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
                fig, ax = plt.subplots(figsize=(14,9))
                plt.hist(values, bins=100)
                plt.axvline(x=np.median(values),ymin =0, ymax=1, label=f"Median: {median}", color="red", linestyle="--", linewidth=2)
                #plt.title(f"Distribution of {self.param_labels[i]} in the top {len(selection)} scores out of {len(sets)}.")
                plt.xlabel("Parameter value")
                plt.ylabel("Frequency")
                plt.legend()
                plt.savefig(f"{folder}/{self.param_labels[i]}_{len(selection)}_{len(sets)}.svg", dpi= 300)
                plt.close()
            
            #Simulate the median parameter set for concentrations.
            FLUOD_outputs, simtime = self.simulate_concentrations(median_params, self.concentrations)
            fig, ax = plt.subplots(figsize=(14,9))
            plt.errorbar(self.concentrations_data, self.dose_response, self.dose_response_error, label="Data", color=colors['Red'])
            plt.plot(self.concentrations, FLUOD_outputs[-1,:], label="Simulated fit", color=colors['Green'])
            plt.xlabel("Lactate concentration (mM)")
            plt.xscale("log") 
            plt.ylabel("Fluorescence/OD600 (a.u.)")
            #plt.title("Comparing simulation with median params to data")
            plt.legend()
            plt.savefig(f"{folder}/median_parameters_{len(selection)}_{len(sets)}.svg", dpi= 300)
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
            fig, ax = plt.subplots(figsize=(14,9))
            plt.errorbar(self.concentrations_data, self.dose_response, self.dose_response_error, label="Data", color=colors['Red'])
            plt.errorbar(self.concentrations, outputs[-1], errors, label="Simulation", color=colors['Green'])
            plt.xlabel("Lactate concentration (mM)")
            plt.xscale("log") 
            plt.ylabel("Fluorescence/OD600 (a.u.)")
            #plt.title("100 best simulations params vs data - Dose response curve")
            plt.legend()
            plt.savefig(f"{folder}/100_averaged_simulations_{len(selection)}_{len(sets)}.svg", dpi= 300)
            plt.close()
        else:
            best_params = selection
        print(f"The best parameters were: {best_params} and gave a score of: {scores[best_score]}")
            
        #Compare simulation vs data for the best parameter set
        FLUOD_outputs, simtime = self.simulate_concentrations(best_params, self.concentrations)
        #Plot the simulation
        fig, ax = plt.subplots(figsize=(14,9))
        plt.errorbar(self.concentrations_data, self.dose_response, self.dose_response_error, label="Data", color=colors['Red'], capsize=5)
        plt.plot(self.concentrations_data, FLUOD_outputs[-1,:], label="Simulation", color=colors['Green'])
        #plt.title("Dose-response comparison of our model to data")
        plt.xlabel("Lactate concentration (mM)")
        plt.xscale("log")
        plt.ylabel("Fluorescence/OD600 (a.u.)")
        plt.legend()
        plt.savefig(f"{folder}/simulation_vs_data_{len(selection)}_{len(sets)}.svg", dpi= 300)
        plt.close()


        #For the best simulation, make a time-series plot for each concentration
        for i in range(len(self.concentrations)):
            sim_time, sim_output, sim_time2, sim_output2 = self.simulate_experiment(best_params, self.init, 0, self.concentrations[i])
            fig, ax = plt.subplots(figsize=(14,9))
            for j in range(len(sim_output2[0])):
                #if j > 0:
                plt.plot(sim_time2, sim_output2[:,j], label=self.labels[j], marker=self.markers[j], markevery=3600)
            plt.legend(loc="upper left")
            plt.xlabel("Time (h)")
            plt.ylabel("Concentration (mM)")
            #plt.title(f"Time series for lactate concentration {self.concentrations[i]} mM")
            plt.savefig(f"{folder}/time_series_{self.concentrations[i]}mM_{len(selection)}_{len(sets)}.svg", dpi= 300)
            plt.close()

        #Also make a series for the OD
        sim_time, sim_output, sim_time2, sim_output2 = self.simulate_experiment(best_params, self.init, 0, 0)
        fig, ax = plt.subplots(figsize=(14,9))
        plt.plot(sim_time2, sim_output2[:,-1], label="OD", marker="o", markevery=3600, color=colors['Green'])
        plt.errorbar(self.time, self.OD_data_p3[:,5], self.OD_error_p3[:,5], label="Data", color=colors['Red'])
        plt.legend(loc="upper left")
        plt.xlabel("Time (h)")
        plt.ylabel("Optical density at 600nm")
        #plt.title("Time series for OD")
        plt.savefig(f"{folder}/time_series_OD_{len(selection)}_{len(sets)}.svg", dpi= 300)
        plt.close()

        #Make a figure for the "overnight" simulation
        fig, ax = plt.subplots(figsize=(14,9))
        for j in range(len(sim_output[0])):
            #if j > 0:
            plt.plot(sim_time, sim_output[:,j], label=self.labels[j], marker=self.markers[j], markevery=3600)
        plt.legend(loc="upper left")
        plt.xlabel("Time (h)")
        plt.ylabel("Concentration (mM)")
        #plt.title(f"Time series for overnight growth (0mM Lactate)")
        plt.savefig(f"{folder}/time_series_overnight_{len(selection)}_{len(sets)}.svg", dpi= 300)
        plt.close()

        #Also make a series for the GFP
        for i in range(len(self.concentrations)):
            sim_time, sim_output, sim_time2, sim_output2 = self.simulate_experiment(best_params, self.init, 0, self.concentrations[i])
            fig, ax = plt.subplots(figsize=(14,9))
            plt.plot(sim_time2, sim_output2[:,-2]*best_params[-1], label="Simulated", marker="o", markevery=3600, color=colors['Green'])#/sim_output2[:,-1]*best_params[-1]
            plt.errorbar(self.time, self.FLUOD_data_p3[:,i], self.FLUOD_error_p3[:,i], label="Data", color=colors['Red'])
            plt.legend(loc="upper left")
            plt.xlabel("Time (h)")
            plt.ylabel("Fluorescence/OD600 (a.u.)")
            #plt.title(f"Time series for GFP at {self.concentrations[i]} mM Lactate")
            plt.savefig(f"{folder}/time_series_GFP_{self.concentrations[i]}mM_{len(selection)}_{len(sets)}.svg", dpi= 300)
            plt.close()
        
        #Write the function that we used for our model.
        import inspect
        with open(f"{folder}/deriv.txt", "w") as f:
            f.write("This model was used for these results:\n")
            f.write("______________________________________\n")
            f.write(inspect.getsource(self.deriv))



if __name__ == '__main__':
    #Create an instance of the model class
    model = Bassystem()

    for i, param in enumerate(model.params):
        print(f"Index {i}: {model.param_labels[i]}: {param}")

    sim_time, sim_output, sim_time2, sim_output2 = model.simulate_experiment(model.params, model.init, 0, 10)
    #Plot the results
    #model.plot(sim_time2, sim_output2, model.labels)

    # #Plot each of the components
    # for i in range(len(model.labels)):
    #     fig, ax = plt.subplots()
    #     plt.plot(sim_time2, sim_output2[:,i])
    #     #plt.title(model.labels[i])
    #     plt.xlabel('Time (hours)')
    #     plt.ylabel(model.labels[i])
    #     plt.show()
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 16

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=12)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)
    # Simulate a dose-response curve
    concentrations = np.logspace(-1, 1.7, 50)
    ec, ec_response, responses, simulations = model.evaluate_dose_response(model.params, concentrations)
    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)

    plt.plot(concentrations, np.array(responses), label="Simulated", color=colors['Green'])
    plt.plot(ec, np.array(ec_response), 'o', label="EC5,50,95", color=colors['Red'])
    plt.axvspan(1.5,3, color=colors['Green'], alpha=0.3, label='Healthy')
    plt.axvspan(10,25, color=colors['Red'], alpha=0.3, label='Cancer')
    plt.axvline(1.5,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2)
    plt.axvline(3,ymin =0, ymax=1, color=colors['Green'], linestyle=dashline, linewidth=2)
    plt.axvline(10,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2)
    plt.axvline(25,ymin =0, ymax=1, color=colors['Red'], linestyle=dashline, linewidth=2)
    #plt.title("Predicted dose-response curve for the full system")
    plt.xscale("log")
    plt.xlabel("Lactate concentration (mM)")
    plt.ylabel("Fluorescence/OD600 (a.u.)")
    plt.legend()
    plt.savefig(f"figures/dose_response_curve_dilutioncorrected.svg", dpi= 300)

    #Create a scaled fluorescence plot of the overnight growth
    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
    plt.plot(sim_time, sim_output[:,-2]*model.params[-1], label="Simulated", color=colors['Green'])
    plt.xlabel("Time (h)")
    plt.ylabel("Fluorescence/OD600 (a.u.)")
    #plt.title("Overnight growth")
    plt.legend()
    plt.savefig(f"figures/overnight_fluorescence_dilutioncorrected.svg", dpi= 300)
    plt.close()

    #Create a scaled fluorescence plot
    fig, ax = plt.subplots(figsize=(8, 4.5), tight_layout = True)
    plt.plot(sim_time2, sim_output2[:,-2]*model.params[-1], label = "Simulated fluorescence", color='orange')
    plt.xlabel('Time (hours)')
    plt.ylabel('Fluorescence (a.u.)')
    #plt.title('Simulated fluorescence with combined models - 1mM induction')
    plt.legend()
    plt.savefig('figures/scaled_fluorescence_dilutioncorrected.svg', dpi = 200)
    plt.close()
    
    model.oaat_analysis_mp(folder="figures/oaat_analysis_fullsystem_dilutioncorrected")
