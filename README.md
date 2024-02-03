# Model driven analysis and design of a lactate sensing genetic circuit
## Optimization of a biosensor to differentiate between healthy and colon cancer environments
#### Project of Josia Pool for iGEM 2022 - supervised by Robert Smith.

## Requirements
Python 3.6.9 to 3.10.6 (3.11 may work as well but will need updated packages.)

## Setting up a virtual environment
I ended up using Venv, not anaconda for running python environments separate from the main installation. A virtual environment ensures you can use all dependencies and package versions that you need without creating conflicts with any other projects you might be working on. I would highly advise you to do the same.

Instructions on how to set up an environment can be found in "Using virtual environments.docx"

## Running models and optimizing
Models are located in the models folder.
- ODEModelv2: Contains a general ODE model class with functions used in every other model that builds upon this. The code for the CRISPRi/asRNA model is also in here.
- Lactatemodel_succinate_p3: The model that was used to fit to the lactate biosensor data from the lab
- Bassystem_final: The combined model that was used to perform one-at-a-time sensitivity analysis
- Bassystem_GFP_pALPAGA: The combined model with the promoter for GFP switched from a constitutive one to a lactate dependent promoter, creating a feed forward loop motif.
- Bassystem_GFP_pALPAGA_sRNA: The combined model with a lactate dependent sRNA, decreasing leaky expression in low lactate concentrations.

In order to sample parameter sets used for fitting the model, check out this example:
```python
#Add the models folder to path
import sys
sys.path.append("models/")

#Import the model
from Lactatemodel_succinate_p3 import Lactate_model_succinate_p3

#Initialize the model
model = Lactate_model_succinate_p3()

#Sample 1 million parameter samples
model.sample(1000000)

#Save samples
model.save_samples(1000000, "samplename") 
#The sample name can be anything to identify the samples by.
#Samples should be saved to a /samples folder (which should be created on initializing the model if its not available)
#The samples are split into 10 seperate files.
```

### Simulation of the samples and optimization
One script is used for simulation of the samples: "optim_wrapper.py"
This script is called using a command line terminal, an example is shown below:
```
python3 optim_wrapper.py --model p3model --method samplename --file 1
```
Arguments are as follows:

- --model: The model you would like to optimize, options are 'dCas9' for the CRISPRi model and 'p3model' for the lactate biosensor model.
- --method: The sample name which is the same when using save_samples.
- --file: The file number, from 1 to 10.

It will take a while to simulate all of these, so it is advised to leave the 10 sample lists running parallel on a server.

Once all samples are simulated, the sample lists are moved to samples/simulated and the files will now contain the scores of the simulated parameter sets.
In order to process these simulated samples and continue with the minimization step, we use "process_optim_results.py".
This script is also called using a command line terminal:
```
python3 process_optim_results.py --model p3model --method samplename --file 1000000 --cores 8
```
Arguments are as follows:
- --model: The model you would like to optimize, options are 'dCas9' for the CRISPRi model and 'p3model' for the lactate biosensor model.
- --method: The sample name which is the same when using save_samples.
- --file: The number of samples in total, as passed in save_samples.
- --cores: This script uses multiprocessing in order to speed up optimization, in order to take advantage of the ssb server. (This argument is optional, if this isnt used it will use 8 cores.)

This script will first report the best score and create some figures on the simulated latin hypercube samples, saved in lhs_results/.
Afterwards the best 1000 parameter sets are optimized by L-BFGS-B, and the results of this are reported again. These results are saved in minimization_results/.
My results can also be seen in these folders.

## Description of jupyter notebook files
These are short descriptions for the jupyter notebook files in this repository.
More details are given in the files themselves.
- BasDataExploration: This file processes all raw data from Bas and create plots to visualize the data.
- GrowthFit: This file contains the code for the preliminary fitting of the growth equation before fitting the entire lactate model
- Model_timedelay: This model contains the code used to evaluate if a time delay improves the fit of the CRISPRi/asRNA model
- BackgroundComparison: This model contains the code used to create Figure 18 and Table 1 of the thesis, comparing system improvements for the reduction of background expression.
- Test_alpaga: This code contains some plots which compare the Lactate model to the data. It also contains a plot of the 'desired' dose-response curve.

## Final remarks
This documentation should give enough information to get started with my code, but if you have any questions you can always reach out to me