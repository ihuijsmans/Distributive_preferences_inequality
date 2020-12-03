import sys, scipy, os

print(sys.path)
sys.path.append("/opt/anaconda3/5.0.0/lib/python3.6/site-packages")
print(sys.path)
print(sys.version)

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import scipy.stats 

from costFunctions import two_norms_costfun
from inequalityaversion import two_norms_model

if sys.platform == 'win32':
    base_dir = 'P:/3014018.13/experiment_3_DG_UG'
elif sys.platform == 'linux':
    base_dir = '/project/3014018.13/experiment_3_DG_UG'
else:
    base_dir ='/Users/rhmhuijsmans/Inge/OneDrive - Harvard/experiment_3_DG_UG'
    

# Arguments
#iterrun = int(sys.argv[2])
#niter = int(sys.argv[2])
#fakepps = int(sys.argv[1])
iterrun = 1
fakepps = 1
niter = 10000

print(iterrun, fakepps)
    
results = pd.DataFrame(columns=['iterrun','fakepps','trueTheta','truePhi','recovTheta','recovPhi','cost','r'])
ntrials = 3

upBoundPhi = 0.5


#Set trueTheta truePhi
trueTheta, truePhi = np.array([scipy.random.uniform(),scipy.random.uniform()*upBoundPhi])

# Make trial set
BP1 = [0.19,0.75,3]
BP2 = [0.19,0.75,3]

# We are estimating based on ntrials
BP1BP2 = np.repeat(BP1,len(BP2*ntrials))
BP1BP2 = pd.DataFrame(BP1BP2)
BP1BP2.columns = ['BP1']
BP1BP2['BP2'] = np.tile(BP1,len(BP2)*ntrials)

# Simulate data
trueDat = BP1BP2.copy()
for j,trial in trueDat.iterrows():
	BP1,BP2 = trial
	trueDat.loc[j,'choice'] = two_norms_model(BP1,BP2,trueTheta,truePhi)

# Fit
# DIFFERENT FAIRNESS PREFERNCES MODEL
fitIters = np.zeros([niter,5])
for i in range(niter):
	#How to bound parameters?
	param0 = np.array([scipy.random.uniform(),scipy.random.uniform()*upBoundPhi])
	fitIters[i,0:2] = param0
	
	res_lsq = least_squares(two_norms_costfun, param0, args=(trueDat,),
							kwargs={'printStep':False,'resid_share':False},
							#diff_step=.01,
							#bounds=([0,-.1],[.5,.1]),
							)
	
	theta,phi = res_lsq.x
	cost = res_lsq.cost
	fitIters[i,2:5] = [theta,phi,cost]

cost_selected = np.min(fitIters[:,4]) #Minimal cost
recovTheta = fitIters[fitIters[:,4]==cost_selected,2][0] # First theta with minimal cost
recovPhi = fitIters[fitIters[:,4]==cost_selected,3][0] # First phi with minimal cost

print('trueTheta = %.3f truePhi = %.3f theta = %.3f phi = %.3f '%(trueTheta, truePhi, recovTheta, recovPhi))

# Simulate with fitted
recovDat = BP1BP2.copy()
for j,trial in recovDat.iterrows():
	BP1,BP2 = trial
	recovDat.loc[j,'choice'] = two_norms_model(BP1,BP2,recovTheta,recovPhi)
r = scipy.stats.pearsonr(trueDat['choice'],recovDat['choice'])[0]

# STORE RESULTS
results = results.append(pd.DataFrame([[iterrun,fakepps,trueTheta,truePhi,recovTheta,recovPhi,cost_selected,r]],columns=
	results.columns))

folder = '%s/analyses/results/parameterrecovery%i' % (base_dir, ntrials)
if not os.path.exists(folder):
    os.makedirs(folder)
results.to_csv('%s/ThetaPhi_%i_%i.csv' % (folder, ntrials,iterrun))
