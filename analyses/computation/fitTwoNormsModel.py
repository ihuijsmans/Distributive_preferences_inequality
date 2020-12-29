import sys, os, scipy
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import time

import costFunctions
import penalizedModelFit

if sys.platform == 'win32':
    base_dir = 'P:/3014018.13/experiment_3_DG_UG'
elif sys.platform == 'linux':
    base_dir = '/project/3014018.13/experiment_3_DG_UG'
else:
    base_dir ='/Users/rhmhuijsmans/Inge/OneDrive - Harvard/experiment_3_DG_UG'
  
data_dir = '%s/data/distributors/' % base_dir
filename = 'data_demo.csv'

# Arguments for cluster
sub = int(sys.argv[1]) # This takes the subject number
niter = int(sys.argv[2]) # This takes the number of iterations for the fitting algorithm
jobs_iteration = int(sys.argv[3]) # This takes the number of job iterations / output folder index
game = str(sys.argv[4]) # This takes the game we are estimating

#sub = 10
#niter = 100
#jobs_iteration = 1
#game = 'UD'

# Load data


dat = pd.read_csv(os.path.join(data_dir,filename),index_col=None)

Subject = np.unique(dat['Subject'])[sub]
subDat = dat.loc[dat['Subject']==Subject,:].reset_index(drop=True)
subDat = subDat.loc[subDat['game']==game,:].reset_index(drop=True)
subDat['choice'] = 1-subDat['choiceMe']

print(subDat.groupby(['BP1', 'BP2'], as_index=False)['choice','expectations'].mean())


print(Subject, niter, jobs_iteration, subDat.shape[0])


# Fit
residShareChoice=False
results = pd.DataFrame(columns=['Subject','game','model','theta','phi','SSE','AIC','BIC'])


# TWO NORMS MODEL
model = 'two_norms_model'
start = time.time()
print ('subject %s model %s' %(Subject,model))

fitIters = np.zeros([niter,5])
for i in range(niter):

    #How to bound parameters?
    param0 = np.array([scipy.random.uniform(),scipy.random.uniform()*0.5])
    fitIters[i,0:2] = param0
    
    res_lsq = least_squares(costFunctions.two_norms_costfun, param0, args=(subDat,),
                            kwargs={'printStep':False,'resid_share':residShareChoice},
                            #diff_step=.1,
                            #bounds=([0,1],[0,1]),
                            )
    theta,phi = res_lsq.x
    cost = res_lsq.cost
    fitIters[i,2:5] = [theta,phi,cost]

cost_selected = np.min(fitIters[:,4]) #Minimal cost
theta = fitIters[fitIters[:,4]==cost_selected,2][0] # First theta with minimal cost
phi = fitIters[fitIters[:,4]==cost_selected,3][0] # First phi with minimal cost
SSE = cost_selected*2
if SSE == 0:
    AIC = float('inf')
    BIC = float('inf')
else:
    AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,2)
    BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,2)

results = results.append(pd.DataFrame([[Subject,game,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))

print('took %.2f seconds to converge on theta = %.3f phi = %.3f with with BIC = %.2f, SSE = %.3f'%(time.time() - start,theta,phi,BIC, SSE))



# STORE RESULTS
results = results.reset_index(drop=True)
folder = os.path.join(base_dir,'analyses/results/Iteration_%i'%jobs_iteration)
if not os.path.exists(folder):
    os.makedirs(folder)
    
print('%s/Results_%s_%s_sub-%05d.csv'%(folder,model, game,Subject))
results.to_csv('%s/Results_%s_%s_sub-%05d.csv'%(folder,model, game,Subject))


