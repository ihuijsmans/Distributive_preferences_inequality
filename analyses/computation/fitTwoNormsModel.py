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
os.listdir(data_dir)
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


# DIFFERENT FAIRNESS PREFERNCES MODEL
model = 'inequality_maximizer'
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


## TOTAL INEQUALITY ONLY MODEL
#model = 'inequality_total'
#theta = np.nan
#start = time.time()
#print ('subject %s model %s' %(Subject,model))
#
#fitIters = np.zeros([niter,5])
#for i in range(niter):
#    #How to bound parameters?
#    
#    param0 = np.array([scipy.random.uniform()]).dot(10)
#    fitIters[i,0] = param0[0]
#    
#    res_lsq = least_squares(costFunctionsIH.inequity_total_costfun, param0, args=(subDat,),
#                            kwargs={'printStep':False,'resid_share':residShareChoice},
#                            #diff_step=.1,
#                            #bounds=([0,1],[0,1]),
#                            )
#    
#    phi = res_lsq.x
#    cost = res_lsq.cost
#    fitIters[i,[2,4]] = [phi,cost]
#
#cost_selected = np.min(fitIters[:,4]) #Minimal cost
#phi = fitIters[fitIters[:,4]==cost_selected,2][0] # First phi with minimal cost
#SSE = cost_selected*2
#AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,1)
#BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,1)
#
#results = results.append(pd.DataFrame([[Subject,game,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))
#
#print('took %.2f seconds to converge on phi = %.3f with SSE = %.3f'%(time.time() - start,phi,SSE))
#
## BEHAVIORAL INEQUALITY ONLY MODEL
#model = 'inequality_behavior'
#theta = np.nan
#start = time.time()
#print ('subject %s model %s' %(Subject,model))
#
#fitIters = np.zeros([niter,5])
#for i in range(niter):
#    #How to bound parameters?
#    
#    param0 = np.array([scipy.random.uniform()]).dot(10)
#    fitIters[i,0] = param0[0]
#    
#    res_lsq = least_squares(costFunctionsIH.inequity_behavior_costfun, param0, args=(subDat,),
#                            kwargs={'printStep':False,'resid_share':residShareChoice},
#                            #diff_step=.1,
#                            #bounds=([0,1],[0,1]),
#                            )
#    
#    phi = res_lsq.x
#    cost = res_lsq.cost
#    fitIters[i,[2,4]] = [phi,cost]
#
#cost_selected = np.min(fitIters[:,4]) #Minimal cost
#phi = fitIters[fitIters[:,4]==cost_selected,2][0] # First phi with minimal cost
#SSE = cost_selected*2
#AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,1)
#BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,1)
#
#results = results.append(pd.DataFrame([[Subject,game,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))
#
#print('took %.2f seconds to converge on phi = %.3f with SSE = %.3f'%(time.time() - start,phi,SSE))


# STORE RESULTS



results = results.reset_index(drop=True)
folder = os.path.join(base_dir,'analyses/results/Iteration_%i'%jobs_iteration)
if not os.path.exists(folder):
    os.makedirs(folder)
    
print('%s/Results_%s_%s_sub-%05d.csv'%(folder,model, game,Subject))
results.to_csv('%s/Results_%s_%s_sub-%05d.csv'%(folder,model, game,Subject))








## GA MODEL (pre-programmed second-order expectations)
#model = 'GA_ppSOE'
#phi = np.nan
#start = time.time()
#print('subject %s model %s' %(sub,model))
#
#fitIters = np.zeros([niter,3])
#for i in range(niter):
#    theta0 = scipy.random.uniform()*10000
#    fitIters[i,0] = theta0
#    res_lsq = least_squares(costFunctions.GA_ppSOE_costfun, theta0, args=(subDat,),
#                            kwargs={'printStep':False,'resid_share':residShareChoice},
#                            diff_step=.1,bounds=([0],[10000]),)
#    theta = res_lsq.x
#    cost = res_lsq.cost
#    fitIters[i,1:3] = [theta,cost]
#cost_selected = np.min(fitIters[:,2]) #Minimal cost
#theta = fitIters[fitIters[:,2]==cost_selected,1][0] # First theta with minimal cost
#SSE = cost_selected*2
#AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,1)
#BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,1)
#results = results.append(pd.DataFrame([[sub,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))
#
#print('took %.2f seconds to converge on theta = %.3f with SSE = %.3f'%(time.time() - start,theta,SSE))
#
## IA MODEL
#model = 'IA'
#phi = np.nan
#start = time.time()
#print('subject %s model %s' %(sub,model))
#
#fitIters = np.zeros([niter,3])
#for i in range(niter):
#    theta0 = scipy.random.uniform()*10000
#    fitIters[i,0] = theta0
#    res_lsq = least_squares(costFunctions.IA_costfun, theta0, args=(subDat,),
#                            kwargs={'printStep':False,'resid_share':residShareChoice},
#                            diff_step=.1,bounds=([0],[10000]),)
#    theta = res_lsq.x
#    cost = res_lsq.cost
#    fitIters[i,1:3] = [theta,cost]
#cost_selected = np.min(fitIters[:,2]) #Minimal cost
#theta = fitIters[fitIters[:,2]==cost_selected,1][0] # First theta with minimal cost
#SSE = cost_selected*2
#AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,1)
#BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,1)
#results = results.append(pd.DataFrame([[sub,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))
#
#print('took %.2f seconds to converge on theta = %.3f with SSE = %.3f'%(time.time() - start,theta,SSE))
#
## MP MODEL
#model = 'MP'
#start = time.time()
#print('subject %s model %s' %(sub,model))
#
#fitIters = np.zeros([niter,5])
#for i in range(niter):
#    param0 = [scipy.random.uniform()/2,scipy.random.uniform()/5-0.1]
#    fitIters[i,0:2] = param0[:]
#    res_lsq = least_squares(costFunctions.MP_costfun, param0, args=(subDat,),
#                            kwargs={'printStep':False,'resid_share':residShareChoice},
#                            diff_step=.05,bounds=([0,-.1],[.5,.1]),)
#    theta,phi = res_lsq.x
#    cost = res_lsq.cost
#    fitIters[i,2:5] = [theta,phi,cost]
#cost_selected = np.min(fitIters[:,4]) #Minimal cost
#theta = fitIters[fitIters[:,4]==cost_selected,2][0] # First theta with minimal cost
#phi = fitIters[fitIters[:,4]==cost_selected,3][0] # First theta with minimal cost
#SSE = cost_selected*2
#AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,2)
#BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,2)
#results = results.append(pd.DataFrame([[sub,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))
#
#print('took %.2f seconds to converge on theta = %.3f and phi = %.3f with SSE = %.2f'%(
#    time.time() - start,theta,phi,SSE))
#
## MP MODEL (pre-programmed second-order expectations)
#model = 'MP_ppSOE'
#start = time.time()
#print 'subject %s model %s' %(sub,model),
#
#fitIters = np.zeros([niter,5])
#for i in range(niter):
#    param0 = [scipy.random.uniform()/2,scipy.random.uniform()/5-0.1]
#    fitIters[i,0:2] = param0[:]
#    res_lsq = least_squares(costFunctions.MP_ppSOE_costfun, param0, args=(subDat,),
#                            kwargs={'printStep':False,'resid_share':residShareChoice},
#                            diff_step=.05,bounds=([0,-.1],[.5,.1]),)
#    theta,phi = res_lsq.x
#    cost = res_lsq.cost
#    fitIters[i,2:5] = [theta,phi,cost]
#cost_selected = np.min(fitIters[:,4]) #Minimal cost
#theta = fitIters[fitIters[:,4]==cost_selected,2][0] # First theta with minimal cost
#phi = fitIters[fitIters[:,4]==cost_selected,3][0] # First theta with minimal cost
#SSE = cost_selected*2
#AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,2)
#BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,2)
#results = results.append(pd.DataFrame([[sub,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))
#
#print 'took %.2f seconds to converge on theta = %.3f and phi = %.3f with SSE = %.2f'%(
#    time.time() - start,theta,phi,SSE)
#
## MP MODEL (pre-programmed second-order expectations)
#model = 'MP_ppSOE_noDiffStep'
#start = time.time()
#print 'subject %s model %s' %(sub,model),
#
#fitIters = np.zeros([niter,5])
#for i in range(niter):
#    param0 = [scipy.random.uniform()/2,scipy.random.uniform()/5-0.1]
#    fitIters[i,0:2] = param0[:]
#    res_lsq = least_squares(costFunctions.MP_ppSOE_costfun, param0, args=(subDat,),
#                            kwargs={'printStep':False,'resid_share':residShareChoice},
##                             diff_step=.05,
#                            bounds=([0,-.1],[.5,.1]),)
#    theta,phi = res_lsq.x
#    cost = res_lsq.cost
#    fitIters[i,2:5] = [theta,phi,cost]
#cost_selected = np.min(fitIters[:,4]) #Minimal cost
#theta = fitIters[fitIters[:,4]==cost_selected,2][0] # First theta with minimal cost
#phi = fitIters[fitIters[:,4]==cost_selected,3][0] # First theta with minimal cost
#SSE = cost_selected*2
#AIC = penalizedModelFit.compute_AIC(subDat.shape[0],SSE,2)
#BIC = penalizedModelFit.compute_BIC(subDat.shape[0],SSE,2)
#results = results.append(pd.DataFrame([[sub,model,theta,phi,SSE,AIC,BIC]],columns=results.columns))
#
#print 'took %.2f seconds to converge on theta = %.3f and phi = %.3f with SSE = %.2f'%(
#    time.time() - start,theta,phi,SSE)

