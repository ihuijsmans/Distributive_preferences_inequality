import numpy as np
from inequalityaversion import two_norms_model


def two_norms_costfun(param,subDat,printStep=False,printPredictions=False):
    theta = param[0]
    phi = param[1]

    for trial in range(subDat.shape[0]):
        subDat.loc[trial,'prediction'] = two_norms_model(
            subDat.loc[trial,'BP1'],
            subDat.loc[trial,'BP2'],
            theta, phi)
        residuals = subDat.loc[:,'choice'] - subDat.loc[:,'prediction']
    
    residuals = residuals.astype('float')
    SSE = np.sum(np.square(residuals))
    
    if printStep==True:
        print ('theta = %.2f, phi = %.2f, SSE = %.2f' % (theta,phi,SSE))
    if printPredictions == True:
        print (subDat)
    return residuals

