# -*- coding: utf-8 -*-
#!/usr/bin/env python

#Inequality aversion model
#Inge Huijsmans
#Distributors MTurk data
#16/5/2019 

import numpy as np


def two_norms_model(BP1, BP2, theta, phi):
    BP1 = float(BP1); BP2 = float(BP2); theta = float(theta); phi = float(phi);
        
    # Compute terms
    totalAmt = 1 #amount to share
    choiceOptions = np.arange(0,(totalAmt+0.1)*10)/10
    P1choice = totalAmt - choiceOptions
    
    divider = (np.sqrt(abs(BP1 - BP2))+1) 
    P1_totalshare =  (((BP1 - BP2) + P1choice)/divider)

    inequity_behavior = (P1choice-.5)**2
    inequity_total = (P1_totalshare-.5)**2 
    
    utility = (theta)*P1choice - (1-theta)*(phi*inequity_behavior + (1-phi)*inequity_total) 
    choice = choiceOptions[utility==np.max(utility)]
        
    return choice

