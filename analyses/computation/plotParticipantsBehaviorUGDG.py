# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:10:34 2019

@author: inghui
"""

import pandas as pd
import sys
import os
import numpy as np
from inequalityaversion import two_norms_model
import seaborn as sns
from scipy.cluster.hierarchy import linkage
import scipy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


if sys.platform == 'win32':
    base_dir = 'P:/3014018.13/experiment_3_DG_UG'
elif sys.platform == 'linux':
    base_dir = '/project/3014018.13/experiment_3_DG_UG'
else:
    base_dir ='/Users/rhmhuijsmans/Inge/OneDrive - Harvard/experiment_3_DG_UG'

#Set colors
twocolors = ["#0093BA", "#5E9D45"]
threecolors= ["#ba1319", "#f36f21", "#fff100"]  
sevencolors = ['#C70039','#FF5733', '#FF8D1A','#FFC300', '#EDDD53', '#ADD45C','#57C785']
fourcolors = ['#C70039','#FF8D1A','#EDDD53','#57C785']

              
#Load parameters for participants behavior
parametersDirUDDD = '%s/analyses/results/Iteration_3'%base_dir
parametersFileUDDD = os.listdir(parametersDirUDDD)

dataParam = pd.concat([pd.read_csv('%s/%s' % (parametersDirUDDD, f)) for f in parametersFileUDDD ])
dataParam.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
dataParamMax = dataParam.loc[dataParam['model']=='inequality_maximizer',:].reset_index(drop=True)

#Recode DG/UG to 0/1
dataParamMax['n_game'] = 1
dataParamMax.loc[dataParamMax['game'] == 'DD','n_game'] = 0    

dataParamMax['theta'] = dataParamMax['theta'] /10
dataParamMax['phi'] = dataParamMax['phi'] /10


#Load raw behavior data
data_dir = '%s/data/distributors/' % base_dir
filename = 'data_demo.csv'
dat = pd.read_csv(os.path.join(data_dir,filename),index_col=None)

clusterFit = True
saveClusterToData = False

saveFigs = False
plot3DDGUG = True


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


if clusterFit:
    
    precision = 100
    phi = np.arange(0,0.5+.001,0.5/precision)
    theta = np.arange(0,1+.001,1/precision)
    BP1 = [0.19,0.75,3]
    BP2 = [0.19,0.75,3]
    
    #Cluster people
    
    thetaPhi = np.repeat(theta,len(phi))
    thetaPhi = pd.DataFrame(thetaPhi)
    thetaPhi.columns = ['theta']
    thetaPhi['phi'] = np.tile(phi,len(theta))
    
    BP1BP2 = [ (x,y) for x in BP1 for y in BP2]
    
    simulations = np.zeros((len(thetaPhi),len(BP1BP2)))
    
    for combo in range(len(thetaPhi)):
        theta0 = thetaPhi.iloc[combo,0]
        phi0 = thetaPhi.iloc[combo,1]
        singleSim = np.array([two_norms_model(BP1, BP2, theta0,phi0) for (BP1, BP2) in BP1BP2]).flatten()
        simulations[combo,:] = singleSim
    
    
    metric = 'sqeuclidean'
    #.8 = 2 clusters (social/non-social)
    #.6 = 3 clusters
    #.5 = 3 clusters
    #.4 = 3 clusters
    #.3 = 4 clusters
    #.25= 5 clusters
    #.2 = 6 clusters
    cluster_threshold = 0.33 # Cutoff for four clusters
    
    sim_distance = scipy.spatial.distance.pdist(simulations,metric)
    Z = linkage(sim_distance,method='average')
    
    clusters = scipy.cluster.hierarchy.fcluster(Z, cluster_threshold, criterion='distance')
    nclusters = len(np.unique(clusters))
    print(nclusters)
    thetaPhi['cluster'] = clusters.T
    thetaPhiPivot = thetaPhi.pivot('theta','phi','cluster').T

    # Add clusters parameters
    key = {1:'Pro-Self',2:'Total Egalitarian',3:'Table Egalitarian',4:'Moral Opportunist'}
    
    dataParamMax['cluster'] = 0
    for i in range(len(dataParamMax)):
        thetaPhiDist = thetaPhi.copy()
        thetaPhiDist['euclidist'] = np.sqrt(np.square(dataParamMax['theta'][i] - thetaPhi['theta']) +
                                            np.square(dataParamMax['phi'][i] - thetaPhi['phi']))
        nearest = np.where(thetaPhiDist['euclidist'] == min(thetaPhiDist['euclidist']))[0][0]
        dataParamMax.loc[i,'cluster'] = thetaPhiDist['cluster'].iloc[nearest]
    dataParamMax['ClustName'] = dataParamMax['cluster'].map(key)
    
    dataParamMax['n_theta'] = dataParamMax['theta']
    selfishPP = np.unique(dataParamMax.loc[dataParamMax['phi']>.4,'Subject'])
    socialPP = np.unique(dataParamMax.loc[dataParamMax['phi']<.4,'Subject'])
    
    socialPPdf = dataParamMax.loc[dataParamMax['Subject'].isin(socialPP),:] 
    selfishPPdf = dataParamMax.loc[dataParamMax['Subject'].isin(selfishPP),:] 
    
    socialPPdfDD = socialPPdf.loc[socialPPdf['game']=='DD',:].reset_index(drop=True)
    socialPPdfUD = socialPPdf.loc[socialPPdf['game']=='UD',:].reset_index(drop=True)
    dataParamMaxDD = dataParamMax.loc[dataParamMax['game']=='DD',:].reset_index(drop=True)
    dataParamMaxUD = dataParamMax.loc[dataParamMax['game']=='UD',:].reset_index(drop=True)
    

    ### Plots
    
    #How are theta and phi distributed?
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(socialPPdfDD['theta'], 10, alpha = 0.5, label = 'DG')
    ax.hist(socialPPdfUD['theta'], 10, alpha = 0.6, label = 'UG')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Theta distribution')
    ax.legend(loc='upper left')
    
    figname = '%s/analyses/plots/Theta_UG_DG.png' % (base_dir)
    if saveFigs:
        plt.savefig(figname,bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(5,5))
    ax.hist(dataParamMaxDD['phi']*2, 10, alpha = 0.5, label = 'DG')
    ax.hist(dataParamMaxUD['phi']*2, 10, alpha = 0.6, label = 'UG')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Phi distribution')
    ax.legend(loc='upper right')
    
    figname = '%s/analyses/plots/Phi_UG_DG.png' % (base_dir)
    if saveFigs:
        plt.savefig(figname,bbox_inches='tight')
    
    #Heatmap, where do participants fit?
    
    # Define colors
    ncols = len(np.unique(thetaPhi['cluster']))
    #sns.set_palette('Spectral',ncols)
    sns.set_palette(sns.color_palette(fourcolors))
    colorMap = sns.color_palette()[0:ncols]
    keyColor = dict(zip(range(1,ncols+1), colorMap))
    dataParamMax['ClustColor'] = dataParamMax['cluster'].map(keyColor)
    
    #How many ppts in each cluster?
    print(dataParamMaxDD.cluster.value_counts())
    print(dataParamMaxUD.cluster.value_counts())
    

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(thetaPhiPivot,cmap=colorMap,square=True, ax = ax,
                #Legend smaller no annotations
                cbar_kws={"shrink": 0.5},annot=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%i'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%i'))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_ylim([-1,101])
    ax.set_xlim([-1,101])
    
    # Define translation of scatter plot onto heatmap space
    x_num = np.size(thetaPhiPivot,0)
    y_num = np.size(thetaPhiPivot,1)
    x_range = np.float(thetaPhiPivot.columns[x_num-1])-np.float(thetaPhiPivot.columns[0])
    y_range = np.float(thetaPhiPivot.index[y_num-1])-np.float(thetaPhiPivot.index[0])
    y_start = 0

    for i in np.arange(len(np.unique(dataParamMax['Subject']))):
        subjPlot = np.unique(dataParamMax['Subject'])[i]
        subData = dataParamMax.loc[dataParamMax['Subject'] == subjPlot,:].reset_index(drop = True)
        
        xpoints = .5+subData['theta']/x_range*(x_num-1)+np.random.rand(1)
        ypoints = .5+(y_start+subData['phi'])/y_range*(y_num-1)

        ax.plot(xpoints,
                   ypoints,'-', color ='black', alpha = 0.15,label = '' )
#    xpointsX = .5+dataParamMax['theta']/x_range*(x_num-1)+np.random.rand(1)
#    ypointsX = .5+(y_start+dataParamMax['phi'])/y_range*(y_num-1)
#    ax.scatter(xpointsX, ypointsX, color ='black', s = 200, 
#               marker = '*', edgecolor = 'black', 
#               label ='DG')
    xpointsX = .5+dataParamMaxDD['theta']/x_range*(x_num-1)+np.random.rand(1)
    ypointsX = .5+(y_start+dataParamMaxDD['phi'])/y_range*(y_num-1)
    ax.scatter(xpointsX, ypointsX, color ='black', s = 200, 
               marker = '*', edgecolor = 'black', 
               label ='DG')
    xpointsX = .5+dataParamMaxUD['theta']/x_range*(x_num-1)+np.random.rand(1)
    ypointsX = .5+(y_start+dataParamMaxUD['phi'])/y_range*(y_num-1)
    ax.scatter(xpointsX, ypointsX, color ='white', s = 200, 
               marker = '*', edgecolor = 'black', 
               label ='UG')
    ax.set_ylim([0, 100])
    ax.legend(scatterpoints = 1,loc=(1.04,0.1))
    
    
#    markers = ['*','*','*','*']
#    colors = sns.color_palette(fourcolors)
#    colors = sns.color_palette(fourcolors)
#    edgecolors = np.divide(colorMap,2)
#    sizes = [200,200,200,200]
#    for i in np.arange(1,5,1):
#        listInd = i-1
#        marker = markers[listInd]; color = colors[listInd];
#        size = sizes[listInd]; edgecolor = edgecolors[listInd];
#        print(i,marker,size)
#        thetaCur = dataParamMax['theta'][dataParamMax['cluster']==i]
#        phiCur = dataParamMax['phi'][dataParamMax['cluster']==i]
#        ax.scatter(x=.5+thetaCur/x_range*(x_num-1)+np.random.rand(1),
#                   y=.5+(y_start+phiCur)/y_range*(y_num-1),
#                   marker=marker,color=color,edgecolor=edgecolor,s=size,lw=1)
#           
#        
    figname = '%s/analyses/plots/Cluster_%i_ThetaPhi_UGDG.png' % (base_dir, nclusters)
    if saveFigs:
        plt.savefig(figname,bbox_inches='tight')
        
# STORE RESULTS
if saveClusterToData:    
    dataParamMax.to_csv('%s/ParticipantClustering.csv'% data_dir)

#3d plot of change in behavior from DG to UG
#For participants who have a phi higher than 8 in the DG, assume that their theta
#will stay the same.
if plot3DDGUG:

    colors = sns.color_palette(fourcolors)
    colors = sns.color_palette(fourcolors)
    
    rotate = range(-90, 100, 10)
    for a in range(len(rotate)):
        fig = plt.figure(figsize=(10,10))
        ax = plt.axes(projection="3d")
    
        for i in np.arange(len(np.unique(dataParamMax['Subject']))):
            subjPlot = np.unique(dataParamMax['Subject'])[i]
            subData = dataParamMax.loc[dataParamMax['Subject'] == subjPlot,:].reset_index(drop = True)
            
            x_points = .5+subData['theta']/x_range*(x_num-1)+np.random.rand(1)
            y_points = .5+(y_start+subData['phi'])/y_range*(y_num-1)
            
            z_points = subData['n_game']
            cDG = colors[int(subData.loc[subData['game'] == 'DD', 'cluster']-1)]
            cUG = colors[int(subData.loc[subData['game'] == 'UD', 'cluster']-1)]
            
            if (y_points[0] < 80) & (y_points[1] < 80):
                ax.plot3D(x_points, y_points, z_points, 'black', alpha = 0.3)  
            
            ax.scatter3D(x_points, y_points, z_points, marker = '*', s = [200,200], depthshade=False,
                         color = [cDG, cUG], edgecolor='black', alpha = 0.5);

            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.view_init(azim=rotate[a], elev = 40)
        ax.set_xlabel('Theta')
        ax.set_ylabel('Phi')
        ax.set_zlabel('Game: 0 = Dictator, 1 = Ultimatum')
        ax.set_zticks([0, 1])
        ax.invert_yaxis()
        figname = '%s/analyses/plots/Cluster_3d_ThetaPhi_%i_%i_UGDG.png' % (base_dir, a, rotate[a])
        if saveFigs:
            plt.savefig(figname,bbox_inches='tight')
            
    
    
    
    
    
    
    
    
    
    
    
    
    