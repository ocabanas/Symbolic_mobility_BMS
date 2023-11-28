import sys
import numpy as np 
import pandas as pd
import warnings
import gc
import os
#from memory_profiler import profile
warnings.filterwarnings('default')
#gc.disable()
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from copy import deepcopy, copy
#from ipywidgets import IntProgress
#from IPython.display import display,display_latex,Latex
import time
import pickle
import matplotlib.gridspec as gs

sys.path.append('/export/home/shared/Projects/City_flows/repository/rguimera-machine-scientist/')
from mcmc import *
from parallel import *

sys.path.append('/export/home/shared/Projects/City_flows/repository/rguimera-machine-scientist/Prior/')
from fit_prior import read_prior_par

priors={
    'v1_p4':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np4.2017-10-18 18:07:35.214923.dat',
    'v1_p3':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np3.2017-10-18 18:07:35.262530.dat',
    'v1_p2':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np2.2017-10-18 18:07:35.163458.dat',
    'v1_p5':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np5.2017-10-18 18:07:35.227360.dat',
    'v1_p6':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np6.2017-10-18 18:07:35.224405.dat',
    'v1_p7':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np7.2017-10-18 18:07:35.230340.dat',
    'v1_p8':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np8.2017-10-18 18:07:35.261518.dat',
    'v1_p9':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv1.np9.2017-10-18 18:07:35.163833.dat',
    'v2_p3':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv2.np3.2016-09-09 18:49:42.927679.dat',
    'v2_p4':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv2.np4.2016-09-09 18:49:43.056910.dat',
    'v3_p3':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv3.np3.2017-06-13 08:55:24.082204.dat',
    'v3_p6': '/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv3.np6.maxs50.2021-12-14 09:51:44.438445.dat',
    'v4_p8': '/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv4.np8.maxs200.2019-12-03 09:30:20.580307.dat',
    'v5_p10':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv5.np10.2016-07-11 17:12:38.129639.dat',
    'v5_p12':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv5.np12.2016-07-11 17:12:37.338812.dat',
    'v6_p12':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv6.np12.2016-07-11 17:20:51.957121.dat',
    'v7_p14':'/export/home/shared/Projects/Bacterial_growth/BMS_Ecology/bin/bacterial_growth/rguimera-machine-scientist/Prior/final_prior_param_sq.named_equations.nv7.np14.2016-06-06 16:43:26.130179.dat'
}



def machinescientist(x,y,XLABS,n_params,resets=1,
                    steps_prod=1000,
                    Ts= [1] + [1.04**k for k in range(1, 20)],
                    log_scale_prediction=False,
                    ensemble_avg=None
                    ):
    """
    pms = Parallel(
        Ts,
        variables=XLABS,
        parameters=['a%d' % i for i in range(n_params)],
        x=x, y=y,
        prior_par=prior_par,
    )"""
    #print(list(priors.keys()))
    #print(Ts)
    if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
        print(f'v{len(XLABS)}_p{str(n_params)}')
        raise ValueError
    if ensemble_avg != None:
        list_ens_mdls=[]
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    #del prior_par['Nopi_abs']
    #del prior_par['Nopi2_abs']
    OPS = {
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'exp': 1,
    'log': 1,
    'sinh' : 1,
    'cosh' : 1,
    'tanh' : 1,
    'pow2' : 1,
    'pow3' : 1,
    'sqrt' : 1,
    'fac' : 1,
    '-' : 1,
    '+' : 2,
    '*' : 2,
    '/' : 2,
    '**' : 2,
    }
    best_description_lengths,lowest_mdl, best_model = [],np.inf, None
    all_mdls={}
    #Start some MCMC
    runs=0
    while runs < resets:
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Parallel(
                Ts,#ops=OPS,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=x, y=y,
                prior_par=prior_par,
            )
            # MCMC 
            description_lengths, mdl, mdl_model = [], np.inf, None
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 1000
            mc_start=time.time()
            for i in range(1,steps_prod+1):
                start = time.time()
                # MCMC update
                pms.mcmc_step() # MCMC step within each T
                pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
                # Add the description length to the trace
                description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                if pms.t1.E < mdl:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    mdl, mdl_model = copy(pms.t1.E), deepcopy(pms.t1)
                    delattr(mdl_model, 'x')
                    delattr(mdl_model, 'y')
                    delattr(mdl_model, 'et_space')
                    delattr(mdl_model, 'fit_par')
                    delattr(mdl_model, 'representative')
                    delattr(mdl_model, 'ets')
                    delattr(mdl_model, 'n_dist_par')
                    #gc.collect()
                # Save step of model
                for f in pms.trees.values():
                    last_seen_by_can[f.canonical()] = i
                    last_seen_by_str[str(f)] = i
                # Clean up old representatives and fitted parameters to speed up
                # sampling and save memory
                if (i % NCLEAN) == 0:
                    to_remove = []
                    for represent in pms.t1.representative:
                        try:
                            if last_seen_by_can[represent] < (i - NCLEAN):
                                to_remove.append(represent)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(represent)
                    for t in to_remove: 
                        del pms.t1.representative[t]
                        if t in last_seen_by_can:
                            del last_seen_by_can[t]
                    to_remove = []
                    for string in pms.t1.fit_par:
                        try:
                            if last_seen_by_str[string] < (i - NCLEAN):
                                to_remove.append(string)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove: 
                        del pms.t1.fit_par[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                if (i % 10) == 0:
                    end = time.time()
                    print(f'Progress: {int(float(i*100)/float(steps_prod))}%  | {round(1./(end-start),2)} MCs/s | Time left: {round(float(steps_prod-i)*float(end-mc_start)/(60.*(i)),2)}min.  |  PT rate:{round(pms.accepted12/float(pms.proposed12)*100,3)}%', end='\r')
                if ensemble_avg != None and i>= ensemble_avg[0] and (i%ensemble_avg[1])==0:
                    if pms.t1.E==float('NaN'): print('NaN in ensemble average')
                    list_ens_mdls+=[deepcopy(pms.t1)]
                    delattr(list_ens_mdls[-1], 'x')
                    delattr(list_ens_mdls[-1], 'y')
                    delattr(list_ens_mdls[-1], 'et_space')
                    delattr(list_ens_mdls[-1], 'fit_par')
                    delattr(list_ens_mdls[-1], 'representative')
                    delattr(list_ens_mdls[-1], 'ets')
                    delattr(list_ens_mdls[-1], 'n_dist_par')
                    #gc.collect()
            print()
            
            # End MCMC
            runs+=1
            if best_model==None:
                best_description_lengths,lowest_mdl,best_model=description_lengths,mdl, deepcopy(mdl_model)
            if mdl<lowest_mdl:
                best_description_lengths=deepcopy(description_lengths)
                lowest_mdl=deepcopy(mdl)
                best_model=deepcopy(mdl_model)
            all_mdls[mdl_model.latex()]=deepcopy(description_lengths)

            print('-'*20)
            print(f"Run {runs}")
            print('-'*20)
            print(f'Parallel Tempering rate:{round(pms.accepted/float(pms.proposed)*100,3)}%')
            print(f'Parallel Tempering 12 proposed:{pms.proposed12}')
            print(f'Parallel Tempering 12 rate:{round(pms.accepted12/float(pms.proposed12)*100,3)}%')
            print('Mdl for training data:',copy(mdl))
            print("Model",mdl_model)
            print(mdl_model.latex())
        except Exception as e:
            print('Error during MCMC evolution:')
            print(e)
            print('Current model',pms.t1)
            print('Current energy',pms.t1.E)
            print('Restarting MCMC')
    fig=plt.figure(figsize=(15, 5))
    
    g = gs.GridSpec(2,1)
    ax = fig.add_subplot(g[0])
    for i,j in all_mdls.items():
        if i!=best_model.latex():
            line='--'
        else:
            line='-'
        ax.plot(j,line,label=f'${i}$')
    ax.set_xlabel('MCMC step', fontsize=14)
    ax.set_ylabel('Description length', fontsize=14)
    ax.set_title('MDL model all MC runs')
    ax1=fig.add_subplot(g[1])
    h, l = ax.get_legend_handles_labels() 
    ax1.legend(h, l,fontsize='large')
    ax1.axis('off')
    print('#'*40)
    print('Lowest mdl for training data:',copy(lowest_mdl))
    print('Model:',copy(best_model))
    if ensemble_avg != None:
        return copy(best_model) , copy(list_ens_mdls), fig
    else: 
        return copy(best_model), copy(lowest_mdl), fig
def annealing(x,y,XLABS,n_params,resets=1,
                    steps_prod=1000,
                    T0=1.,
                    BT_cte=None,PT_cte=None
                    ):
    #print(list(priors.keys()))
    #print(Ts)
    if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
        print(f'v{len(XLABS)}_p{str(n_params)}')
        raise ValueError
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    del prior_par['Nopi_abs']
    del prior_par['Nopi2_abs']
    best_description_lengths,lowest_mdl, best_model = [],np.inf, None
    OPS = {
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'exp': 1,
    'log': 1,
    'sinh' : 1,
    'cosh' : 1,
    'tanh' : 1,
    'pow2' : 1,
    'pow3' : 1,
    'sqrt' : 1,
    'fac' : 1,
    '-' : 1,
    '+' : 2,
    '*' : 2,
    '/' : 2,
    '**' : 2,
    }
    all_mdls={}
    #Start some MCMC
    runs=0
    mdl, mdl_model = np.inf, []
    while runs < resets:
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Tree(
                variables=deepcopy(XLABS),
                parameters=['a%d' % i for i in range(n_params)],
                x=x, y=y,ops=OPS,
                prior_par=prior_par,BT=T0)
            print('Initial model:',str(pms))
            # MCMC 
            description_lengths=[]
            last_seen_by_can, last_seen_by_str = {}, {}
            """
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0"""
            NCLEAN = 1000
            mc_start=time.time()
            #steps equilibrium
            #for i in range(1,1000):
            #    pms.mcmc_step()
            #print('Model after 1000 steps:',str(pms))
            for i in range(1,steps_prod+1):
                start = time.time()
                # MCMC update
                pms.mcmc_step() # MCMC step within each T
                # Add the description length to the trace
                description_lengths.append(pms.E)
                #
                if BT_cte != None:
                    pms.BT=BT_cte*pms.BT
                if PT_cte != None:
                    pms.PT=PT_cte*pms.PT
                # Check if this is the MDL expression so far
                # Save step of model
                """
                for f in pms.trees.values():
                    last_seen_by_can[f.canonical()] = i
                    last_seen_by_str[str(f)] = i
                # Clean up old representatives and fitted parameters to speed up
                # sampling and save memory
                if (i % NCLEAN) == 0:
                    to_remove = []
                    for represent in pms.t1.representative:
                        try:
                            if last_seen_by_can[represent] < (i - NCLEAN):
                                to_remove.append(represent)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(represent)
                    for t in to_remove: 
                        del pms.t1.representative[t]
                        if t in last_seen_by_can:
                            del last_seen_by_can[t]
                    to_remove = []
                    for string in pms.t1.fit_par:
                        try:
                            if last_seen_by_str[string] < (i - NCLEAN):
                                to_remove.append(string)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove: 
                        del pms.t1.fit_par[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                if (i % 10) == 0:
                    end = time.time()
                    print(f'Progress: {int(float(i*100)/float(steps_prod))}%  | {round(1./(end-start),2)} MCs/s | Time left: {round(float(steps_prod-i)*float(end-mc_start)/(60.*(i)),2)}min.  |  PT rate:{round(pms.accepted12/float(pms.proposed12)*100,3)}%', end='\r')"""
            print()
            
            # End MCMC
            runs+=1
            all_mdls[pms.latex()]=deepcopy(description_lengths)
            mdl_model.append(deepcopy(pms))
            
            print('-'*20)
            print(f"Run {runs}|BIC temp:{pms.BT}")
            print(f"Run {runs}|PRIO temp:{pms.PT}")
            print('-'*20)
            #print(f'Parallel Tempering rate:{round(pms.accepted/float(pms.proposed)*100,3)}%')
            #print(f'Parallel Tempering 12 proposed:{pms.proposed12}')
            #print(f'Parallel Tempering 12 rate:{round(pms.accepted12/float(pms.proposed12)*100,3)}%')
            print("Model",pms,pms.E)
            print(pms.par_values['A3'])
            print(mdl_model[-1].latex())
            if best_model==None:
                best_model=deepcopy(pms)
            elif pms.E<best_model.E:
                best_model=deepcopy(pms)
        except Exception as e:
            print('Error during MCMC evolution:')
            print(e)
            print('Current model',pms)
            print('Current energy',pms.E)
            print('Restarting MCMC')
    fig=plt.figure(figsize=(15, 5))
    
    g = gs.GridSpec(2,1)
    ax = fig.add_subplot(g[0])
    def sortKey(val):
        return float(val.E)
    mdl_model.sort(key=sortKey)
    if len(mdl_model)>=5:
        top5_latex=[k.latex() for k in mdl_model[0:4]]
    else:
        top5_latex=[k.latex() for k in mdl_model]
    for i,j in all_mdls.items():
        if i!=best_model.latex():
            line='--'
        else:
            line='-'
        if i in top5_latex: ax.plot(j,line,label=f'${i}$');
        else: ax.plot(j,line);
    ax.set_xlabel('MCMC step', fontsize=14)
    ax.set_ylabel('Description length', fontsize=14)
    ax.set_title('MDL model all MC runs')
    ax1=fig.add_subplot(g[1])
    h, l = ax.get_legend_handles_labels() 
    ax1.legend(h, l,fontsize='large')
    ax1.axis('off')
    print('#'*40)
    print('Lowest mdl for training data:',copy(best_model.E))
    print('Model:',copy(best_model))
    return copy(best_model), fig
#@profile
def annealing_sampling(x,y,XLABS,n_params,resets=1,
                    steps_prod=1000,
                    Ts= [1] + [1.04**k for k in range(1, 20)],
                    ):
    """
    pms = Parallel(
        Ts,
        variables=XLABS,
        parameters=['a%d' % i for i in range(n_params)],
        x=x, y=y,
        prior_par=prior_par,
    )"""
    #print(list(priors.keys()))
    #print(Ts)
    if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
        print(f'v{len(XLABS)}_p{str(n_params)}')
        raise ValueError
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    #del prior_par['Nopi_abs']
    #del prior_par['Nopi2_abs']
    OPS = {
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'exp': 1,
    'log': 1,
    'sinh' : 1,
    'cosh' : 1,
    'tanh' : 1,
    'pow2' : 1,
    'pow3' : 1,
    'sqrt' : 1,
    'fac' : 1,
    '-' : 1,
    '+' : 2,
    '*' : 2,
    '/' : 2,
    '**' : 2,
    }
    lowest_mdl, best_model = np.inf, None
    lowest_ann, best_ann = np.inf, None
    #Start some MCMC
    runs=0
    while runs < resets:
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Parallel(
                Ts,#ops=OPS,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=x, y=y,
                prior_par=prior_par,
            )
            # MCMC 
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 100
            mc_start=time.time()
            for i in range(1,steps_prod+1):
                start = time.time()
                # MCMC update
                pms.mcmc_step() # MCMC step within each T
                pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
                # Add the description length to the trace
                # Check if this is the MDL expression so far
                if pms.t1.E < lowest_mdl:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    lowest_mdl, best_model = copy(pms.t1.E), deepcopy(pms.t1)
                    #gc.collect()
                # Save step of model
                for f in pms.trees.values():
                    last_seen_by_can[f.canonical()] = copy(i)
                    last_seen_by_str[str(f)] = copy(i)
                # Clean up old representatives and fitted parameters to speed up
                # sampling and save memory
                if (i % NCLEAN) == 0:
                    to_remove = []
                    for represent in pms.t1.representative:
                        try:
                            if last_seen_by_can[represent] < (i - NCLEAN):
                                to_remove.append(represent)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(represent)
                    for t in to_remove: 
                        del pms.t1.representative[t]
                        if t in last_seen_by_can:
                            del last_seen_by_can[t]
                    to_remove = []
                    for string in pms.t1.fit_par:
                        try:
                            if last_seen_by_str[string] < (i - NCLEAN):
                                to_remove.append(string)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove: 
                        del pms.t1.fit_par[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                    del to_remove
                    print(i,len(pms.t1.representative),len(pms.t1.fit_par),str(pms.t1))
                #if i>(steps_prod+1-1000) and i%100==0:
            annealing_model=deepcopy(best_model)
            print(f'Initial ann model(step{i}/{steps_prod}):', str(annealing_model),f'[{annealing_model.E}|{annealing_model.BT}]')
            while annealing_model.BT>=0.001:
                annealing_model.BT=0.995*annealing_model.BT
                annealing_model.mcmc_step()
                annealing_model.representative={}
                annealing_model.fit_par={}
                if annealing_model.E < lowest_ann:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    lowest_ann, ann_model = copy(annealing_model.E), deepcopy(str(annealing_model))
            print('Final ann model:', str(annealing_model),f'[{annealing_model.E}|{annealing_model.BT}]')
            print('Lowest ann model:', str(ann_model),lowest_ann)
            del annealing_model
            runs+=1
        except Exception as e:
            print('Error during MCMC evolution:')
            print(e)
            print('Current model',pms.t1)
            print('Current energy',pms.t1.E)
            print('Restarting MCMC')
    print('Best Sampling:',best_model,f'({lowest_mdl})')
    print('Best Annealing:',ann_model,f'({lowest_ann})')
    return  best_model,ann_model
        
def plot_predicted_model(prediction=None,real=None,title="Data vs Model prediction",n_box=20,log_scale=False,
                         min_xy=None,max_xy=None):
    if log_scale:
        prediction=[i if i>1e-6 else 1e-6 for i in prediction]
        real=[i if i>1e-6 else 1e-6 for i in real]
    data=pd.DataFrame(data={'x':real,'y':prediction})
    data.replace(0, 1e-6)
    if len(prediction)<n_box:
        n_box=len(prediction)
    if log_scale:
        bins=np.logspace(np.log10(data.x.min()),np.log10(data.x.max()+1e-6), n_box)
        
    else:
        bins=np.linspace(data.x.min(),data.x.max()+1e-6, n_box)
    #print(bins)
    fig,ax1= plt.subplots(1,1,figsize=(8, 8))
    for i in range(len(bins)-1):
        sub_sample=data[(data.x>=bins[i])&(data.x<bins[i+1])]
        mean=np.mean(sub_sample.y.values)
        center=(bins[i]+bins[i+1])*0.5
        #ax1.scatter(sub_sample.x.values,sub_sample.y.values,c='gainsboro')
        #ax2.scatter(sub_sample.x.values,sub_sample.y.values)
        #print(sub_sample.x.mean(),sub_sample.x.mean())
        #w = .75/float(n_box)
        #width = 10**(np.log10(bins[i])+w/2.)-10**(np.log10(bins[i])-w/2.)
        #width = w
        width = (bins[i+1]-bins[i])/2.
        ax1.boxplot([sub_sample.y.values], positions=[center],showfliers=False,widths=[width])
        ax1.plot(center,mean,"pr")
    if min_xy==None:
        min_xy=min(data.x.min(),data.y.min())
    if max_xy==None:
        max_xy=max(data.x.max(),data.y.max())
    if log_scale==True:
        min_xy=min_xy/10.
        max_xy=max_xy*10.
    ax1.plot((min_xy, max_xy), (min_xy,max_xy))
    ax1.scatter(data.x.values,data.y.values,c='gainsboro')
    if log_scale==True:
        ax1.set_yscale('log')
        ax1.set_xscale('log')
    ax1.set_title(title)
    ax1.set_xlabel('Real', fontsize=14)
    ax1.set_ylabel('Predicted', fontsize=14)
    #ax2.set_xlabel('Real', fontsize=14)
    ax1.axis('equal')
    ax1.tick_params(axis = 'both')
    ax1.set_xlim(min_xy,max_xy)
    ax1.set_ylim(min_xy,max_xy)
    """
    locs, labels = plt.yticks()
    locs1, labels1 = plt.xticks()
    locs1=locs
    labels1=labels
    """
    #if log_scale==True: plt.ticklabel_format(style='sci', axis='x')#, scilimits=(0,0))
    #ax1.set(adjustable='box', aspect='equal')
    fig.show()
    #print('Len data:',len(prediction))
    #if len(prediction)<2:
    #    for i,j in zip(real,prediction):
            #print(f'({i,j})')
def from_string_model(x,y,string_model,n_vars,n_params,vars,silence=False):
    prior_par = read_prior_par(priors[f'v{str(n_vars)}_p{str(n_params)}'])
    model=Tree(prior_par=deepcopy(prior_par), from_string=string_model , x=x,y=y,variables=vars)
    if silence==False:
        print('Model summary')
        print('Par_values:',model.par_values)
        print(model.BT,model.PT)
        print('bic:',model.bic)
        print('E:',model.E)
        print('EB:',model.EB)
        print('EP:',model.EP)
        print('Representative:',model.representative)
        print('Variables:',model.variables)
        print('Parameters:',model.parameters)
    return model
def from_string_DL(x,y,string_model,n_vars,n_params,par_values,fit_params=True,silence=False):
    if 'd0' in par_values==False:
        raise KeyError("Key 'do' is not in par_values")
    prior_par = read_prior_par(priors[f'v{str(n_vars)}_p{str(n_params)}'])
    #Initialize String model
    model=Tree(prior_par=deepcopy(prior_par), from_string=deepcopy(string_model) , x=x,y=y)
    if silence==False: print('E(A):',model.E)
    if silence==False: print('Par_values(A):',model.par_values)#['d0'])
    #Change par_values
    model.par_values=par_values
    #model.fit_par={string_model:par_values}
    if silence==False: print('Par_values(B):',model.par_values)#['d0'])
    #Fit and SSE
    model.get_sse(fit=fit_params,verbose=True)
    model.get_bic(verbose=True)
    model.get_energy(bic=True, reset=True,verbose=True)
    if silence==False: print('Par_values(C):',model.par_values)#['d0'])
    if silence==False: print('E(C):',model.E)
    if silence==False:
        print('Model summary')
        print(model.BT,model.PT)
        print('bic:',model.bic)
        print('E:',model.E)
        print('EB:',model.EB)
        print('EP:',model.EP)
        print('Representative:',model.representative)
        print('Variables:',model.variables)
        print('Parameters:',model.parameters)
    return model
def MCMC_save_strings(x,y,XLABS,n_params,resets=1,
                    steps_prod=1000,
                    log_scale_prediction=False,
                    file_name=None
                    ):
    Ts= [1] + [1.04**k for k in range(1, 20)]
    x_test=None
    y_test=None
    ensemble_avg=None
    """
    pms = Parallel(
        Ts,
        variables=XLABS,
        parameters=['a%d' % i for i in range(n_params)],
        x=x, y=y,
        prior_par=prior_par,
    )"""
    if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
        raise ValueError
    if ensemble_avg != None:
        list_ens_mdls=[]
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    best_description_lengths,lowest_mdl, best_model = [],np.inf, None
    best_model_all_data=None
    all_mdls=[]
    if (x_test is None and y_test is not None)or(x_test is not None and y_test is None):
        raise Exception("Missing x_test or y_test.")
    #Start some MCMC
    runs=0
    while runs < resets:
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Parallel(
                Ts,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=x, y=y,
                prior_par=prior_par,
            )

            # MCMC 
            description_lengths, mdl, mdl_model = [], np.inf, None
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 1000
            mc_start=time.time()
            for i in range(1,steps_prod+1):
                start = time.time()
                # MCMC update
                pms.mcmc_step() # MCMC step within each T
                pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
                # Add the description length to the trace
                #description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                #if pms.t1.E < mdl:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                #    mdl, mdl_model = pms.t1.E, deepcopy(pms.t1)
                # Save step of model
                for f in pms.trees.values():
                    last_seen_by_can[f.canonical()] = i
                    last_seen_by_str[str(f)] = i
                # Clean up old representatives and fitted parameters to speed up
                # sampling and save memory
                if (i % NCLEAN) == 0:
                    to_remove = []
                    for represent in pms.t1.representative:
                        try:
                            if last_seen_by_can[represent] < (i - NCLEAN):
                                to_remove.append(represent)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(represent)
                    for t in to_remove: 
                        del pms.t1.representative[t]
                        if t in last_seen_by_can:
                            del last_seen_by_can[t]
                    to_remove = []
                    for string in pms.t1.fit_par:
                        try:
                            if last_seen_by_str[string] < (i - NCLEAN):
                                to_remove.append(string)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove: 
                        del pms.t1.fit_par[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                #Append string to pickle
                with open(file_name, "a") as file_object:
                    file_object.write(str(deepcopy(pms.t1))+' '+str(deepcopy(pms.t1.E)))
                    file_object.write("\n")
            # End MCMC
            runs+=1
        except Exception as e:
            print('Error during MCMC evolution:')
            print(e)
            print('Current model',pms.t1)
            print('Current energy',pms.t1.E)
            print('Restarting MCMC')
def machinescientist_parallel(x,y,XLABS,n_params,resets=1,
                    steps_prod=1000,
                    Ts= [1] + [1.04**k for k in range(1, 20)],
                    log_scale_prediction=False
                    ):
    from multiprocessing import Pool
    from contextlib import closing
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    input_MCMC=(x,y,XLABS,n_params,steps_prod,Ts,prior_par)
    global MC_evolution1
    def MC_evolution1(x,y,XLABS,n_params,steps_prod,Ts,prior_par):
        if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
            raise ValueError
        #Start some MCMC
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Parallel(
                Ts,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=x, y=y,
                prior_par=prior_par,
            )
            # MCMC 
            description_lengths, mdl, mdl_model = [], np.inf, None
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 1000
            mc_start=time.time()
            for i in range(0,steps_prod+1):
                start = time.time()
                # MCMC update
                pms.mcmc_step() # MCMC step within each T
                pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
                # Add the description length to the trace
                #description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                if pms.t1.E < mdl:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    mdl, mdl_model = copy(pms.t1.E), deepcopy(pms.t1)
                    delattr(mdl_model, 'x')
                    delattr(mdl_model, 'y')
                    delattr(mdl_model, 'et_space')
                    delattr(mdl_model, 'fit_par')
                    delattr(mdl_model, 'representative')
                    delattr(mdl_model, 'ets')
                    delattr(mdl_model, 'n_dist_par')
                    #gc.collect()
                # Save step of model
                for f in pms.trees.values():
                    last_seen_by_can[f.canonical()] = i
                    last_seen_by_str[str(f)] = i
                # Clean up old representatives and fitted parameters to speed up
                # sampling and save memory
                if (i % NCLEAN) == 0:
                    to_remove = []
                    for represent in pms.t1.representative:
                        try:
                            if last_seen_by_can[represent] < (i - NCLEAN):
                                to_remove.append(represent)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(represent)
                    for t in to_remove: 
                        del pms.t1.representative[t]
                        if t in last_seen_by_can:
                            del last_seen_by_can[t]
                    to_remove = []
                    for string in pms.t1.fit_par:
                        try:
                            if last_seen_by_str[string] < (i - NCLEAN):
                                to_remove.append(string)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove: 
                        del pms.t1.fit_par[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                    end = time.time()
            """
            print("MCMC evolution")
            plt.figure(figsize=(15, 5))
            plt.plot(description_lengths)
            plt.xlabel('MCMC step', fontsize=14)
            plt.ylabel('Description length', fontsize=14)
            plt.title('MDL model: $%s$' % mdl_model.latex())
            plt.show()
            """
        except Exception as e:
            print('Error during MCMC evolution:')
            print(e)
            print('Current model',pms.t1)
            print('Current energy',pms.t1.E)
            print('Restarting MCMC')
        return copy(mdl_model)
    #with closing(Pool(resets)) as p:
    #    result=p.starmap(MC_evolution1, [input_MCMC for k in range(resets)])
    pool = Pool(resets)
    result=pool.starmap(MC_evolution1, [input_MCMC for k in range(resets)])
    pool.close()
    pool.join()
    best_energy=np.inf
    for item in result:
        if best_energy>item.E:
            best_energy=copy(item.E)
            best_model=deepcopy(item)
    print('Lowest mdl for training data:',copy(best_model.E))
    print('Model:',copy(best_model))
    print('Par values:',copy(best_model.par_values))
    return copy(best_model)
#@profile
def machinescientist_ensemble_parallel(x,y,XLABS,n_params,resets=1,
                    steps_prod=1000,
                    Ts= [1] + [1.04**k for k in range(1, 20)],
                    log_scale_prediction=False,
                    ensemble_avg=None
                    ):
    from multiprocessing import Pool
    from contextlib import closing
    prior_par = read_prior_par(priors[f'v{len(XLABS)}_p{str(n_params)}'])
    input_MCMC=(x,y,XLABS,n_params,steps_prod,Ts,prior_par,ensemble_avg)
    global MC_evolution2
    #@profile
    def MC_evolution2(x,y,XLABS,n_params,steps_prod,Ts,prior_par,ensemble_avg):
        if f'v{len(XLABS)}_p{str(n_params)}' not in list(priors.keys()):
            raise ValueError
        if ensemble_avg != None:
            list_ens_mdls=[]
        
        #Start some MCMC
        try: #Sometimes a NaN error appears. Therefore we forget the current MCMC and start again.
            # Initialize the parallel machine scientist
            pms = Parallel(
                Ts,
                variables=XLABS,
                parameters=['a%d' % i for i in range(n_params)],
                x=x, y=y,
                prior_par=prior_par,
            )
            # MCMC 
            description_lengths, mdl, mdl_model = [], np.inf, None
            last_seen_by_can, last_seen_by_str = {}, {}
            for f in pms.trees.values():
                last_seen_by_can[f.canonical()] = 0
                last_seen_by_str[str(f)] = 0
            NCLEAN = 1000
            mc_start=time.time()
            for i in range(0,steps_prod+1):
                start = time.time()
                # MCMC update
                pms.mcmc_step() # MCMC step within each T
                pms.tree_swap() # Attempt to swap two randomly selected consecutive temps
                # Add the description length to the trace
                #description_lengths.append(pms.t1.E)
                # Check if this is the MDL expression so far
                if pms.t1.E < mdl:
                    #if pms.t1.E==float('NaN'): print('NaN in best model mdl')
                    mdl, mdl_model = copy(pms.t1.E), deepcopy(pms.t1)
                    delattr(mdl_model, 'x')
                    delattr(mdl_model, 'y')
                    delattr(mdl_model, 'et_space')
                    delattr(mdl_model, 'fit_par')
                    delattr(mdl_model, 'representative')
                    delattr(mdl_model, 'ets')
                    delattr(mdl_model, 'n_dist_par')
                    #gc.collect()
                # Save step of model
                for f in pms.trees.values():
                    last_seen_by_can[f.canonical()] = i
                    last_seen_by_str[str(f)] = i
                # Clean up old representatives and fitted parameters to speed up
                # sampling and save memory
                if (i % NCLEAN) == 0:
                    to_remove = []
                    for represent in pms.t1.representative:
                        try:
                            if last_seen_by_can[represent] < (i - NCLEAN):
                                to_remove.append(represent)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(represent)
                    for t in to_remove: 
                        del pms.t1.representative[t]
                        if t in last_seen_by_can:
                            del last_seen_by_can[t]
                    to_remove = []
                    for string in pms.t1.fit_par:
                        try:
                            if last_seen_by_str[string] < (i - NCLEAN):
                                to_remove.append(string)
                        except KeyError: # This tree was tested but not visited anyway!
                            to_remove.append(string)
                    for t in to_remove: 
                        del pms.t1.fit_par[t]
                        if t in last_seen_by_str:
                            del last_seen_by_str[t]
                    end = time.time()
                if i>= ensemble_avg[0] and (i%ensemble_avg[1])==0:
                    if pms.t1.E==float('NaN'): print('NaN in ensemble average')
                    list_ens_mdls+=[deepcopy(pms.t1)]
                    delattr(list_ens_mdls[-1], 'x')
                    delattr(list_ens_mdls[-1], 'y')
                    delattr(list_ens_mdls[-1], 'et_space')
                    delattr(list_ens_mdls[-1], 'fit_par')
                    delattr(list_ens_mdls[-1], 'representative')
                    delattr(list_ens_mdls[-1], 'ets')
                    delattr(list_ens_mdls[-1], 'n_dist_par')
                    #gc.collect()
            """
            print("MCMC evolution")
            plt.figure(figsize=(15, 5))
            plt.plot(description_lengths)
            plt.xlabel('MCMC step', fontsize=14)
            plt.ylabel('Description length', fontsize=14)
            plt.title('MDL model: $%s$' % mdl_model.latex())
            plt.show()
            """
        except Exception as e:
            print('Error during MCMC evolution:')
            print(e)
            print('Current model',pms.t1)
            print('Current energy',pms.t1.E)
            print('Restarting MCMC')
        del x,y
        return [copy(mdl_model) , copy(list_ens_mdls)]
    #with closing( Pool(resets) ) as p:
    #    result=p.starmap(MC_evolution2, [deepcopy(input_MCMC) for k in range(resets)])
    pool = Pool(resets)
    result=pool.starmap(MC_evolution2, [deepcopy(input_MCMC) for k in range(resets)])
    pool.close()
    pool.join()
    #print(result)
    ensemble=[]
    best_energy=np.inf
    for item in result:
        ensemble+=item[1]
        if best_energy>item[0].E:
            best_energy=copy(item[0].E)
            best_model=deepcopy(item[0])
    del result
    print('Lowest mdl for training data:',copy(best_model.E))
    print('Model:',copy(best_model))
    print('Par values:',copy(best_model.par_values))
    return copy(best_model) , copy(ensemble)
#@profile
def main():
    name=open('../City2City_DataFrames/x_train_list_sample_2022_06_21-05_07_34.pkl', "rb")
    x_train_list_sample=pickle.load(name)
    name.close()
    #
    name=open('../City2City_DataFrames/y_train_list_sample_2022_06_21-05_07_34.pkl', "rb")
    y_train_list_sample=pickle.load(name)
    name.close()
    best_model_train, list_ensemble_train = machinescientist_ensemble_parallel(x=x_train_list_sample['NewYork'],
                                                                           y=y_train_list_sample['NewYork'],
                                                                           XLABS=['d','m_o','m_d'],n_params=6,
                                                                           resets=5,
                                                                           steps_prod=500,
                                                                           log_scale_prediction=True,
                                                                           ensemble_avg=[100,10]
                                                                          )
    print(len(list_ensemble_train))

if __name__ == "__main__":
    main()
    
