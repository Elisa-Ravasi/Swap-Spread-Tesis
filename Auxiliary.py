# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:51:29 2020

@author: Elisa
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import math
import time
import matplotlib.pyplot as plt


def Read_Data(path_gral, file_ss, file_drivers):
    ############## Swap Sprad Data ############################
    
    file_data_ss = os.path.join(path_gral,'Data/{}'.format(file_ss) ) 
    data_ss = pd.read_excel(file_data_ss, index_col=None)
    # convert to percent
    data_ss[data_ss.select_dtypes(include=['number']).columns] *= 0.01
    data_ss['Date']=data_ss['Date'].dt.date    
    ############# Drivers Data ####################################
    file_data_drivers = os.path.join(path_gral,'Data/{}'.format(file_drivers) ) 
    data_drivers = pd.read_excel(file_data_drivers, index_col=None)
    #### Convert to percent the slope 2/10 and scale MBS for 10
    data_drivers['US 2/10 Treasury Slope'] = data_drivers['US 2/10 Treasury Slope'].apply(lambda x: x/100)
    data_drivers['MBS Effective Duration'] = data_drivers['MBS Effective Duration'].apply(lambda x: x/10)
    data_drivers['USD/JPY'] = data_drivers['USD/JPY'].apply(lambda x: x/100)
    data_drivers['Date']=data_drivers['Date'].dt.date  

    return data_ss, data_drivers

def CalcFactors(maturities,lambda0,data_ss, data_drivers):
    
    yields=data_ss[data_ss.columns[2:]]
    Drivers=data_drivers[data_drivers.columns[1:]]
    
    #Matriz Lambda
    col1X = np.ones(len(maturities))
    col2X = np.array([(1-np.exp(-lambda0*maturities[i]))/(lambda0*maturities[i])\
                  for i in range(len(maturities))])
    col3X = np.array([col2X[i]-np.exp(-lambda0*maturities[i])\
                  for i in range(len(maturities))])
    X = np.vstack([col1X, col2X, col3X]).T
    
    
    #Calculo de los factores a traves de OLS.
    factors=np.zeros((len(yields),3))
    res = np.zeros((len(yields),len(maturities)))
    
    for i in range(len(yields)):
       result = sm.OLS(endog=yields.iloc[i], exog=X).fit()
       factors[i,:] = result.params
       res[i,:]=result.resid
    
    ##Series de las primeras diferencias de los factores.\n",
    diff_factors = np.zeros((len(yields)-1,3))
    for i in range(len(yields)-1):
        for j in range(3):
            diff_factors[i,j]=factors[i+1,j]-factors[i,j]
            
    ##Series de las primeras diferencias de los drivers.\n",
    diff_drivers = np.zeros((len(Drivers)-1,len(Drivers.columns)))
    for i in range(len(Drivers)-1):
        for j in range(len(Drivers.columns)):
            diff_drivers[i,j]=Drivers[Drivers.columns[j]][i+1]-Drivers[Drivers.columns[j]][i]
    
    
    factors_df=pd.DataFrame(factors)
    factors_df.rename(columns={0:'Level',1:'Slope',2:'Curv'}, inplace=True)
    factors_df['Date']=data_ss['Date']
    factors_df.set_index('Date',  inplace=True)
    
        
    diff_factors_df=pd.DataFrame(diff_factors)
    diff_factors_df.rename(columns={0:'Level',1:'Slope',2:'Curv'}, inplace=True)
    diff_factors_df.set_index(data_ss['Date'][1:],  inplace=True)
     
    Drivers.set_index(data_ss['Date'],  inplace=True)
    
    diff_drivers_df=pd.DataFrame(diff_drivers, columns=Drivers.columns)    
    diff_drivers_df.set_index(data_ss['Date'][1:],  inplace=True)
        
    name_factors=['Level', 'Slope', 'Curv']
    
    return name_factors, factors_df, diff_factors_df, Drivers, diff_drivers_df
    
### Integrationn: Augmented Dickey Fuller test
def ADF(v, crit, reg, fd):
            """ Augmented Dickey Fuller test
        
            Parameters
            ----------
            v: ndarray matrix
            residuals matrix
            
            Returns
            -------
            bool: boolean
            true if v pass the test 
               adfuller:
               -------------------------------------------
               maxlag (int) – Maximum lag which is included in test, default 12*(nobs/100)^{1/4}
               regression ({'c','ct','ctt','nc'}) –
               Constant and trend order to include in regression
               ’c’ : constant only (default)
               ’ct’ : constant and trend
               ’ctt’ : constant, and linear and quadratic trend
               ’nc’ : no constant, no trend
               autolag ({'AIC', 'BIC', 't-stat', None}) –
               if None, then maxlag lags are used
               if ‘AIC’ (default) or ‘BIC’, then the number of lags is chosen to minimize the corresponding informa
               
               ## Return:
               Critical values: Critical values for the test statistic at the 1 %, 5 %, and 10 % levels. Based on MacKinnon (2010).
            """
            boolean = False    
            adf = adfuller(v,maxlag=None, regression=reg, autolag='BIC')
            if(adf[0] < adf[4][crit]):
                pass
            else:
                boolean = True
            
            if fd[0]==0:
                print(adf)
            else:
                fd[1].write('%.2f'% (adf[0]))
    
            return boolean

def AnalysisRootUnit( Option, crit, reg, ini_dat_str, fin_dat_str, n_drivers, factors_df, all_drivers, Drivers, diff_factors_df, diff_drivers_df):
   
    print('\n')
    print('####### UNIT ROOT ANALYSIS ##############','\n') 
   
    # # Testing Unit root   
     
    if Option=='Specif data':
        ini_dat=datetime.strptime(ini_dat_str, '%Y-%m-%d').date()
        fin_dat=datetime.strptime(fin_dat_str, '%Y-%m-%d').date()
    else:
        pass     
    
    #Factors
    print("LEVEL")
    print('Hip unit root-ADF test-Level=',ADF(factors_df['Level'][ini_dat:fin_dat],crit,reg,[0,0]),'\n')   
    print("SLOPE")
    print('Hip unit root-ADF test-Slope=',ADF(factors_df['Slope'][ini_dat:fin_dat],crit,reg,[0,0]),'\n')   
    print("CURVATURE")
    print('Hip unit root-ADF test-Curv=',ADF(factors_df['Curv'][ini_dat:fin_dat],crit,reg,[0,0]),'\n')   
    
    delta = timedelta( days=1)
    print("Delta Level")
    print('Hip unit root-ADF test- Diff Level=',ADF(diff_factors_df['Level'][ini_dat+delta:fin_dat],crit,reg,[0,0]),'\n') 
    print("Delta Slope")
    print('Hip root unit-ADF test-Diff Slope=',ADF(diff_factors_df['Slope'][ini_dat+delta:fin_dat],crit,reg,[0,0]),'\n') # Hipótesis de raíz unitaria es rechazada al 99% de confianza -> I(0)
    print("Delta CURVATURE")
    print('Hip unit root-ADF test-Diff Curvature=',ADF(diff_factors_df['Curv'][ini_dat+delta:fin_dat],crit,reg,[0,0]),'\n') # Hipótesis de raíz unitaria es rechazada al 99% de confianza -> I(0)
       
    #Drivers
    print("DRIVER")
    for i in n_drivers:
        print('Hip unit root-ADF test-Driver {}='.format(all_drivers[i]),ADF(Drivers[all_drivers[i]][ini_dat:fin_dat],crit,reg,[0,0]),'\n')   
        print('Hip unit root-ADF test-Diff Driver {}='.format(all_drivers[i]),ADF(diff_drivers_df[all_drivers[i]][ini_dat+delta:fin_dat],crit,reg,[0,0]),'\n') 
        print('\n')
  
    print('###################################')

def get_johansen(y, k_ar_diff, det_order, test):
    """
    Get the cointegration vectors at 90%, 95% or 99% level of significance
    given by the trace statistic test and max eigenvalue statistic.
    """

    N, l = y.shape
    jres = coint_johansen(y, det_order, k_ar_diff)
    
    """
    coint_johansen(endog, det_order, k_ar_diff)
    *endog (array-like (nobs_tot x neqs)) – The data with presample.
    *det_order (int) –
       -1 - no deterministic terms
        0  - constant term
        1 - linear trend
    *k_ar_diff (int, nonnegative) – Number of lagged differences in the model. 
                                   EN REALIDAD k_ar_dif = p ó lag del VAR original.
                                   Ver paginas 247-248 Luktepol.
    """
    
    if test=='Trace':
        stat = jres.lr1               # trace statistic
        signf = jres.cvt              # critical values
        #print("Critical values(90%, 95%, 99%) of trace_stat\n",signf,'\n')
        #print("stat", stat)
    elif test=='Max Eig':
        stat = jres.lr2               # max eigen statistic
        signf = jres.cvm             # critical values
    
    for i in range(l):
        if stat[i] > signf[i, 1]:     # 0: 90%  1:95% 2: 99%
      #      print('stat', stat[i], '\n ', signf,'\n')
            r = i + 1        
    jres.r = r
    jres.evecr = jres.evec[:, :r]

    return jres

  
def AnalysisCointegration(Option, factors_df, Drivers, n_drivers, n_factors, all_drivers, name_factors, ini_dat_str, fin_dat_str, det_order, trend_lag, test):
        
    spec_drivers= [all_drivers[index] for index in n_drivers]
    spec_factor= [name_factors[index] for index in n_factors]

    Vec_Model=pd.merge(factors_df[spec_factor], Drivers[spec_drivers], on="Date")
    #ec_Model.set_index('Date',  inplace=True)

    print('\n')
    print('####### COINTEGRATION ANALYSIS ##############','\n')
    print('Dimension of Model Vector:',len(Vec_Model.columns))    
    
    if Option=='Specif data':
        ini_dat=datetime.strptime(ini_dat_str, '%Y-%m-%d').date()
        fin_dat=datetime.strptime(fin_dat_str, '%Y-%m-%d').date()
        Vec_Model=Vec_Model[ini_dat:fin_dat]
    else:
        pass
    
    modelVAR = VAR(Vec_Model.to_numpy())

    ########################################################################
    #######  Analizamos el lag del VECM con selected_orders: ###############
    '''
     VAR.select_order(maxlags=None, trend='c')
     Compute lag order selections based on each of the available information criteria
     Parameters
        maxlagsint
            if None, defaults to 12 * (nobs/100.)**(1./4)
        trendstr {“nc”, “c”, “ct”, “ctt”}
                “nc” - no deterministic terms
                “c” - constant term
                “ct” - constant and linear term
                “ctt” - constant, linear, and quadratic term

    '''
    CIS = modelVAR.select_order(maxlags=10,trend=trend_lag)
    print('\n')
    print('VAR Lag Order-Critera ({})'.format(trend_lag))
    print(CIS.selected_orders,'\n')
    print('BIC Lag Order: ', CIS.bic)
    print('\n')
    #k_ar_diff tiene que ser >=1. Si es =1 no tenemos Gammas.  
    #elegimos el criterio BIC.
    k_ar_diff = CIS.bic - 1
        
    
    jres = get_johansen(Vec_Model,k_ar_diff, det_order, test)

    print ('There are', jres.r, 'cointegration vectors for all data =', len(Vec_Model), '\n')
    print('The Hip of rank {} is rejected at 95%'.format(str(jres.r+1)) )

    print('##########################################','\n')

    return Vec_Model, k_ar_diff, jres.r


def Simulation( path_gral, Option, factors_df, Drivers, n_drivers, n_factors, name_factors, all_drivers, ini_dat_str, fin_dat_str, k_ar_diff,coint_rank, m_pre, accuracy, determ, crit, reg):
    
    spec_drivers= [all_drivers[index] for index in n_drivers]
    spec_factor= [name_factors[index] for index in n_factors]

    Vec_Model=pd.merge(factors_df[spec_factor], Drivers[spec_drivers], on="Date")
 
    print('\n')
    print('####### SIMULATION ##############','\n')
    print('Dimension of Model Vector:',len(Vec_Model.columns))    
    print('\n')
    
    #Restriction to data for specific data
    if Option=='Specif data':
        ini_dat=datetime.strptime(ini_dat_str, '%Y-%m-%d').date()
        fin_dat=datetime.strptime(fin_dat_str, '%Y-%m-%d').date()
        Vec_Model=Vec_Model[ini_dat:fin_dat]
        
    
    # Restriction of sample data to period for estimation
    Vec_Model_pre=Vec_Model[:-m_pre]
    #Construction of VAR MODEl with data less last days to predict
    modelVAR_pre = VAR(Vec_Model_pre.to_numpy())
    #Construction of VECM Model with data less last days to predict
    modelVECM_pre = VECM(Vec_Model_pre.to_numpy(), deterministic=determ,  k_ar_diff = k_ar_diff, coint_rank=coint_rank) 
    
    dim_factors=len(n_factors)
    dim_var=len(Vec_Model.columns)
    
    fd = open(os.path.join(path_gral, 'Sim_Results_VAR'), 'w')
    ######### VAR MODEL SIMULATION ###############################
    ### Estimation of model parameters from data sample less m_pre days. 
    print('####  Parameters Estimation VAR Model ##########', '\n')  
    fd.write('%s \n'% ('####  Parameters Estimation VAR Model ########## \n')) 
     
    FitModelVAR_pre= modelVAR_pre.fit(k_ar_diff+1)
    cte=FitModelVAR_pre.intercept
    ## Extract A
    A=[]
    for i in range(0,k_ar_diff+1 ):
        A.append(FitModelVAR_pre.coefs[i])
        print('A{}='.format(i+1),A[i],'\n')
        fd.write('A{}='.format(i+1))
        for line in A[i]:
            fd.write("{}\n".format(line))
        fd.write('\n')
    #Extract Sigma
    Sigma_u= FitModelVAR_pre.sigma_u   
    print('SIGMA =', Sigma_u, '\n')
    fd.write('Sigma=')
    for line in Sigma_u:
         fd.write("{}\n".format(line))
    fd.write('\n')
    #Extract Cte
    print('cte =', cte) 
    fd.write('Cte=[')
    for line in cte:
        fd.write("{}\n".format(line))
    fd.write(']\n')
    fd.write('\n')
    print('\n')    
    
    ###### VAR Simulation ######################
    #Initialization of simulation data. 
    modelVAR_sim=np.zeros((len(Vec_Model_pre),dim_var))
    #Initializing modelVAR_sim
    for i in range(0, k_ar_diff+1):
        modelVAR_sim[i,:] = Vec_Model.iloc[i].copy()
    
    n_VAR = 0
    media_sim_VAR= []
    for i in range(0, m_pre):
        media_sim_VAR.append([])
        for j in range(0,dim_var):
            media_sim_VAR[i].append([])
    med_muestral=np.zeros((dim_var,m_pre))
    med_muestral_ant=np.zeros((dim_var,m_pre))
    var_estim=np.zeros((dim_var,m_pre))
    aux_mpre=np.zeros(m_pre)
    
    start_time_VAR=time.time()
    while n_VAR < 100 or   (np.max( [ np.max( [var_estim[i][j]/n_VAR for i in range(0,dim_factors)] ) for j in range(0,m_pre)])>accuracy**2 ):   
        for t in range(0,len(Vec_Model_pre)-k_ar_diff-1):
            u = np.random.multivariate_normal([0]*dim_var,Sigma_u,1)  
            modelVAR_sim[t+k_ar_diff+1,:]= cte + u
            for j in range(0, k_ar_diff+1):
                modelVAR_sim[t+k_ar_diff+1,:] += A[j]@modelVAR_sim[t+k_ar_diff-j,:]
        #Predigo m_pre valores del VAR utilizando estos m_est valores que acabo de simular
        modelVAR_pre_sim = VAR(modelVAR_sim)
        FitModelVAR_pre_sim = modelVAR_pre_sim.fit(k_ar_diff+1)
        forecastVAR=FitModelVAR_pre_sim.forecast(modelVAR_sim,steps=m_pre)
        n_VAR += 1
        for k in range(dim_var):
            if n_VAR == 1:
                for j in range(0, m_pre):
                    med_muestral[k,j] = forecastVAR[j,k]
            else:
                for j in range(0, m_pre):
                    med_muestral_ant[k,j] = med_muestral[k,j]
                    med_muestral[k,j] = med_muestral[k,j] + (forecastVAR[j,k]-med_muestral[k,j])/n_VAR
                    var_estim[k,j] = (1-1/(n_VAR-1)) * var_estim[k,j] + n_VAR * (med_muestral[k,j]-med_muestral_ant[k,j])**2
            for j in range(0, m_pre):
                media_sim_VAR[j][k].append(forecastVAR[j,k])
        ##Print if some step of prediction reached the tolerance
        for j in range(0, m_pre): 
            if aux_mpre[j]==0:
                if n_VAR >= 100 and np.max( [var_estim[i][j]/n_VAR for i in range(0,dim_factors)]) < accuracy**2:
                    aux_mpre[j]=1
                    print('------ PREDICTION-STEP: {} --------'.format(j+1) )   
                    fd.write('\n')
                    fd.write('------ PREDICTION-STEP: {} -------- \n'.format(j+1) )
                    print('Nr of Simulation - VAR = {}-Tolerance= {} \n'.format( n_VAR, accuracy))
                    fd.write('Nr of Simulation - VAR = {}-Tolerance= {} \n'.format( n_VAR, accuracy))
                    fd.write('\n')
                    print('\n')            
                    for k in range(0,len(n_factors)):                  
                        print('mean of factor {}: estimated mean={}, bias={} \n'.format(name_factors[n_factors[k]], round(med_muestral[k,j],4), round(med_muestral[k,j] - Vec_Model[name_factors[n_factors[k]]][-m_pre+j],4)))
                        fd.write('mean of factor {}: estimated mean={}, bias={} \n'.format(name_factors[n_factors[k]], round(med_muestral[k,j],4), round(med_muestral[k,j] - Vec_Model[name_factors[n_factors[k]]][-m_pre+j],4)))
                    print('\n')    
                    fd.write('\n')
                    for k in range(0,len(n_factors)): 
                        print('std error of factor {}: std estimator={} \n'.format(name_factors[n_factors[k]], round(math.sqrt(var_estim[k,j]/n_VAR),4)))
                        fd.write('std error of factor {}: std estimator={} \n'.format(name_factors[n_factors[k]], round(math.sqrt(var_estim[k,j]/n_VAR),4)))
                    print('\n')    
                    fd.write('\n')
                    if n_drivers != None:
                        for k in range(0,len(n_drivers)):                       
                            print('mean of driver {}: estimated mean={}, bias={} \n'.format(all_drivers[n_drivers[k]], round(med_muestral[k+len(n_factors),j],4) , round(med_muestral[k+len(n_factors),0] -  Vec_Model[all_drivers[n_drivers[k]]][-m_pre+j],4)  ))
                            fd.write('mean of driver {}: estimated mean={}, bias={} \n'.format(all_drivers[n_drivers[k]], round(med_muestral[k+len(n_factors),j],4) , round(med_muestral[k+len(n_factors),0] -  Vec_Model[all_drivers[n_drivers[k]]][-m_pre+j],4)  ))
                        print('\n')    
                        fd.write('\n')
                        for k in range(0,len(n_drivers)):  
                           print('std error of driver {}:  std estimator={} \n'.format(all_drivers[n_drivers[k]], round(math.sqrt(var_estim[k+len(n_factors),j]/n_VAR),4)))
                           fd.write('std error of driver {}:  std estimator={} \n'.format(all_drivers[n_drivers[k]], round(math.sqrt(var_estim[k+len(n_factors),j]/n_VAR),4)))
                        print('\n')    
                        fd.write('\n')
                    print('\n') 
                    fd.write('\n')
                else:
                    pass
            else:
                pass
                
    stop_VAR=(time.time() - start_time_VAR)
    pd_VarSim=pd.DataFrame(np.concatenate((modelVAR_sim, forecastVAR)), index=Vec_Model.index, columns=Vec_Model.columns)
    print("--- %s seconds VAR---" , stop_VAR, '\n')
    fd.write('--- %s seconds VAR---: {} \n'.format(stop_VAR))
    fd.close()

    
    ########## VECM MODEL SIMULATION ###########################
    ### Estimation of model parameters from data sample less m_pre days.
    fd = open(os.path.join(path_gral, 'Sim_Results_VECM'), 'w')
    print('##### Parameters Estimation VECM Model ##### ', '\n')
    fd.write('##### Parameters Estimation VECM Model ##### \n')
    fd.write('\n')
    FitModelVECM_pre = modelVECM_pre.fit()
    Alpha=FitModelVECM_pre.alpha
    Beta=FitModelVECM_pre.beta.T
    #AlphaBeta=Alpha @ Beta
    C=FitModelVECM_pre.det_coef_coint
    Co=FitModelVECM_pre.det_coef
    Sigma_u=FitModelVECM_pre.sigma_u
    print('tr(Beta)=', Beta, '\n') 
    fd.write('tr(Beta)=')
    for line in Beta:
        fd.write("{}\n".format(line))
    fd.write('\n')    
    print('Alpha=', Alpha,'\n')
    fd.write('Alpha=')
    for line in Alpha:
        fd.write("{}\n".format(line))
    fd.write('\n')    
    if determ[0:2]=='ci':
        print('C =',C, '\n')
        fd.write('C=')
        for line in C:
            fd.write("{}\n".format(line))
        fd.write('\n')    
    else:
        print('Co =',Co, '\n')
        fd.write('C=')
        for line in Co:
            fd.write("{}\n".format(line))
        fd.write('\n')
    Gamma_all=FitModelVECM_pre.gamma
    Gamma=[]
    Gamma.append(Gamma_all[:,0:dim_var])
    print('Gamma{}'.format(1), Gamma[0])
    fd.write('Gamma{}='.format(1))
    for line in Gamma[0]:
        fd.write("{}\n".format(line))
    fd.write('\n')    
    for i in range(1, k_ar_diff):
        Gamma.append(Gamma_all[:,i*len(dim_var):(i+1)*len(modelVECM_sim.columns)])
        print('Gamma{}'.format(i+1), Gamma[i])
        fd.write('Gamma{}'.format(i+1))
        for line in Gamma[i]:
            fd.write("{}\n".format(line))
    print('\n')
    fd.write('\n')
    fd.write('Sigma_u=')
    for line in Sigma_u:
        fd.write("{}\n".format(line))
    fd.write('\n')   
    for i in range(0,coint_rank):
        print('Test ADF for (tr(Beta)*X)[{}]= \n'.format(i), ADF( Vec_Model_pre.dot(np.transpose(Beta))[i],crit, reg,[0,0]))
        fd.write('%s \n'% ('Test ADF for (tr(Beta)*X)[{}]= '.format(i)))
        ADF( Vec_Model_pre.dot(np.transpose(Beta))[i],crit, reg,[1,fd])
        print('\n')
        fd.write('\n')
    
    ###### VECM Simulation ######################
    
    #Initialization of simulation data. 
    modelVECM_sim=np.zeros((len(Vec_Model_pre),dim_var))
    ####################################################    
    #Initializing modelVEC_sim
    for i in range(0, k_ar_diff+1):
        modelVECM_sim[i,:] = Vec_Model.iloc[i].copy()
    
    n_VECM = 0
    media_sim_VECM= []
    for i in range(0, m_pre):
        media_sim_VECM.append([])
        for j in range(0,dim_var):
            media_sim_VECM[i].append([])
    med_muestral=np.zeros((dim_var,m_pre))
    med_muestral_ant=np.zeros((dim_var,m_pre))
    var_estim=np.zeros((dim_var,m_pre))
    aux_mpre=np.zeros(m_pre)
    
    start_time_VECM=time.time()
    
    while n_VECM < 100 or   (np.max( [ np.max( [var_estim[i][j]/n_VECM for i in range(0,dim_factors)] ) for j in range(0,m_pre)])>accuracy**2 ):   
        for t in range(0,len(Vec_Model_pre)-k_ar_diff-1):
            u = np.random.multivariate_normal([0]*dim_var,Sigma_u,1)   
            if determ=='ci':
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+(Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:]+ C)).T + u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1,:]+= Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:])
            elif determ=='li':
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+(Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:]+ C*(t+k_ar_diff))).T + u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1]+=  Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:]) 
            elif determ=='cili':  
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+(Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:]+ C.T@np.array([1,t+k_ar_diff]))).T + u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1]+=  Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:]) 
            elif determ=='co':
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:])+ Co.T + u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1,:]+= Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:])
            elif determ=='lo':
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:])+ Co.T*(t+k_ar_diff+1) + u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1]+=  Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:])  
            elif determ=='colo':  
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:] ) + Co@np.array([1,t+k_ar_diff+1]) + u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1]+=  Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:])    
            elif determ=='coli':  
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+(Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:] + C*(t+k_ar_diff))).T+ Co.T + u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1]+=  Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:]) 
            #cilo
            else:
                modelVECM_sim[t+k_ar_diff+1,:]= modelVECM_sim[t+k_ar_diff,:]+(Alpha@(Beta@modelVECM_sim[t+k_ar_diff,:]+ C) + Co*(t+k_ar_diff+1)).T+ u
                for j in range(0, k_ar_diff): 
                    modelVECM_sim[t+k_ar_diff+1]+=  Gamma[j]@(modelVECM_sim[t+k_ar_diff,:]-modelVECM_sim[t+k_ar_diff-1,:]) 
            
        modelVECM_pre_sim = VECM(modelVECM_sim, deterministic=determ,  k_ar_diff=k_ar_diff, coint_rank=coint_rank)
        FitModelVECM_pre_sim = modelVECM_pre_sim.fit()
        
        #Forecast m_pre values from VECM using m_est values (just simulated). 
        forecastVECM=FitModelVECM_pre_sim.predict(steps=m_pre)
        n_VECM += 1
        
        for k in range(dim_var):
            if n_VECM == 1:
                for j in range(0, m_pre):
                    med_muestral[k,j] = forecastVECM[j,k]
            else:
                for j in range(0, m_pre):
                    med_muestral_ant[k,j] = med_muestral[k,j]
                    med_muestral[k,j] = med_muestral[k,j] + (forecastVECM[j,k]-med_muestral[k,j])/n_VECM
                    var_estim[k,j] = (1-1/(n_VECM-1)) * var_estim[k,j] + n_VECM * (med_muestral[k,j]-med_muestral_ant[k,j])**2
            for j in range(0, m_pre):
                media_sim_VECM[j][k].append(forecastVECM[j,k])
        ##Print if some step of prediction reached the tolerance
        for j in range(0, m_pre):
            if aux_mpre[j]==0:
                if n_VECM >= 100 and np.max( [var_estim[i][j]/n_VECM for i in range(0,dim_factors)]) < accuracy**2:
                    aux_mpre[j]=1
                    print('------ PREDICTION-STEP: {} -------- \n'.format(j+1) )  
                    fd.write('------ PREDICTION-STEP: {} -------- \n'.format(j+1))
                    print('Nr of Simulation - VECM = {}-Tolerance= {} \n'.format( n_VECM, accuracy))
                    fd.write('Nr of Simulation - VECM = {}-Tolerance= {} \n'.format( n_VECM, accuracy))
                    print('\n')            
                    fd.write('\n')
                    for k in range(0,len(n_factors)):                  
                        print('mean of factor {}: estimated mean={}, bias={} \n'.format(name_factors[n_factors[k]], round(med_muestral[k,j],4), round(med_muestral[k,j] - Vec_Model[name_factors[n_factors[k]]][-m_pre+j],4)))
                        fd.write('mean of factor {}: estimated mean={}, bias={} \n'.format(name_factors[n_factors[k]], round(med_muestral[k,j],4), round(med_muestral[k,j] - Vec_Model[name_factors[n_factors[k]]][-m_pre+j],4)))
                    print('\n')    
                    fd.write('\n')
                    for k in range(0,len(n_factors)): 
                        print('std_error of factor {}: std estimator={} \n'.format(name_factors[n_factors[k]], round(math.sqrt(var_estim[k,j]/n_VAR),4)))
                        fd.write('std error of factor {}: std estimator={} \n'.format(name_factors[n_factors[k]], round(math.sqrt(var_estim[k,j]/n_VAR),4)))
                    print('\n')    
                    fd.write('\n')
                    if n_drivers != None:
                        for k in range(0,len(n_drivers)):                       
                            print('mean of driver {}: sample mean={}, bias={} \n'.format(all_drivers[n_drivers[k]], round(med_muestral[k+len(n_factors),j],4) , round(med_muestral[k+len(n_factors),0] -  Vec_Model[all_drivers[n_drivers[k]]][-m_pre+j],4)  ))
                            fd.write('mean of driver {}: sample mean={}, bias={} \n'.format(all_drivers[n_drivers[k]], round(med_muestral[k+len(n_factors),j],4) , round(med_muestral[k+len(n_factors),0] -  Vec_Model[all_drivers[n_drivers[k]]][-m_pre+j],4)  ))
                        for k in range(0,len(n_drivers)):  
                           print('std error of driver {}:  std estimator={} \n'.format(all_drivers[n_drivers[k]], round(math.sqrt(var_estim[k+len(n_factors),j]/n_VECM),4)))
                           fd.write('std error of driver {}:  std estimator={} \n'.format(all_drivers[n_drivers[k]], round(math.sqrt(var_estim[k+len(n_factors),j]/n_VECM),4)))
                    print('\n') 
                    fd.write('\n')
                else:
                    pass
            else:
                pass
      
    stop_VECM=(time.time() - start_time_VECM)
    pd_VecmSim=pd.DataFrame(np.concatenate((modelVECM_sim, forecastVECM)), index=Vec_Model.index,columns=Vec_Model.columns)
    
    
    print("--- %s seconds VECM---" , stop_VECM)
    fd.write('--- %s seconds VECM---: {}'.format( stop_VECM))
    fd.close() 
    
    return pd_VarSim, pd_VecmSim, modelVAR_pre, modelVECM_pre, modelVAR_sim, modelVECM_sim, Vec_Model, Beta
    
def Graphics_Factors(factors_df, diff_factors_df):    
    
    #Graphics of factors and their first differences. 
    plt.subplots(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(factors_df.index,factors_df['Level'], lw = 1.5, label = 'Level')    
    plt.ylabel('%')
    plt.legend(loc='upper left')
    plt.grid(True) 
    plt.subplot(1,2,2)
    plt.plot(diff_factors_df.index[1:],diff_factors_df['Level'][1:], lw = 1.5, label = 'Diff Level' )    
    plt.ylabel('%')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('FactorsSerie_Level.jpg')
    plt.show()
    
    
    fig= plt.subplots(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(factors_df.index,factors_df['Slope'], lw = 1.5, label = 'Slope')    
    plt.ylabel('%')
    plt.legend(loc='upper left')
    plt.grid(True) 
    plt.subplot(1,2,2)
    plt.plot(diff_factors_df.index[1:],diff_factors_df['Slope'][1:], lw = 1.5, label = 'Diff Slope' )    
    plt.ylabel('%')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('FactorsSerie_Slope.jpg')
    plt.show()
    
    fig= plt.subplots(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(factors_df.index, factors_df['Curv'], lw = 1.5, label = 'Curvature')    
    plt.ylabel('%')
    plt.legend(loc='upper left')
    plt.grid(True) 
    plt.subplot(1,2,2)
    plt.plot(diff_factors_df.index[1:], diff_factors_df['Curv'][1:], lw = 1.5, label = 'Diff Curvature' )    
    plt.ylabel('%')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig('FactorsSerie_Curvature.jpg')
    plt.show()
    
def Graphics_Sim(n_factors, coint_rank, Vec_Model, Beta, modelVAR_sim, pd_VarSim, pd_VecmSim):    
        
    ###plot last simulation of factors together the actual sample
    ### last Simulation of VAR 
    for i in range(0, len(n_factors)):
        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(pd_VarSim[pd_VarSim.columns[i]], lw = 1.5, label = 'SIM VAR:{}'.format(pd_VarSim.columns[i]))    
        ax.plot(Vec_Model[Vec_Model.columns[i]],lw = 1.5, label = 'FACTOR {}'.format(Vec_Model.columns[i]))
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        ax.grid(True) 
        plt.savefig('Sim_var {}.jpg'.format(Vec_Model.columns[i]))
        plt.show()
    
    ### last Simulation of VECM
    for i in range(0, len(n_factors)):
        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(pd_VecmSim[pd_VecmSim.columns[i]], lw = 1.5, label = 'SIM VECM:{}'.format(pd_VecmSim.columns[i]))    
        ax.plot(Vec_Model[Vec_Model.columns[i]],lw = 1.5, label = 'FACTOR {}'.format(Vec_Model.columns[i]))
        ax.set_xlabel('Time')
        ax.legend(loc='upper left')
        ax.grid(True) 
        plt.savefig('Sim_vecm {}.jpg'.format(Vec_Model.columns[i]))
        plt.show()
    
    ###plot regression beta@X
    ################################################
    for i in range(0, coint_rank):
        Vec_Model.dot(np.transpose(Beta))[i]
        x=np.arange(len(Vec_Model))
        x = sm.add_constant(x)
        results=sm.OLS(Vec_Model.dot(np.transpose(Beta))[i],x).fit()
        stat=Vec_Model.dot(np.transpose(Beta))[i]
        a=results.params
        s=np.dot(x,a)
        s_pd=pd.Series( s, index=stat.index)
    
        fig, ax = plt.subplots(figsize=(10,8))
        plt.subplot(2,1,1)
        plt.plot(stat, lw = 1.5, label = 'BETA@VECM ({}-th linear comb)'.format(i+1))    
        plt.plot(s_pd,lw = 1.5, label = 'TREND')
        ax.set_xlabel('Time')
        plt.legend(loc='upper left')
        plt.grid(True) 
        plt.subplot(2,1,2)
        plt.plot(stat-s_pd, lw = 1.5, label = 'STAT ADF: ({:,.3f})'.format(adfuller( stat-s,maxlag=None, regression='nc', autolag='BIC')[0] ))    
        ax.set_xlabel('Time')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig('Beta@Vecm{}.png'.format(i+1))
        plt.show()
        
    
    return s_pd, stat
