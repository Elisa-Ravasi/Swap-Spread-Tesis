# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 10:26:01 2020

@author: Elisa
"""

import os
import numpy as np
import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
from Auxiliary import Read_Data,CalcFactors,  Graphics_Factors, AnalysisRootUnit, AnalysisCointegration, Simulation, Graphics_Sim
 

####################################################################
########## Data Read ###############################################
####################################################################
path_gral=os.getcwd()
file_ss='data_ss.xlsx'
file_drivers='data_drivers.xlsx'

data_ss, data_drivers= Read_Data(path_gral, file_ss, file_drivers)

#extremo en 8 años
lambda0 = 0.01868002083  #extremo en 14 años: lambda0=0.0106744048;
#extremo en 16,5 años: lambda0=0.0090570707;
    
# maturities in months
maturities = np.array([24, 36, 60, 84, 120, 360]).T

### Drivers
all_drivers=['Libor - Repo O/N', 'Libor','Libor - Effective', 'VIX', \
'Budget Balance (% GDP)',  'S&P Index', 'Repo', 'US 2/10 Treasury Slope', \
'MBS Effective Duration', 'Global Financial (USFICORP)', 'Corporate Purpose Issuance', \
'USD/JPY']


######################################################################
############## Calculations Factors ##################################
######################################################################

##### Factors Calculation ##############################
name_factors,factors_df, diff_factors_df, Drivers, diff_drivers_df= CalcFactors(maturities,lambda0,data_ss, data_drivers)
######################################################################

######## Plots factors serie #######################
Graphics_Factors(factors_df, diff_factors_df)

'''
######################################################################
############### Initial Analysis #####################################
######################################################################
'''
###### Total sample to Initial Analysis: ####### 
#level
ini_dat_str='2011-08-25'
fin_dat_str='2014-08-04'


##########################################################
############ Analysis Unit Root ##########################
#Critical values for the ADF test statistic at the 1 %, 5 %, and 10 % levels.
crit='1%'

#Type of regresion: Constant and trend order to include in regression
#’c’ : constant only (default)
#’ct’ : constant and trend
#’ctt’ : constant, and linear and quadratic trend
#’nc’ : no constant, no trend
reg='ct'

#Run analysis for all data or for Specific data give previously in 'Total sample for simulation'
#Option='All data' 
Option='Specif data'

#Choose a vector of drivers to analyse:
# 0: Libor - Repo O/N,	1 :Libor;	2: Libor - Effective; 3: VIX; 
# 4: Budget Balance (% GDP); 5: S&P Index; 6: Repo; 7: US 2/10 Treasury Slope;
# 8: MBS Effective Duration; 9: Global Financial (USFICORP); 10: Corporate Purpose Issuance;
# 11: USD/JPY 
n_drivers = [8] 

AnalysisRootUnit(Option, crit, reg , ini_dat_str, fin_dat_str, n_drivers, factors_df, all_drivers, Drivers, diff_factors_df, diff_drivers_df)

#####################################################
######## Cointegration Rank ##########################
Option='Specif data'

#Choose a vector of factors to analyse:
#0: Level, 1: Slope, 2: Curvature 
n_factors=[0]

### Type of regresion: Constant and trend order to include in regression 
# (for VAR.select_order function). 
###       “nc” - no deterministic terms
###       “c” - constant term
###       “ct” - constant and linear term
###       “ctt” - constant, linear, and quadratic term
trend_lag='ct'

#Choose to consider in get_johansen
#det_order (int) –
#       -1 - no deterministic terms
#        0  - constant term
#        1 - linear trend
det_order=1

## Johansen Statistic test: 'Trace'/'Max Eig'
test='Trace'

Vec_Model, k_ar_diff, coint_rank=AnalysisCointegration(Option, factors_df, Drivers, n_drivers, n_factors, all_drivers, name_factors, ini_dat_str, fin_dat_str, det_order, trend_lag, test)
#######################################################################

'''  
#######################################################################
############ Simulation ###############################################
#######################################################################
'''
###### Total sample to Simulation Study: ####### 
#level
ini_dat_str='2011-08-25'
fin_dat_str='2014-08-04'



#Choose vector of factors to add model:    
# 0:level; 1:slope; 2:curvature
n_factors=[0] 

#Choose a vector of drivers to add model:
# 0: Libor - Repo O/N,	1 :Libor;	2: Libor - Effective; 3: VIX; 
# 4: Budget Balance (% GDP); 5: S&P Index; 6: Repo; 7: US 2/10 Treasury Slope;
# 8: MBS Effective Duration; 9: Global Financial (USFICORP); 10: Corporate Purpose Issuance;
# 11: USD/JPY 
n_drivers = [8] 

#lag-1. By default, take the return 'k_ar_diff' in AnalysisCointegration.
#k_ar_diff=

#Cointegration Rank. By default, take the return 'coint_rank' in AnalysisCointegration.
#coint_rank=

###### Definition VECM(p) ##########################
#deterministicstr {"nc", "co", "ci", "lo", "li"}
#        "nc" - no deterministic terms
#        "co" - constant outside the cointegration relation
#        "ci" - constant within the cointegration relation
#        "lo" - linear trend outside the cointegration relation
#        "li" - linear trend within the cointegration relation
determ="colo"   

# nr step backward in sample to predict
m_pre=1

# accuracy to variance goal.
accuracy=0.03

pd_VarSim, pd_VecmSim, modelVAR_pre, modelVECM_pre, modelVAR_sim, modelVECM_sim, Vec_Model, Beta=Simulation( path_gral, Option, factors_df, Drivers, n_drivers, n_factors,  name_factors, all_drivers, ini_dat_str, fin_dat_str, k_ar_diff,coint_rank, m_pre, accuracy, determ, crit, reg)
#######################################################################

#######################################################################
############ Graphics ################################################# 
#######################################################################
s, stat=Graphics_Sim(n_factors, coint_rank, Vec_Model, Beta, modelVAR_sim, pd_VarSim, pd_VecmSim) 
#######################################################################