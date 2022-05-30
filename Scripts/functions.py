#Functions for the BOOM TephraDataSet exploration
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#--------------------- Function for CheckNormalizations notebook --------------------------------------

def renormalizing (data):

    data_renormalized = data.copy()
    data_renormalized = data_renormalized.reset_index()
    data = data.reset_index()
    data_renormalized['MnO'] = data_renormalized['MnO'].replace('-',-1).astype(float)
    data_renormalized['P2O5'] = data_renormalized['P2O5'].replace('-',-1).astype(float)
    data_renormalized['Cl'] = data_renormalized['Cl'].replace('-',-1).astype(float)

    #Defining some variables which we will plot later to understand the variability of the re normalized data 
    data_renormalized['MnO + P2O5 + Cl'] = 'default'
    data_renormalized['Analytical Total without LOI'] = 'default'

    for i in range(0,len(data_renormalized.Total)):
        sum_ = np.nansum([data_renormalized.SiO2[i],
                      data_renormalized.TiO2[i],
                      data_renormalized.Al2O3[i],
                      data_renormalized['FeOT'][i], #the samples tested have been analyzed by EMP, thus FeO corresponds to FeOT
                      data_renormalized.MgO[i],
                      data_renormalized.CaO[i],
                      data_renormalized.Na2O[i],
                      data_renormalized.K2O[i]])
    
        data_renormalized.loc[i,'MnO + P2O5 + Cl'] = np.nansum([data_renormalized.MnO[i],
                                                              data_renormalized.Cl[i],
                                                              data_renormalized.P2O5[i]])
        data_renormalized.loc[i,'Analytical Total without LOI'] = np.nansum([data_renormalized.Total[i],
                                                                           - data_renormalized.LOI[i]])
    
        for elemento in ['SiO2','TiO2','Al2O3','FeOT','MgO','CaO','Na2O','K2O']:
            data_renormalized.loc[i,elemento] = data_renormalized[elemento][i]*100/sum_

    return data_renormalized


#--------------------- Function for UncertaintyAndGeostandards notebook --------------------------------------

def estimating_accuracy(BOOM_geostandards,BOOM_geostandards_ref):
# Estimating Accuracy: Measured Average/ Certified Value for each analyzed element for each secondary standard
    MeasuredVsRef = pd.DataFrame(0, index = np.arange(len(BOOM_geostandards.StandardID)) ,columns = ['MeasurementRun','StandardID','SiO2','TiO2','Al2O3','MnO','MgO','Fe2O3T','FeOT','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U'])
    elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','Fe2O3T','FeOT','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','Pb','Th','U']
    
    # removing non numerical strings from data set 
    BOOM_geostandards = BOOM_geostandards.replace(np.nan,-1) 
    BOOM_geostandards = BOOM_geostandards.replace('<0.01',-1) 
    BOOM_geostandards = BOOM_geostandards.replace('<0.002',-1)
    BOOM_geostandards = BOOM_geostandards.replace('Over range',-1)
    BOOM_geostandards = BOOM_geostandards.replace('<5',-1)
    BOOM_geostandards = BOOM_geostandards.replace('> 10000',-1)
    
    # filtrating measurement runs where certified geostandards have been analyzed
    BOOM_geostandards_certified = BOOM_geostandards[BOOM_geostandards.StandardID.isin(BOOM_geostandards_ref[BOOM_geostandards_ref.error.isin(["95%CL","SD"])].StandardID.unique().tolist())].copy()

    for elemento in elementos:
        MeasuredVsRef[elemento] = MeasuredVsRef[elemento].astype('float64')
    
    i=0
    for run in BOOM_geostandards_certified.MeasurementRun.unique():
        #print(run)
        temp = BOOM_geostandards_certified[BOOM_geostandards_certified.MeasurementRun==run]
        for std in temp.StandardID.unique():
            #print(std)
            MeasuredVsRef.loc[i,'MeasurementRun'] = run
            MeasuredVsRef.loc[i,'StandardID'] = std
        
            temp1 = temp[temp.StandardID == std]
            index1 = temp1.first_valid_index()
            
            temp2 = BOOM_geostandards_ref[BOOM_geostandards_ref.StandardID == std]
            index2 = temp2.first_valid_index()
        
            for elemento in elementos:
                
                if (temp1[elemento][index1] != -1) & (temp2[elemento][index2] != -1):
                    #print(type(temp1[elemento][index1]))
                    #print(type(temp2[elemento][index2]))                
                    MeasuredVsRef.loc[i,elemento] = temp1[elemento][index1]/temp2[elemento][index2]
                    #print(type(Standards_Color[elemento][i]))
                    #print(temp1[elemento][index1]/temp2[elemento][index2])
            i=i+1            
             
    MeasuredVsRef = MeasuredVsRef.replace(-1,np.nan)
    MeasuredVsRef = MeasuredVsRef.replace(0,np.nan)
    MeasuredVsRef = MeasuredVsRef.dropna(subset = ['StandardID'],axis=0)
    MeasuredVsRef.loc[:,'MeasurementRun'] = MeasuredVsRef.loc[:,'MeasurementRun'].astype('str')
    
    return MeasuredVsRef

def simbología_std(std):
    simbología = pd.read_excel('../Data/Standards_Reference.xlsx')
    temp = simbología.loc[simbología['StandardID'] == std]
    coloR = temp.values[0,1]
    return coloR

def plot_accuracy_MeasurementRun(Accuracy_data,save=False,ymin=0.4,ymax=1.6):
# Plot the accuracy for all the elements analyzed for each Standards in each MeasurementRun

    # Here we choose which set of elements we want to analyze
    elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeOT','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
    #elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeOT','CaO','Na2O','K2O','P2O5','Cl']
    linea1 = np.empty(len(elementos))
    linea1.fill(1.05)
    linea2 = np.empty(len(elementos))
    linea2.fill(0.95)

    for run in Accuracy_data.MeasurementRun.unique():
        plt.figure(figsize=(12,5))
        ax = plt.axes()        
        temp = Accuracy_data[Accuracy_data.MeasurementRun==run]
        for std in temp.StandardID.unique():
            temp2 = temp[temp.StandardID == std]
            temp2 = temp2.reset_index(drop=True)
            index2 = temp2.first_valid_index()
            #print(index2)
            #print(temp2)
            Color = simbología_std(std)
            plt.plot(elementos, linea1,color = 'lightgrey')
            plt.plot(elementos, linea2,color = 'lightgrey')
            plt.plot(elementos, temp2[elementos].iloc[0,:],marker = 'o',color = Color,label = std)
        
        leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=14, title="Analyzed Standards")
        plt.ylim(ymin,ymax)
        ax.set_title('Measurement Run: '+ run ,fontsize=16)
        ax.tick_params(labelsize = 14,direction='in',axis='x',rotation=45)
        plt.ylabel("Measured/Certified", fontsize = 16)
        ax.grid(axis ='y')
        
        if save:
            plt.savefig('../Plots/Accuracy_'+run+'.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
        plt.show()

def plot_accuracy_BOOM(Accuracy_data,save=False,ymin=0.4,ymax=1.6):
# Plot the accuracy for all the elements analyzed for each Standards in each MeasurementRun grafico
    # Here we choose which set of elements we want to analyze
    elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeOT','CaO','Na2O','K2O','P2O5','Cl','Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
    #elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeOT','CaO','Na2O','K2O','P2O5']
    linea1 = np.empty(len(elementos))
    linea1.fill(1.05)
    linea2 = np.empty(len(elementos))
    linea2.fill(0.95)
    Accuracy_data = Accuracy_data.sort_values(by=['StandardID'])
    plt.figure(figsize=(12,5))
    ax = plt.axes()        

    for std in Accuracy_data.StandardID.unique():
        temp = Accuracy_data[Accuracy_data.StandardID==std]
        #print(std)
        #print(len(temp))
        Color = simbología_std(std)
        if len(temp.SiO2) > 3:
            for elemento in elementos:
                #print(elemento)
                temp2 = temp.dropna(axis = 'rows',subset=([elemento]))
                temp2 = temp2.reset_index(drop=True)
                index2 = temp2.first_valid_index()
                if temp2[elemento].notnull().sum()>1:
                    ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std(),temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5)
            ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std(),temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5,label = std +' (' + str(len(temp))+' MRs)')
        
        if len(temp.SiO2) <= 3:
            plt.plot(elementos, temp[elementos].iloc[0,:],marker = 'o',linestyle='None',ms=4,color = Color,label = std)    
    
    plt.plot(elementos, linea1,color = 'grey')
    plt.plot(elementos, linea2,color = 'grey')            
    leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1.1),ncol=1,fontsize=13, title="Analyzed Standards")
    plt.ylim(ymin,ymax)
    ax.tick_params(labelsize = 15,direction='in',axis='x',rotation=75)
    ax.tick_params(labelsize = 15,direction='in',axis='y')
    plt.ylabel("Analyzed/Certified: Accuracy", fontsize = 16)
    ax.grid(axis ='y')
    if save:
        plt.savefig('../Plots/AccuracyTDS.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
    plt.show()

def plot_RSD_MeasurementRun(TDB_standards,save=False,ymin=0,ymax=50):
###### Plot the presicion for all the elements analyzed for each Standards in each MeasurementRun
    #first filter the data for which n, SD and thus RSD have not been reported:
    TDB_standards = TDB_standards[TDB_standards.n != 'Not reported']
    
    elementos = [ 'SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'Cl', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd','Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
#    elementos = ['SiO2','TiO2','Al2O3','MnO','MgO','FeO*','CaO','Na2O','K2O','P2O5']
    elementos_RSD = ['RSD_SiO2','RSD_TiO2','RSD_Al2O3','RSD_FeOT','RSD_MnO','RSD_MgO','RSD_CaO','RSD_Na2O','RSD_K2O','RSD_P2O5','RSD_Cl','RSD_Rb','RSD_Sr','RSD_Y','RSD_Zr','RSD_Nb','RSD_Cs','RSD_Ba','RSD_La','RSD_Ce','RSD_Pr','RSD_Nd','RSD_Sm','RSD_Eu','RSD_Gd','RSD_Tb','RSD_Dy','RSD_Ho','RSD_Er','RSD_Tm','RSD_Yb','RSD_Lu','RSD_Hf','RSD_Ta','RSD_Pb','RSD_Th','RSD_U']
    
    linea1 = np.empty(len(elementos))
    linea1.fill(5)
    linea2 = np.empty(len(elementos))
    linea2.fill(10)

    for run in TDB_standards.MeasurementRun.unique():
        plt.figure(figsize=(12,5))
        ax = plt.axes()
        temp = TDB_standards[TDB_standards.MeasurementRun==run]
        for std in temp.StandardID.unique():
            temp2 = temp[temp.StandardID == std]
            temp2 = temp2.reset_index(drop=True)
            index2 = temp2.first_valid_index()
            Color = simbología_std(std)
            
            plt.plot(elementos, linea1,color = 'lightgrey')
            plt.plot(elementos, linea2,color = 'lightgrey')
            plt.plot(elementos, temp2[elementos_RSD].iloc[0,:],marker = 'o',color = Color,label = std)
        
        leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=12, title="Analyzed Standards")
        #plt.ylim(0,50)
        ax.set_title('Measurement Run: '+ run ,fontsize=16)
        ax.tick_params(labelsize = 13,direction='in',axis='x',rotation=45)
        ax.tick_params(labelsize = 13,direction='in',axis='y')
        plt.ylabel("RSD (%)", fontsize = 16)
        ax.grid(axis ='y')
    
        if save:
            plt.savefig('../Plots/RSD_'+run+'.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
        plt.show()
    
def plot_RSD_BOOM(TDB_standards,save=False,ymin=0,ymax=50):
# Plot the presición for all the elements analyzed for each Standards in each MeasurementRun

    #first filter the data for which n, SD and thus RSD have not been reported:
    TDB_standards = TDB_standards[TDB_standards.n != 'Not reported']
    
    # Here we choose which set of elements we want to analyze
    elementos = ['SiO2','TiO2','Al2O3','FeOT','MnO','MgO','CaO','Na2O','K2O','P2O5','Cl',
                 'Rb','Sr','Y','Zr','Nb','Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Pb', 'Th', 'U']
    elementos_RSD = ['RSD_SiO2','RSD_TiO2','RSD_Al2O3','RSD_FeOT','RSD_MnO','RSD_MgO','RSD_CaO','RSD_Na2O','RSD_K2O','RSD_P2O5',
                     'RSD_Cl','RSD_Rb','RSD_Sr','RSD_Y','RSD_Zr','RSD_Nb','RSD_Cs','RSD_Ba','RSD_La','RSD_Ce','RSD_Pr','RSD_Nd','RSD_Sm','RSD_Eu','RSD_Gd','RSD_Tb','RSD_Dy','RSD_Ho','RSD_Er','RSD_Tm','RSD_Yb','RSD_Lu','RSD_Hf','RSD_Ta','RSD_Pb','RSD_Th','RSD_U']
    linea1 = np.empty(len(elementos_RSD))
    linea1.fill(5)
    linea2 = np.empty(len(elementos_RSD))
    linea2.fill(10)
    TDB_standards = TDB_standards.sort_values(by=['StandardID'])
    plt.figure(figsize=(12,5)) 
    ax = plt.axes()        

    for std in TDB_standards.StandardID.unique():
        temp = TDB_standards[TDB_standards.StandardID==std]
        #print(std)
        #print(len(temp))
        Color = simbología_std(std)
        if len(temp.SiO2) > 3:
            for elemento in elementos_RSD:
                #print(elemento)
                temp2 = temp.dropna(axis = 'rows',subset=([elemento]))
                temp2 = temp2.reset_index(drop=True)
                index2 = temp2.first_valid_index()
                if temp2[elemento].notnull().sum()>1:
                    ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std() ,temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5)
            ax.vlines(elemento,temp2[elemento].mean()-temp2[elemento].std() ,temp2[elemento].mean()+temp2[elemento].std(),colors=Color,linewidth=3.5,label = std +' (' + str(len(temp))+' MRs)')        
                    
        if len(temp.SiO2) <= 3:
            plt.plot(elementos_RSD, temp[elementos_RSD].iloc[0,:],marker = 'o',linestyle='None',ms=4,color = Color,label = std)    
       
    plt.plot(elementos_RSD, linea1,color = 'grey')
    plt.plot(elementos_RSD, linea2,color = 'grey')            
    leg=plt.legend(fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=13, title="Analyzed Standards")
    plt.ylim(ymin,ymax)
    ax.tick_params(labelsize = 15,direction='in',axis='x',rotation=75)
    ax.tick_params(labelsize = 15,direction='in',axis='y')
    
    ax.set_xticklabels(elementos)
    plt.ylabel("RSD (%)", fontsize = 16)
    ax.grid(axis ='y')
    if save:
        plt.savefig('../Plots/RSD_TDS.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)
    plt.show()