import numpy as np
import pandas as pd
#import tasplot
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
import seaborn as sns
import matplotlib.patches as ptch
from matplotlib.ticker import AutoMinorLocator
#import cartopy.crs as ccrs
#import cartopy
from PIL import Image as PImage
import scipy.stats
import os
plt.style.use('seaborn-white')

def SampleIDD(Data):

    PorMientras_SID = Data['SampleID'].copy()
    Data = Data.fillna(-1)

    for i in range(0,np.size(Data['SampleID'])):
        if (Data['SampleObservationID'][i] == -1):
            #print("1 --  SampleObservationID: {}, SampleID: {}".format(Data['SampleObservationID'][i],Data['Sample ID'][i]))
            PorMientras_SID[i]=Data['SampleID'][i]
        else:
            #print("2 --  SampleObservationID: {}, SampleID: {}".format(Data['SampleObservationID'][i],Data['Sample ID'][i]))
            PorMientras_SID[i]=Data['SampleObservationID'][i] 
			
    Data['SampleObservationID']=PorMientras_SID
    Data=Data.replace(to_replace=-1, value=np.nan)
    return Data

def graficar_versus_core(AA,BB,Data,Data_cores='default',Xmin='default',Xmax='default',Ymin='default',Ymax='default',save=False,nombre='default'):

    plt.figure(figsize=(5,4))
    ax = plt.axes()
    
    #plot identifyed glass shards
    Data = Data.dropna(subset=[AA,BB])
    Data = Data.reset_index(drop=True)
	
#----------------------------- PLOT Data base ---------------------------------
    MarkerSize = 90; Alpha = 0.7
    for Volcano in Data.Volcano.unique():
        #print('Volcano {}'.format(Volcano))
        temp0 = Data[Data.Volcano == Volcano]
        #print(Volcano)
        for Event in temp0.Event.unique():
            #print(Event,temp0.Event.unique())
            temp = temp0[temp0.Event== Event]
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            Color, Marker  = simbologia(temp.Volcano[Index],temp.Event[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color,s=MarkerSize, marker = Marker, alpha=Alpha)# , label=Event
#---------------------------------------plot core data when given
    if isinstance(Data_cores,pd.DataFrame):   
        Data_cores = Data_cores.dropna(subset=[AA,BB])
        Data_cores = Data_cores.reset_index(drop=True)
        MarkerSize = 130; Alpha = 0.6
        
        for label in Data_cores.Label.unique():
            temp = Data_cores[Data_cores.Label == label]
            #print(Volcano)
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            #print('Label {} Core {} Depth {}'.format(label,temp.Core[Index],temp.Depth[Index]))
            Color, Marker  = simbologia_core(temp.Core[Index],temp.Depth[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color, marker = Marker,s=MarkerSize, edgecolors ='black',label=Data_cores.Depth[Index])#
			
             
    if (Xmax!='default')&(Xmin!='default'):
        plt.xlim(Xmin,Xmax)

    if (Ymin!='default')&(Ymax!='default'):
        plt.ylim(Ymin,Ymax)		
		
    plt.xlabel(AA + ' (wt %)', fontsize = 22)
    plt.ylabel(BB + ' (wt %)', fontsize = 22)
    #plt.xlabel("La/Yb", fontsize = 22)
    #plt.ylabel("Zr/Nb", fontsize = 22)
    #ax.set_xticks([0,5,10,15,20])
    #ax.set_yticks([10,20,30,40,50])
    #plt.xlabel(r"SiO$_{\rm 2}$", fontsize = 22)
    #plt.ylabel(r"K$_{\rm 2}$O", fontsize = 22)
    #ax.set_xticks([50,60,70,80])
    #ax.set_yticks([1,2,3,4])
    ax.tick_params(labelsize = 22,direction='in',axis='both')#,visible = True
	
    #ax.grid(axis ='x')
 #   if Data.Event.unique().size > 90:
 #       leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
 #   if (Data.Event.unique().size > 45)&(Data.Event.unique().size < 90):
 #       leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.8,1),ncol=2,fontsize=13)
    #if (Data.Event.unique().size < 45):
    leg=plt.legend(loc='lower right', fancybox=True, ncol=1,fontsize=15,bbox_to_anchor=(1,1))#bbox_to_anchor=(1,1)
    
    #leg.get_frame().set_alpha(1)
    
    if save:
        plt.savefig('../Plots/'+nombre+'.png',dpi = 300,bbox_inches='tight'
    )
    
    plt.show()
	
def graficar_versus_core_rock(AA,BB,Data,Data_cores='default',Xmin='default',Xmax='default',Ymin='default',Ymax='default',save=False,nombre='default'):

    plt.figure(figsize=(5,4))
    ax = plt.axes()
    
    #plot identifyed glass shards
    Data = Data.dropna(subset=[AA,BB])
    Data = Data.reset_index(drop=True)
    Data_glass = Data[(Data.TypeOfRegister=='Pyroclastic material')&(Data.TypeOfAnalysis=='Micro Analytical')]
    Data_bulk = Data[(Data.TypeOfRegister=='Pyroclastic material')&(Data.TypeOfAnalysis=='Bulk')]
    Data_lava = Data[(Data.TypeOfRegister=='Effusive material')]
#----------------------------- PLOT Data base ---------------------------------
    MarkerSize = 90; Alpha = 0.7
    for Volcano in Data.Volcano.unique():
        #print('Volcano {}'.format(Volcano))
        temp0_glass = Data_glass[Data_glass.Volcano == Volcano]
        temp0_bulk = Data_bulk[Data_bulk.Volcano == Volcano]
        temp0_lava = Data_lava[Data_lava.Volcano == Volcano]
        #print(Volcano)
        for Event in temp0_glass.Event.unique():
            #print(Event,temp0.Event.unique())
            temp_glass = temp0_glass[temp0_glass.Event== Event]
            A_glass = temp_glass[AA].values;B_glass = temp_glass[BB].values
            Index = temp_glass.first_valid_index()
            Color, Marker  = simbologia(temp_glass.Volcano[Index],temp_glass.Event[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A_glass,B_glass, color = Color,s=MarkerSize, marker = Marker, alpha=0.7, label='glass '+ Event)# 
			
			
        for Event in temp0_lava.Event.unique(): 
            temp_lava = temp0_lava[temp0_lava.Event== Event]
            Index = temp_lava.first_valid_index()
            A = temp_lava[AA].values;B = temp_lava[BB].values
            Color, Marker  = simbologia(temp_lava.Volcano[Index],temp_lava.Event[Index])
            plt.scatter(A,B, color = 'black',s=MarkerSize, marker = Marker,edgecolors= 'black', alpha= 0.5, label='effusive material '+ Event)# 

        for Event in temp0_bulk.Event.unique():			
            temp_bulk = temp0_bulk[temp0_bulk.Event== Event]
            A = temp_bulk[AA].values;B = temp_bulk[BB].values
            Index = temp_bulk.first_valid_index()
            Color, Marker  = simbologia(temp_bulk.Volcano[Index],temp_bulk.Event[Index])
            plt.scatter(A,B, color = Color,s=MarkerSize, marker = Marker,edgecolors= 'black', alpha=0.5, label='bulk tephra '+ Event)# 

#---------------------------------------plot core data when given
    if isinstance(Data_cores,pd.DataFrame):   
        Data_cores = Data_cores.dropna(subset=[AA,BB])
        Data_cores = Data_cores.reset_index(drop=True)
        MarkerSize = 130; Alpha = 0.6
        
        for label in Data_cores.Label.unique():

            temp = Data_cores[Data_cores.Label == label]
            #print(Volcano)
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            #print('Label {} Core {} Depth {}'.format(label,temp.Core[Index],temp.Depth[Index]))
            Color, Marker  = simbologia_core(temp.Core[Index],temp.Depth[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color, marker = Marker,s=MarkerSize, edgecolors ='black',label=Data_cores.Label[Index])#
			
             
    if (Xmax!='default')&(Xmin!='default'):
        plt.xlim(Xmin,Xmax)

    if (Ymin!='default')&(Ymax!='default'):
        plt.ylim(Ymin,Ymax)		
		
    plt.xlabel(AA + ' (wt %)', fontsize = 22)
    plt.ylabel(BB + ' (wt %)', fontsize = 22)
    #plt.xlabel("La/Yb", fontsize = 22)
    #plt.ylabel("Zr/Nb", fontsize = 22)
    #ax.set_xticks([0,5,10,15,20])
    #ax.set_yticks([10,20,30,40,50])
    #plt.xlabel(r"SiO$_{\rm 2}$", fontsize = 22)
    #plt.ylabel(r"K$_{\rm 2}$O", fontsize = 22)
    #ax.set_xticks([50,60,70,80])
    #ax.set_yticks([1,2,3,4])
    ax.tick_params(labelsize = 22,direction='in',axis='both')#,visible = True
	
    #ax.grid(axis ='x')
 #   if Data.Event.unique().size > 90:
 #       leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
 #   if (Data.Event.unique().size > 45)&(Data.Event.unique().size < 90):
 #       leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.8,1),ncol=2,fontsize=13)
    if (Data.Event.unique().size < 45):
      leg=plt.legend(loc='upper left', fancybox=True, ncol=1,fontsize=15,bbox_to_anchor=(1,1))#
    
    #leg.get_frame().set_alpha(1)
    
    if save:
        plt.savefig('../Plots/'+nombre+'.png',dpi = 300,bbox_inches='tight',bbox_to_anchor=(1,1)
    )
    
    plt.show()	
	
def graficar2(AA,BB,Data,Data_cores='default',Alpha=0.5,MarkerSize=110,Xmin='default',Xmax='default',Ymin='default',Ymax='default',save=False,nombre='default'):

    plt.figure(figsize=(6,5))
    ax = plt.axes()
    
    #plot identifyed glass shards
    Data = Data.dropna(subset=[AA,BB])
    Data = Data.reset_index(drop=True)

#----------------------------- PLOT Data base ---------------------------------
    
    for Volcano in Data.Volcano.unique():
        print('Volcano {}'.format(Volcano))
        temp0 = Data[Data.Volcano == Volcano]
        #print(Volcano)
        for Event in temp0.Event.unique():
            print(Event,temp0.Event.unique())
            temp = temp0[temp0.Event== Event]
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            Color, Marker  = simbologia(temp.Volcano[Index],temp.Event[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color,s=MarkerSize, marker = Marker, alpha=Alpha, label=Event)#
            
#---------------------------------------plot core data when given
    if isinstance(Data_cores,pd.DataFrame):   
        Data_cores = Data_cores.dropna(subset=[AA,BB])
        Data_cores = Data_cores.reset_index(drop=True)
        MarkerSize = 130
        
        for label in Data_cores.Label.unique():
            #print('Volcano {}'.format(Volcano))
            temp = Data_cores[Data_cores.Label == label]
            #print(Volcano)
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            Color, Marker  = simbologia_core(temp.Core[Index],temp.Depth[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color, marker = Marker,s=MarkerSize, edgecolors ='black', label=Data_cores.Depth[Index])#
            #print(temp.SiO2[Index])
    if (Xmax!='default')&(Xmin!='default'):
        plt.xlim(Xmin,Xmax)

    if (Ymin!='default')&(Ymax!='default'):
        plt.ylim(Ymin,Ymax)		
		
    plt.xlabel(AA, fontsize = 22)
    plt.ylabel(BB, fontsize = 22)
    #plt.xlabel("La/Yb", fontsize = 22)
    #plt.ylabel("Zr/Nb", fontsize = 22)
    #ax.set_xticks([0,5,10,15,20])
    #ax.set_yticks([10,20,30,40,50])
    #plt.xlabel(r"Al$_{\rm 2}$O$_{\rm 3}$", fontsize = 22)
    plt.ylabel(r"K$_{\rm 2}$O", fontsize = 22)
    #ax.set_xticks([50,60,70,80])
    #ax.set_yticks([1,2,3,4])
	
	#putting ticks if necesary
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.xaxis.set_minor_locator(AutoMinorLocator())

    #ax.tick_params(axis='both', which='major', labelsize=13, length=7, width=1.3)
    #ax.tick_params(which='minor', length=5)
    #ax.set_yticks(np.linspace(Ymin,Ymax,11))
    #ax.set_xticks(np.linspace(Xmin,Xmax,9))
	
    ax.tick_params(labelsize = 22,direction='in',axis='both')
    #ax.grid(axis ='x')
    #if Data.Event.unique().size > 90:
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
    #if (Data.Event.unique().size > 45)&(Data.Event.unique().size < 90):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.8,1),ncol=2,fontsize=13)
    #if (Data.Event.unique().size < 45):
    leg=plt.legend(loc='upper left', fancybox=True, ncol=1,fontsize=18)#bbox_to_anchor=(1,1)
    
    #leg.get_frame().set_alpha(1)
    
    if save:
        plt.savefig('../Plots/'+nombre+'.png',dpi = 300,bbox_inches='tight')
    
    plt.show()
	
def graficar_sections(AA,BB,Data,Data_cores='default',Alpha=0.5,MarkerSize=110,MarkeR='o',Xmin='default',Xmax='default',Ymin='default',Ymax='default',save=False,nombre='default'):

    plt.figure(figsize=(5,4))
    ax = plt.axes()
    
    #plot identifyed glass shards
    Data = Data.dropna(subset=[AA,BB])
    Data = Data.reset_index(drop=True) 
#---------------------------------------plot core data when given
    if isinstance(Data_cores,pd.DataFrame):   
        Data_cores = Data_cores.dropna(subset=[AA,BB])
        Data_cores = Data_cores.reset_index(drop=True)
        MarkerSize = 130
        
        for label in Data_cores.Label.unique():
            #print('Volcano {}'.format(Volcano))
            temp = Data_cores[Data_cores.Label == label]
            #print(Volcano)
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            Color, Marker  = simbologia_core(temp.Core[Index],temp.Depth[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color, marker = Marker,s=MarkerSize, edgecolors ='black',alpha=0.7,label=Data_cores.Label[Index])#
            #print(temp.SiO2[Index])
			
#----------------------------- PLOT Data base ---------------------------------
    
    for Volcano in Data.Volcano.unique():
        #print('Volcano {}'.format(Volcano))
        temp0 = Data[Data.Volcano == Volcano]
        #print(Volcano)
        for seccion in temp0.Seccion.unique():
            #print(Event,temp0.Event.unique())
            temp = temp0[temp0.Seccion == seccion]
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            Color, Marker  = simbologia(temp.Volcano[Index],temp.Event[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color,s=MarkerSize, marker = MarkeR,edgecolors ='black', alpha=0.8, label=Data.SubSeccion[Index])#
            			
    if (Xmax!='default')&(Xmin!='default'):
        plt.xlim(Xmin,Xmax)

    if (Ymin!='default')&(Ymax!='default'):
        plt.ylim(Ymin,Ymax)		
		
    plt.xlabel(AA, fontsize = 22)
    plt.ylabel(BB, fontsize = 22)
    #plt.xlabel("La/Yb", fontsize = 22)
    #plt.ylabel("Zr/Nb", fontsize = 22)
    #ax.set_xticks([0,5,10,15,20])
    #ax.set_yticks([10,20,30,40,50])
    plt.xlabel(r"Al$_{\rm 2}$O$_{\rm }$", fontsize = 22)
    plt.ylabel(r"K$_{\rm 2}$O", fontsize = 22)
    #ax.set_xticks([50,60,70,80])
    #ax.set_yticks([1,2,3,4])
	
	#putting ticks if necesary
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
    #ax.xaxis.set_minor_locator(AutoMinorLocator())

    #ax.tick_params(axis='both', which='major', labelsize=13, length=7, width=1.3)
    #ax.tick_params(which='minor', length=5)
    #ax.set_yticks(np.linspace(Ymin,Ymax,11))
    #ax.set_xticks(np.linspace(Xmin,Xmax,9))
	
    ax.tick_params(labelsize = 22,direction='in',axis='both')
    #ax.grid(axis ='x')
    #if Data.Event.unique().size > 90:
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
    #if (Data.Event.unique().size > 45)&(Data.Event.unique().size < 90):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.8,1),ncol=2,fontsize=13)
    if (Data.Event.unique().size < 45):
        leg=plt.legend(loc='lower right', fancybox=True, ncol=1,fontsize=14)#bbox_to_anchor=(1,1)
    
    #leg.get_frame().set_alpha(1)
    
    if save:
        plt.savefig('../Plots/'+nombre+'.png',dpi = 300,bbox_inches='tight')
    
    plt.show()
	
def simbologia(volcano,event):

    simbología = pd.read_excel('../Scripts/Simbologia.xlsx')
    Event = simbología.loc[simbología['Volcano'] == volcano]
    Event = Event.loc[Event['Event'] == event]
    coloR = Event.values[0,2]
    markeR = Event.values[0,3]
    return coloR, markeR

def simbologia_core(testigo,profundidad):

    simbología = pd.read_excel('../Scripts/SimbologiaTestigos.xlsx')
    Event = simbología.loc[simbología['Testigo'] == testigo]
    Event = Event.loc[Event['Profundidad'] == profundidad]
    Index = Event.first_valid_index()
    coloR = Event.Color[Index]
    markeR = Event.Simbología[Index]
    return coloR, markeR
	          	          	
def grafico_edades(Data,Data_cores ='default',save=False,nombre='default'):
    plt.figure(figsize=(2.5,9))
    ax = plt.axes()
    
    Data_magnitud = Data.copy()
    Data_magnitud.Magnitude = Data_magnitud.Magnitude.replace(np.nan,0)
    Data = Data.dropna(subset=['14C_Age'],axis=0,how='any')
    Data = Data.reset_index(drop=True)
    Data['14C_Age'] = Data['14C_Age'].values/1000
    Data['14C_Age_Error'] = Data['14C_Age_Error'].values/1000
    Data['Edad'] = Data['14C_Age']
    Data['ErrorEdad'] = Data['14C_Age_Error']
    
    Events = Data['Event'].values
    Volcanoes = Data['Volcano'].values
    Sample = Data['SampleID'].values
    i = 0
    j = 0
    k = 0
    #print(Events)
    while i < np.size(Events):

        #print("0 {}, {}".format(Events[i],Volcanoes[i]))
        Color, Marker  = simbologia(Volcanoes[i],Events[i])
        marker_edge = 'black'
        Data_Event = Data.loc[Data['Volcano'] == Volcanoes[i]]  
        Data_Event = Data_Event.sort_values(by=['Event'])	
        Data_Event = Data_Event.loc[Data_Event['Event'] == Events[i]]
        Data_magnitud_Event = Data_magnitud.loc[Data_magnitud['Volcano'] == Volcanoes[i]]
        Data_magnitud_Event = Data_magnitud_Event.loc[Data_magnitud_Event['Event'] == Events[i]]
        Magnitud = scipy.stats.mode(Data_magnitud_Event.Magnitude)
        Magnitud = Magnitud[0]
        
        if Data_Event.shape[0] != 1: #Events con más de una datación 14C
            
            #print("1 {}, {}, {}, {}".format(Events[i],Volcanoes[i],Data.SampleObservationID[i],Magnitud))
            if Events[i] != 'Unknown': #Es un Event con un nombre asignado 
                if Magnitud==0: #la magnitud del Event es desconocida
                    #print("2 {}, {}, {}, {}, {}".format(Events[i],Volcanoes[i],Data.SampleObservationID[i],Data_Event['Edad'].min(),Data_Event['Edad'].max()))
                    rect = ptch.Rectangle((-1.5,Data_Event['Edad'].min()),1.5,Data_Event['Edad'].max()-Data_Event['Edad'].min(), facecolor = Color,alpha=0.4,linewidth = 0.25)
                    simbolo = ax.plot(-2,Data_Event['Edad'].min() +(Data_Event['Edad'].max()-Data_Event['Edad'].min())/2, color = Color, marker = Marker, markersize=8,markeredgecolor=marker_edge, label=Events[i],alpha=0.8)
                    #ax.text(-4,Data_Event['Edad'].min() +(Data_Event['Edad'].max()-Data_Event['Edad'].min())/2+.07,Events[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
                    while j < Data_Event.shape[0]:
                        rect = ptch.Rectangle((-1.5,Data_Event.Edad.values[j]),1.5,Data_Event.ErrorEdad.values[j],facecolor = Color,edgecolor='black',linewidth = 0.25)
                        j=j+1
                        ax.add_patch(rect) 
                else: #la magnitud del Event es conocida
                    #print("3 {}, {}, {}, {}, {}".format(Events[i],Volcanoes[i],Data.SampleObservationID[i],Data_Event['Edad'].min(),Data_Event['Edad'].max()))
                    rect = ptch.Rectangle((0,Data_Event['Edad'].min()),Magnitud,Data_Event['Edad'].max()-Data_Event['Edad'].min(),facecolor = Color,alpha=0.4,linewidth = 0.25)
                    simbolo = ax.plot(Magnitud+.7,Data_Event['Edad'].min() +(Data_Event['Edad'].max()-Data_Event['Edad'].min())/2,color = Color, marker = Marker, markersize=10,markeredgecolor=marker_edge, label=Events[i],alpha=0.8)
                    #ax.text(Magnitud+1.6,Data_Event['Edad'].min() +(Data_Event['Edad'].max()-Data_Event['Edad'].min())/2+.07,Events[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
                    while j < Data_Event.shape[0]:
                        rect = ptch.Rectangle((0,Data_Event.Edad.values[j]),Magnitud,Data_Event.ErrorEdad.values[j],facecolor = Color,alpha=.7,edgecolor='black',linewidth = 0.25)
                        j=j+1
                        ax.add_patch(rect)
            else: #son varias dataciones de un Volcano pero no se sabe el Event
                 while j < Data_Event.shape[0]:
                    #print("4 {}, {}, {}".format(Events[i],Volcanoes[i],Data.SampleObservationID[i]))
                    rect = ptch.Rectangle((0,Data_Event.Edad.values[j]),Magnitud,Data_Event.ErrorEdad.values[j], facecolor = Color,linewidth = 0.25)
                    simbolo = ax.plot(0,Data_Event.Edad.values[j], color = Color, marker = Marker, markersize=8,markeredgecolor='black',alpha=0.7)
                    j=j+1
                    ax.add_patch(rect)
        else: #Events con una datación 14C
            #print("-0 {}".format(Events[i]))			
            if Events[i] != 'Unknown': #Es un Event con un nombre asignado 
                if Magnitud==0: #la magnitud del Event es desconocida
                    #print("5 {}, {}, {}".format(Events[i],Volcanoes[i],Data.SampleObservationID[i]))
                    simbolo = ax.plot(-2,Data_Event['Edad'].min(),color = Color, marker = Marker, markersize=7,markeredgecolor=marker_edge, alpha=0.8, label=Events[i])
                    rect = ptch.Rectangle((-1.5,Data_Event['Edad'].min()),1.5,Data_Event.ErrorEdad.values[j], facecolor = Color,edgecolor='black',linewidth = 0.25) #este rectangulo es para que el codig no reclame por el addpatch
                    #ax.text(-4,Data_Event['Edad'].min() +(Data_Event['Edad'].max()-Data_Event['Edad'].min())/2+.07,Events[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
                else: #la magnitud del Event es conocida
                    #print("6 {}, {}, {}".format(Events[i],Volcanoes[i],Data.SampleObservationID[i]))
                    rect = ptch.Rectangle((0,Data_Event['Edad'].min()), Magnitud, Data_Event.ErrorEdad.values[j], facecolor = Color,linewidth = 0.25)
                    simbolo = ax.plot(Magnitud+.7, Data_Event['Edad'].min(), color = Color, marker = Marker, markersize=10,markeredgecolor=marker_edge, alpha=0.8, label=Events[i])
                    #ax.text(Magnitud+1.6,Data_Event['Edad'].min()+.07,Events[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
            else: #una datación de un Event unknown
                    #print("7 {}, {}, {}".format(Events[i],Volcanoes[i],Data.SampleObservationID[i]))
                    rect = ptch.Rectangle((0,Data_Event.Edad.values[j]), 0.05 ,Data_Event.ErrorEdad.values[j], facecolor = Color,alpha=1,edgecolor='black',linewidth = 0.25)
                    simbolo = ax.plot(0,Data_Event.Edad.values[j], color = Color, marker = Marker, markersize=8,markeredgecolor='black',alpha=0.8)
                    ax.add_patch(rect)
                    
        #ax.add_patch(rect)
        #txt = ax.text(Data_Event['Edad'].min(),Magnitud[i]+1.5 ,Event[i],weight='bold',color='black')
        #txt.set_rotation(45)
        i = i+ Data_Event.shape[0]
        j = 0
        #print(i)		
    #print(Data_cores)
	
    if isinstance(Data_cores, pd.DataFrame):
        Data_cores = Data_cores.dropna(axis = 'rows',subset=(['Edad']))
        Data_cores = Data_cores.reset_index(drop=True)
        Data_cores.Edad = Data_cores.Edad.values/1000
        Data_cores.ErrorEdad = Data_cores.ErrorEdad.values/1000
        Depth = Data_cores['Depth']
        Core = Data_cores['Core']	
    
        while k < (len(Data_cores)):
            Color, Marker  = simbologia_core(Core[k],Depth[k])
            simbolo = ax.plot( 0,Data_cores.Edad.values[k], color = Color, marker = Marker, markersize=12,markeredgecolor='black',alpha=1,label=Data_cores.Label[k])
            #txt = ax.text(0.1,Data_cores.Edad.values[k]-0.1,Data_cores.Depth.values[k],rotation=90,weight='bold',color='black',fontsize=17)
            k = k+1    
    
    #tags de qué representa cada línea
    #conMagnitud = ax.text(18000,4,'Events con Magnitud estimada',weight='bold',color='black',fontsize=15)
    #sinMagnitud = ax.text(18000,-2,'Event sin Magnitud estimada',weight='bold',color='black',fontsize=15)
    #unknown = ax.text(18000,-4,'Event Unknown',weight='bold',color='black',fontsize=15)
    
        
    ax.set_ylabel('14C age (kyears BP)', fontsize=12)
    ax.set_xlabel('Magnitud', fontsize=12)
    ax.invert_yaxis()
    #ax.set_xticklabels((np.linspace(0,15,6),0))
    ax.tick_params(labelsize = 12)

    #if Data.Event.unique().size > 60:
    #    leg=ax.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.7,1),ncol=3,fontsize=19)
        
    #if (Data.Event.unique().size > 20)&(Data.Event.unique().size < 60):
    #    leg=ax.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.45,1),ncol=2,fontsize=19)
        
    #if (Data.Event.unique().size < 20):
    #    leg=ax.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.27,1),ncol=1,fontsize=19)
     
    ax.legend(loc='lower left', fancybox=True, bbox_to_anchor=(1,0),ncol=2,fontsize=10)	 
    #leg.get_frame().set_alpha(0.5)
    plt.yticks(np.linspace(0,16,17))
    plt.xticks(np.linspace(0,6,7))
    ax.grid(axis ='y')
    #ax[0].set_yscale("log")
    ax.set_xlim(-5,9)
        
    if save:
        plt.savefig('../Plots/' + nombre +'.pdf',dpi = 300,bbox_inches='tight')
    
    plt.show()

SVZ="../Scripts/Images/ZVS.png" 
SVZ= PImage.open(SVZ)
AVZ="../Scripts/Images/AVZ2.jpg" 
AVZ= PImage.open(AVZ)
Ambos="../Scripts/Images/Ambos2.jpg" 
Ambos= PImage.open(Ambos)

def grafico_posicion(Datas,zona,VolcanoES='default', texto='no',save=False,nombre='default'):


    if zona == 'Ambos':
        MarkerSize = 6
    else:
        MarkerSize = 8
		
    plt.figure(figsize=(10,10))
    ax = plt.axes(projection = ccrs.PlateCarree())
    Datas = Datas.dropna(how='any',subset=['Latitud']) 
    Datas = Datas.reset_index(drop=True)
    Volcanoes = Datas.Volcano
    Events = Datas.Event
    Lon =  Datas.Longitud
    Lat =  Datas.Latitud
    
    for Volcano in Datas.Volcano.unique():
        #print('Volcano {}'.format(Volcano))
        temp0 = Datas[Datas.Volcano == Volcano]
        #print(Volcano)
        for Event in temp0.Event.unique():
            #print(Event,temp0.Event.unique())
            temp = temp0[temp0.Event== Event]
            for seccion in temp.Latitud.unique():
                temp2 = temp[temp.Latitud == seccion]
                Index = temp2.first_valid_index()
                #print(Datas.SampleObservationID[Index])
                x,y = (temp2.Longitud[Index],temp2.Latitud[Index])
                Color, Marker  = simbologia(temp2.Volcano[Index],temp2.Event[Index])
                if Event == 'Unknown':
                    #print("1 {}, {}, i: {}".format(Event,Volcano,seccion))
                    marker_edge = 'white'
                else:
                    #print("2 {}, {}, i: {}".format(Event,Volcano,seccion))
                    marker_edge = 'black'
                plt.plot(x,y, color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize, alpha=.8)#,markeredgecolor='black'
            
            x,y = (temp2.Longitud[Index],temp2.Latitud[Index])
            Color, Marker  = simbologia(temp2.Volcano[Index],temp2.Event[Index])
            #print('Event {}'.format(Event))
            plt.plot(x,y, color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize, alpha=1,label=Event)#,markeredgecolor='black'
        
            
        x,y = (temp0.Longitud[Index],temp0.Latitud[Index])
        Color, Marker  = simbologia(temp0.Volcano[Index],temp0.Event[Index])
        #print('Volcano {}'.format(Volcano))
        #plt.plot(x,y, color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor=marker_edge, alpha=0.6,label=Volcano)
                		
    #plt.plot(Lon[np.size(Events)-1],Lat[np.size(Events)-1], color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor=marker_edge, alpha=0.6])
	
    if zona == 'Ambos':
        MarkerSize = 18
        FontSize=10
    else:
        MarkerSize = 28
        FontSize=24
		
    VolcanoES = VolcanoES.reset_index(drop=True)	
    for i in range(0,np.size(VolcanoES.Volcano)):
        Color, Marker  = simbologia(VolcanoES.Volcano[i],'Unknown')
        x,y = (VolcanoES.Longitud[i],VolcanoES.Latitud[i])
        if (VolcanoES.Volcano[i] == 'MD07-3098')|(VolcanoES.Volcano[i] == 'MD07-3100')|(VolcanoES.Volcano[i] == 'MD07-3081')|(VolcanoES.Volcano[i] == 'MD07-3082')|(VolcanoES.Volcano[i] == 'MD07-3088')|(VolcanoES.Volcano[i] == 'MD07-3119'):
            #print(VolcanoES.Volcano[i])
            MarkerSize = 15			
            plt.plot(x,y, color = Color, marker = 'o', transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor='black',alpha=0.7)#,markeredgewidth=1 + i/10
        else:
            #print(VolcanoES.Volcano[i])
            plt.plot(x,y, color = Color, marker = '^', transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor='black',alpha=0.7)#,markeredgewidth=1 + i/10
        
        if texto == 'sí':
            ax.text(x+ 1.2, y+0.22,VolcanoES.Volcano[i], transform=ccrs.PlateCarree(),weight='bold',color='black',fontsize=11)
        
		
    if zona == 'SVZ':
	    #extent : scalars (left, right, bottom, top), optional
        ax.imshow(SVZ,extent=[-77.0304,-68.0880,-46.9696,-37.8877],origin='upper', transform = ccrs.PlateCarree(),alpha=0.55)    
        ax.set_extent([-77.0304,-68.0880,-46.9696,-37.8877],crs = ccrs.PlateCarree())
                        
    if zona == 'AVZ':
        ax.imshow(AVZ,extent=[-76.5000,-64.7930,-56.1270,-45.0176],origin='upper', transform = ccrs.PlateCarree(),alpha=0.55)
		
        ax.set_extent([-76.5000,-64.7930,-56.1270,-45.0176],crs = ccrs.PlateCarree())
    if zona == 'Ambos':
        ax.imshow(Ambos,extent=[-77.2699,-64.8598,-56.3484,-37.6805],origin='upper', transform = ccrs.PlateCarree(),alpha=0.55)    
        ax.set_extent([-77.2699,-64.8598,-56.3484,-37.6805],crs = ccrs.PlateCarree())		

	
    ax.coastlines(resolution='10m')
    plt.xlabel('Lon')
    plt.ylabel('Lat')
    #plt.title('Identified Tephras Not identified samples',fontsize=25)
    #if Datas.Event.unique().size > 90:
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.83,1),ncol=3,fontsize=13)
    #if (Datas.Event.unique().size > 45)&(Datas.Event.unique().size < 90):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.5,1),ncol=2,fontsize=13)
    #if (Datas.Event.unique().size < 45):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.6,1),ncol=1,fontsize=15)
		
    #leg.get_frame().set_alpha(0.5)
    
    if save:
        if zona == 'SVZ':
            plt.savefig('../Plots/'+nombre+'.pdf',dpi = 300,bbox_inches='tight')#bbox_extra_artists=(leg,)
        if zona == 'AVZ':
            plt.savefig('../Plots/'+nombre+'.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)        
        if zona == 'Ambos':
            plt.savefig('../Plots/'+nombre+'.pdf',dpi = 300,bbox_inches='tight')    
		
    plt.show()
def grafico_posicion_section(Datas,zona,VolcanoES='default', texto='no',save=False,nombre='default',MarkerSize=9):

    plt.figure(figsize=(10,10))
    ax = plt.axes(projection = ccrs.PlateCarree())
    Datas = Datas.dropna(how='any',subset=['Latitud']) 
    Datas = Datas.reset_index(drop=True)
    Volcanoes = Datas.Volcano
    Events = Datas.Event
    Lon =  Datas.Longitud
    Lat =  Datas.Latitud
    markeR = itertools.cycle(('o','s','v','p','d','<','^','X','+','*','D','x'))

    for Volcano in Datas.Volcano.unique():
        #print('Volcano {}'.format(Volcano))
        temp0 = Datas[Datas.Volcano == Volcano]
        #print(Volcano)
        for seccion in temp0.Seccion.unique():
            #print(Event,temp0.Event.unique())
            temp = temp0[temp0.Seccion== seccion]
            Index = temp.first_valid_index()
            #print(Datas.SampleObservationID[Index])
            x,y = (temp.Longitud[Index],temp.Latitud[Index])
            Color, Marker  = simbologia(temp.Volcano[Index],'Unknown')
            plt.plot(x,y, color = Color, marker = next(markeR), transform = ccrs.PlateCarree(),markersize=MarkerSize, alpha=.8,label=seccion,markeredgecolor='black')#,markeredgecolor='black'
 	
    if zona == 'Ambos':
        MarkerSize = 18
        FontSize=10
    else:
        MarkerSize = 28
        FontSize=24
		
    VolcanoES = VolcanoES.reset_index(drop=True)	
    for i in range(0,np.size(VolcanoES.Volcano)):
        Color, Marker  = simbologia(VolcanoES.Volcano[i],'Unknown')
        x,y = (VolcanoES.Longitud[i],VolcanoES.Latitud[i])
        if (VolcanoES.Volcano[i] == 'MD07-3098')|(VolcanoES.Volcano[i] == 'MD07-3100')|(VolcanoES.Volcano[i] == 'MD07-3081')|(VolcanoES.Volcano[i] == 'MD07-3082')|(VolcanoES.Volcano[i] == 'MD07-3088')|(VolcanoES.Volcano[i] == 'MD07-3119'):
            #print(VolcanoES.Volcano[i])
            MarkerSize = 15			
            plt.plot(x,y, color = Color, marker = 's', transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor='black',alpha=0.7)#,markeredgewidth=1 + i/10
        else:
            #print(VolcanoES.Volcano[i])
            plt.plot(x,y, color = Color, marker = '^', transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor='black',alpha=0.7)#,markeredgewidth=1 + i/10
        
        if texto == 'sí':
            ax.text(x+ 1.2, y+0.22,VolcanoES.Volcano[i], transform=ccrs.PlateCarree(),weight='bold',color='black',fontsize=11)
        
    if zona == 'SVZ':
	    #extent : scalars (left, right, bottom, top), optional
        ax.imshow(SVZ,extent=[-77.0304,-68.0880,-46.9696,-37.8877],origin='upper', transform = ccrs.PlateCarree(),alpha=0.55)    
        ax.set_extent([-77.0304,-68.0880,-46.9696,-37.8877],crs = ccrs.PlateCarree())
                        
    if zona == 'AVZ':
        ax.imshow(AVZ,extent=[-76.5000,-64.7930,-56.1270,-45.0176],origin='upper', transform = ccrs.PlateCarree(),alpha=0.55)
		
        ax.set_extent([-76.5000,-64.7930,-56.1270,-45.0176],crs = ccrs.PlateCarree())
    if zona == 'Ambos':
        ax.imshow(Ambos,extent=[-77.2699,-64.8598,-56.3484,-37.6805],origin='upper', transform = ccrs.PlateCarree(),alpha=0.55)    
        ax.set_extent([-77.2699,-64.8598,-56.3484,-37.6805],crs = ccrs.PlateCarree())		

	
    ax.coastlines(resolution='10m')
    plt.xlabel('Lon')
    plt.ylabel('Lat')
    #plt.title('Identified Tephras Not identified samples',fontsize=25)
    #if Datas.Event.unique().size > 90:
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.83,1),ncol=3,fontsize=13)
    #if (Datas.Event.unique().size > 45)&(Datas.Event.unique().size < 90):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.5,1),ncol=2,fontsize=13)
    #if (Datas.Event.unique().size < 45):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.6,1),ncol=1,fontsize=15)
		
    #leg.get_frame().set_alpha(0.5)
    
    if save:
        if zona == 'SVZ':
            plt.savefig('../Plots/'+nombre+'.pdf',dpi = 300,bbox_inches='tight')#bbox_extra_artists=(leg,)
        if zona == 'AVZ':
            plt.savefig('../Plots/'+nombre+'.pdf',dpi = 300,bbox_inches='tight')#,bbox_extra_artists=(leg,)        
        if zona == 'Ambos':
            plt.savefig('../Plots/'+nombre+'.pdf',dpi = 300,bbox_inches='tight')    
		
    plt.show()	
def TAS(Data,Data_cores='default'):
       
    plt.figure(figsize=(13,13))
    ax = plt.axes()
    #plot identifyed glass shards
    Data = Data.dropna(subset=['Na2O','K2O','SiO2'])
    Data = Data.reset_index(drop=True)
    
    Na2O = Data['Na2O'].values
    K2O = Data['K2O'].values
    SiO2 = Data['SiO2'].values
    
    Volcanoes = Data.Volcanoo.values
    Events = Data.Event.values
    Volcano = Volcanoes[0]
    Event = Events[1]
    
    for i in range(0,np.size(Events)):
        if Event == Events[i]:
            Color, Marker  = simbologia(Volcanoes[i],Events[i])
            plt.plot(SiO2[i], Na2O[i]+K2O[i], color = Color, marker = Marker,markersize=12, alpha=0.4,markeredgecolor = 'black')
        else:
            if Volcanoes[i]==Volcano:
                #print(Volcanoes[i],Events[i])
                Color, Marker  = simbologia(Volcanoes[i],Events[i])
                plt.plot(SiO2[i], Na2O[i]+K2O[i], color = Color, marker = Marker, markersize = 12,alpha=0.4, label = Events[i] , markeredgecolor = 'black')
                Event = Events[i]

            else:
                Color, Marker  = simbologia(Volcanoes[i],Events[i])
                plt.plot(SiO2[i], Na2O[i]+K2O[i], color = Color, marker = Marker,markersize=12,alpha=0.4,label=Events[i] , markeredgecolor = 'black')
                Event = Events[i]
                Volcano = Volcanoes[i]
                
#---------------------------------------plot core data when given
    if isinstance(Data_cores,pd.DataFrame):   
        Depth = Data_cores['Depth'].values
        Na2O = Data_cores['Na2O'].values
        K2O = Data_cores['K2O'].values
        SiO2 = Data_cores['SiO2'].values
        Core = Data_cores['Core'].values
        Depth0 = 0
        for i in range(0,np.size(Depth)-1):
            Color, Marker  = simbologia_core(Core[i],Depth[i])
            if Depth[i] == Depth0:
                plt.plot(SiO2[i], Na2O[i]+K2O[i],marker = Marker, color = Color, markersize=13,alpha=1, markeredgecolor = 'black')
            else:
                plt.plot(SiO2[i], Na2O[i]+K2O[i],marker = Marker, color = Color, markersize=13,alpha=1, markeredgecolor = 'black', label = Data_cores.Depth[i])
                Depth0 = Depth[i]                

    #plt.xlim(45,85)
    #plt.ylim(0,5)
    tasplot.add_LeMaitre_fields(ax)
    plt.xlabel('SiO2', fontsize = 22)
    plt.ylabel('Na2O + K2O', fontsize = 22)
    ax.tick_params(labelsize = 21)
    #if Data.Event.unique().size > 90:
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
    #if (Data.Event.unique().size > 45)&(Data.Event.unique().size < 90):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.5,1),ncol=2,fontsize=13)
    #if (Data.Event.unique().size < 45):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=15)
    #leg.get_frame().set_alpha(1)
    plt.savefig('../Plots/TAS.pdf',dpi = 300,bbox_extra_artists=(leg,),bbox_inches='tight')
    plt.show()
	
def MissingMechanism(DATA):
#---------------------------
#   This function assignes a label to each sample which indicates what kind of measurement was made, the aim of this function is to provide information to discriminatee 
#       the missing mechanisms of the data, according by the definition by Rubin (1976)

    Data = SampleIDD(DATA)
    Data['MissingMechanism'] = 'default'
    Data = Data.replace(np.nan,-1)
    
    for i in range(0, np.size(Data['SampleID'])):
        if (Data['SiO2'][i]!=-1) & ((Data['Sr'][i]==-1) | (Data['La'][i]==-1) | (Data['Nb'][i]==-1)):
            Data['MissingMechanism'][i] = 'Mayores'
        if (Data['SiO2'][i]==-1)&((Data['Sr'][i]!=-1) | (Data['La'][i]!=-1) | (Data['Nb'][i]!=-1)):
            Data['MissingMechanism'][i] = 'Traza'
        if (Data['SiO2'][i]!=-1) & ((Data['Sr'][i]!=-1) | (Data['La'][i]!=-1) | (Data['Nb'][i]!=-1) | (Data['Sm'][i]!=-1)):
            Data['MissingMechanism'][i] = 'Ambos'
        if (Data['SiO2'][i]==-1) & (Data['Sr'][i]==-1) & (Data['La'][i]==-1) & (Data['Nb'][i]==-1)& (Data['Sm'][i]==-1):
            Data['MissingMechanism'][i] = 'Edad'

    Data = Data.replace(-1,np.nan)
    return Data
	
def Histogramas(Data_temp):

	for elemento in Data_temp.columns:
		if (elemento != 'Volcano')&(elemento != 'Event'):  
			print(" ")
			print('\033[1m'+ elemento + '\033[0m')
			for i in Data_temp.Volcano.unique():
				print(i)
				coloR, Marker  = simbologia(i,'Unknown')
				Data_Volcano =Data_temp[Data_temp['Volcano']==i]
				if pd.isnull(Data_Volcano[elemento]).all():
					print('Volcán without information {}'.format(i))              
				else:
					plt.hist(Data_Volcano[elemento] ,label= i, color = coloR, alpha= 0.3,range=(np.nanmin(Data_Volcano[elemento]), np.nanmax(Data_Volcano[elemento])))  
    
			plt.gca().set(xlabel=elemento,ylabel='Frequency')
			plt.legend(bbox_to_anchor=(1,1),ncol=2)
			plt.show()
			
def Colores(Y,df):
    Dpal = {}
    for i, ID in enumerate(np.unique(Y)):
        volcan = df.Volcano.cat.categories[ID]
        #print(volcan)
        color, marker = simbologia(volcan,'Unknown')
        Dpal[volcan] = color
    return Dpal

def graficar_imputing(est,X,X_imp,df,y,A,B):
    dpal = Colores(y,df)
    fig, axes = plt.subplots(1, 2, figsize=(15,5),sharex=True,sharey=True)
    sns.scatterplot(X.loc[:, A], X.loc[:, B],hue=df.Volcano.cat.categories[y], alpha=0.7, palette=dpal, ax=axes[0])
    axes[0].set_title("Original data",fontsize=14)
    axes[0].legend(loc='center left', bbox_to_anchor=(0, -0.4), ncol=3)
    #axes[0].set_xlim([40,80]);axes[0].set_ylim([0,5])
    #sns.scatterplot(X.loc[:, A], X.loc[:, B],
    #            hue=df.Volcan.cat.categories[y], alpha=0.7, palette=dpal, ax=axes[1])
    sns.scatterplot(X_imp.loc[:, A], X_imp.loc[:, B],alpha=0.7, hue=df.Volcano.cat.categories[y], ax=axes[1], palette=dpal,s=30)
    #sns.scatterplot(X_imp.loc[:, A], X_imp.loc[:, B],
    #            alpha=0.7, ax=axes[1], marker='x', color='k',s=30)
    axes[1].set_title(est,fontsize=14)
    axes[1].legend(loc='center left', bbox_to_anchor=(0, -0.6), ncol=3)
    #axes[1].set_xlim([40,80]);axes[0].set_ylim([0,5])
    fig.show()