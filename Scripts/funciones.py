import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import itertools
import matplotlib.patches as ptch
import cartopy.crs as ccrs
import cartopy
from PIL import Image as PImage
import scipy.stats
import os
plt.style.use('seaborn-white')

def SampleIDD(Data):

    PorMientras_SID = Data['SampleID'].copy()
    Data = Data.fillna(-1)

    for i in range(0,np.size(Data['SampleID'])):
        if (Data['SamplePoint'][i] == -1):
            #print("1 --  SamplePoint: {}, SampleID: {}".format(Data['SamplePoint'][i],Data['Sample ID'][i]))
            PorMientras_SID[i]=Data['SampleID'][i]
        else:
            #print("2 --  SamplePoint: {}, SampleID: {}".format(Data['SamplePoint'][i],Data['Sample ID'][i]))
            PorMientras_SID[i]=Data['SamplePoint'][i] 
			
    Data['SamplePoint']=PorMientras_SID
    Data=Data.replace(to_replace=-1, value=np.nan)
    return Data

def graficar_versus_core(AA,BB,Data,Data_cores='default',Xmin='default',Xmax='default',Ymin='default',Ymax='default',save=False):

    plt.figure(figsize=(7,7))
    ax = plt.axes()
    
    #plot identifyed glass shards
    Data = Data.dropna(subset=[AA,BB])
    Data = Data.reset_index(drop=True)
    MarkerSize = 11; Alpha = 0.5

#----------------------------- PLOT Data base ---------------------------------
    
    for volcan in Data.Volcan.unique():
        #print('Volcan {}'.format(volcan))
        temp0 = Data[Data.Volcan == volcan]
        #print(volcan)
        for evento in temp0.Evento.unique():
            #print(evento,temp0.Evento.unique())
            temp = temp0[temp0.Evento== evento]
            A = temp[AA].values
            B = temp[BB].values
            Index = temp.first_valid_index()
            Color, Marker  = simbologia(temp.Volcan[Index],temp.Evento[Index])
            #print(A); print(B);print(Marker, Color)
            plt.scatter(A,B, color = Color, marker = Marker, alpha=Alpha,label=evento)
        
#---------------------------------------plot core data when given
    if isinstance(Data_cores,pd.DataFrame):   
        Data_cores = Data_cores.dropna(subset=[AA,BB])
        Data_cores = Data_cores.reset_index(drop=True)
        Depth = Data_cores['Depth']
        A = Data_cores[AA].values
        B = Data_cores[BB].values
        Core = Data_cores['Core']
        Depth0 = 0
        Core0 = 0
        markers = itertools.cycle(('o','o','v','^','p','<','h','o','v','s','^','p','<','h','>','h','>','X','D','d', 'o','v','s','^','p','d','>','X','D',))
        colors = itertools.cycle(('black','white'))
        Color = next(colors)
        Marker = next(markers)

        for i in range(0,np.size(Depth)):
            if (Depth[i] == Depth0)&(Core[i] == Core0):
                Color, Marker  = simbologia_core(Core[i],Depth[i])
                plt.scatter(A[i], B[i],marker = Marker, color = Color, markersize=20, markeredgecolor = 'black', alpha=1)

            else:
                Color, Marker  = simbologia_core(Core[i],Depth[i])
                plt.scatter(A[i], B[i],marker = Marker, color = Color, markersize=20,label=Core[i]+ ' '+ Depth[i],markeredgecolor = 'black', alpha=1)
                Depth0 = Depth[i]
                Core0 = Core[i]
	#    if i == (np.size(Depth)-1):
	#	    print(i)
	#        plt.plot(A[i], B[i],marker = Marker, color = Color, markersize=13,label=Core[i]+ ''+ Depth[i],markeredgecolor = 'black', alpha=1)
            
    if (Xmax!='default')&(Xmin!='default'):
        plt.xlim(Xmin,Xmax)

    if (Ymin!='default')&(Ymax!='default'):
        plt.ylim(Ymin,Ymax)		
		
    plt.xlabel(AA, fontsize = 22)
    plt.ylabel(BB, fontsize = 22)
    ax.tick_params(labelsize = 22,axis='both', which='major')
    #if Data.Evento.unique().size > 90:
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
    #if (Data.Evento.unique().size > 45)&(Data.Evento.unique().size < 90):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.8,1),ncol=2,fontsize=13)
    #if (Data.Evento.unique().size < 45):
    #    leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.62,1),ncol=1,fontsize=15)
    
    #leg.get_frame().set_alpha(1)
    
    if save:
        plt.savefig('../Plots/'+AA+'vs'+BB+'.png',dpi = 300,bbox_inches='tight')
    
    plt.show()

def graficar5x5(AA,BB,Data,Data_cores='default',Xmin='default',Xmax='default',Ymin='default',Ymax='default',save=False):

    plt.figure(figsize=(6,6))
    ax = plt.axes()
    
    #plot identifyed glass shards
    Data = Data.dropna(subset=[AA,BB])
    Data = Data.reset_index(drop=True)
    Datas = Data.values
    A = Data[AA].values
    B = Data[BB].values
    Evento = 0
    Volcan = 0
    Eventos = Data['Evento'].values
    Volcanes = Data['Volcan'].values    
    
    for i in range(0,np.size(Datas[:,1])):
	
        if Evento == Eventos[i]:
            if Volcanes[i] == Volcan:
                #print("1 Volcan {}, Core {} ".format(Volcanes[i],Eventos[i]))
                Color, Marker  = simbologia(Volcanes[i],Eventos[i])
                plt.plot(A[i],B[i], color = Color, marker = Marker,markersize=12, alpha=0.7,markeredgecolor = Color)
            else:
                Color, Marker  = simbologia(Volcanes[i],Eventos[i])
                plt.plot(A[i],B[i], color = Color, marker = Marker, markersize = 12,alpha=0.7, label = Eventos[i], markeredgecolor = Color)
                Volcan = Volcanes[i]

        else:
            #print("2 Volcan {}, Evento {} ".format(Volcanes[i],Eventos[i]))
            Color, Marker  = simbologia(Volcanes[i],Eventos[i])
            plt.plot(A[i],B[i], color = Color, marker = Marker, markersize = 12,alpha=0.7, label = Eventos[i], markeredgecolor = Color)
            Evento = Eventos[i]
            Volcan = Volcanes[i]
                           
   #---------------------------------------plot core data when given
    if isinstance(Data_cores,pd.DataFrame):   
        Data_cores = Data_cores.dropna(subset=[AA,BB])
        Data_cores = Data_cores.reset_index(drop=True)
        Depth = Data_cores['Depth']
        A = Data_cores[AA].values
        B = Data_cores[BB].values
        Core = Data_cores['Core']
        Depth0 = 0
        Core0 = 0
        markers = itertools.cycle(('o','o','v','^','p','<','h','o','v','s','^','p','<','h','>','h','>','X','D','d', 'o','v','s','^','p','d','>','X','D',))
        colors = itertools.cycle(('black','white'))
        Color = next(colors)
        Marker = next(markers)
	
        for i in range(0,np.size(Depth)):
            if (Depth[i] == Depth0)&(Core[i] == Core0):
                Color, Marker  = simbologia_core(Core[i],Depth[i])
                plt.plot(A[i], B[i],marker = Marker, color = Color, markersize=13, markeredgecolor = 'black', alpha=1)

            else:
                Color, Marker  = simbologia_core(Core[i],Depth[i])
                plt.plot(A[i], B[i],marker = Marker, color = Color, markersize=13,label=Core[i]+ ' '+ Depth[i],markeredgecolor = 'black', alpha=1)
                Depth0 = Depth[i]
                Core0 = Core[i]
	#    if i == (np.size(Depth)-1):
	#	    print(i)
	#        plt.plot(A[i], B[i],marker = Marker, color = Color, markersize=13,label=Core[i]+ ''+ Depth[i],markeredgecolor = 'black', alpha=1)
            
    if (Xmax!='default')&(Xmin!='default'):
        plt.xlim(Xmin,Xmax)

    if (Ymin!='default')&(Ymax!='default'):
        plt.ylim(Ymin,Ymax)		
		
    plt.xlabel(AA, fontsize = 20)
    plt.ylabel(BB, fontsize = 20)
    ax.tick_params(labelsize = 20)

    if Data.Evento.unique().size > 90:
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
    if (Data.Evento.unique().size > 45)&(Data.Evento.unique().size < 90):
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.8,1),ncol=2,fontsize=13)
    if (Data.Evento.unique().size < 45):
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.8,1),ncol=1,fontsize=15)
    
    leg.get_frame().set_alpha(1)
    
    if save:
        plt.savefig('../Plots/'+AA+'vs'+BB+'.png',dpi = 300,bbox_extra_artists=(leg,),bbox_inches='tight')
    
    plt.show()

def simbologia(volcan,evento):

    simbología = pd.read_excel('../Scripts/Simbologia.xlsx')
    EventO = simbología.loc[simbología['Volcan'] == volcan]
    EventO = EventO.loc[EventO['Evento'] == evento]
    coloR = EventO.values[0,2]
    markeR = EventO.values[0,3]
    return coloR, markeR

def simbologia_core(testigo,profundidad):

    simbología = pd.read_excel('../Scripts/SimbologiaTestigos.xlsx')
    EventO = simbología.loc[simbología['Testigo'] == testigo]
    EventO = EventO.loc[EventO['Profundidad'] == profundidad]
    coloR = EventO.values[0,2]
    markeR = EventO.values[0,3]
    return coloR, markeR
	          	          
def grafico_edades(Data,Data_cores ='default',save=False):
    plt.figure(figsize=(2.5,9))
    ax = plt.axes()
    
    Data_magnitud = Data.copy()
    Data['Historic'] = Data['Edad'].str.contains('Historic')
    Data = Data[Data['Historic']!= True]
    Data = Data.dropna(subset=['Edad'])
    Data = Data.reset_index(drop=True)
    Data.Edad = Data.Edad.values/1000
    Data.ErrorEdad = Data.ErrorEdad.values/1000
    Eventos = Data['Evento'].values
    Volcanes = Data['Volcan'].values
    Sample = Data['SampleID'].values
    i = 0
    j = 0
    k = 0
    #print(Eventos)
    while i < np.size(Eventos):

        #print("0 {}, {}".format(Eventos[i],Volcanes[i]))
        Color, Marker  = simbologia(Volcanes[i],Eventos[i])
        marker_edge = 'black'
        Data_evento = Data.loc[Data['Volcan'] == Volcanes[i]]  
        Data_evento = Data_evento.sort_values(by=['Evento'])	
        Data_evento = Data_evento.loc[Data_evento['Evento'] == Eventos[i]]
        Data_magnitud_evento = Data_magnitud.loc[Data_magnitud['Volcan'] == Volcanes[i]]
        Data_magnitud_evento = Data_magnitud_evento.loc[Data_magnitud_evento['Evento'] == Eventos[i]]
        Magnitud = scipy.stats.mode(Data_magnitud_evento.Magnitud)
        Magnitud = Magnitud[0]

        if Data_evento.shape[0] != 1: #Eventos con más de una datación 14C
            
            #print("1 {}, {}, {}".format(Eventos[i],Volcanes[i],Data.SamplePoint[i]))
            if Eventos[i] != 'Unknown': #Es un evento con un nombre asignado 
                if Magnitud==0: #la magnitud del evento es desconocida
                    #print("2 {}, {}, {}, {}, {}".format(Eventos[i],Volcanes[i],Data.SamplePoint[i],Data_evento['Edad'].min(),Data_evento['Edad'].max()))
                    rect = ptch.Rectangle((-1.5,Data_evento['Edad'].min()),1.5,Data_evento['Edad'].max()-Data_evento['Edad'].min(), facecolor = Color,alpha=0.4,linewidth = 0.25)
                    simbolo = ax.plot(-2,Data_evento['Edad'].min() +(Data_evento['Edad'].max()-Data_evento['Edad'].min())/2, color = Color, marker = Marker, markersize=8,markeredgecolor=marker_edge, label=Eventos[i],alpha=0.8)
                    ax.text(-4,Data_evento['Edad'].min() +(Data_evento['Edad'].max()-Data_evento['Edad'].min())/2+.07,Eventos[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
                    while j < Data_evento.shape[0]:
                        rect = ptch.Rectangle((-1.5,Data_evento.Edad.values[j]),1.5,Data_evento.ErrorEdad.values[j],facecolor = Color,edgecolor='black',linewidth = 0.25)
                        j=j+1
                        ax.add_patch(rect) 
                else: #la magnitud del evento es conocida
                    #print("3 {}, {}, {}, {}, {}".format(Eventos[i],Volcanes[i],Data.SamplePoint[i],Data_evento['Edad'].min(),Data_evento['Edad'].max()))
                    rect = ptch.Rectangle((0,Data_evento['Edad'].min()),Magnitud,Data_evento['Edad'].max()-Data_evento['Edad'].min(),facecolor = Color,alpha=0.4,linewidth = 0.25)
                    simbolo = ax.plot(Magnitud+.7,Data_evento['Edad'].min() +(Data_evento['Edad'].max()-Data_evento['Edad'].min())/2,color = Color, marker = Marker, markersize=10,markeredgecolor=marker_edge, label=Eventos[i],alpha=0.8)
                    ax.text(Magnitud+1.6,Data_evento['Edad'].min() +(Data_evento['Edad'].max()-Data_evento['Edad'].min())/2+.07,Eventos[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
                    while j < Data_evento.shape[0]:
                        rect = ptch.Rectangle((0,Data_evento.Edad.values[j]),Magnitud,Data_evento.ErrorEdad.values[j],facecolor = Color,alpha=.7,edgecolor='black',linewidth = 0.25)
                        j=j+1
                        ax.add_patch(rect)
            else: #son varias dataciones de un Volcan pero no se sabe el evento
                 while j < Data_evento.shape[0]:
                    #print("4 {}, {}, {}".format(Eventos[i],Volcanes[i],Data.SamplePoint[i]))
                    rect = ptch.Rectangle((0,Data_evento.Edad.values[j]),Magnitud,Data_evento.ErrorEdad.values[j], facecolor = Color,linewidth = 0.25)
                    simbolo = ax.plot(0,Data_evento.Edad.values[j], color = Color, marker = Marker, markersize=8,markeredgecolor='black',alpha=0.7)
                    j=j+1
                    ax.add_patch(rect)
        else: #Eventos con una datación 14C
            #print("-0 {}".format(Eventos[i]))			
            if Eventos[i] != 'Unknown': #Es un evento con un nombre asignado 
                if Magnitud==0: #la magnitud del evento es desconocida
                    #print("5 {}, {}, {}".format(Eventos[i],Volcanes[i],Data.SamplePoint[i]))
                    simbolo = ax.plot(-2,Data_evento['Edad'].min(),color = Color, marker = Marker, markersize=7,markeredgecolor=marker_edge, alpha=0.8, label=Eventos[i])
                    rect = ptch.Rectangle((-1.5,Data_evento['Edad'].min()),1.5,Data_evento.ErrorEdad.values[j], facecolor = Color,edgecolor='black',linewidth = 0.25) #este rectangulo es para que el codig no reclame por el addpatch
                    ax.text(-4,Data_evento['Edad'].min() +(Data_evento['Edad'].max()-Data_evento['Edad'].min())/2+.07,Eventos[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
                else: #la magnitud del evento es conocida
                    #print("6 {}, {}, {}".format(Eventos[i],Volcanes[i],Data.SamplePoint[i]))
                    rect = ptch.Rectangle((0,Data_evento['Edad'].min()), Magnitud, Data_evento.ErrorEdad.values[j], facecolor = Color,linewidth = 0.25)
                    simbolo = ax.plot(Magnitud+.7, Data_evento['Edad'].min(), color = Color, marker = Marker, markersize=10,markeredgecolor=marker_edge, alpha=0.8, label=Eventos[i])
                    ax.text(Magnitud+1.6,Data_evento['Edad'].min()+.07,Eventos[i],rotation=0,color='black',fontsize=8)
                    ax.add_patch(rect)
            else: #una datación de un evento unknown
                    #print("7 {}, {}, {}".format(Eventos[i],Volcanes[i],Data.SamplePoint[i]))
                    rect = ptch.Rectangle((0,Data_evento.Edad.values[j]), 0.05 ,Data_evento.ErrorEdad.values[j], facecolor = Color,alpha=1,edgecolor='black',linewidth = 0.25)
                    simbolo = ax.plot(0,Data_evento.Edad.values[j], color = Color, marker = Marker, markersize=8,markeredgecolor='black',alpha=0.8)
                    ax.add_patch(rect)
                    
        #ax.add_patch(rect)
        #txt = ax.text(Data_evento['Edad'].min(),Magnitud[i]+1.5 ,Evento[i],weight='bold',color='black')
        #txt.set_rotation(45)
        i = i+ Data_evento.shape[0]
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
            simbolo = ax.plot( 0,Data_cores.Edad.values[k], color = Color, marker = Marker, markersize=12,markeredgecolor='black',alpha=1,label=Core[k]+' '+Depth[k])
            #txt = ax.text(0.1,Data_cores.Edad.values[k]-0.1,Data_cores.Depth.values[k],rotation=90,weight='bold',color='black',fontsize=17)
            k = k+1    
    
    #tags de qué representa cada línea
    #conMagnitud = ax.text(18000,4,'Eventos con Magnitud estimada',weight='bold',color='black',fontsize=15)
    #sinMagnitud = ax.text(18000,-2,'Evento sin Magnitud estimada',weight='bold',color='black',fontsize=15)
    #unknown = ax.text(18000,-4,'Evento Unknown',weight='bold',color='black',fontsize=15)
    
        
    ax.set_ylabel('14C age (kyears BP)', fontsize=12)
    ax.set_xlabel('Magnitud', fontsize=12)
    ax.invert_yaxis()
    #ax.set_xticklabels((np.linspace(0,15,6),0))
    ax.tick_params(labelsize = 12)

    #if Data.Evento.unique().size > 60:
    #    leg=ax.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.7,1),ncol=3,fontsize=19)
        
    #if (Data.Evento.unique().size > 20)&(Data.Evento.unique().size < 60):
    #    leg=ax.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.45,1),ncol=2,fontsize=19)
        
    #if (Data.Evento.unique().size < 20):
    #    leg=ax.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.27,1),ncol=1,fontsize=19)
     
    ax.legend(loc='lower left', fancybox=True, bbox_to_anchor=(1,0),ncol=1,fontsize=8)	 
    #leg.get_frame().set_alpha(0.5)
    plt.yticks(np.linspace(0,16,17))
    plt.xticks(np.linspace(0,6,7))
    ax.grid(axis ='y')
    #ax[0].set_yscale("log")
    ax.set_xlim(-5,9)
        
    if save:
        plt.savefig('../Plots/Dataciones.pdf',dpi = 300,bbox_inches='tight')
    
    plt.show()

SVZ="../Scripts/Images/ZVS.png" 
SVZ= PImage.open(SVZ)
AVZ="../Scripts/Images/AVZ2.jpg" 
AVZ= PImage.open(AVZ)
Ambos="../Scripts/Images/Ambos2.jpg" 
Ambos= PImage.open(Ambos)

def grafico_posicion(Datas,zona,VOLCANES='default', texto='no',save=False):


    if zona == 'Ambos':
        MarkerSize = 10
    else:
        MarkerSize = 10
		
    plt.figure(figsize=(20,20))
    ax = plt.axes(projection = ccrs.PlateCarree())
    Datas = Datas.dropna(how='any',subset=['Latitud']) 
    Datas = Datas.reset_index(drop=True)
    Volcanes = Datas.Volcan
    Eventos = Datas.Evento
    Lon =  Datas.Longitud
    Lat =  Datas.Latitud
    
    for volcan in Datas.Volcan.unique():
        #print('Volcan {}'.format(volcan))
        temp0 = Datas[Datas.Volcan == volcan]
        #print(volcan)
        for evento in temp0.Evento.unique():
            #print(evento,temp0.Evento.unique())
            temp = temp0[temp0.Evento== evento]
            for seccion in temp.Latitud.unique():
                temp2 = temp[temp.Latitud == seccion]
                Index = temp2.first_valid_index()
                #print(Datas.SamplePoint[Index])
                x,y = (temp2.Longitud[Index],temp2.Latitud[Index])
                Color, Marker  = simbologia(temp2.Volcan[Index],temp2.Evento[Index])
                if evento == 'Unknown':
                    #print("1 {}, {}, i: {}".format(evento,volcan,seccion))
                    marker_edge = 'white'
                else:
                    #print("2 {}, {}, i: {}".format(evento,volcan,seccion))
                    marker_edge = 'black'
                plt.plot(x,y, color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor=Color , alpha=1)
            
            x,y = (temp2.Longitud[Index],temp2.Latitud[Index])
            Color, Marker  = simbologia(temp2.Volcan[Index],temp2.Evento[Index])
            #print('evento {}'.format(evento))
            plt.plot(x,y, color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor=Color, alpha=1,label=evento)
        
            
        x,y = (temp0.Longitud[Index],temp0.Latitud[Index])
        Color, Marker  = simbologia(temp0.Volcan[Index],temp0.Evento[Index])
        #print('Volcan {}'.format(volcan))
        #plt.plot(x,y, color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor=marker_edge, alpha=0.6,label=volcan)
                		
    #plt.plot(Lon[np.size(Eventos)-1],Lat[np.size(Eventos)-1], color = Color, marker = Marker, transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor=marker_edge, alpha=0.6])
	
    if zona == 'Ambos':
        MarkerSize = 27
        FontSize=20
    else:
        MarkerSize = 45
        FontSize=24
    VOLCANES = VOLCANES.reset_index(drop=True)	
    for i in range(0,np.size(VOLCANES.Volcan)):
        Color, Marker  = simbologia(VOLCANES.Volcan[i],'Unknown')
        x,y = (VOLCANES.Longitud[i],VOLCANES.Latitud[i])
        if (VOLCANES.Volcan[i] == 'MD07-3098')|(VOLCANES.Volcan[i] == 'MD07-3100')|(VOLCANES.Volcan[i] == 'MD07-3081')|(VOLCANES.Volcan[i] == 'MD07-3082')|(VOLCANES.Volcan[i] == 'MD07-3088')|(VOLCANES.Volcan[i] == 'MD07-3119'):
            #print(VOLCANES.Volcan[i])
            MarkerSize = 15			
            plt.plot(x,y, color = Color, marker = 'o', transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor='black',alpha=0.5)#,markeredgewidth=1 + i/10
        else:
            #print(VOLCANES.Volcan[i])
            plt.plot(x,y, color = Color, marker = '^', transform = ccrs.PlateCarree(),markersize=MarkerSize,markeredgecolor='black',alpha=0.7)#,markeredgewidth=1 + i/10
        
        if texto == 'sí':
            ax.text(x- 0.2, y+0.22,VOLCANES.Volcan[i], transform=ccrs.PlateCarree(),weight='bold',color='w',fontsize=24)
        
		
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
    if Datas.Evento.unique().size > 90:
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.83,1),ncol=3,fontsize=13)
    if (Datas.Evento.unique().size > 45)&(Datas.Evento.unique().size < 90):
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.5,1),ncol=2,fontsize=13)
    if (Datas.Evento.unique().size < 45):
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=15)
		
    leg.get_frame().set_alpha(0.5)
    
    if save:
        if zona == 'SVZ':
            plt.savefig('../Plots/PosiciónDataSVZ.pdf',dpi = 300,bbox_extra_artists=(leg,),bbox_inches='tight')
        if zona == 'AVZ':
            plt.savefig('../Plots/PosiciónDataAVZ.pdf',dpi = 300,bbox_extra_artists=(leg,),bbox_inches='tight')        
        if zona == 'Ambos':
            plt.savefig('../Plots/PosiciónSamples.pdf',dpi = 300,bbox_extra_artists=(leg,),bbox_inches='tight')    
		
    plt.show()
	
def TAS(Data,Data_cores):
    
       
    plt.figure(figsize=(13,13))
    ax = plt.axes()
            
    #plot identifyed glass shards
    Data = Data.dropna(subset=['Na2O','K2O','SiO2'])
    Data = Data.reset_index(drop=True)
    
    Na2O = Data['Na2O'].values
    K2O = Data['K2O'].values
    SiO2 = Data['SiO2'].values
    
    Volcanes = Data.Volcan.values
    Eventos = Data.Evento.values
    Volcan = Volcanes[0]
    Evento = Eventos[1]
    
    for i in range(0,np.size(Eventos)):
        if Evento == Eventos[i]:
            Color, Marker  = simbologia(Volcanes[i],Eventos[i])
            plt.plot(SiO2[i], Na2O[i]+K2O[i], color = Color, marker = Marker,markersize=12, alpha=0.4,markeredgecolor = 'black')
        else:
            if Volcanes[i]==Volcan:
                Color, Marker  = simbologia(Volcanes[i],Eventos[i])
                plt.plot(SiO2[i], Na2O[i]+K2O[i], color = Color, marker = Marker, markersize = 12,alpha=0.4, label = Eventos[i] , markeredgecolor = 'black')
                Evento = Eventos[i]

            else:
                Color, Marker  = simbologia(Volcanes[i],Eventos[i])
                plt.plot(SiO2[i], Na2O[i]+K2O[i], color = Color, marker = Marker,markersize=12,alpha=0.4,label=Eventos[i] , markeredgecolor = 'black')
                Evento = Eventos[i]
                Volcan = Volcanes[i]
                
 #---------------------------------------plot core data when given
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
    plt.xlabel('SiO2', fontsize = 22)
    plt.ylabel('Na2O + K2O', fontsize = 22)
    ax.tick_params(labelsize = 21)
    if Data.Evento.unique().size > 90:
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(2,1),ncol=3,fontsize=11)
    if (Data.Evento.unique().size > 45)&(Data.Evento.unique().size < 90):
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1.5,1),ncol=2,fontsize=13)
    if (Data.Evento.unique().size < 45):
        leg=plt.legend(loc='upper right', fancybox=True, bbox_to_anchor=(1,1),ncol=1,fontsize=15)
    leg.get_frame().set_alpha(1)
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