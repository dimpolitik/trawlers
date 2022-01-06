# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 11:20:59 2021
@author: ΔΗΜΗΤΡΗΣ
"""
# https://towardsdatascience.com/quickly-build-and-deploy-an-application-with-streamlit-988ca08c7e83
# https://www.analyticsvidhya.com/blog/2021/06/build-web-app-instantly-for-machine-learning-using-streamlit/
# https://docs.streamlit.io/library/api-reference
# https://towardsdatascience.com/data-visualization-using-streamlit-151f4c85c79a
# https://carpentries-incubator.github.io/python-interactive-data-visualizations/07-add-widgets/index.html
# https://docs.streamlit.io/library/api-reference/layout/st.expander
# https://towardsdatascience.com/deploying-a-web-app-with-streamlit-sharing-c320c79ae350
# https://share.streamlit.io/dimpolitik/trawlers/main/myapp.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from operator import itemgetter
import shapefile as shp
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import math

st.set_page_config(layout="wide")

# Functions
def land_profiles(df_input, gear, cols_LLS):
    # create landing profiles    
    df_gear = df_input.copy()
    df_gear = df_gear.loc[(df_gear['Q'] == 'A') | (df_gear['Q'] == 'B') |  (df_gear['Q'] == 'C') |  (df_gear['Q'] == 'G')]
              
    #https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby
    appended_cols = [] 
    appended_cols.extend(cols_LLS)
    appended_cols.append('SPECIES')
        
    c = df_gear.groupby(appended_cols).agg({'CATCH_GR':'sum'})
              
    # https://stackoverflow.com/questions/47876663/pandas-divide-two-multi-index-series
    # https://stackoverflow.com/questions/23377108/pandas-percentage-of-total-with-groupby
    rat = c.div(c.groupby(cols_LLS).transform('sum'))
    rat.reset_index().to_excel(CS + '_Land_prof_' + gear + '_daily.xlsx',index = False)
    
    df_gear['CATCH_GR'] = df_input['CATCH_GR'] * 60. / df_input['DURATION']    
    
    c1 = df_gear.groupby(appended_cols).agg({'CATCH_GR':'sum'})
    c1.reset_index().to_excel(CS + '_Catch_prof_' + gear + '_daily.xlsx',index = False)

def fill_dicts(d1,d2):
    diff_fill2 = set(d1) - set(d2)
    if list(diff_fill2):
        for i in list(diff_fill2):
            d2[i] = 0
           
    diff_fill1 = set(d2) - set(d1)
    if list(diff_fill1):
        for i in list(diff_fill1):
            d1[i] = 0
                
    sorted_d1 = sorted(d1.items())
    sorted_d2 = sorted(d2.items())
    return dict(sorted_d1), dict(sorted_d2)
    
def metier_metanal(dframe):
    # Metanalysis
    df_meta = dframe.copy()
    df_g = df_meta.loc[(df_meta['Q'] == 'A') | (df_meta['Q'] == 'B') |  (df_meta['Q'] == 'C') |  (df_meta['Q'] == 'G')]
    df_discards = df_meta.loc[(df_meta['Q'] == 'D') | (df_meta['Q'] == 'E') ]     
       
    mon =  df_g.groupby(['MONTH'])['CATCH_GR'].sum().transform(lambda x: 100*x/x.sum())
    day_night = df_g['DAY_NIGHT'].value_counts().transform(lambda x: 100*x/x.sum()) 
    
    # Efficiency    
    EUE = 100 # * df_discards['CATCH_GR'].sum() / (df_g['CATCH_GR'].sum() + df_discards['CATCH_GR'].sum())
    # shannon
    s = df_g.groupby(['SPECIES'])['CATCH_GR'].sum().transform(lambda x: x/x.sum())
    shannon = s * np.log(s)
    shannon = - shannon.sum()
    
    s_disc = df_discards.groupby(['SPECIES'])['CATCH_GR'].sum().transform(lambda x: x/x.sum())
    shannon_disc = s_disc * np.log(s_disc)
    shannon_disc = - shannon_disc.sum()
    
    depth_init = df_g['DEPTH_ST']
    depth_end = df_g['DEPTH_END']
    
    # Coeeficient of variation (Total catches per trip)
    land_species = df_g.groupby(['SPECIES'])['CATCH_GR'].sum()
    disc_species = df_discards.groupby(['SPECIES'])['CATCH_GR'].sum()
    
    species_depth_st = df_g.groupby(['SPECIES'])['DEPTH_ST'].apply(list).to_dict()
    species_depth_end = df_g.groupby(['SPECIES'])['DEPTH_END'].apply(list).to_dict()
    
    land_species_f, disc_species_f = fill_dicts(land_species.to_dict(), disc_species.to_dict())
    
    return (mon.to_dict(),  dict(day_night), dict(s), 
            dict(s_disc), EUE, shannon, shannon_disc, depth_init, depth_end, land_species_f, disc_species_f, species_depth_st, species_depth_end)

@st.cache
def load_data():
    xls = pd.ExcelFile('ntokos_maren_upd3.xlsx') 
    df = pd.read_excel(xls)
    df['MONTH'] = pd.DatetimeIndex(df['SAMPL_DATE']).month
    df = df.dropna(subset=['N', 'MESH_SIZE', 'DEPTH_ST', 'DEPTH_END']) 
    df['CATCH_GR'] = df['CATCH_GR'].astype(float)
    df['DEPTH_ST'] = df['DEPTH_ST'].astype(float)
    df['DEPTH_END'] = df['DEPTH_END'].astype(float)
    #df['MESH_SIZE'] = df['MESH_SIZE'].astype(int)
    df['CATCH_GR'] = df['CATCH_GR'] * 60. / df['DURATION']     
    df['DCF_AREA'] = df['DCF_AREA'].str.replace('N-ION','North Ionion')
    df['DCF_AREA'] = df['DCF_AREA'].str.replace('S-ION','South Ionion')
    df['DCF_AREA'] = df['DCF_AREA'].str.replace('C-ION','Central Ionion')
    return df

df = load_data()

st.title('Visualisation of trawlers landing data in the Ionian Sea')

st.markdown("Explore landing data in the Ionian Sea, Greece")
            #"HCMR")

# user choices
with st.sidebar:
    st.subheader("Select")
    area = st.sidebar.selectbox('Region', [None, "North Ionion", "Central Ionion", "South Ionion"])
    mesh_size = st.sidebar.selectbox('Mesh size', [None, 40,50])
      
    df_q = df.copy()
    df_q = df_q[(df_q['DCF_AREA'] == area) & (df_q['MESH_SIZE'] == mesh_size)]
    
    if (area == 'South Ionion'):
        depth = st.sidebar.selectbox('Depth', [None, '<= 350 m', '> 350 m']) 
        
        if (depth == '<= 350 m'): 
            df_q = df_q[ (df_q['DEPTH_ST'] <= 350) & (df_q['DEPTH_END'] <= 350)]
        elif (depth == '> 350 m'):
            df_q = df_q[ (df_q['DEPTH_ST'] > 350) & (df_q['DEPTH_END'] > 350)]  
        
    select_species = st.multiselect('Species (multiple choices)', sorted(list(set(df_q['SPECIES']))))
    
    lst_species = sorted(list(set(df_q['SPECIES'])))
    lst_species.insert(0, 'None')
    
    species_spatial = st.sidebar.selectbox('Species for spatial plot (one option each time)', lst_species) 

    st.image('urk-fishing-trawlers.jpg')

    months, day_nights, species, discards, eue, shannon_com, shannon_disc, dinit, dend, land, disc, sp_depth_st, sp_depth_end = metier_metanal(df_q)

if mesh_size is not None:
    
    col1, col2, col3  = st.columns(3)
    
    with col1:
        st.subheader('Landings per month')      
        fig, ax = plt.subplots() #figsize=(8,5)
        keys = months.keys()
        values = months.values()
        ax.bar(keys, values)
        plt.xticks(np.arange(1, 13))
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Month')
        st.pyplot(fig)
    
    with col2:
        st.subheader('Starting depth of landings')      
        fig, ax = plt.subplots()
        plt.hist(dinit, bins =[25,50,75,100,125,150,200,300,600])
        ax.set_xlabel('Meters')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
    
    with col3:
        st.subheader('Ending depth of landings')      
        fig, ax = plt.subplots() 
        plt.hist(dend, bins =[25,50,75,100,125,150,200,300,600])  
        ax.set_xlabel('Meters')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)
            
    # Species plots
    if len(select_species) > 0: 
        desired_order_list1 = select_species
        desired_order_list = select_species
        
        v1 = {k: land[k] for k in desired_order_list}
        v2 = {k: disc[k] for k in desired_order_list}
        
        v3 = {k: sp_depth_st[k] for k in desired_order_list1}
        v4 = {k: sp_depth_end[k] for k in desired_order_list1}
        
        labels1, data1 = v3.keys(), v3.values()
        labels2, data2 = v4.keys(), v4.values()
                      
        col1, col2, col3  = st.columns(3)
        
        with col1:
            st.subheader('Selected species')      
            fig, ax = plt.subplots()
            X = np.arange(len(v1))
            ax.bar(X, 100 * np.array(list(v1.values())) / ( np.array(list(v1.values()))+ np.array(list(v2.values()))), width=0.75, color="blue", alpha=0.8, bottom=0, align='center')
            ax.bar(X, 100 * np.array(list(v2.values())) / ( np.array(list(v1.values()))+ np.array(list(v2.values()))), width=0.75, color="orange", alpha=0.8, bottom=100 * np.array(list(v1.values())) / ( np.array(list(v1.values()))+ np.array(list(v2.values()))), align='center')
            ax.legend(['Landings rate', 'Discards rate'])
            plt.xticks(X, v1.keys())
            ax.set_xticklabels(v1.keys(), rotation = 60) # size =12
            ax.grid(axis='y')
            ax.set_ylabel('Percentage (%)')
            plt.setp(ax.xaxis.get_majorticklabels(), ha='right')     
            st.pyplot(fig)
            plt.savefig('test.jpg')   # <-- save first

        
        with col2:       
            st.subheader('Starting depth')           
            fig, ax = plt.subplots()   
            X = np.arange(1, len(labels1)+1)
            plt.boxplot(data1, showfliers=False)
            plt.xticks(X, labels1, rotation=60)
            plt.grid(axis='y')
            #plt.ylim([dmin, dmax])
            plt.ylabel('Depth start (m)')
            st.pyplot(fig)
         
        with col3:
            st.subheader('End depth')
            fig, ax = plt.subplots()
            X = np.arange(1, len(labels2)+1)
            plt.boxplot(data2, showfliers=False)
            plt.xticks(X, labels2, rotation=60)
            plt.grid(axis='y')
            #plt.ylim([dmin, dmax])
            plt.ylabel('Depth end (m)')
            st.pyplot(fig)
     
    if (species_spatial != 'None'):
        df_sp = df_q.copy()
        df_sp = df_sp.loc[df_sp['SPECIES'] == species_spatial]
        
        
        xstep=0.1
        ystep=0.1
        xxleft=19
        yybot=36
        im=math.floor((24.5-19)/xstep) + 1
        jm=math.floor((40-36)/xstep) + 1
        aloni =  np.zeros((jm,im))
        alati =  np.zeros((jm,im))
        pop =  np.zeros((jm,im))
        for i in range(im):
            for j in range(jm):
                aloni[j,i]=xxleft+ i*xstep
                alati[j,i]=yybot+ j*ystep
                pop[j,i]= 0
        
        st.subheader('Spatial landings for ' + species_spatial)  
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            
            #ax.set_aspect('equal')
            
            sf = shp.Reader('coastLine.shp')
            
            for shape in sf.shapeRecords():
                
                # end index of each components of map
                l = shape.shape.parts
                
                len_l = len(l)  # how many parts of countries i.e. land and islands
                x = [i[0] for i in shape.shape.points[:]] # list of latitude
                y = [i[1] for i in shape.shape.points[:]] # list of longitude
            
                l.append(len(x)) # ensure the closure of the last component
                for k in range(len_l):
                    # draw each component of map.
                    # l[k] to l[k + 1] is the range of points that make this component
                    plt.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-') 
                    plt.fill(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'black', alpha=0.3)
            plt.xlim([19.0, 23])
            plt.ylim([36, 40])

            sp_sum = df_sp.groupby(['LONGITUDE_ST','LATITUDE_ST'])['CATCH_GR'].sum()
            df_lon_lat = sp_sum.to_frame()
            df_lon_lat.reset_index(inplace=True)
            df_lon_lat.columns = ['LONGITUDE_ST',  'LATITUDE_ST', 'Catch']
            plt.scatter(df_lon_lat['LONGITUDE_ST']/10000, df_lon_lat['LATITUDE_ST']/10000, c = df_lon_lat['Catch'], cmap=plt.cm.get_cmap("jet", 256))
            plt.colorbar()
            st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots()
                for index, row in df_lon_lat.iterrows():
                    iloc= math.floor((row['LONGITUDE_ST']/10000-xxleft)/xstep)
                    jloc= math.floor((row['LATITUDE_ST']/10000-yybot)/ystep)
                    pop[jloc,iloc]= pop[jloc,iloc] + row['Catch']
                    
                #st.write(pop)
                pop[pop == 0] = 'nan' 
                
                sf = shp.Reader('./coastLine/coastLine.shp')
                #ax.set_aspect('equal')
                for shape in sf.shapeRecords():
                    
                    # end index of each components of map
                    l = shape.shape.parts
                    
                    len_l = len(l)  # how many parts of countries i.e. land and islands
                    x = [i[0] for i in shape.shape.points[:]] # list of latitude
                    y = [i[1] for i in shape.shape.points[:]] # list of longitude
                
                    l.append(len(x)) # ensure the closure of the last component
                    for k in range(len_l):
                        # draw each component of map.
                        # l[k] to l[k + 1] is the range of points that make this component
                        plt.plot(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'k-') 
                        plt.fill(x[l[k]:l[k + 1]],y[l[k]:l[k + 1]], 'black', alpha=0.3)
                                
                plt.pcolor(aloni + xstep/2, alati + ystep/2, np.log10(pop+1), edgecolors='k', cmap = plt.cm.get_cmap("jet", 256), shading = 'auto')
                clb = plt.colorbar()
                clb.set_label('log-form')
                plt.xlim([19.0, 23])
                plt.ylim([36, 40])
                st.pyplot(fig)
                
    #df =bb.to_frame()
    #df.reset_index(inplace=True)
    #df.columns = ['LONGITUDE_ST',  'LATITUDE_ST', 'Catch']
    #plt.scatter(df['LONGITUDE_ST']/10000, df['LATITUDE_ST']/10000, c = df['Catch']/1000, cmap=plt.cm.get_cmap("jet", 256))
    #plt.colorbar()
    #plt.clim(0,600)
    
    #st.subheader('Landing profiles & Discards')      
    #fig, ax = plt.subplots(figsize=(18,16))
    #plt.subplot(1, 2, 1)
    #species_sorted = dict(sorted(species.items(), key = itemgetter(1), reverse = True)[:10])
    #labels = list(species_sorted.keys())
    #values = list(species_sorted.values())
    #plt.pie(values, labels=labels, autopct='%1.1f%%', normalize = False, textprops={'fontsize': 12})
    #ax.set_ylabel('Profiles (%)')
    ##st.pyplot(fig)
    
    ##st.subheader('Discards')    
    #plt.subplot(1, 2, 2)
    ##fig, ax = plt.subplots(figsize=(8,5))
    #discards_sorted = dict(sorted(discards.items(), key = itemgetter(1), reverse = True)[:10])
    #labels = list(discards_sorted.keys())
    #values = list(discards_sorted.values())
    #plt.pie(values, labels=labels, autopct='%1.1f%%', normalize = False, textprops={'fontsize': 12})
    #ax.set_ylabel('%')
    #st.pyplot(fig)
   
