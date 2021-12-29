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

st.write("""
     ## Visualisation of trawlers landing data in the Ionian Sea
     """)

st.markdown("Explore landing data in the Ionian Sea, Greece")
            #"HCMR")

# user choices
with st.sidebar:
    st.subheader("Select")
    area = st.sidebar.selectbox('Region', [None, "North Ionion","Central Ionion","South Ionion"])
    mesh_size = st.sidebar.selectbox('Mesh size', [None, 40,50])
      
    df_q = df.copy()
    df_q = df_q[(df_q['DCF_AREA'] == area) & (df_q['MESH_SIZE'] == mesh_size)]
    
    if (area == 'South Ionion'):
        depth = st.sidebar.selectbox('Depth', [None, '<= 350 m', '> 350 m']) 
        
        if (depth == '<= 350 m'): 
            df_q = df_q[ (df_q['DEPTH_ST'] <= 350) & (df_q['DEPTH_END'] <= 350)]
        elif (depth == '> 350 m'):
            df_q = df_q[ (df_q['DEPTH_ST'] > 350) & (df_q['DEPTH_END'] > 350)]  
        
    select_species = st.multiselect('Species', list(set(df_q['SPECIES'])))

months, day_nights, species, discards, eue, shannon_com, shannon_disc, dinit, dend, land, disc, sp_depth_st, sp_depth_end = metier_metanal(df_q)

if mesh_size is not None:
    
     # Species plots
    if len(select_species) > 0:        
        # ------------------------------------------ #
        st.subheader('Selected species')      
        fig, ax = plt.subplots(figsize=(6,4))
        desired_order_list = select_species
        v1 = {k: land[k] for k in desired_order_list}
        v2 = {k: disc[k] for k in desired_order_list}
        X = np.arange(len(v1))
        ax.bar(X, 100 * np.array(list(v1.values())) / ( np.array(list(v1.values()))+ np.array(list(v2.values()))), width=0.75, color="blue", alpha=0.8, bottom=0, align='center')
        ax.bar(X, 100 * np.array(list(v2.values())) / ( np.array(list(v1.values()))+ np.array(list(v2.values()))), width=0.75, color="orange", alpha=0.8, bottom=100 * np.array(list(v1.values())) / ( np.array(list(v1.values()))+ np.array(list(v2.values()))), align='center')
        ax.legend(['Landings rate', 'Discards rate'])
        plt.xticks(X, v1.keys())
        ax.set_xticklabels(v1.keys(), rotation = 90) # size =12
        ax.grid(axis='y')
        ax.set_ylabel('Percentage (%)')
        plt.setp(ax.xaxis.get_majorticklabels(), ha='right')     
        st.pyplot(fig)
    
        # ------------------------------------------- #
        st.subheader('Depth start/end for selected species')      
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8,6))
        desired_order_list1 = select_species
        v1 = {k: sp_depth_st[k] for k in desired_order_list1}
        v2 = {k: sp_depth_end[k] for k in desired_order_list1}
        
        labels1, data1 = v1.keys(), v1.values()
        labels2, data2 = v2.keys(), v2.values()
        
        X = np.arange(1, len(labels1)+1)
        plt.subplot(2, 1, 1)
        plt.boxplot(data1, showfliers=False)
        plt.xticks(X, labels1)
        plt.grid(axis='y')
        plt.ylabel('Depth start')
               
        plt.subplot(2, 1, 2)
        X = np.arange(1, len(labels2)+1)
        plt.boxplot(data2, showfliers=False)
        plt.xticks(X, labels2)
        plt.grid(axis='y')
        plt.ylabel('Depth end')
        st.pyplot(fig)
    
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
        
    
    st.subheader('Landings per month')      
    fig, ax = plt.subplots(figsize=(8,5))
    keys = months.keys()
    values = months.values()
    plt.bar(keys, values)
    plt.xticks(np.arange(1, 13))
    ax.set_ylabel('%')
    ax.set_xlabel('Month')
    st.pyplot(fig)
    
    st.subheader('depth of landings (START and END)')      
    fig, ax = plt.subplots(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.hist(dinit, bins =[25,50,75,100,125,150,200,300,600])
    ax.set_xlabel('Meters')
    #ax.set_title('Starting Depth')
    plt.subplot(1, 2, 2)
    plt.hist(dend, bins =[25,50,75,100,125,150,200,300,600])
    ax.set_xlabel('Meters')
    #ax.set_title('End Depth')
    st.pyplot(fig)
 
