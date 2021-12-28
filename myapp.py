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
       
    return (mon.to_dict(),  dict(day_night), dict(s), 
            dict(s_disc), EUE, shannon, shannon_disc, depth_init, depth_end)

@st.cache
def load_data():
    xls = pd.ExcelFile('ntokos_maren_upd3.xlsx') 
    df = pd.read_excel(xls)
    df['MONTH'] = pd.DatetimeIndex(df['SAMPL_DATE']).month
    df = df.dropna(subset=['N', 'MESH_SIZE', 'DEPTH_ST', 'DEPTH_END']) 
    df['CATCH_GR'] = df['CATCH_GR'].astype(float)
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

months, day_nights, species, discards, eue, shannon_com, shannon_disc, dinit, dend = metier_metanal(df_q)

if mesh_size is not None:
    st.subheader('Landing profiles & Discards')      
    fig, ax = plt.subplots(figsize=(18,16))
    
    plt.subplot(1, 2, 1)
    species_sorted = dict(sorted(species.items(), key = itemgetter(1), reverse = True)[:10])
    labels = list(species_sorted.keys())
    values = list(species_sorted.values())
    plt.pie(values, labels=labels, autopct='%1.1f%%', normalize = False, textprops={'fontsize': 12})
    ax.set_ylabel('Profiles (%)')
    #st.pyplot(fig)
    
    #st.subheader('Discards')    
    plt.subplot(1, 2, 2)
    #fig, ax = plt.subplots(figsize=(8,5))
    discards_sorted = dict(sorted(discards.items(), key = itemgetter(1), reverse = True)[:10])
    labels = list(discards_sorted.keys())
    values = list(discards_sorted.values())
    plt.pie(values, labels=labels, autopct='%1.1f%%', normalize = False, textprops={'fontsize': 12})
    ax.set_ylabel('%')
    st.pyplot(fig)
    
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
    
    
    
    