import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
st.set_page_config(page_title="Gym Guru", page_icon="💪")
st.title("CMU Fitness Facility Capacity Dashboard")

def _max_width_(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

_max_width_()

st.subheader('Current Crowd Levels')
images = ['images/cuc.jpg', 'images/tepper2.jpg']
#st.image(images, width=500)

col1, mid, col2, mid2, col3 = st.columns([20,1,20,1,20])
with col1:
    st.write("Cohon University Center Fitness Center")
    st.image('images/cuc.jpg', width=350)
    st.metric(label="Current Capacity", value="50/50", delta="8 in queue", delta_color="inverse")
    st.metric(label="Estimated Waiting Time", value="10 min")
with col2:
    st.write("Tepper Fitness Center")
    st.image('images/tepper2.jpg', width=350)
    st.metric(label="Current Capacity", value="8/50", delta="0 in queue")
    st.metric(label="Estimated Waiting Time", value="0 min")
with col3:
    st.write("Swimming & Diving Pool")
    st.image('images/pool.jpg', width=350)
    st.metric(label="Current Capacity", value="12/50", delta="0 in queue")
    st.metric(label="Estimated Waiting Time", value="0 min")

