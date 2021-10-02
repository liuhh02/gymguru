import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
import pickle
import joblib

st.set_page_config(page_title="Gym Guru", page_icon="💪", initial_sidebar_state="collapsed")
st.title("CMU Fitness Facility Capacity Dashboard")

def _max_width_(prcnt_width:int = 75):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                .sidebar .sidebar-content {{
                    width: 100px;
                }}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

_max_width_()

def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second


pageview = st.sidebar.selectbox(
    "Select what you want to do!",
    ("Check Current Crowd Levels", "Plan my Workout", 
    "Visualize Some Cool Charts", "Build my Own Prediction Model")
)

if pageview == "Check Current Crowd Levels":
    st.subheader('Current Crowd Levels')
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

elif pageview == "Plan my Workout":
    st.subheader('Predict Crowd Levels')
    st.write("Planning your visit to the gym? Enter the time you intend to go at and figure out how crowded the facility will likely be!")
    def user_input_data():
        date = st.date_input("Day of Visit", datetime.date(2021, 10, 1))
        time = st.time_input('Time of Visit', datetime.time(8, 00))
        #month = date.month
        #day = date.day
        startsem = 0
        schoolsem = 1
        fallstart = datetime.date(2021, 8, 30)
        fallend = datetime.date(2021, 12, 17)
        springstart = datetime.date(2022, 1, 17)
        springend = datetime.date(2022, 5, 17)
        delta1 = date - fallstart
        delta2 = date - springstart
        #print(delta.days)
        #schoolsem from Aug 30 - Dec 17, Jan 17 - May 17
        if (date < fallstart):
            schoolsem = 0
        elif (date > fallend and date < springstart):
            schoolsem = 0
        elif (date > springend):
            schoolsem = 0
        if (delta1.days >= 0 and delta1.days <=21):
            startsem = 1
        elif (delta2.days >= 0 and delta2.days <=21):
            startsem = 1
        day_of_week = date.weekday()+1
        is_weekend = 0
        if(day_of_week == 6 or day_of_week == 7):
            is_weekend = 1
        #st.write("Date is: ", date)
        #st.write("Time is: ", time)
        timestamp = time_to_seconds(time)
        data = {'date': date, 'timestamp':timestamp, 'day_of_week': day_of_week,
        'is_weekend': is_weekend, 'temperature': 70, 'is_start_of_semester': startsem, 
        'is_during_semester': schoolsem, 'month':int(date.month), 'hour':int(time.hour)}
        return data
    daydict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
    df=user_input_data()
    #model = pickle.load(open('rf.pkl', 'rb'))
    model = joblib.load("./rf.joblib")
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    if st.button('Predict!'):
        semstatus = 'School Semester'
        if (df['is_start_of_semester'] == 1):
            semstatus = 'Start of Semester'
        elif (df['is_during_semester'] == 0):
            semstatus = 'Semester Break'
        col4, col6, col7 = st.columns(3)
        col4.metric(label="Date", value=str(df['date']), delta=daydict[df['day_of_week']])
        #col5.metric(label="Day", value=daydict[df['day_of_week']])
        col6.metric(label="Semester", value=semstatus)
        col7.metric(label="Projected Temperature", value=f"{df['temperature']} °F")
        #col5.metric()
        pdf = pd.DataFrame(df, index=[0])
        #st.write(df)
        pdf.drop("date", axis=1, inplace=True)
        pdf = scaler.transform(pdf)
        predicted = model.predict(pdf)
        print(predicted)
        st.subheader("The estimated occupancy is")
        st.title(int(round(predicted[0], 0)))
        if predicted < 10:
            st.write("This is the prime time to gym! Go ahead and get those gainz (though you should first make sure the gym is open!)")
        elif predicted < 30:
            st.write("This seems ok, go ahead with your workout plans!")
        elif predicted < 40:
            st.write("The gym seems relatively crowded, you can try going but maybe change up your plans a little?")
        else:
            st.write("The gym seems to be pretty crowded at that time, maybe visit it at another time?")
    
elif pageview == "Visualize Some Cool Charts":
    st.subheader('Analyze Historical Crowd Data')
    st.write("Unleash your inner geek and gain some insights on crowd data!")
    

