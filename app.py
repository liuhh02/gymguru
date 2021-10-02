import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
import pickle
import joblib
import calplot
from trainmodel import (generate_snippet, model_selector, 
                        load_dataset, get_model_info, 
                        get_model_url, train_model)

st.set_page_config(page_title="Gym Guru", page_icon="💪")
st.title("CMU Fitness Facility Capacity Dashboard")

def _max_width_(prcnt_width:int = 80):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                .sidebar .sidebar-content {{
                    width: 100;
                }}
                </style>    
                """, 
                unsafe_allow_html=True,
    )

_max_width_()

hide_footer_style = """
    <style>
    .reportview-container .main footer {visibility: hidden;}    
    """
st.markdown(hide_footer_style, unsafe_allow_html=True)

hide_menu = """
    <style>
    #MainMenu {
        visibility:hidden;
    }
    </style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

def time_to_seconds(time):
    return time.hour * 3600 + time.minute * 60 + time.second

def user_input_data():
    date = st.date_input("Day of Visit", datetime.date(2021, 10, 3))
    time = st.time_input('Time of Visit', datetime.time(14, 00))
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

pageview = st.sidebar.selectbox(
    "Select what you want to do!",
    ("Check Current Crowd Levels", "Plan my Workout", 
    "Visualize Some Cool Charts", "Build my Own Prediction Model")
)

def footer():
    st.sidebar.header("")
    
    st.sidebar.markdown("---")
    st.sidebar.write("Made with ❤️ by Haohui for HackCMU")
    # st.sidebar.markdown(
    #     """
    #     [<img src='data:image/png;base64,{}' class='img-fluid' width=25 height=25>](https://github.com/ahmedbesbes/playground) <small> Made with 0.1.0 | April 2021</small>""".format(
    #         img_to_bytes("./images/github.png")
    #     ),
    #     unsafe_allow_html=True,
    # )

footer()

if pageview == "Check Current Crowd Levels":
    st.subheader('Current Crowd Levels')
    st.write("")
    #today = datetime.date.today()
    now = datetime.datetime.now() - datetime.timedelta(hours=3, minutes=0)
    currday = now.strftime("%B %d, %Y")
    currtime = now.strftime("%I: %M %p")

    st.metric(label="Time Now:", value=currday, delta = currtime)
    st.write("")
    st.write("")
    col1, mid, col2, mid2, col3 = st.columns([20,1,20,1,20])
    with col1:
        st.markdown("##### Cohon University Center Fitness Center")
        st.image('images/cuc.jpg', width=350)
        st.metric(label="Current Capacity", value="50/50", delta="8 in queue", delta_color="inverse")
        st.metric(label="Estimated Waiting Time", value="10 min")
        st.markdown("#### Wiegand Gym")
        st.image('images/wiegand.jpg', width=350)
        st.metric(label="Current Capacity", value="24/50", delta="0 in queue")
        st.metric(label="Estimated Waiting Time", value="0 min")
    with col2:
        st.markdown("##### Tepper Fitness Center")
        st.image('images/tepper2.jpg', width=350)
        st.metric(label="Current Capacity", value="8/50", delta="0 in queue")
        st.metric(label="Estimated Waiting Time", value="0 min")
        st.markdown("#### Skibo Gym")
        st.image('images/skibo.jpg', width=350)
        st.metric(label="Current Capacity", value="CLOSED", delta="FOR CONSTRUCTION", delta_color="inverse")
        st.metric(label="Estimated Waiting Time", value="-")
    with col3:
        st.markdown("##### Swimming & Diving Pool")
        st.image('images/pool.jpg', width=350)
        st.metric(label="Current Capacity", value="12/50", delta="0 in queue")
        st.metric(label="Estimated Waiting Time", value="0 min")

elif pageview == "Plan my Workout":
    st.subheader('Predict Crowd Levels')
    st.write("Planning your visit to the gym? Enter the time you intend to go at and figure out how crowded the facility will likely be!")
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
        pdf = pd.DataFrame(df, index=[0])
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
    df = pd.read_csv('data.csv')
    daydict = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['day_of_week'].replace(daydict, inplace=True)
    df['date'] = pd.to_datetime(df['date'], utc=True)
    dfdt = df.set_index('date')
    dfdt = dfdt['number_people']
    dfdt = pd.Series(dfdt)
    cola, mid, colb = st.columns([10, 1, 10])
    with cola:
        st.subheader("Crowd Distribution Across Days")
        fig = px.histogram(df, x="day_of_week",
                   width=500, 
                   height=400,
                   y="number_people", histfunc='sum',
                   color_discrete_map={
                       "Monday": "RebeccaPurple", "Sunday": "lightsalmon",
                       },
                   template="simple_white", 
                   category_orders={"day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]}
                   )
        colors = ['lightgray',] * 7 
        colors[6] = 'lightseagreen'
        colors[5] = 'lightseagreen'
        colors[1] = 'crimson'
        colors[0] = 'crimson'
        colors[2] = 'crimson'

        fig.update_traces(marker_color=colors, marker_line_color='seagreen',
                        marker_line_width=2.5, opacity=0.5)
        st.write(fig)
        st.write("We can see that the number of gym-goers generally decreases as the week progresses, with crowd levels markedly lower during the weekend.")

    with colb:
        st.subheader("Crowd Distribution Across Time")
        fig = px.histogram(df, x="hour",
                   width=500, 
                   height=400,
                   y="number_people", histfunc='sum',
                   template="simple_white"
                   )
        # custom color
        colors = ['lightgray',] * 24
        colors[16] = 'crimson'
        colors[17] = 'crimson'
        colors[18] = 'crimson'
        colors[19] = 'crimson'
        colors[0] = 'lightseagreen'
        colors[1] = 'lightseagreen'
        colors[5] = 'lightseagreen'
        colors[6] = 'lightseagreen'
        colors[7] = 'lightseagreen'
        colors[8] = 'lightseagreen'
        colors[23] = 'lightseagreen'

        fig.update_traces(marker_color=colors, marker_line_color='white',
                        marker_line_width=2.5, opacity=0.5)
                        
        st.write(fig)
        st.write("It seems like students generally prefer working out later in the day as compared to in the morning.")
    st.title("Crowd Heatmap")
    st.write("We may also visualize the heatmap to get more intuition about the data")

    subset = df[['hour','number_people','day_of_week']]

    #Group by time and day
    heatmap = subset.groupby(['hour','day_of_week'], as_index = False).number_people.mean().pivot('day_of_week','hour', 'number_people').fillna(0)

    fig = px.imshow(heatmap, labels=dict(y="Day of Week", x="Time", color="Average Crowd Levels"), color_continuous_scale="RdBu_r",
                    y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    x=['12am', '1am', '2am', '3am','4am','5am','6am','7am','8am','9am','10am','11am','12pm',
                    '1pm', '2pm', '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'], width=1000)
    #fig.update_xaxes(side="top")
    st.write(fig)
    st.write("Again, this fits with our earlier observation that the gym is more crowded in the late afternoon with little traffic from 1am - 5am (which indicates the gym is closed)")
    
    st.subheader("Rate of Change of Crowd Levels")
    st.write("Maybe we want to also visualize the rate of change of the crowd for each time of the day to gather when the gym is getting busier (we do this using our dear friend calculus; but don't worry you don't have to do the calculations yourself!)")
    roc = np.gradient(heatmap, edge_order = 2)[1]
    heatmapc = pd.DataFrame(roc, columns=heatmap.columns, index = heatmap.index)

    fig = px.imshow(heatmapc, labels=dict(y="Day of Week", x="Time", color="Change in Crowd Levels"), color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                    y=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                    x=['12am', '1am', '2am', '3am','4am','5am','6am','7am','8am','9am','10am','11am','12pm',
                    '1pm', '2pm', '3pm','4pm','5pm','6pm','7pm','8pm','9pm','10pm','11pm'], width=1000)
    #fig.update_xaxes(side="top")
    st.write(fig)

    st.subheader("Calendar Heatmap")
    st.write("Lastly, let us also visualize the heatmap using a calendar. You can see that the data ends in March 2020, so let's collect more data after HackCMU and make this even better!")
    fig, ax = calplot.calplot(dfdt, cmap='YlGn')
    st.pyplot(fig)

elif pageview == "Build my Own Prediction Model":
    st.subheader("Train My Own Model!")
    st.write("Unconvinced by this model? Want to try your hand at training a more accurate model? Here's the place to do it!")
    model_type, model = model_selector()
    duration_placeholder = st.empty()
    score_placeholder = st.empty()
    model_url_placeholder = st.empty()
    code_header_placeholder = st.empty()
    snippet_placeholder = st.empty()
    info_header_placeholder = st.empty()
    info_placeholder = st.empty()

    X_train, X_test, y_train, y_test, scaler = load_dataset()
    
    model_url = get_model_url(model_type)
    (
        model,
        train_score,
        test_score,
        duration,
    ) = train_model(model, X_train, y_train, X_test, y_test)

    snippet = generate_snippet(model, model_type)

    model_info = get_model_info(model_type)

    duration_placeholder.warning(f"Training took {duration:.3f} seconds")
    coltrain, coltest = score_placeholder.columns(2)
    coltrain.metric("Training Score", train_score)
    coltest.metric("Test Score", test_score, np.round(test_score-train_score, 3))
    model_url_placeholder.markdown(model_url)
    code_header_placeholder.header("**Retrain the same model in Python**")
    snippet_placeholder.code(snippet)
    info_header_placeholder.header(f"**Info on {model_type} 💡 **")
    info_placeholder.info(model_info)
    
    st.subheader("Test My Model")
    model_testing_container = st.expander("Does your model have what it takes? Put it to the test!", False)
    with model_testing_container:
    #if (st.button("Does your model have what it takes? Put it to the test!")):
        daydict = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
        df=user_input_data()
        if st.button('Predict!'):
            semstatus = 'School Semester'
            if (df['is_start_of_semester'] == 1):
                semstatus = 'Start of Semester'
            elif (df['is_during_semester'] == 0):
                semstatus = 'Semester Break'
            col4, col6, col7 = st.columns(3)
            col4.metric(label="Date", value=str(df['date']), delta=daydict[df['day_of_week']])
            col6.metric(label="Semester", value=semstatus)
            pdf = pd.DataFrame(df, index=[0])
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