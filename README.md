# GymGuru
Link to hosted website: https://share.streamlit.io/liuhh02/gymguru/main/app.py

## Inspiration
Social distancing is one of the artefacts of the COVID-19 pandemic. Facilities such as gyms now often face long queues of people waiting to enter the facility due to capacity restrictions. The CMU Cohon University Center's Fitness Center in particular is always plagued with long queues lasting up to 20 minutes or more. As a result, many students have had to spend unnecessary time in line, or even had to forgo their workout for the day due to the queues. 

Wouldn't it be great if you could check the current capacity in the gym before heading over, or even better, predict hours or even days ahead of time what the crowd levels will be like so you can better plan your workouts for the week? 

## What it does
Introducing GymGuru, your machine learning powered workout buddy! GymGuru consists of four functionalities:
1. A **dashboard showing live crowd levels** at each fitness facility on campus
2. A **machine learning model that forecasts crowd levels** to help you plan your workouts ahead of time
3. An **interactive data visualization platform** displaying and **extracting insights from crowd data**
4. A **machine learning playground** for YOU to build your own prediction models

The dashboard shows the current crowd levels in the indoor fitness facilities on campus. Facilities which are full will also display an estimated waiting time for users to get a sense of how long they will have to wait to enter.

For the second functionality, I trained a machine learning model from scratch on data on the crowd levels of a university gym. When predicting crowd levels, the model takes into account information including whether it is at the start of the school semester where the gyms are typically more crowded, or during school breaks where many students are off campus.

Thirdly, the data visualization helps users gain a better understanding of the trends in crowd levels and thus enables users to make better decisions on when is the best time to visit the gym.

Lastly, users can also take part in building their own model on the crowd data. This is a machine learning playground which automatically generates training code for the model based on user configured parameters, as well as provides a brief description on the characteristics of each model. Thus GymGuru **not only provides users with predictions on crowd levels**, it also **gives users the power to develop their own predictive models while also learning about machine learning**. So the next time you find yourself waiting in line to enter the gym, why not play around with this tool and gain some machine learning knowledge!

## How I built it
I trained a machine learning model from scratch using the scikit-learn library to predict crowd levels based on the date, day of the week, and time of semester (whether it's the start of semester, during school break, etc). I created a full stack website using Streamlit and connected the machine learning backend to the frontend. I used Plotly to visualize the dataset and extract insights from the data. I then hosted the website, so please go ahead and [play around with it](https://share.streamlit.io/liuhh02/gymguru/main/app.py)!

## Challenges I ran into
Due to the short amount of time available for the hackathon, I was unable to obtain data specific to CMU's fitness facilities (because the hackathon started on Friday night I could not obtain data so quickly from CMU Facilities Management Services). Hence I first obtained data online on the crowd levels at another university's gym and trained a machine learning model on the data to show the proof of concept.

## Accomplishments that I'm proud of
I'm proud of training a machine learning model from scratch and creating a fully functional full stack website in such a short amount of time!

## What I learned
Being my first time going solo for a hackathon, I learned a lot about the entire process of coding the frontend, training the machine learning model, linking the frontend and backend together and finally hosting it online. It is a lot of work, but it really helped provide a big picture overview of the entire web stack, and I'm excited to learn even more!

## What's next for GymGuru
The first step would be to link the app to CMU's data so that whenever someone swipes their card to enter the gym, it will be automatically reflected on the dashboard to show the current occupancy in the facility. Such a system preserves the privacy of the gym users as it does not require a video camera to capture and count the crowd numbers but rather only requires anonymized card swipes data that only keeps track of the numerical count.

Beyond the data, I plan to further improve the website's functionalities and make it production-ready for CMU's use!
