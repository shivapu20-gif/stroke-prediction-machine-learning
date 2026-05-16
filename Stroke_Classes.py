# Libraries
import streamlit as st 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import time
from streamlit_extras.colored_header import colored_header
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import graphviz
import zipfile
import os




# Data import
path = "C:/Users/Dellb/OneDrive/Desktop/stroke_data.csv/stroke_data.csv"
data = pd.read_csv(path)

# Change the name of the column
data.rename(columns={'sex': 'gender'}, inplace=True)

# Data Labelling for Visualization
data = data.copy()

# Mapping dictionaries for each column
gender_mapping = {0: 'Female', 1: 'Male'}
hypertension_mapping = {0: 'No hypertension', 1: 'hypertension'}
heart_disease_mapping = {0: 'No heart disease', 1: 'heart disease'}
ever_married_mapping = {0: 'No', 1: 'Yes'}
work_type_mapping = {0: 'Govt job', 1: 'Private', 2: 'Self-employed'}
residence_type_mapping = {0: 'Urban', 1: 'Rural'}
smoking_status_mapping = {0: 'Never smoked', 1: 'Smokes'}
stroke_mapping = {0: 'No', 1: 'Yes'}

# Use the replace() method to map the numerical codes to the corresponding labels
data['gender'] = data['gender'].replace(gender_mapping)
data['hypertension'] = data['hypertension'].replace(hypertension_mapping)
data['heart_disease'] = data['heart_disease'].replace(heart_disease_mapping)
data['ever_married'] = data['ever_married'].replace(ever_married_mapping)
data['work_type'] = data['work_type'].replace(work_type_mapping)
data['Residence_type'] = data['Residence_type'].replace(residence_type_mapping)
data['smoking_status'] = data['smoking_status'].replace(smoking_status_mapping)






# Web Application
st.set_page_config(
    page_title="Stroke Prediction", 
    page_icon="⚕️",
    )





# Page 1 - Homepage
def home_page():
    colored_header(
        label = "Welcome to Early Diagnosis of Stroke:brain:",
        description = "Hello there!:wave:",
        color_name = "red-70"
    )
    #st.title("Welcome to Early Diagnosis of Stroke:brain:")
    #color_name = "violet-70"
    
    st.write("Welcome to our Stroke Prediction Web App, where you can explore and predict stroke risk.")
    st.write("We're excited to have you on board. To get started, please enter your name below.")
    
    user_name = st.text_input("Please Enter Your Name:")
    if user_name:
        st.write(f"Hello, {user_name}!")
        st.markdown("Did you know, Stroke is a neurological disorder which occurs when blood flow to the brain is interrupted or disrupted.")
        st.markdown("This can happen when a blood clot or plaque blocks the artery carrying oxygen and nutrients to the brain or when the artery ruptures or bursts preventing blood from flowing into the brain.")
        st.markdown("Stroke is a major health concern globally and ranks second and third cause of death and disability respectively, accounting for approximately 6 million deaths and 15 million new cases annually. 1 in 4 adults with age over 25 years will have a stroke at least once in their lifetime.")
        st.markdown("For an indepth information, please watch this video.:loud_sound:")
        # Embed a video
        st.video("C:/Users/Dellb/Stroke Info.mp4")
        
        #st.subheader("How To Use The App:")
        
        colored_header(
            label = "How To Use The App:",
            description = "Look to the left:point_left:",
            color_name = "red-70"
        )


        st.write("1. Navigation:compass::")
        st.write("   - Use the sidebar on the left to navigate between different pages: Welcome, Explore, and Predict.")
        st.write("   - Click on the page you want to explore or use the dropdown menu to select it.")
        
        st.write("2. Explore Page:mag::")
        st.write("   - On the 'Explore' page, you can view and analyze the Exploratory Data Analysis (EDA) of the stroke data.")
        st.write("   - Select variables you want to explore from the dropdown.")
        st.write("   - Choose questions to answer based on the available checkboxes.")
        
        st.write("3. Predict Page:dart::")
        st.write("   - The 'Predict' page allows you to predict stroke outcomes using the model.")
        st.write("   - Choose the prediction method:")
        st.write("       - 'Single Row': Enter details for a single patient to get a prediction.")
        st.write("       - 'Upload Data': Upload a CSV file with patient data to get predictions.")
        st.write("   - After prediction, you can see the results and model evaluation.")
        
        st.write("4. Enjoy exploring and predicting!")
        
        
    
    
    
    
    
    
    
    
    
        
    


# Page 2 - Explore 
def explore_page():
    #st.title("Explore Data and Questions")
    
    colored_header(
        label = "Explore Data and Questions:mag:",
        description = "Welcome to the Explore page!:wave:",
        color_name = "red-70"
    )
    
    #st.write("Welcome to the Explore page!")
    st.write("Here, you can dive into the details of the stroke data and answer interesting questions.")
    st.subheader("How to Explore:")

    st.write("1. Select Variables:")
    st.write("   - Use the dropdown menu to select variables you want to explore.")
    st.write("   - Choose from options such as age, hypertension, bmi, and more.")
    st.write("   - Visualizations:bar_chart: and insights will be generated based on your selections.")
    
    # Dropdown variables
    selected_variables = st.multiselect("Select variables to explore:", ["Gender", "Heart Disease", "Age", "Hypertension", "BMI", "Marriage", "Work Type", "Residence Type", "Average Glucose Level", "Smoking"])
    
    # Explore Button
    explore_button = st.button("Explore")
    if explore_button:      
        
        with st.spinner("Please wait..."):
           time.sleep(0.1)
        
        st.success("Visualization Successful:white_check_mark:")
         
        # Display the visualizations based on selected variables
        if selected_variables:
            for variable in selected_variables:
                    # st.subheader(f"{variable}")
                    if variable == "Gender":
                        st.subheader("Which gender is more likely to suffer from stroke?")
                        sns.set(style="whitegrid")
                        plt.figure(figsize=(8, 6))
                        ax = sns.countplot(x='gender', hue='stroke', data=data)
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
                        plt.xlabel('Gender')
                        plt.ylabel('Count')
                        plt.title('Stroke Distribution by Gender')
                        st.pyplot(plt)
                        st.write("Explanation: In the dataset, there are more males than females but percentage of females suffering from stroke is higher. Specifically, 44% of males have suffered from stroke while females are at 56%. This observation aligns with the study done by Rexrode, et al., (2022). The study mentions that the lifetime risk of stroke for a woman is higher than man especially after the age of 25 as statistically women live longer than men. Furthermore, the higher risk can be explained by gender specific risk factors such as hormonal influences, age at menopause, reproductive health variables such as pregnancy outcomes, all play a significant role in shaping the stroke risk. Moreover, health conditions such as diabetes and hypertension affect woman different than men. ")

                        
                    if variable == "Marriage":
                        st.subheader("Do married individuals have a higher risk of stroke?")
                        sns.set(style="whitegrid") 
                        plt.figure(figsize=(8, 6))
                        ax = sns.countplot(x='ever_married', hue='stroke', data=data)
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
                        plt.xlabel('Ever Married')
                        plt.ylabel('Count')
                        plt.title('Stroke Distribution by Ever Married')
                        st.pyplot(plt)
                        st.write("Explanation: In the dataset, 18,179 out of 33,552 married patients have suffered from stroke while only 2,223 out of 7,300 unmarried patients have suffered stroke. This translates to 54% of married patients suffer from stroke while only 30% of unmarried patients have suffered from stroke. Based on these statistics, it appears that married patients have a higher risk of stroke. ")
                        
                      
                    if variable == "Work Type":
                        st.subheader("Which work type has a higher risk of getting stroke?")
                        sns.set(style="whitegrid") 
                        plt.figure(figsize=(8, 6)) 
                        ax = sns.countplot(x='work_type', hue='stroke', data=data) 
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
                        plt.xlabel('Work Type')
                        plt.ylabel('Count')
                        plt.title('Stroke Distribution by Work Type')
                        st.pyplot(plt)
                        st.write("Explanation: In the dataset, 60% of patients who are self-employed suffered from stroke, 48% of patients with Private jobs, 45% of patients with Govt job and 0% of children and patients who have never worked have suffered stroke. We don’t have a lot of data on patients who never worked and one that we do, have never suffered from stroke. This is also the case for the children category. This does not mean that these categories do not suffer from stroke but rather, it simply indicated that we have small information on them. Of course, children are extremely unlikely to suffer from stroke. The percentage of working patients who have suffered from stroke in the dataset is 50%. It is inconclusive that working or any specific work type increases the risk of stroke.")
                   
                        
                    if variable == "Residence Type":
                        st.subheader("Which residence type has higher risk of getting stroke?")
                        sns.set(style="whitegrid") 
                        plt.figure(figsize=(8, 6)) 
                        ax = sns.countplot(x='Residence_type', hue='stroke', data=data) 
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
                        plt.xlabel('Residence Type')
                        plt.ylabel('Count')
                        plt.title('Stroke Distribution by Residence Type')
                        st.pyplot(plt)
                        st.write("Explanation: From the figure, 10,621 out of 21,031 patients living in Rural area have suffered from stroke which is 50.5% of the patients. This is also the case for Urban residence type here 49.5% of the patients in Urban residences have suffered from stroke. There is no indication whether urban or rural residents have a higher risk of stroke.")
                        
                        
                    if variable == "Hypertension":
                        st.subheader("How does hypertension influence the likelihood of getting a stroke?")
                        sns.set(style="whitegrid") 
                        plt.figure(figsize=(8, 6)) 
                        ax = sns.countplot(x='hypertension', hue='stroke', data=data) 
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
                        plt.xlabel('Hypertension')
                        plt.ylabel('Count')
                        plt.title('Stroke Distribution by Hypertension')
                        st.pyplot(plt)
                        st.write("Explanation: The number of observation of patients with no hypertension is significantly higher than with hypertension. However, over 70% of patients who have hypertension have suffered from stroke while only 43% of patients who do not have hypertension have suffered from stroke. Therefore, it is conclusive to state that having hypertension does increase the risk of having stroke. It is one of the most important risk factors. ")
                  
                    
                    if variable == "Heart Disease":
                        st.subheader("Do patients with previously diagnosed heart disease have a higher risk of stroke?")
                        sns.set(style="whitegrid") 
                        plt.figure(figsize=(8, 6)) 
                        ax = sns.countplot(x='heart_disease', hue='stroke', data=data) 
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
                        plt.xlabel('Heart Disease')
                        plt.ylabel('Count')
                        plt.title('Stroke Distribution by Heart Disease')
                        st.pyplot(plt)
                        st.write("Explanation: Heart disease have the same case as hypertension, where the number of observations of patients with no heart disease is significantly higher than with heart disease. But over 79% of patients with heart disease have suffered stroke while 45% of patients with no heart disease have suffered stroke. Therefore, it is conclusive to state that having heart disease does increase the risk of having stroke. It is one of the most important risk factors.")
                  
                    
                    if variable == "Smoking":
                        st.subheader("Does smoking increase the likelihood of getting a stroke?")
                        sns.set(style="whitegrid") 
                        plt.figure(figsize=(8, 6)) 
                        ax = sns.countplot(x='smoking_status', hue='stroke', data=data) 
                        for p in ax.patches:
                            height = p.get_height()
                            ax.annotate(f'{height}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')
                        plt.xlabel('Smoking Status')
                        plt.ylabel('Count')
                        plt.title('Stroke Distribution by Smoking Status')
                        st.pyplot(plt)
                        st.write("Explanation: The number of observations for smokes and never smokes is approximately equal at 19,966 and 20,886 observations. Over 10,673 patients who smoke have suffered a stroke which is 53.5% while 9,764 patients who do not smoke have suffered stroke which 46%. There does seem to be a slight association between smoking and stroke. This is because smoking causes buildup of fatty substances in the main artery which supplied blood to the brain. Additionally, nicotine increases the blood pressure and the carbon monoxide reduces oxygen levels in the blood. This thickens the blood, making it easier to clot.")
                        
                    
                    if variable == "Age":
                        st.subheader("Which age group is more likely to suffer from stroke?")
                        stroke_data = data[data['stroke'] == 1]
                        no_stroke_data = data[data['stroke'] == 0]
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.kdeplot(data=stroke_data['age'], shade=True, label='Stroke', color='blue')
                        sns.kdeplot(data=no_stroke_data['age'], shade=True, label='No Stroke', color='orange')
                        plt.xlabel('Age')
                        plt.ylabel('Density')
                        plt.title('Density Plot of Age by Stroke Event')
                        plt.legend(title='Stroke Event', labels=['Stroke', 'No Stroke'])
                        st.pyplot(fig)
                        st.write("Explanation: Looking at the density plot, there does not seem to be any strong association between age and stroke. The density plot for both stroke and no stroke follow the same distribution and both have peaks at age 40 to 60 range but that is because of the high concentration of data points within that range. However, there does seem to be a small association between older age and stroke as there is higher density for stroke towards the older age. ")
                        
                        
                    if variable == "Average Glucose Level":
                        st.subheader("Do subjects with higher average glucose level have higher risk of getting a stroke?")
                        stroke_data = data[data['stroke'] == 1]
                        no_stroke_data = data[data['stroke'] == 0]
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.kdeplot(data=stroke_data['avg_glucose_level'], shade=True, label='Stroke', color='blue')
                        sns.kdeplot(data=no_stroke_data['avg_glucose_level'], shade=True, label='No Stroke', color='orange')
                        plt.xlabel('Average Glucose Level')
                        plt.ylabel('Density')
                        plt.title('Density Plot of Average Glucose Level by Stroke Event')
                        plt.legend(title='Stroke Event', labels=['Stroke', 'No Stroke'])
                        st.pyplot(fig)
                        st.write("Explanation: From the density plot, it is clear that high glucose level increases the risk of stroke. Starting from glucose level of 180, the density of stroke cases increases significantly. This spike in density can be explained to diabetes. According to American Diabetes Association, having glucose level of 180-200 is considered prediabetic and levels over 200 is diabetic.  ")
                      
                    
                    if variable == "BMI":
                        st.subheader("Does BMI have impact on the risk of getting a stroke?")
                        plt.figure(figsize=(8, 6))
                        sns.boxplot(x='stroke', y='bmi', data=data, palette='Set2')
                        plt.xlabel('Stroke Event')
                        plt.ylabel('BMI')
                        plt.title('Box Plot of BMI by Stroke Event')
                        plt.xticks(ticks=[0, 1], labels=['No Stroke', 'Stroke'])
                        st.pyplot(plt)
                        st.write("Explanation: From the box plot, BMI does not have correlation with stroke as BMI levels of stroke and no stroke events are approximately equal. Studies suggest that high BMI increases the risk of stroke because it indicates high boy fatness which may cause health problems. However, according to the box plot, patients with low BMI have suffered stroke and patients with extremely high BMI have no suffered stroke. Therefore, there is no strong evidence to prove that BMI have a direct impact on stroke.")
                        
    
                   
            
    #QUIZ!!!
    def display_correlation():
        image8=Image.open("C:/Users/Dellb/feature selection.png")
        st.write("The bar chart belows shows the correlation between the Features and Stroke.")
        st.write("As you can see, Hypertension, Heart Disease and Average Glucose Level are some of the highest correlated features with stroke.")
        st.image(image8, use_column_width=True)
        st.write("Correlation measures the strength and direction of a linear relationship between two variables.")
        st.write("A correlation value can range from -1 to 1:")
        st.write("- A value close to 1 indicates a strong positive correlation, where as one variable increases, the other tends to increase.")
        st.write("- A value close to -1 indicates a strong negative correlation, where as one variable increases, the other tends to decrease.")
        st.write("- A value close to 0 (like 0.2) indicates a weak positive correlation, meaning that there is some tendency for the variables to increase together, but it's not very strong.")
        

    # Define the correct answer
    correct_answer = ["Hypertension", "Heart Disease", "Average Glucose Level"]

    # Display the quiz question
    #st.subheader("Quiz!!!")
    
    colored_header(
        label = "Quiz!!!",
        description = "Hint: There are :three: risk factors:shushing_face:",
        color_name = "red-70"
    )
    st.subheader("After exploring the features, what do you think are the biggest risk factors of stroke?")
    options = ["Gender", "Heart Disease", "Age", "Hypertension", "BMI", "Marriage", "Work Type", "Residence Type", "Average Glucose Level", "Smoking"]
    user_answers = st.multiselect("Select the features you think are the biggest risk factors:", options)
    
    # Check user's answers and display result
    if st.button("Submit"):
        correct_guesses = [answer for answer in user_answers if answer in correct_answer]
        
        if len(correct_guesses) == len(correct_answer):
            st.sidebar.balloons()
            st.success("🎉🎉🎉 Congratulations! You guessed all the correct answers! 🎉🎉🎉")
            display_correlation()
        elif len(correct_guesses) >= 1:
            st.write("You're almost there!")
            st.write("The correct answers are:", ", ".join(correct_answer))
            display_correlation()
            
        else:
            st.write("Wrong guess. The correct answers are:", ", ".join(correct_answer))
            display_correlation()











   

            
# Load the saved model
loaded_model = joblib.load("DecisionTree_model.joblib")


# Function topass user inputs to model for prediction
def predict_stroke(user_inputs):
    probabilities = loaded_model.predict_proba(user_inputs)
    probability_of_stroke = probabilities[0][1]
    return probability_of_stroke              
                

# Page 3 - Predict
def predict_page():
    colored_header(
        label="Predict Risk of Stroke:timer_clock:",
        description="",
        color_name="red-70"
    )
    
    st.write("There are two options to choose from:")
    st.write("- Single Row: Predict the stroke risk of a single individual.")
    st.write("- Upload Data: You can upload a test data and use the algorithm to predict stroke.")
    
    option = st.radio("Choose prediction method:", ["Single Row", "Upload Data"])
    
    if option == "Single Row":
        colored_header(
            label="Single Row",
            description="",
            color_name="red-70"
        )
        st.write("Enter details for a single individual to predict their stroke risk.")
        
        st.title("Stroke Prediction")
        st.write("Please provide the following details to predict your stroke outcome.")
        
        # Mapping
        hypertension_mapping = {0: 'No', 1: 'Yes'}
        heart_disease_mapping = {0: 'No', 1: 'Yes'}
        ever_married_mapping = {0: 'No', 1: 'Yes'}
        work_type_mapping = {0: 'Govt job', 1: 'Private', 2: 'Self-employed'}
        gender_mapping =  {0: 'Female', 1: 'Male'}
        
        # Label Encoding
        hypertension_encoder = LabelEncoder()
        hypertension_encoder.fit(list(hypertension_mapping.values()))
        
        heart_disease_encoder = LabelEncoder()
        heart_disease_encoder.fit(list(heart_disease_mapping.values()))
        
        ever_married_encoder = LabelEncoder()
        ever_married_encoder.fit(list(ever_married_mapping.values()))
        
        work_type_encoder = LabelEncoder()
        work_type_encoder.fit(list(work_type_mapping.values()))
        
        gender_encoder = LabelEncoder()
        gender_encoder.fit(list(gender_mapping.values()))
    
        # User Inputs
        gender = st.radio("What is your gender?", list(gender_mapping.values()))
        hypertension = st.radio("Do you have Hypertension?", list(hypertension_mapping.values()))
        heart_disease = st.radio("Do you have any Heart Disease?", list(heart_disease_mapping.values()))
        ever_married = st.radio("Are you Married?", list(ever_married_mapping.values()))
        work_type = st.radio("What kind of work do you do?", list(work_type_mapping.values()))
        avg_glucose_level = st.slider("What is your average glucose level", 50, 300, 25 )
        
        # Convert categorical inputs to numerical values
        gender_num = gender_encoder.transform([gender])[0]
        hypertension_num = hypertension_encoder.transform([hypertension])[0]
        heart_disease_num = heart_disease_encoder.transform([heart_disease])[0]
        ever_married_num = ever_married_encoder.transform([ever_married])[0]
        work_type_num = work_type_encoder.transform([work_type])[0]
        
        
        # Defining risk_factors based on hypertension, heart_disease and avg_glucose_level
        if hypertension_num not in [0, 1] or heart_disease_num not in [0, 1]:
            # Handle unexpected values for hypertension or heart_disease
            st.error("Invalid values for hypertension or heart_disease")
        else:
            # Calculate health_condition based on hypertension and heart_disease
            if hypertension_num == 0 and heart_disease_num == 0:
                health_condition = "No Health Condition"
            else:
                health_condition = "Health Condition"
        
            # Calculate risk_factors based on health_condition and avg_glucose_level
            if health_condition == "No Health Condition" and avg_glucose_level < 180:
                risk_factors = "No Risk"
            elif health_condition == "Health Condition" and avg_glucose_level < 180:
                risk_factors = "Moderate Risk"
            elif health_condition == "No Health Condition" and avg_glucose_level >= 180:
                risk_factors = "Moderate Risk"
            elif health_condition == "Health Condition" and avg_glucose_level >= 180:
                risk_factors = "High Risk"

        # Make prediction when user clicks the "Predict" button
        if st.button("Predict"):
            user_inputs = [
                [gender_num, hypertension_num, heart_disease_num, ever_married_num, work_type_num] + [avg_glucose_level]
            ]
            stroke = predict_stroke(user_inputs)
            
            if stroke == 0 and risk_factors == "No Risk":
                stroke_class = "No Risk"
            elif stroke == 0 and risk_factors == "Moderate Risk":
                stroke_class = "Low Risk"
            elif stroke == 0 and risk_factors == "High Risk":
                stroke_class = "Moderate Risk"
            elif stroke == 1 and risk_factors == "No Risk":
                stroke_class = "Moderate Risk"
            elif stroke == 1 and risk_factors == "Moderate Risk":
                stroke_class = "High Risk"
            elif stroke == 1 and risk_factors == "High Risk":
                stroke_class = "High Risk"
                
            if stroke_class == "No Risk":
                st.success("Great news! You are NOT LIKELY to suffer from a stroke.")
                st.markdown("For valuable information on stroke prevention, we recommend exploring the following resources:")
                st.markdown("[National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) - Stroke Prevention](https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke)")
                st.markdown("[National Health Service (NHS) - Stroke Prevention](https://www.nhs.uk/conditions/stroke/prevention/#:~:text=The%20best%20way%20to%20help,clogged%20with%20fatty%20substances%20(atherosclerosis))")
                
            elif stroke_class == "Low Risk":
                st.warning("You have a LOW Risk of suffering from a stroke.")
                st.markdown("To learn more about stroke prevention, please consider reviewing this informative resource:")
                st.markdown("[National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) - Stroke Prevention](https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke)")
                st.markdown("[National Health Service (NHS) - Stroke Prevention](https://www.nhs.uk/conditions/stroke/prevention/#:~:text=The%20best%20way%20to%20help,clogged%20with%20fatty%20substances%20(atherosclerosis))")
                
            elif stroke_class == "Moderate Risk":
                st.warning("You have a MODERATE Risk of suffering from a stroke.")
                st.markdown("For further insights into stroke prevention, we encourage you to explore these valuable resources:")
                st.markdown("[National Health Service (NHS) - Stroke Prevention](https://www.nhs.uk/conditions/stroke/prevention/#:~:text=The%20best%20way%20to%20help,clogged%20with%20fatty%20substances%20(atherosclerosis))")
                st.markdown("[National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) - Stroke Prevention](https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke)")
                
            elif stroke_class == "High Risk":
                st.warning("You are at a HIGH RISK of suffering from a stroke.")
                st.markdown("For urgent information on stroke prevention, please explore these crucial resources:")
                st.markdown("[National Health Service (NHS) - Stroke Prevention](https://www.nhs.uk/conditions/stroke/prevention/#:~:text=The%20best%20way%20to%20help,clogged%20with%20fatty%20substances%20(atherosclerosis))")
                st.markdown("[National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) - Stroke Prevention](https://www.niddk.nih.gov/health-information/diabetes/overview/preventing-problems/heart-disease-stroke)")

                 
                 
      # Display model accuracy
        if st.button("View Accuracy of Prediction"):
            curve = Image.open("C:/Users/Dellb/decision_tree_curve.png")
            st.image(curve, caption="Precision-Recall Curve", use_column_width=True)               
            st.write("Explanation: The precision-recall curve is a representation of model’s performance across different threshold. The ideal scenario would for the curve to reach or hit the top-right corner of (1,1).")
            st.write("To understand how the prediciton was made, you can download the Decision Tree below.")
            

            # Function to create the zip file
            def create_zip():
                # Create a zip file
                with zipfile.ZipFile('decision_tree_visualization.zip', 'w') as zipf:
                    zipf.write('decision_tree_visualization.png')
            
            # Check if the zip file exists
            if not os.path.exists('decision_tree_visualization.zip'):
                create_zip()
            
            # Display a download button
            if st.button("Click if you want to download the Decision Tree"):
                with open('decision_tree_visualization.zip', 'rb') as f:
                    bytes_data = f.read()
                st.download_button(
                    label="Download Decision Tree",
                    data=bytes_data,
                    file_name="decision_tree_visualization.zip",
                    mime="application/zip"
                )
            
            # PIL Image Resampling
            img = Image.open("decision_tree_visualization.png")
            resized_img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
            resized_img.save("decision_tree_visualization_resized.png")
                        
            

    # Option 2 - Upload Data
    if option == "Upload Data":
        colored_header(
            label="Upload Data",
            description="",
            color_name="red-70"
        )
        st.write("Upload a test data to predict their stroke risk.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded_file is not None:
            # Load the uploaded CSV file into a DataFrame
            uploaded_data = pd.read_csv(uploaded_file)
            
            # Change the name of the column
            uploaded_data.rename(columns={'sex': 'gender'}, inplace=True)
            
            uploaded_data = uploaded_data.copy()

            # Mapping dictionaries for each column
            gender_mapping = {0: 'Female', 1: 'Male'}
            hypertension_mapping = {0: 'No hypertension', 1: 'hypertension'}
            heart_disease_mapping = {0: 'No heart disease', 1: 'heart disease'}
            ever_married_mapping = {0: 'No', 1: 'Yes'}
            work_type_mapping = {0: 'Never worked', 1: 'Children', 2: 'Govt job', 3: 'Self-employed', 4: 'Private'}
            residence_type_mapping = {0: 'Urban', 1: 'Rural'}
            smoking_status_mapping = {0: 'Never smoked', 1: 'Smokes'}
            stroke_mapping = {0: 'No', 1: 'Yes'}
            
            # Use the replace() method to map the numerical codes to the corresponding labels
            uploaded_data['gender'] = uploaded_data['gender'].replace(gender_mapping)
            uploaded_data['hypertension'] = uploaded_data['hypertension'].replace(hypertension_mapping)
            uploaded_data['heart_disease'] = uploaded_data['heart_disease'].replace(heart_disease_mapping)
            uploaded_data['ever_married'] = uploaded_data['ever_married'].replace(ever_married_mapping)
            uploaded_data['work_type'] = uploaded_data['work_type'].replace(work_type_mapping)
            uploaded_data['Residence_type'] = uploaded_data['Residence_type'].replace(residence_type_mapping)
            uploaded_data['smoking_status'] = uploaded_data['smoking_status'].replace(smoking_status_mapping)
            
                       
            # Preprocessing
            uploaded_data.dropna(inplace=True)
            uploaded_data.drop_duplicates(inplace=True)
            uploaded_data = uploaded_data[(uploaded_data['work_type'] != 'Children') & (uploaded_data['work_type'] != 'Never worked')]
            
            # List of categorical variables to be label encoded
            categorical_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
            
            LE = LabelEncoder()
            
            # Initialize the encoding mapping dictionary
            encoding_mapping = {}
            
            # Apply label encoding 
            for i in categorical_features:
                uploaded_data[i] = LE.fit_transform(uploaded_data[i])
                encoding_mapping[i] = dict(zip(range(len(LE.classes_)), LE.classes_))
            

            # Drop unnecessary columns
            uploaded_data = uploaded_data.drop(['age', 'Residence_type', 'bmi', 'smoking_status'], axis=1)
            
            st.write(uploaded_data.head(10))
        
            # Make predictions when user clicks the "Predict" button
            if st.button("Predict"):
                # Drop the 'stroke' column if it's in the uploaded data
                if 'stroke' in uploaded_data.columns:
                    data_for_prediction = uploaded_data.drop('stroke', axis=1)
                else:
                    data_for_prediction = uploaded_data  # Keep all columns for prediction
                
                predictions = predict_stroke(data_for_prediction)
            
                # Create a DataFrame with predictions and the corresponding index from uploaded_data
                result_df = pd.DataFrame({'Prediction': predictions}, index=uploaded_data.index)
            
                st.write("Predicted Stroke Risks:")
                st.write("- 0 stands for No Stroke")
                st.write("- 1 stands for Stroke")
                st.dataframe(result_df)

# Create navigation menu
menu = ["Home", "Explore", "Predict"]
choice = st.sidebar.selectbox("Select Page", menu)

# Display selected page content
if choice == "Home":
    home_page()
elif choice == "Explore":
    explore_page()
elif choice == "Predict":
    predict_page()
