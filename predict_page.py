import streamlit as st
import pickle 
import pandas as pd 
import numpy as np 

def preprocess(X, major_input, year_input):
    # gender 
    X["Gender"] = X["Gender"].replace({"Male" : 1, "Female" : 0})

    # year and major
    majors = ['Accounting', 'Applied Liberal Arts', 'Arabic as a Second Language',
       'Banking', 'Biomedical science', 'Biotechnology',
       'Business Administration', 'Business Information Technology',
       'Certified Technology Specialist', 'Communications', 'Computer Science',
       'Economics', 'Engineering', 'English as a Second Language',
       'Environmental and Natural Resource Management', 'Human Resources',
       'Human Sciences', 'Information Technology', 'Islamic Education',
       'Islamic Revealed Knowledge and Human Sciences', 'Law',
       'Marine science', 'Masters in Health Science', 'Mathematics', 'Nursing',
       'Pharmaceuticals', 'Philosophy', 'Principles of Islam', 'Psychology',
       'Radiography', 'Religion']
    years = ["First-year", "Sophomore", "Junior", "Senior"]
    for y in years:
        if y == year_input:
            X[y] = 1
        else:
            X[y] = 0
    for m in majors:
        if m == major_input:
            X[m] = 1
        else:
            X[m] = 0
    X = X.drop(columns = ["Major", "Year"])
    
    # gpa
    X.loc[X["GPA"] == "0 - 1.99", "GPA"] = 0
    X.loc[X["GPA"] == "2.00 - 2.49", "GPA"] = 1
    X.loc[X["GPA"] == "2.50 - 2.99", "GPA"] = 2
    X.loc[X["GPA"] == "3.00 - 3.49", "GPA"] = 3
    X.loc[X["GPA"] == "3.50 - 4.00", "GPA"] = 4

    # marital status
    X["Marital Status"] = X["Marital Status"].replace({"Married": 1, "Not Married": 0})

    # anxiety
    X["Anxiety"] = X["Anxiety"].replace({"I have anxiety": 1, "I do not have anxiety": 0})

    # panic
    X["Panic"] = X["Panic"].replace({"I experience panic attacks": 1, "I do not experience panic attacks": 0})

    # treated 
    X["Treated"] = X["Treated"].replace({"I am being treated for my disorder(s)": 1, "I am not being treated for my disorder(s)": 0})

    return X.to_numpy()


def load_model():
    with open("saved_model.pkl", "rb") as file:
        data = pickle.load(file)

    return data

data = load_model()
model = data["model"]

def show_predict_page():
    st.title("Are you Depressed?")
    st.write("""###### Input the data that best represents yourself. I will predict if you're depressed.""")
    st.write("""###### \n\n\n\n\n Training data sourced from https://www.kaggle.com/datasets/shariful07/student-mental-health/data""")

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", value = None, min_value=17, max_value=24)
    major = st.selectbox("Major", ['Accounting', 'Applied Liberal Arts', 'Arabic as a Second Language',
       'Banking', 'Biomedical science', 'Biotechnology',
       'Business Administration', 'Business Information Technology',
       'Certified Technology Specialist', 'Communications', 'Computer Science',
       'Economics', 'Engineering', 'English as a Second Language',
       'Environmental and Natural Resource Management', 'Human Resources',
       'Human Sciences', 'Information Technology', 'Islamic Education',
       'Islamic Revealed Knowledge and Human Sciences', 'Law',
       'Marine science', 'Masters in Health Science', 'Mathematics', 'Nursing',
       'Pharmaceuticals', 'Philosophy', 'Principles of Islam', 'Psychology',
       'Radiography', 'Religion'])
    year = st.selectbox("Year in School", ["First-year", "Sophomore", "Junior", "Senior"])
    gpa = st.selectbox("GPA", ["0 - 1.99", "2.00 - 2.49", "2.50 - 2.99", "3.00 - 3.49", "3.50 - 4.00"])
    marital_status = st.selectbox("Marital Status", ["Married", "Not Married"])
    anxiety = st.selectbox("Anxiety", ["I have anxiety", "I do not have anxiety"])
    panic = st.selectbox("Panic Attacks", ["I experience panic attacks", "I do not experience panic attacks"])
    treated = st.selectbox("Treatment", ["I am being treated for my disorder(s)", "I am not being treated for my disorder(s)"])
    give_prediction = st.button("Submit")
    
    if give_prediction == True:
        X = pd.DataFrame( {"Gender": gender, "Age": age, "GPA": gpa, "Marital Status": marital_status, "Anxiety": anxiety, "Panic": panic, "Treated": treated , "Major": major, "Year": year}, index = [0])
        X = preprocess(X, major, year)
        X = X.astype(float)

        pred = model.predict(X)
        if pred == 1:
            st.subheader("You are likely depressed")
        else:
            st.subheader("You are likely not depressed")
        st.write(f"###### Prediction confidence: 84.62 \%")
