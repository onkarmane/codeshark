import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('model_pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
	return 'Welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
# def prediction(age,sex,	cp,	trestbps,	chol,	fbs	,restecg,	thalach	,exang,	oldpeak	,slope,	ca,	thal):
def prediction(df):
	prediction = classifier.predict(df)
	print(prediction)
	return prediction

    
	
# This is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("CodeShark Heart disease prediction App")

	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	left_column, right_column = st.columns(2)
	age = left_column.slider("age")
	sex = left_column.number_input("sex",min_value=0, max_value=1)

	cp = left_column.number_input("cp",min_value=0, max_value=3)
	trestbps = left_column.number_input("trestbps")
	chol = left_column.number_input("chol")
	fbs = left_column.number_input("fbs")
	restecg = left_column.number_input("restecg",min_value=0, max_value=2)
	thalach = left_column.number_input("thalach")
	exang = left_column.number_input("exang",min_value=0, max_value=1)
	oldpeak = left_column.number_input("oldpeak")
	slope = left_column.number_input("slope",min_value=0, max_value=2)
	ca = left_column.number_input("ca",min_value=0, max_value=4)
	thal = left_column.number_input("thal",min_value=0, max_value=3)
	result =""
	df = pd.DataFrame(data=[age,sex, cp,	trestbps,	chol,	fbs	,restecg,	thalach	,exang,	oldpeak	,slope,	ca,	thal])
	l = np.array([age,sex, cp,	trestbps,	chol,	fbs	,restecg,	thalach	,exang,	oldpeak	,slope,	ca,	thal])
	l = l.reshape(1,-1)
	
	if right_column.button("Predict"):
		result = prediction(l)
		if result == 0:
			o = "No Heart Disease"
		else:
			o = "Heart Disease"
		right_column.success('The patient has {}'.format(o))
	
if __name__=='__main__':
	main()
