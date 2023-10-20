import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
	return 'Welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(sepal_length, sepal_width, petal_length, petal_width):

	prediction = classifier.predict(
		[[sepal_length, sepal_width, petal_length, petal_width]])
	print(prediction)
	return prediction
	

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("SimplestAI App")

	
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	sepal_length = st.text_input("Sepal Length")
	sepal_width = st.text_input("Sepal Width")
	petal_length = st.text_input("Petal Length")
	petal_width = st.text_input("Petal Width")
	result =""
	
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
    # 'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2}
	if st.button("Predict"):
		result = prediction(sepal_length, sepal_width, petal_length, petal_width)
		if result == 0:
			o = "Iris-setosa"
		elif result == 1:
			o = "Iris-versicolor"
		else:
			o = "Iris-virginica"
		st.success('The flower with given specification is {}'.format(o))
	
if __name__=='__main__':
	main()
