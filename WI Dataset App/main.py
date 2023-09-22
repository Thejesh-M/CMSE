import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv')

st.write("""
# WI Cancer Data Analysis
""")
         
chosen_column = st.selectbox("Choose a column:", df.columns)

if chosen_column:
    st.subheader(f"seaborn plot for {chosen_column}")
    sns.histplot(df, x=chosen_column, hue="diagnosis", kde=True)
    plt.xlabel(chosen_column)
    plt.title(f"Plot of {chosen_column} by diagnosis")
    st.pyplot()
st.write("Basic Information of data:")
st.write(df.describe())
st.write("Distribution Plot:")
st.bar_chart(df["diagnosis"].value_counts())
         
sns.displot(df, x='radius_mean',hue='diagnosis',kde=True)

st.pyplot()