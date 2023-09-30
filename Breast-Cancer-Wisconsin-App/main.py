import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Keep this to avoid unwanted warning on the wen app
st.set_option('deprecation.showPyplotGlobalUse', False)

# Loading dataset
data = pd.read_csv('Breast-Cancer-Wisconsin-App/data.csv')
data.drop(['Unnamed: 32','id'], axis=1, inplace=True)
X = data.drop('diagnosis',axis =1)
y = data['diagnosis']
print(data.columns.tolist())
# Title 
st.title('Breast Cancer Wisconsin (Diagnostic) Analysis')
st.write('Interactive plots on Data')

# Sidebar
st.sidebar.title('Control Panel')
plot_type = st.sidebar.selectbox('Select a plot here:', ('Histogram', 'Box Plot'))

# Interactive Plots
if plot_type == 'Histogram':
    st.sidebar.subheader('Select a feature for the histogram:')
    selected_feature = st.sidebar.selectbox('Select a feature', X.columns.tolist())
    bin_count = st.sidebar.slider('Number of Bins', min_value=1, max_value=100, value=20)
    st.subheader(f'Histogram of {selected_feature}')
    sns.histplot(data= data, x= data[selected_feature], hue='diagnosis', kde=True, bins=bin_count)
    st.pyplot()

elif plot_type == 'Box Plot':
    st.sidebar.subheader('Select a feature for the box plot:')
    numeric_columns = X.select_dtypes(include=['int', 'float']).columns.tolist()
    selected_feature = st.sidebar.selectbox('Select a feature', numeric_columns)
    
    whisker_length = st.sidebar.slider('Whisker Length', min_value=1, max_value=10, value=2) 
    st.subheader(f'Box Plot of {selected_feature}')
    sns.boxplot(x=(y == 1), y=data[selected_feature],palette='Set1', whis=whisker_length)
    plt.xlabel('Diagnosis (0: Benign, 1: Malignant)')
    plt.ylabel(selected_feature)
    st.pyplot()


st.write('Basic Breast Cancer Dataset Information:')
st.write(f'Total Number of Samples: {X.shape[0]}')
st.write(f'Number of Features: {X.shape[1]}')

# Display the dataset
if st.checkbox('Show Raw Data'):
    st.write(pd.DataFrame(data, columns=data.columns))

# Add some explanatory text
st.write('''
This Streamlit app allows the user to explore the Breast Cancer dataset using interactive plots.
You can choose the type of plot and features to visualize in the sidebar on the left.
You can also adjust thresholds and parameters for each plot. Like the whisker length for box plot 
         and number of bins in histogram.
''')
