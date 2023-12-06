import streamlit as st
from imblearn.over_sampling import SMOTE
import seaborn as sns
import pandas as pd
import altair as alt
import plotly.express as px
from PIL import Image
import hiplot as hip
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")


# Add custom CSS to center the content
st.write('<h1 style="text-align:center; vertical-align:middle; line-height:2; color:#046366;">Global Crises Data by Country</h1>', unsafe_allow_html=True)

def train_model(y_name,X_train, y_train):
    rf = RandomForestClassifier(criterion = 'gini', n_estimators=150, max_depth=10)
    rf.fit(X_train, y_train)
    return rf

def predict_crisis(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def generate_roc_plot(fpr, tpr, thresholds):
    roc_df = pd.DataFrame()
    roc_df['fpr'] = fpr
    roc_df['tpr'] = tpr
    roc_df['thresholds'] = thresholds
    roc_line = alt.Chart(roc_df).mark_line(color = 'red').encode(
                                                        alt.X('fpr', title="false positive rate"),
                                                        alt.Y('tpr', title="true positive rate"))
    roc = alt.Chart(roc_df).mark_area(fillOpacity = 0.5, fill = 'red').encode(alt.X('fpr', title="false positive rate"),
                                                            alt.Y('tpr', title="true positive rate"))
    baseline = alt.Chart(roc_df).mark_line(strokeDash=[20,5], color = 'black').encode(alt.X('thresholds', scale = alt.Scale(domain=[0, 1]), title=None),
                                                        alt.Y('thresholds', scale = alt.Scale(domain=[0, 1]), title=None))
    c = roc_line + roc + baseline.properties(title='ROC Curve').interactive()
    return c


def get_plots(mod, X_train, X_test, y_train, y_test):
    mod.fit(X_train, y_train)
    y_pred = mod.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, mod.predict_proba(X_test)[:,1])
    cm = confusion_matrix(y_pred, y_test)
    c = generate_roc_plot(fpr, tpr, thresholds)
    fig = plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True,fmt='g')

    precision = round(precision_score(y_test, y_pred),3)
    recall = round(recall_score(y_test, y_pred),3)
    accuracy = round(accuracy_score(y_test, y_pred),3)
    f1 = round(f1_score(y_test, y_pred),3)
    
    return c, fig, precision, recall, f1, accuracy


df = pd.read_csv("Global-Crisis/african_crises.csv")
df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])
df = df[df["currency_crises"]<=1]
df["year"] = pd.to_datetime(df.year, format='%Y')
scaled_df = df.copy()


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["About Data", "Crisis Over Time", "Interactive Plots","Predicting Crises","Make Predictions","About me"])

with tab1 :
    image_path = "Global-Crisis/bg.png"  # Replace with the actual file path

    # Check if the image file exists at the specified path
    try:
        with open(image_path, "rb") as image_file:
            img = Image.open(image_file)
            img = img.resize((img.width, 300))
            st.image(img, caption="Global Crisis", use_column_width=True)
    except FileNotFoundError:
        pass

    st.write('Welcome to our web app dedicated to exploring the economic landscape of African countries. Africa, a continent of incredible diversity and promise, has faced its share of economic challenges over the years. This platform is your gateway to understanding the different types of crises that have shaped the economic trajectory of African nations.')
    st.write('From financial meltdowns to currency turmoil and banking crises, this interactive visualization dives deep into the data, allowing you to uncover patterns, trends, and insights that shed light on the economic realities of the continent. With the power to explore crisis dynamics across regions and time, this tool empowers you to draw valuable conclusions and insights from the data.')

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.write('This dataset is a derivative of Reinhart et. Global Financial Stability dataset which can be found online at: https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx The dataset will be valuable to those who seek to understand the dynamics of financial stability within the African context.')
    st.write('The dataset specifically focuses on the Banking, Debt, Financial, Inflation and Systemic Crises that occurred, from 1860 to 2014, in 13 African countries, including: Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia and Zimbabwe.')
    st.write('This dataset consists a total of 1059 records and 14 features to study the crisis. This data is collected by behavioral Finance & Financial Stability students at harvard business school. Some of the important features in the dataset are country, year, exchange rates in USD, domestic debt, sovereign external debt, gdp, annual inflation.')

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write('If you want to check the raw data, you can see by clicking on the Show Raw data button below. If you are interested in the numbers, check the statistics section below for a detailed breakdown of features and there statistical analysis.')
    checks = st.columns(2)
    # Display the dataset
    with checks[0]:
        with st.expander("Show Raw Data"):
            # if st.checkbox('Show Raw Data'):
            st.write(pd.DataFrame(df, columns=df.columns))
            st.write('Basic Global Crisis Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

    with checks[1]:
        with st.expander("Show Statistics about Data"):
            st.write(df.describe())
            st.write('Basic Global Crisis Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

    st.write('Understand more about the  data and deep dive by going through the interactive plots and visualizations in the next tabs.')

with tab2:

    st.header("Which countries are more vulnerable to crises across time")

    st.write('Use the below visualization to observe how economic crises are distributed across African countries. Hover over each country to see details of crisis the country got affected on and number of occurences. Also you have an option to select the range of time period using the slider below which helps you to focus on crisis in the particular time period.')

    # st.header("Varying trends across time and countries")
    # Time Slider
    values = st.slider(
        'Select a range of years',
        1870, 2013, (1870, 2013))

    # Slicing data as per selected time
    selector = alt.selection_single(encodings=['x', 'color'])
    df_y = df[(df["year"]>=f"01-01-{values[0]}")&(df["year"]<=f"01-01-{values[-1]}")]

    # Plot with selection, gray color used for showing focus on selected part
    c2 = alt.Chart(df_y).transform_fold(
    ['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis'],
    as_=['column', 'value']
    ).mark_bar().encode(
    x=alt.X('country:N', title="Countries"),
    y=alt.Y('sum(value):Q', title="Total no. of crises"),
    color=alt.condition(selector, 'column:N', alt.value('#FF5733'))
    ).add_selection(
        selector
    ).interactive()

    st.altair_chart(c2, use_container_width=True)

    p1_obs = ['1. The country that shows the highest count for systemic crisis is Central African Republic followed by Zimbabwe and Kenya.',
              '2. Angolo, Zambia and Zimbabwe suffered more number of inflations compared to all other african countries.',
              '3. South Africa has the lowest number of inflation crisis.',
              '4. Zimbabwe has higher number of crisis (Systemic, Currency, Inflation, Banking)']
    for point in p1_obs:
        st.write(point)

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    st.header("Crisis Over Time")
    st.write('These Visualization shows how the particular crisis occured over time across different countries in africa. Select the type of Crisis you want check that occurred over time across the countries. You can use the below dropdown option to select the crisis and there is play and stop options for the below visualization aswell as a slider. ')
    crisis_type = st.selectbox("Select Crisis : ",['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis'])
    df['year'] = df['year'].dt.year
    fig = px.choropleth(df, locations='cc3', animation_frame='year',
                        labels={f'{crisis_type}': f'{crisis_type}', 'cc3': 'Code', 'country': 'Country'},
                        animation_group='country', color=f'{crisis_type}',
                        color_continuous_scale=px.colors.sequential.Bluered, template='plotly_dark')

    # Remove the legend
    fig.update_layout(showlegend=False)

    fig.update_layout(width=1200, height=800)

    # Display the figure using Streamlit
    st.plotly_chart(fig)


    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    countries_list = list(df["country"].unique())
    st.header("Varying trends across time and countries")
    country_name = st.selectbox("Select a country : ",countries_list)

    # Multiselect dropdown for types of crises
    crisis_options = st.multiselect(
        'Types of crises:',
        ['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'],
        default=['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'])

    # Multiselect dropdown for Economic parameters
    eco_ops = st.multiselect(
        'Macroeconomic parameters:',
        ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence'],
        default = ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'inflation_annual_cpi', 'independence'])

    ###
    # Multi-Line chart 
    ###
    # preprocessing
    country_df = scaled_df[scaled_df["country"]==country_name]
    country_df.set_index("year", inplace=True)
    source = country_df.drop(columns=["country","cc3","case"])
    column_list = [i for i in list(country_df) if i not in []]
    source = source[crisis_options+eco_ops]
    source = source.reset_index().melt('year', var_name='category', value_name='y')

    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['year'], empty='none')

    # The basic line
    line = alt.Chart(source).mark_line(interpolate='basis').encode(
        x='year:T',
        y='y:Q',
        color='category:N',
        strokeDash=alt.condition(
            (alt.datum.category == 'exch_usd') | (alt.datum.category == 'domestic_debt_in_default') | (alt.datum.category == 'sovereign_external_debt_default') | (alt.datum.category == 'gdp_weighted_default') | (alt.datum.category == 'inflation_annual_cpi') | (alt.datum.category == 'independence'),
            alt.value([5, 5]),  # dashed line: 5 pixels  dash + 5 pixels space
            alt.value([0]),  # solid line
        )
    )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='year:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    # Draw text labels near the points, and highlight based on selection
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'y:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='year:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    c = alt.layer(
        line, selectors, points, rules, text
    ).properties(
        width=600, height=300
    ).interactive()

    st.altair_chart(c, use_container_width=True)

    observations = [
        '1. Some countries have relatively lower exchange rate than other countries. Countries like South Africa, Zambia, Egypt and Morocco has relatively lower exchange rate.',
'2. The exchange rate is almost zero for all the countries before 1940. This is because, most of the countries would have opted for new currency system after independece. For example, Tunisian dinar was introduced in 1960 and the Algerian dinar was introduced in 1964 (Reference: Wikipedia).',
'3. There are tremendous spikes in the exchange rate Angola and Zimbabwe.',
 '4. Egypt has been an independent country since 1850s. However its exchange rate has started increasing from 1970s. Lets consider Egypt as a special case in respective to independence.',
'5. The exchange rate had gone up after the independence for almost all the countries expect Tunisia. Except Tunisia and Ivory coast, the exchange rate for all the countries have been increasing from the independence with some fluctuations. There are some sudden spikes in the exchange rate. Angolan Kwanza - In 1999, a second currency was introduced in Angola called the kwanza and it suffered early on from high inflation (Wikipedia). Tunisian dinar was introduced in 1960, hence a spike.'
    ]

    st.write('<h3>Observations :</h3>', unsafe_allow_html=True)
    for point in observations:
        st.write(point)

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write('Now Lets move on to some Interactive plots...!!!')


with tab3:
    st.header("What causes a crisis?")
    st.write("The below plot helps us see if there's a connection between two things. If the dots are all over the place, there might not be a strong connection. If most points cluster together and go up and to the right, it suggests that, in general, as one characteristic increases, the other tends to increase too. Scatter plots help us see patterns and connections in data, making it easier to spot relationships between different factors or variables. Now you can select the feature and the crisis that you are focusing. ")

    # Country dropdown
    countries_list = list(df["country"].unique())
    country = st.selectbox("Select a country: ",countries_list)

    # preprocessing
    country_df = scaled_df[scaled_df["country"]==country]
    country_df.set_index("year", inplace=True)
    source = country_df.drop(columns=["country","cc3","case"])
    column_list = [i for i in list(country_df) if i not in []]
    # source = source[crisis_options+eco_ops]
    # source = source.reset_index().melt('year', var_name='category', value_name='y')



    # Radio button
    crisis = st.radio(
        "Select the type of crises to analyse (use this radio button for chloropeth map as well(last))",
        ('inflation_crises', 'systemic_crisis', 'banking_crisis','currency_crises'))

    # 
    c3 = alt.Chart(country_df.reset_index()).mark_circle(size=60).encode(
        x = 'inflation_annual_cpi:Q',
        y = 'exch_usd:Q',
        color=f'{crisis}:N',
        tooltip=['sovereign_external_debt_default:N', 'domestic_debt_in_default:N','year:T']
    ).properties(
        width=300,
        height=300
    ).interactive()


    c4 = alt.Chart(country_df.reset_index()).mark_circle(size=60).encode(
        x = 'inflation_annual_cpi:Q',
        y = 'gdp_weighted_default:Q',
        color=f'{crisis}:N',
        tooltip=['sovereign_external_debt_default:N', 'domestic_debt_in_default:N','year:T']
    ).properties(
        width=300,
        height=300
    ).interactive()

    hc = alt.hconcat(c3, c4)
    st.altair_chart(hc, use_container_width=True)

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Feature Vs Crisis :')
    col_1,col_2=st.columns(2,gap='small')
    st.write("The below plot helps you quickly see where most of the data falls, where the middle value is, and whether there are any unusual values. It's like a snapshot of your data, making it easier to understand and compare different sets of numbers. Any data points that fall much higher or lower than the rest are shown as dots. These are called outliers, and they can help you identify unusual or extreme values in your data. Now you can select the feature and the crisis that you are focusing. So you can observe some interesting facts.")
    numeric_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
    selected_feature = col_1.selectbox('Select a feature', numeric_columns)
    crisis_type = col_2.selectbox('Type of Crisis:', ['banking_crisis', 'systemic_crisis', 'inflation_crises', 'currency_crises'])
    y = df[crisis_type]

    st.subheader(f'Box Plot of {selected_feature} w.r.t {crisis_type}')

    # Create an interactive box plot using Plotly Express
    fig = px.box(df, x=(y == 1), y=selected_feature,color=crisis_type, labels={selected_feature: selected_feature})
    fig.update_xaxes(categoryorder='total ascending')
    fig.update_traces(marker=dict(size=5), boxpoints='all', jitter=0.3)
    fig.update_layout(xaxis_title=crisis_type, yaxis_title=selected_feature, showlegend=False)
    fig.update_layout(height=600, width=800)
    # Display the interactive plot
    st.plotly_chart(fig)

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Decoding Crises: The Financial Story')
    st.write("The below colorful chart is like a magic lens that helps you to understand better about the data. The bars you see represent different groups within our data, each with a different color. Each bar shows us how many times something happened (like a crisis) and how it relates to something else (like a financial factor). The height of the bars tells us how often it occurred, and you can hover your mouse over them to see specific numbers. The way the bars overlap, like they're hugging each other, shows how these things are connected. If one group's bars are mostly on one side and another group's bars are on the other side, it tells us that they're different in some way. Feeling Like I explained much 😅. Go on play with the data.")
    col1,col2=st.columns(2,gap='small')
    st.subheader('Select a feature for the histogram:')
    selected_feature = col1.selectbox('Select a feature', df.columns.tolist())
    crisis_format = col2.selectbox('Type of Crisis :', ['banking_crisis', 'systemic_crisis', 'inflation_crises', 'currency_crises'])
    bin_count = st.slider('Number of Bins', min_value=1, max_value=100, value=20)

    st.subheader(f'Histogram of {selected_feature}')

    # Create an interactive histogram using Plotly Express
    fig = px.histogram(df, x=selected_feature, color=crisis_format, nbins=bin_count)
    fig.update_xaxes(title_text=selected_feature)
    fig.update_yaxes(title_text='Count')
    fig.update_traces(marker=dict(line=dict(width=2)))
    fig.update_layout(height=700, width=900)
    # Display the interactive plot
    st.plotly_chart(fig)


    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)


    #visualization with HiPlot
    def save_hiplot_to_html(exp):
        output_file = "hiplot_plot_1.html"
        exp.to_html(output_file)
        return output_file
    
    st.header("Visualization with HiPlot")
    selected_columns = st.multiselect("Select columns to visualize", df.columns,default = ['exch_usd', 'banking_crisis', 'inflation_annual_cpi'])
    selected_data = df[selected_columns]
    if not selected_data.empty:
        experiment = hip.Experiment.from_dataframe(selected_data)
        hiplot_html_file = save_hiplot_to_html(experiment)
        st.components.v1.html(open(hiplot_html_file, 'r').read(), height=1500, scrolling=True)
    else:
        st.write("No data selected. Please choose at least one column to visualize.")

    
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.header('Thankyou for Visiting..!!!')


with tab4:
    st.write("The Predictive Analysis tab in the application uses advanced machine learning models like Logistic Regression and Support Vector Machine to demystify the complexities of global crises. It offers a diverse set of models, enabling varied analytical perspectives on crisis data.")
    st.write("This tab transforms intricate data into accessible, interactive insights, allowing users to experiment with data and see immediate results. This approach not only aids in predicting and understanding global crises but also empowers a wide range of users, from decision-makers to the general public, to engage with and respond to these critical issues proactively. Users can tailor the analysis to specific types of crises, like banking or systemic crises. This customization means that users can focus on the most relevant aspects of a global crisis, gaining insights that are directly applicable to their concerns or areas of interest.")
    st.write(" By offering a range of models, the tab allows users to apply different analytical perspectives to the same data. This variety is crucial because different models can highlight different aspects of a crisis, such as its likelihood, severity, or potential impact on different economic indicators.")
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    if 'knn' not in st.session_state:
	    st.session_state.knn = 0
    if 'lr' not in st.session_state:
	    st.session_state.lr = 0
    if 'svc' not in st.session_state:
	    st.session_state.svc = 0
    if 'mlp' not in st.session_state:
	    st.session_state.mlp = 0
    if 'rf' not in st.session_state:
	    st.session_state.rf = 0

    df = pd.read_csv("Global-Crisis/african_crises.csv")
    df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
    df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
    df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])

    st.subheader("Let's begin with selecting the crisis that we want to predict :")

    y_name = st.selectbox("Select the Crisis you need to predict: ",['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'])
    other_cols = [i for i in ['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'] if i != y_name]

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    
    st.write("Now let's select Features that we are interested in to create a model for predicting the above selected crisis.")
    
     # Multiselect dropdown for Economic parameters
    col_names = st.multiselect(
        'Select Input features:',
        ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence']+other_cols,
        default = ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence']+other_cols)
    

    st.write("Tick the below box to encode the categorical features if you selected any")
    # Label Encoder
    agree = st.checkbox('Encode categorical variables?')
    if agree:
        col_names.append("country_cat")

    df["country_cat"] = LabelEncoder().fit_transform(df["country"])
    df.drop(columns=["cc3","case","year","country"],inplace=True)

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    st.subheader("What about the data distribution of crisis ?")

    c11 = alt.Chart(df).mark_bar().encode(
    alt.Y(f'{y_name}:N'),
    alt.X(f'count({y_name}):Q'))
    st.altair_chart(c11, use_container_width=True)

    st.write("Oops..! Seems like the data is not balanced. Don't worry, we have solution below to make it balanced.")

    df.dropna(inplace=True)


    # Oversample
    st.subheader("Let's Balance the data")
    st.write("In our data, there are 'majority' and 'minority' classes, referring to the most and least common outcomes in our analysis, like 'no crisis' vs. 'crisis'. Often, the minority class (like 'crisis') doesn't have as many examples as the majority class, which can make our predictions less accurate for it. This is where the slider comes in. It's a tool that lets you balance out these classes. By moving the slider, you're essentially adjusting how much we 'oversample' the minority class. Oversampling is like adding more examples of the less common outcome to our data. This helps the models learn about it better.")
    st.write("So, go ahead and adjust the slider! See how it changes the balance between common and rare events in our data, and observe how it might improve our predictions.")
    percent_os = st.slider('Set Minority class-Majority class ratio:', 0.00, 1.00, 1.00, step = 0.05)
    X = df.drop(columns = [y_name])
    y = df[y_name]
    if percent_os > 0.05:
        X_resampled, y = SMOTE(sampling_strategy=percent_os, random_state =42).fit_resample(X, y)
        X = pd.DataFrame(X_resampled, columns=X.columns)
    else:
         X = pd.DataFrame(X, columns=X.columns)

    c12 = alt.Chart(pd.DataFrame(y)).mark_bar().encode(
    alt.Y(f'{y_name}:N'),
    alt.X(f'count({y_name}):Q'))
    st.altair_chart(c12, use_container_width=True)

    
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Splittng Data")
    st.write("We have a bunch of data that we can use to 'train' our predictive models, teaching them to recognize patterns and make forecasts. But, we also need to know how good these models are – that's where 'testing' comes in. The slider you see is for deciding how much of our data is used for training, and how much for testing")
    # Train Test split
    percent_test = st.slider('Set Train size:', 0.00, 1.00, 0.30, step = 0.05)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size= 1 - percent_test,random_state=42)

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Feature Transformation")
    st.write("As the data could be various scales, ML Models learns better if all features are in similar scale. You can select one of the feature scaling apporach below.")
    scaler_type = st.radio(
        "Select the type of scaling operation/normalization -",
        ('No scaling', 'MinMax', 'Standard', 'Normalize'))

    if scaler_type == "MinMax":
        scaler = MinMaxScaler()
    elif scaler_type == "Standard":
        scaler = StandardScaler()
    elif scaler_type == "Normalize":
        scaler = Normalizer()

    if scaler_type!="No scaling":
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    
    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Train, Evaluate and Tune Estimators ")
    st.write("Think of our data analysis tool as a kitchen with different appliances, each designed for a specific cooking task. In our scenario, these 'appliances' are different machine learning models, and each one has its unique way of 'cooking' or analyzing the data.")
    st.write(" All set ! Now let's apply various models below and evaluate the model performance and it's predictive power. Don't forget to play with the model options ( commonly called as hyperparameters).")
    df.dropna(inplace=True)

    col1, col2, col3, col4, col5 = st.columns(5, gap="medium")

    with col1:
        st.subheader("Logistic Regression")
        parameters_lr = {"penalty":('l2',), "solver":('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'), "C":[1,0.1,0.01,0.001]}
        penalty = st.selectbox('Select penalty -',parameters_lr["penalty"],index=0)
        solver = st.selectbox("Select solver -", parameters_lr["solver"],index=1)
        c = st.select_slider('Select regularization strength -', parameters_lr["C"],value=0.1)

        lr = LogisticRegression(penalty = penalty, solver = solver, C = c)
        c13, fig_lr, precision_lr, recall_lr, f1_lr, accuracy_lr = get_plots(lr, X_train, X_test, y_train, y_test)

        
        if st.button('Evaluate LR'):
            st.session_state.lr += 1

        if st.session_state.lr > 0:       
            st.altair_chart(c13, use_container_width=True)
            st.pyplot(fig_lr)

            if 'lr_max_precision' not in st.session_state:
	            st.session_state.lr_max_precision, st.session_state.lr_max_recall, st.session_state.lr_max_f1, st.session_state.lr_max_accuracy= precision_lr, recall_lr, f1_lr, accuracy_lr                

            st.metric(label="Precision", value=precision_lr, delta = str(round(precision_lr-st.session_state.lr_max_precision,3)))
            st.metric(label="Recall", value=recall_lr, delta = str(round(recall_lr-st.session_state.lr_max_recall,3)))
            st.metric(label="F1-score", value=f1_lr, delta = str(round(f1_lr-st.session_state.lr_max_f1,3)))
            st.metric(label="Accuracy", value=accuracy_lr, delta = str(round(accuracy_lr-st.session_state.lr_max_accuracy,3)))
            st.session_state.lr_max_precision = max(st.session_state.lr_max_precision, precision_lr)
            st.session_state.lr_max_recall = max(st.session_state.lr_max_recall, recall_lr)
            st.session_state.lr_max_f1 = max(st.session_state.lr_max_f1, f1_lr)
            st.session_state.lr_max_accuracy = max(st.session_state.lr_max_accuracy, accuracy_lr)

        if st.button('Tune LR with CV'):
            clf = GridSearchCV(lr, parameters_lr, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col2:
        st.subheader("Support Vector Machine")
        parameters_svc = {"kernel":('linear', 'poly', 'rbf', 'sigmoid'), "gamma":('scale', 'auto'), "C":[1,0.1,0.01,0.001]}
        kernel = st.selectbox('Select kernel -',parameters_svc["kernel"],index=2)
        gamma = st.selectbox("Select gamma -", parameters_svc["gamma"],index=0)
        c = st.select_slider('Select regularization parameter -', options=parameters_svc["C"],value=1)

        svc = SVC(kernel = kernel, gamma = gamma, C = c, probability=True, max_iter=10000)
        c14, fig_svc, precision_svc, recall_svc, f1_svc, accuracy_svc = get_plots(svc, X_train, X_test, y_train, y_test)
        
        if st.button('Evaluate SVC'):
            st.session_state.svc += 1

        if st.session_state.svc > 0:       
            st.altair_chart(c14, use_container_width=True)
            st.pyplot(fig_svc)
            st.metric(label="Precision", value=precision_svc)
            st.metric(label="Recall", value=recall_svc)
            st.metric(label="F1-score", value=f1_svc)
            st.metric(label="Accuracy", value=accuracy_svc)

        if st.button('Tune SVC with CV'):
            clf = GridSearchCV(svc, parameters_svc, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col3:
        st.subheader("K Neighbors Classifier")
        parameters_knn = {"weights":('uniform', 'distance'), "algorithm":('auto', 'ball_tree', 'kd_tree', 'brute'), "n_neighbors":list(range(1,30,2))}
        weights = st.selectbox('Select weight function -',parameters_knn["weights"],index=0)
        algorithm = st.selectbox("Select algorithm -", parameters_knn["algorithm"],index=0)
        n_neighbors = st.select_slider('Select number of neighbors -', options=parameters_knn["n_neighbors"],value=5)
        
        knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights=weights, algorithm=algorithm)
        c15, fig_knn, precision_knn, recall_knn, f1_knn, accuracy_knn = get_plots(knn, X_train, X_test, y_train, y_test)

        if st.button('Evaluate KNN'):
            st.session_state.knn += 1

        if st.session_state.knn > 0:
            st.altair_chart(c15, use_container_width=True)
            st.pyplot(fig_knn)
            if 'knn_max_precision' not in st.session_state:
	            st.session_state.knn_max_precision, st.session_state.knn_max_recall, st.session_state.knn_max_f1, st.session_state.knn_max_accuracy= precision_knn, recall_knn, f1_knn, accuracy_knn                

            st.metric(label="Precision", value=precision_knn, delta = str(round(precision_knn-st.session_state.knn_max_precision,3)))
            st.metric(label="Recall", value=recall_knn, delta = str(round(recall_knn-st.session_state.knn_max_recall,3)))
            st.metric(label="F1-score", value=f1_knn, delta = str(round(f1_knn-st.session_state.knn_max_f1,3)))
            st.metric(label="Accuracy", value=accuracy_knn, delta = str(round(accuracy_knn-st.session_state.knn_max_accuracy,3)))
            st.session_state.knn_max_precision = max(st.session_state.knn_max_precision, precision_knn)
            st.session_state.knn_max_recall = max(st.session_state.knn_max_recall, recall_knn)
            st.session_state.knn_max_f1 = max(st.session_state.knn_max_f1, f1_knn)
            st.session_state.knn_max_accuracy = max(st.session_state.knn_max_accuracy, accuracy_knn)

        if st.button('Tune KNN with CV'):
            clf = GridSearchCV(knn, parameters_knn, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col4:
        st.subheader("Multilayer Perceptron")
        parameters_mlp = {"activation":['identity', 'logistic', 'tanh', 'relu'], "solver":('lbfgs', 'sgd', 'adam'), "hidden_layer_sizes":[(100,),(50,50,),(100,100,)]}
        activation = st.selectbox('Select activation function -', options=parameters_mlp["activation"],index=3)
        solver = st.selectbox('Select solver -',parameters_mlp["solver"], index=2)
        hidden_layer_sizes = st.text_input("Select hidden layer sizes -", "100,")
        
        mlp = MLPClassifier(activation = activation, solver=solver, hidden_layer_sizes=literal_eval(hidden_layer_sizes))
        c16, fig_mlp, precision_mlp, recall_mlp, f1_mlp, accuracy_mlp  = get_plots(mlp, X_train, X_test, y_train, y_test)

        if st.button('Evaluate MLP'):
            st.session_state.mlp += 1

        if st.session_state.mlp > 0:
            st.altair_chart(c16, use_container_width=True)
            st.pyplot(fig_mlp)
            st.metric(label="Precision", value=precision_mlp)
            st.metric(label="Recall", value=recall_mlp)
            st.metric(label="F1-score", value=f1_mlp)
            st.metric(label="Accuracy", value=accuracy_mlp)

        if st.button('Tune MLP with CV'):
            clf = GridSearchCV(mlp, parameters_mlp, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)


    with col5:
        st.subheader("Random Forest")
        parameters_rf = {"criterion":['gini', 'entropy', 'log_loss'], "n_estimators":list(range(100,501,100)), "max_depth": list(range(1,11,3))}
        criterion = st.selectbox('Select criterion for split -', options=parameters_rf["criterion"],index=0)
        n_estimators = st.select_slider('Select number of estimators -', options=parameters_rf["n_estimators"],value=100)
        max_depth = st.select_slider('Select maximum depth -', options=parameters_rf["max_depth"],value=10)
        
        rf = RandomForestClassifier(criterion = criterion, n_estimators=n_estimators, max_depth=max_depth)
        c17, fig_rf, precision_rf, recall_rf, f1_rf, accuracy_rf = get_plots(rf, X_train, X_test, y_train, y_test)

        if st.button('Evaluate RF'):
            st.session_state.rf += 1

        if st.session_state.rf > 0:
            st.altair_chart(c17, use_container_width=True)
            st.pyplot(fig_rf)
            st.metric(label="Precision", value=precision_rf)
            st.metric(label="Recall", value=recall_rf)
            st.metric(label="F1-score", value=f1_rf)
            st.metric(label="Accuracy", value=accuracy_rf)

        if st.button('Tune RF with CV'):
            clf = GridSearchCV(rf, parameters_rf, cv=3)
            clf.fit(X_train, y_train)
            st.json(clf.best_params_)

    st.subheader("How each model understands data ?")

    st.write("Logistic Regression: Think of it as a straightforward detective tool. It looks for direct links, like investigating whether specific economic signs directly point to a crisis. It's great for clear-cut, 'yes or no' type of questions, like 'Is this factor leading us to a financial crisis?'")
    st.write("Support Vector Machine (SVM): This is like a sophisticated detective tool that can draw complex boundaries. It's used to classify data into distinct groups, such as distinguishing between different types of crises. It's like sorting out different puzzle pieces to see which ones fit into the 'crisis' picture.")
    st.write("K-Nearest Neighbors (KNN): Imagine this as analyzing a crisis based on its 'neighbors' or similar past scenarios. It looks at what happened in similar economic conditions to predict if we're heading towards a crisis. It's like learning from history to predict the future.")
    st.write("Multilayer Perceptron (MLP): This is akin to a team of experts analyzing a crisis from multiple angles. With its layers, it delves deep into complex patterns, making it suitable for intricate crisis situations where multiple factors are at play.")
    st.write("Random Forest: Think of this as gathering insights from a diverse panel of experts. Each 'tree' in the forest provides its perspective on whether a situation is heading towards a crisis. The final prediction is a consensus of these various expert opinions, offering a comprehensive view.")

    st.write("Each model offers a unique way to analyze and predict crises. Some are straightforward and great for clear situations, while others are more intricate, perfect for complex scenarios with multiple influencing factors. Experimenting with these models can help you understand which is most effective in forecasting or dissecting a particular type of crisis.")


    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write("Don't forget to checkout the next tab on model inference.")

with tab5:
    st.write('<h2 style="text-align:center; vertical-align:middle; line-height:2; color:#046482;">Crisis Prediction</h2>', unsafe_allow_html=True)

    st.write("Let's begin with selecting the crisis that we want to predict. You can select the crisis below.")

    df = pd.read_csv("Global-Crisis/african_crises.csv")
    df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
    df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
    df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])
    df.dropna(inplace=True)

    X = df.drop(columns = [y_name])
    X = X[['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 
                    'gdp_weighted_default', 'inflation_annual_cpi', 'independence']]
    y = df[y_name]
    X_resampled, y = SMOTE(sampling_strategy=1, random_state =42).fit_resample(X, y)
    X = pd.DataFrame(X_resampled, columns=X.columns)

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size= 0.2,random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)


    y_name = st.selectbox("Select the Crisis you want to predict: ",['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'])

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

    rf_model = train_model(y_name,X_train, y_train)

    st.write("Now Enter the values of the below features and click on the predict button below to check the chances of crisis.")

    # Creating columns for the input fields
    col1, col2, col3 = st.columns(3)
    with col1:
        exch_usd = st.number_input('Exchange Rate to USD', value=0.0)
        domestic_debt_in_default = st.number_input('Domestic Debt in Default', value=0.0)
    with col2:
        sovereign_external_debt_default = st.number_input('Sovereign External Debt Default', value=0.0)
        gdp_weighted_default = st.number_input('GDP Weighted Default', value=0.0)
    with col3:
        inflation_annual_cpi = st.number_input('Annual CPI Inflation Rate', value=0.0)
        independence = st.number_input('Independence', value=0.0)

    col4, col5, col6 = st.columns([1,1,1])
    with col5:
        st.write("")
        st.write("")
        # Button to make prediction
        if st.button('Predict Crisis'):
            # Load model (ensure model is pre-trained and saved)

            # Creating a dataframe from the inputs
            input_df = pd.DataFrame([[
                exch_usd, 
                domestic_debt_in_default, 
                sovereign_external_debt_default, 
                gdp_weighted_default, 
                inflation_annual_cpi, 
                independence
            ]], columns=['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 
                        'gdp_weighted_default', 'inflation_annual_cpi', 'independence'])

            # Predict
            prediction = predict_crisis(rf_model, input_df)

            # Display result
            if prediction[0] == 1:
                st.success("There is a high likelihood of a crisis.")
            else:
                st.success("There is a low likelihood of a crisis.")

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Conclusion")
    st.write("This web application represents a significant stride in democratizing access to advanced predictive analytics in the realm of economic and financial crises. By leveraging intuitive interfaces and powerful machine learning models, we have transformed complex data analysis into a user-friendly experience, accessible to both experts and non-technical users.")
    st.write("The core feature of our application is its ability to analyze various economic indicators – such as exchange rates, debt defaults, and inflation rates – to predict potential financial crises. This tool is not just about numbers and predictions; it's about empowering decision-makers, researchers, and the general public with the insights needed to foresee and mitigate the risks of economic downturns. Through interactive elements like customizable sliders and input fields, users have the flexibility to explore different scenarios, understand intricate data patterns, and receive instant predictions. Our vision is for this tool to become a guiding light for those seeking insights into financial challenges, contributing towards more robust economies and informed communities.")

    st.write("However, it's important to note that this application is a demonstration of potential capabilities and should not be relied upon for real-time crisis prediction or decision-making in its current state. It serves as an educational and exploratory tool, showcasing the possibilities of data science and machine learning in economic analysis.")

    st.markdown("""<hr style="height:3px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.write('<h3 style="text-align:center; vertical-align:middle; line-height:2; color:#046482;">Big shoutout for swinging by our app! You handled it smoother than a ninja navigating a room full of lasers. Keep being awesome and may your Wi-Fi always be strong!</h3>', unsafe_allow_html=True)
with tab6:

    st.title("About the Developer ")

    col1, col2 = st.columns(2)

    col1.subheader("Thejesh Mallidi (he/him)")
    col1.text("Master's in Data Science, MSU")
    col1.write("I am passionate about solving data-driven problems through predictive modeling and applying advanced technical solutions to address real-world challenges. Over the past two years, I have worked as a Machine Learning Engineer, focusing on object detection, classification, and segmentation in various proof-of-concept projects. I have successfully developed and implemented algorithms, leveraging machine learning and statistical modeling techniques to enhance performance, data management, and accuracy. Additionally, I've designed and implemented innovative document extraction systems using a combination of computer vision and natural language processing, primarily in the BFSI domain. My skill set includes expertise in machine learning, deep learning, computer vision, natural language processing, Big data, Python, Nvidia technologies, and proficiency in cloud services.")
    
    col1.write("")
    col1.markdown("###### _Hobbies_")
    col1.text("1. Volley ball⚽️")
    col1.text("2. Video Games🎮")
    col1.text("3. Hang Around🏄🏾‍♂️")
    col1.write("")

    col1.write("")

    try :
        col2.image("Global-Crisis/thejesh.jpeg")
    except:
     pass


    
    




    




