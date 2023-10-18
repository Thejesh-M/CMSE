import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import plotly.express as px
from PIL import Image
import hiplot as hip
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(layout="wide")


# Add custom CSS to center the content
st.write('<h1 style="text-align:center; vertical-align:middle; line-height:2; color:#046366;">Global Crises Data by Country</h1>', unsafe_allow_html=True)



df = pd.read_csv("Global-Crisis/african_crises.csv")
df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])
df = df[df["currency_crises"]<=1]
df["year"] = pd.to_datetime(df.year, format='%Y')
scaled_df = df.copy()


tab1, tab2, tab3 = st.tabs(["About Data", "Crisis Over Time", "Interactive Plots"])

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



