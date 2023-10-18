import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt
import plotly.express as px
from PIL import Image

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
    image_path = "bg.png"  # Replace with the actual file path

    # Check if the image file exists at the specified path
    try:
        with open(image_path, "rb") as image_file:
            img = Image.open(image_file)
            img = img.resize((img.width, 300))
            st.image(img, caption="Global Crisis", use_column_width=True)
    except FileNotFoundError:
        pass

    st.write('This dataset is a derivative of Reinhart et. Global Financial Stability dataset which can be found online at: https://www.hbs.edu/behavioral-finance-and-financial-stability/data/Pages/global.aspx The dataset will be valuable to those who seek to understand the dynamics of financial stability within the African context.')
    st.write('The dataset specifically focuses on the Banking, Debt, Financial, Inflation and Systemic Crises that occurred, from 1860 to 2014, in 13 African countries, including: Algeria, Angola, Central African Republic, Ivory Coast, Egypt, Kenya, Mauritius, Morocco, Nigeria, South Africa, Tunisia, Zambia and Zimbabwe.')
    st.write('This dataset consists a total of 1059 records and 14 features to study the crisis. This data is collected by behavioral Finance & Financial Stability students at harvard business school. Some of the important features in the dataset are country, year, exchange rates in USD, domestic debt, sovereign external debt, gdp, annual inflation.')
    checks = st.columns(2)
    # Display the dataset
    with checks[0]:
        with st.expander("Show Raw Data"):
            # if st.checkbox('Show Raw Data'):
            st.write(pd.DataFrame(df, columns=df.columns))
            st.write('Basic Breast Cancer Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

    with checks[1]:
        with st.expander("Show Statistics about Data"):
            st.write(df.describe())
            st.write('Basic Breast Cancer Dataset Information:')
            st.write(f'Total Number of Samples: {df.shape[0]}')
            st.write(f'Number of Features: {df.shape[1]}')

with tab2:

    st.header("Which countries are more vulnerable to crises across time")
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


    st.header("Crisis Over Time")
    st.write('Select the type of Crisis you want ')
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








    countries_list = list(df["country"].unique())
    st.header("Varying trends across time and countries -")
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
'2. The exchange rate is almost zero for all the countries before 1940. This might be because the value is not recorded or a new currency had been adopted by the countries.',
'3. There are tremendous spikes in the exchange rate Angola and Zimbabwe.'
    ]

    st.write('<h3>Observations :</h3>', unsafe_allow_html=True)
    for point in observations:
        st.write(point)


with tab3:
    st.header("What causes a crisis?")


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