import streamlit as st
import seaborn as sns
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")


# Add custom CSS to center the content
st.write('<h1 style="text-align:center; vertical-align:middle; line-height:2; color:#046366;">Global Crises Data by Country</h1>', unsafe_allow_html=True)



df = pd.read_csv("african_crises.csv")
df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])
df = df[df["currency_crises"]<=1]
df["year"] = pd.to_datetime(df.year, format='%Y')
scaled_df = df.copy()



# Display the dataset
if st.checkbox('Show Raw Data'):
    st.write(pd.DataFrame(df, columns=df.columns))
    st.write('Basic Breast Cancer Dataset Information:')
    st.write(f'Total Number of Samples: {df.shape[0]}')
    st.write(f'Number of Features: {df.shape[1]}')




# Multiselect dropdown for types of crises
# crisis_options = st.multiselect(
#     'Types of crises:',
#     ['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'],
#     default=['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'])

# eco_ops = st.multiselect(
#     'Macroeconomic parameters:',
#     ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence'],
#     default = ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'inflation_annual_cpi', 'independence'])



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
