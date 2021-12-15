# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px

# Read the airline data into pandas dataframe
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv'
spacex_df = pd.read_csv(url,sep=",")
print(spacex_df.head(5))
#spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                # dcc.Dropdown(id='site-dropdown',...)
                                dcc.Dropdown(id='site-dropdown', 
                                    options=[
                                        {'label':'All Sites','value':'ALL'},
                                        {'label':'CCAFS LC-40','value':'CCAFS LC-40'},
                                        {'label':'VAFB SLC-4E','value':'VAFB SLC-4E'},
                                        {'label':'KSC LC-39A','value':'KSC LC-39A'},
                                        {'label':'CCAFS SLC-40','value':'CCAFS SLC-40'}
                                    ], value='ALL',placeholder="Select a Launch Site here",
                                    searchable=True),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                #dcc.RangeSlider(id='payload-slider',...)
                                dcc.RangeSlider(id='payload-slider',
                                                min=0, max=10000, step=1000,
                                                marks={0:'0',2500:'2500',5000:'5000',7500:'7500',10000:'10000'},
                                                value=[min_payload, max_payload]),
                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                                html.H2('My observations & findings:', 
                                        style={'textAlign': 'left', 'color': 'purple',
                                               'font-size': 20}),
                                dcc.Markdown('''
                                    * **Which site has the largest successful launches?** _41.7% of the launches are from KSC LC-39A._
                                    * **Which site has the highest launch success rate?** _KSC LC-39A has 76.9% success rate, the highest among all sites._
                                    * **Which payload range(s) has the highest launch success rate?** _Range 0 to 5300 includes the highest launch success rate 76.9% from KSC LC-39A._
                                    * **Which payload range(s) has the lowest launch success rate?** _Range 1000 to 5000 (payload mass 1952 kg) includes the lowest launch success rate 26.9% from CCAFS LC-40._
                                    * **Which F9 Booster version (v1.0, v1.1, FT, B4, B5, etc.) has the highest launch success rate?** _Highest launch success rate uses FT._
                                    ''', style={'textAlign': 'left', 'color': 'purple','font-size': 20})
                                ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(Output(component_id='success-pie-chart', component_property='figure'),
              Input(component_id='site-dropdown', component_property='value'))

def get_pie_chart(entered_site):
    filtered_df = spacex_df.groupby(['Launch Site','class']).size().reset_index(name='count')
    if entered_site == 'ALL':
        fig = px.pie(spacex_df, values='class', 
                     names='Launch Site', 
                     title='Total Success Launches By Site')
        return fig
    else:
        # return the outcomes piechart for a selected site
        filtered_df = filtered_df[filtered_df['Launch Site']==entered_site]
        fig = px.pie(filtered_df, values='count', 
                     names='class', 
                     title=('Total Success Launches for Site ' + entered_site))
        return fig        
        
# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id='success-payload-scatter-chart', component_property='figure'),
              [Input(component_id='site-dropdown', component_property='value'), 
               Input(component_id="payload-slider", component_property="value")])

def update_chart(entered_site, slider_range):
    low, high = slider_range   
    if entered_site == 'ALL':
        dff = spacex_df
        stl = 'Correlation between Payload and Success for all Sites'
    else:
        dff = spacex_df[spacex_df['Launch Site']==entered_site] 
        stl = 'Correlation between Payload and Success for ' + entered_site           
    mask = (dff['Payload Mass (kg)'] >= low) & (dff['Payload Mass (kg)'] <= high)           
    figs = px.scatter(dff[mask],x='Payload Mass (kg)', y='class',color="Booster Version Category",
                      title=stl)         
    return figs

# Run the app
if __name__ == '__main__':
    app.run_server()
