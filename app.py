#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import plotly
import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio
from plotly.subplots import make_subplots

# from jupyter_dash import JupyterDash
import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


# In[2]:


pio.templates.default = "plotly_dark"


# In[3]:


usecols=['location','date', 'total_cases', 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million', 'incidence', 'mortality', 'iso_code',
       'continent', 'cfr']
conti_usecols=['iso_code', 'location', 'date', 'total_cases', 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million']
world_usecols = ['iso_code', 'location', 'date', 'total_cases', 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million','new_vaccinations',
        'Cases', 'Deaths', 'incidence', 'mortality', 'fatality']
df_usecols=['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases',
       'total_deaths', 'new_deaths', 'total_cases_per_million',
       'new_cases_per_million', 'total_deaths_per_million',
       'new_deaths_per_million','new_vaccinations', 'Cases',
       'Deaths', 'incidence', 'mortality', 'fatality']


# In[4]:


df = pd.read_csv('all.csv', usecols= df_usecols)
world = pd.read_csv('world.csv', usecols=world_usecols)
conti = pd.read_csv('continents.csv',usecols=conti_usecols)
para = pd.read_csv('parameters.csv', usecols=usecols)


# In[5]:


del usecols, conti_usecols, world_usecols, df_usecols


# In[6]:


df.date=pd.to_datetime(df.date)
world.date=pd.to_datetime(world.date)
conti.date=pd.to_datetime(conti.date)
para.date=pd.to_datetime(para.date)


# In[7]:


import math
cases = world[world.date==world.date.max()].iloc[0]['total_cases']
deaths = world[world.date==world.date.max()].iloc[0]['total_deaths']
daily_c = world[world.date==world.date.max()].iloc[0]['new_cases']
daily_d = world[world.date==world.date.max()].iloc[0]['new_deaths']


# #### FUNCTIONS FOR PLOTTING

# 1. Updating Map

# In[8]:


def get_map(col):
    labels = {'incidence':'30-day Cases per Million',
              'mortality':'30-day Deaths per Million',
              'total_cases':'Total Cases',
              'total_deaths':'Total Deaths',
              'new_cases':'New Cases',
              'new_deaths':'New Deaths',
              'cfr':'Case Fatality Rate'}
    world_map = px.choropleth(para, locations="iso_code",
                        color=col,
                        color_continuous_scale='blues',
    #                     title='Total Cases Per Million',
                        hover_name="location",
                        hover_data={'iso_code':False,'incidence':True, 'mortality':True, 'cfr':True},
                        basemap_visible = False,
                        labels=labels,
                        )
    world_map.update_traces(hoverlabel={'bgcolor':'white','font':{'family':'balto'}}
                           )
    world_map.update_layout(autosize= True, margin=dict(l=0, r=0, t=0, b=0),
                            geo=dict(bgcolor='rgba(0,0,0,0)'),
                            paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                            coloraxis_colorbar=dict(title=labels[col],
                                                    thicknessmode="pixels", thickness=10,
                                                    lenmode="pixels", len=300,
                                                    yanchor="middle",)
                           )
    return world_map


# 2. Location-wise trend analysis - crude data and rates 

# In[9]:


def plot_crude(name):
    
    if name == 'World':
        ind=world.copy()

    else:
        ind = df[df.location==name].copy()
        ind.reset_index(drop=True, inplace=True)

    fig1 = make_subplots(rows=3, cols=1,shared_yaxes=False, shared_xaxes=True,#vertical_spacing=0.07,
                        specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]],
                        subplot_titles=("Cumulative Confirmed Cases and Reported Deaths","Daily Confirmed Cases","Daily Reported Deaths")
                       )

    # First graph
    fig1.add_trace(
        go.Scatter(x=ind['date'], y=ind['total_cases'], line=dict(color='grey'), showlegend=True, name="Cases"),
        row=1, col=1,secondary_y=False
    )

    fig1.add_trace(
        go.Scatter(x=ind['date'], y=ind['total_deaths'],line=dict(color='orangered'), showlegend=True, name="Deaths"),
        row=1, col=1,secondary_y=True,
    )
    # Second graph
    fig1.add_trace(
        go.Bar(x=ind['date'], y=ind['new_cases'],marker=dict(color='grey'),showlegend=False, name='Daily Cases'),
        row=2, col=1, 
    )
    fig1.add_trace(
        go.Scatter(x=ind['date'], y=ind.new_cases.rolling(7).mean(),line=dict(color='lightblue'), showlegend=False, name='Moving Avg'),
        row=2, col=1, 
    )

    # Third Garph
    fig1.add_trace(
        go.Bar(x=ind['date'], y=ind['new_deaths'],marker=dict(color='orangered'),showlegend=False, name='Daily Deaths'),
        row=3, col=1
    )
    fig1.add_trace(
        go.Scatter(x=ind['date'], y=ind['new_deaths'].rolling(7).mean(),marker=dict(color='red'),showlegend=False, name='Moving Avg'),
        row=3, col=1
    )
    
    fig1.update_annotations(font_size=12)

    if ind['new_vaccinations'].notnull().any():

        index = ind['new_vaccinations'].first_valid_index()
        x_pos = ind.at[index,'date']

        fig1.add_annotation(x=x_pos, y=1, xref='x', yref='paper', xanchor='left',
                           text='Vaccination Started',font_color='lightgreen', showarrow=False)

        fig1.add_annotation(x=x_pos, y=1, xref='x2', yref='paper', xanchor='left', 
                           text='Vaccination Started',font_color='lightgreen', showarrow=False)

        fig1.add_annotation(x=x_pos, y=1, xref='x3', yref='paper', xanchor='left', 
                           text='Vaccination Started',font_color='lightgreen', showarrow=False)


        fig1.add_vline(x=x_pos, line_width=0.5, opacity=0.8,line_dash="dash",line_color="yellow")

    fig1.update_xaxes(showgrid=False)

    fig1.update_layout( showlegend=False,
                       margin=dict(l=20,r=20,t=20,b=20),
                      yaxis=dict(gridwidth=0.5,title='No. of Cases',
                                 titlefont = dict(size = 10,color = 'lightgrey'),
                                 tickfont= dict(size = 9)),
                      yaxis2=dict(tickfont= dict(size = 9)),
                       yaxis3=dict(tickfont= dict(size = 9)),
                       yaxis4=dict(tickfont= dict(size = 9)),
                      )
    return fig1
print('done')


# In[10]:


def plot_rates(name):
    
    if name == 'World':
        ind=world.copy()
        
    else:
        ind = df[df.location==name].copy()
        ind.reset_index(drop=True, inplace=True)
    
    fig = make_subplots(rows=3, cols=1,shared_yaxes=False,shared_xaxes=True,#vertical_spacing=0.07,
                        specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"secondary_y": False}]],
                        subplot_titles=("30-day Incidence Rate", "30-day Mortality Rate", '30-day Case Fatality Rate (%)'))

    fig.add_trace(
        go.Scatter(x=ind['date'], y=ind['incidence'], line=dict(color='grey'), showlegend=True, name="Incidence Rate"),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=ind['date'], y=ind['mortality'],line=dict(color='orangered'),showlegend=False, name='30-day mortality'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=ind['date'], y=ind['fatality'].rolling(7).mean(),line=dict(color='green'),showlegend=True, name='Case Fatality Rate'),
        row=3, col=1
    )    
    
    fig.update_annotations(font_size=12)
   
    if ind['new_vaccinations'].notnull().any():

        index = ind['new_vaccinations'].first_valid_index()
        x_pos = ind.at[index,'date']

        fig.add_annotation(x=x_pos, y=1, xref='x', yref='paper', xanchor='left',
                           text='Vaccination Started',font_color='lightgreen', showarrow=False)
        
        fig.add_annotation(x=x_pos, y=1, xref='x2', yref='paper', xanchor='left', 
                           text='Vaccination Started',font_color='lightgreen', showarrow=False)
        
        fig.add_annotation(x=x_pos, y=1, xref='x3', yref='paper', xanchor='left', 
                           text='Vaccination Started',font_color='lightgreen', showarrow=False)
    

        fig.add_vline(x=x_pos, line_width=1, line_dash="dash",line_color="yellow", opacity=0.8)
        
    fig.update_yaxes(title_text="Cases per Million", titlefont = dict(size = 10,color = 'lightgrey'),
                                 tickfont= dict(size = 9), row=1, col=1)
    fig.update_yaxes(title_text="Deaths per Million", titlefont = dict(size = 10,color = 'orangered'),
                                 tickfont= dict(size = 9), row=2, col=1)
    fig.update_yaxes(title_text="Percentage", titlefont = dict(size = 10,color = 'lightgreen'),
                                 tickfont= dict(size = 9), row=3, col=1)
    
    fig.update_layout(showlegend=False, 
                      margin=dict(l=20,r=20,t=20,b=20),
                     )

    del ind
    return fig


# In[11]:


def compare_plot(nations, active):
    
    colors=['#636efa','#EF553B','#00cc96','#ab63fa','#FFA15A','#19d3f3']
    color_map = {nations[i]:colors[i] for i in range(len(nations))}
    
    df1 = df[df.location.isin(nations)][['date','location','incidence','mortality']].copy()    
    if active=='first':
        col1 = 'new_cases'
        col2 = 'incidence'
        ytitle = 'Cases per million'
        title1 = 'Daily Cases'
        title2 = '30-day incidence of cases per million population'
    
    if active=='second':
        col1 = 'new_deaths'
        col2 = 'mortality'
        ytitle = 'Deaths per million'
        title1 = 'Daily Deaths'
        title2 = '30-day mortality as deaths per million population'
        
    df2 = df[df.location.isin(nations)].set_index(['date','location']).unstack('location')[col1].rolling(7).mean()
    df2.reset_index(inplace=True)

    df_melt1 = df2.melt(id_vars='date', value_vars=nations)
      
    fig = px.line(df_melt1, x='date', y='value',color='location',
                      color_discrete_map=color_map,title=title1,
                      hover_name='location', hover_data={'location':False},
                      labels={'value':title1})
        
    fig.update_layout(showlegend=True, title={'font':{'size':12}},
                      legend=dict(title='',
                                  orientation="h", 
                                yanchor="bottom", y=1.2, 
                                     xanchor="center", x=0.5,
                                     bordercolor="Black",
                                     borderwidth=1
                                    ),
                     )
    fig.update_yaxes(titlefont={'size':12})
    
    fig1 = px.line(df1, x='date', y=col2,color='location',
                      color_discrete_map=color_map, title=title2,
                      hover_name='location', hover_data={'location':False},
                      labels={'incidence':'Cases per Million','mortality':'Deaths per Million'})
        
    fig1.update_layout(showlegend=True,title={'font':{'size':12}},
                      legend=dict(title='',
                                  orientation="h", 
                                yanchor="bottom", y=1.3, 
                                     xanchor="center", x=0.5,
                                     bordercolor="Black",
                                     borderwidth=1
                                    ),
                     )
        
    fig1.update_yaxes(title_text=ytitle, titlefont={'size':12})

    del df1, df2, df_melt1
    return fig, fig1


# 3. Updating graph in dash data-table

# In[12]:


def data_bars(df, column):
    n_bins = 100
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    ranges = [
        ((df[column].max() - df[column].min()) * i) + df[column].min()
        for i in bounds
    ]
    styles = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        max_bound_percentage = bounds[i] * 100
        styles.append({
            'if': {
                'filter_query': (
                    '{{{column}}} >= {min_bound}' +
                    (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                'column_id': column
            },
            'background': (
                """
                    linear-gradient(90deg,
                    #0074D9 0%,
                    #0074D9 {max_bound_percentage}%,
                    white {max_bound_percentage}%,
                    white 100%)
                """.format(max_bound_percentage=max_bound_percentage)
            ),
            'paddingBottom': 2,
            'paddingTop': 2
        })

    return styles


# #### GRAPHS

# 1. Continental

# In[13]:


current = conti[conti['date']==conti.date.max()].sort_values(by='total_cases')

con = make_subplots(rows=1, cols=2, specs=[[{'type':'scatter'},{'type':'bar'}]], horizontal_spacing=0, column_widths=[0.7, 0.3])

traces =px.line(conti, x='date', y='total_cases', color='location', 
                hover_name = 'location', hover_data = {'location':False,},
                labels={'total_cases':'Total Confirmed Cases','date':'Date'})

for trace in traces.data:
    con.add_trace(trace,1,1)

con.add_trace(go.Bar(y=current.location, x=current['total_cases'],
                     text=current.location, textposition='auto', textfont_color='white',
                     marker=dict(color='rgba(50, 171, 96, 0.6)',
                                 line=dict(color='rgba(50, 171, 96, 1.0)',
                                           width=1),),
                     name='Total Number of Cases',
                     showlegend=False,
                     orientation='h'),
             row=1, col=2)

con.update_layout(xaxis=dict(showgrid=False),
                  legend=dict(title='Continent:', title_side='top',
                              orientation='h',
                              title_font=dict(color='white'),
                             x=0.5,y=1.2, xanchor='center',yanchor='top',
                             font=dict(color='white',size=9), 
                             bordercolor="White", borderwidth=0.5),
                  yaxis2=dict(showticklabels=False)
                 )
print('done')
del current


# #### Creating App

# In[14]:


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN],
                suppress_callback_exceptions=True,
               meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}]
               )
server=app.server
app.title = 'COVID19 Interactive Dashboard'


# #### navbar

# In[15]:


"""navbar component"""

logo = app.get_asset_url('logo1.png')
    
navbar = dbc.Navbar(
        [
            
            html.A([
                    html.Img(src=logo, height="40px", style={'margin-left':'1rem'}),
                    dbc.NavbarBrand("infoPhoenix", style={'margin-left':'1rem'}),
                ]
            ),
            
            html.H5('COVID-19 Interactive Dashboard', className='text-secondary text-uppercase font-weight-bold'), #style={ 'margin-right':'3rem', 'margin-left':'2rem'}   

        ],
        
    color="primary", expand='lg', dark=True, style={'margin':'0rem', 'padding':'0rem'},
)


# #### Cards for glossary

# In[16]:


card_content1 = [
    dbc.CardHeader("Incidence Rate",className="text-primary text-uppercase font-weight-bold"),
    
    dbc.CardBody(
        [
            html.P([
                "It refers to the number of cases occurring, per population at risk, in defined timeframe.",
                html.Br(),
                "It is expressed as ",
                html.B('cases per million population'),
                " with 30-days timeframe for trend analysis.",],
                className="card-text",),
        ]
    ),
]
card_content2 = [
    dbc.CardHeader("Mortality Rate",className="text-primary text-uppercase font-weight-bold"),
    
    dbc.CardBody(
        [
            html.P([
                "It refers to the number of deaths occuring, attributed to a particular disease, per population at risk, in a defined timeframe.",
                html.Br(),
                "It is expressed as ",
                html.B('deaths per million population'),
                " with 30-days timeframe for trend analysis",
            ],
                className="card-text",),
        ]
    ),
]
card_content3 = [
    dbc.CardHeader("Case Fatality Rate", className="text-primary text-uppercase font-weight-bold"),
    
    dbc.CardBody(
        [
            html.P([
                "It refers to the proportion of incident patients dying because of the disease in a given timeframe, expressed as percentage ",
                html.B("(deaths per 100 cases)."),
                html.Br(),
                "It can reflect the seriousness of the condition as well the efficacy of medical interventions.",
            ], className="card-text"),
        ]
    ),
]


# In[17]:


crude_items = [
    {'label':'New Confirmed Cases', 'value':'new_cases'},
    {'label':'New Reported Deaths', 'value':'new_deaths'},
    {'label':'Total Confirmed Cases', 'value':'total_cases'},
    {'label':'Total Reported Deaths', 'value':'total_deaths'}

]

rates_items = [
    {'label':'Incidence Rate', 'value':'incidence'},
    {'label':'Mortality Rate', 'value':'mortality'},
    {'label':'Case Fatality Rate', 'value':'cfr'}
]


# In[18]:


sub_df=para[['location','continent','total_cases_per_million','total_deaths_per_million','total_cases','new_cases', 'total_deaths','new_deaths','cfr']].copy()
sub_df=sub_df.round(2)


# In[19]:


# para.info()


# #### Main content

# In[20]:


""" Main app body components"""
CONTENT_STYLE = {
    "margin": "0rem",
    "padding": "1rem 1rem",
    'background-color':'black'
}


# In[21]:


config={'scrollZoom': False, 'displayModeBar': False}


# In[22]:


content = dbc.Container([
    
    dbc.Row([
            dbc.Col([
                html.H4('Total Cases', className='text-center text-uppercase shadow-sm text-primary'),
                html.H5(str(f'{int(cases):,}'), className='text-center font-weight-bold text-light')
            ],width='auto',#xs=2, sm=2, md=2, lg=2, xl=2,
            ),
            dbc.Col([
                html.H4('Total Deaths', className='text-center text-uppercase shadow-sm text-primary'),
                html.H5(str(f'{int(deaths):,}'), className='text-center font-weight-bold text-light')
            ],width='auto',#xs=2, sm=2, md=2, lg=2, xl=2,
            ),
            dbc.Col([
                html.H4('New Cases', className='text-center text-uppercase shadow-sm text-primary'),
                html.H5(str(f'{int(daily_c):,}'), className='text-center font-weight-bold text-light')
            ],width='auto',#xs=2, sm=2, md=2, lg=2, xl=2,
            ),
            dbc.Col([
                html.H4('New Deaths', className='text-center text-uppercase shadow-sm text-primary'),
                html.H5(str(f'{int(daily_d):,}'), className='text-center font-weight-bold text-light')
            ],width='auto',#xs=2, sm=2, md=2, lg=2, xl=2,
            ),
                ],justify='around'
    ),
    
    html.Hr(style={"height":"1px","border-width":"0"}, className='bg-primary'),
    
    dbc.Row([
        
        dbc.Col([
            dbc.Row(dbc.Col(
            dcc.RadioItems(id = 'radio', 
                           options=[{'label':'Crude Data', 'value':'crude'},
                                    {'label':'Derived Data', 'value':'rates'}],
                           value='crude',
                           labelStyle={'display': 'inline-block'},
                           labelClassName='text-light m-2',
                           inputClassName='p-2 my-1'
                          ),)),
            
            dbc.Row(dbc.Col(
            dcc.Dropdown(
                id='crude_drop', 
                 options=crude_items, 
                 value='new_cases',
                 disabled=False,
                 searchable=False,
                 clearable=False,
                 className="p-2 my-2"),)),

            dbc.Row(dbc.Col(
            dcc.Dropdown(id='rate_drop',
                         options=rates_items,
                         value='incidence',
                         disabled=True,
                         searchable=False,
                         clearable=False,
                         className="p-2 my-2"),)),
            
            dbc.Row(dbc.Col(
            html.A("Check Glossary", href="#glossary")))
            
        ], align='center',width=3, xs=12, sm=12, md=6, lg=3, xl=3,
        ),
        
        dbc.Col(
            dcc.Graph(id='map2',figure={}, 
                      config=config),
            width=9
        ),
    ]),
    
    html.Hr(style={"height":"0.5px", 'background-color':'grey'}),
    
    html.H5('Looking at the trend in various attributes over time - location-wise :', className='text-primary'),
    html.Br(),

    dbc.Row([
        dbc.Col( html.P('Select location here : ', className='d-inline-block text-white'), width='auto'),
        dbc.Col(
            dcc.Dropdown(
                id = 'location_list2',
                options=[{'label': 'World', 'value': 'World'}]+[{'label': i, 'value': i} for i in para.location.unique()],
                value='World',
                clearable=False, 
                className ='d-inline-block w-25 align-center',
            ),
        ),
    ], justify='start'),
    
    html.Br(),
    
#     html.Hr(style={"height":"2px","border-width":"0"}, className='bg-secondary'),

    dbc.Row([
        dbc.Col(
            dcc.Graph(id='crude2',figure={},config=config)
        ),
        dbc.Col(
            dcc.Graph(id='rates2',figure={},config=config),
        ),
    ]),
    
#     html.Hr(style={"height":"0.5px"}),
    
    html.Hr(style={"height":"0.5px", 'background-color':'grey'}),
    
    html.H5('Trend in total COVID19 cases across continents :', className='text-primary'),
    html.Br(),
    
    dbc.Row(dbc.Col(dcc.Graph(id='cont2',figure=con,config=config))),
    
    html.Br(),
    
    html.Hr(style={"height":"0.5px", 'background-color':'grey'}),
    
    html.H5('Updated data on COVID19 as of date '+str(para.iat[0,1].strftime("%d/%m/%Y")), className='text-primary'),
    
    html.Br(),
    
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='table2',
                data = sub_df.to_dict('records'),
                columns = [{'name' : ['','Country'] , 'id' : 'location'},
                           {'name' : ['','Continent'] , 'id' : 'continent'},
                           {'name' : ['Confirmed Cases','Total'] , 'id' : 'total_cases', 'selectable':'last'},
                           {'name' : ['Confirmed Cases','New'] , 'id' : 'new_cases', 'selectable':'last'},
                           {'name' : ['Confirmed Cases','Per Million Pop'] , 'id' : 'total_cases_per_million', 'selectable':'last'},
                           {'name' : ['Reported Deaths', 'Total'] , 'id' : 'total_deaths', 'selectable':'last'},
                           {'name' : ['Reported Deaths','New'] , 'id' : 'new_deaths', 'selectable':'last'},
                           {'name' : ['Reported Deaths','Per Million Pop'], 'id' : 'total_deaths_per_million', 'selectable':'last'},
                           {'name' : ['Case Fatality Rate','Since Pandemic Start'], 'id' : 'cfr', 'selectable':'last'},
                          ],
                merge_duplicate_headers=True,
                sort_action='native',
                sort_mode='multi',
                filter_action='native',
                column_selectable='single',
                selected_columns=['total_cases'],
                cell_selectable=False,
                fixed_rows={'headers': True},   

                style_as_list_view=False,

                style_header={'backgroundColor': '#03396c ',
                              'color':'white',
                              'fontWeight': 'bold',
                              'fontSize':12
                             },
                style_header_conditional=[{'if':{'column_id':i, 'header_index':0},
                                           'textAlign':'center',
                                           'textTransform':'uppercase',
                                           'textDecoration':'underline',
                                           'color':'lightblue',
                                           'fontSize':'14px'} for i in sub_df.columns ],
                
                style_filter={'backgroundColor':'white',
                              'color':'white'},

                style_cell_conditional=[{'if': {'column_id': 'location'},
                                         'textAlign': 'left',
                                         'width':'130px',
                                         'fontWeight':'bold'},
                                        {'if': {'column_id': 'continent'}, 
                                         'textAlign': 'left',
                                        'width':'130px'}],
                style_cell={'padding': '5px',
                            'width':'130px',
                            'textAlign':'center',
                            'backgroundColor': 'rg(90,90,90) ',
                            'color': '#011f4b',
                            'height':'auto',
                            'whiteSpace':'normal',},
            ), width=11
        ), justify='center'
    ),

#     html.Hr(),
    
    html.Hr(style={"height":"0.5px", 'background-color':'grey'}),
    
    html.H5(['Select countries to compare attributes:'], className='text-primary'),
    html.P('(upto 6)',className='text-warning'),
    
    dcc.Dropdown(
        id = 'locations',
        options=[{'label': i, 'value': i} for i in para.location.unique()],
        value=['India','United States','Turkey'],
        className ='d-inline-block w-50 align-center',
        multi=True,
        clearable=False
    ),
    
    dbc.Tabs(
            [
                dbc.Tab(label="Cases", tab_id='first',tab_style={"margin-left": "auto"},label_style={"color": "#00AEF9"}),
                dbc.Tab(label="Deaths", tab_id='second',label_style={"color": "#00AEF9"}),
            ],
        id='tabs',
        active_tab='first'
        ),
    
    dbc.Row([
       dbc.Col([  
           html.Hr(),
           dcc.Graph(id='compare_incidence', figure={},config=config)
           
       ]),
        dbc.Col([
            html.Hr(),
            dcc.Graph(id='compare_mortality', figure={},config=config)
        ])
    ]),
    
#     dbc.Row(dbc.Col(dcc.Graph(id='cfr_bar', figure={}), width=6), justify='center'),
    
    html.Hr(),
    
    html.H5('GLOSSARY', id='glossary', className='text-primary border-info text-center', style={'border':'1px solid'}),
    
    dbc.CardDeck([
        dbc.Card(card_content1, color='secondary'),
        dbc.Card(card_content2, color='secondary'),
        dbc.Card(card_content3, color='secondary'),
    ], className='m-1'),
           
    html.Hr(style={"height":"1px"}, className='bg-dark'),

    dbc.Row(
        dbc.Col(
            html.P([u"\u00A9"+" Sayali Bachhav       @infoPhoenix"], className='text-white text-center'),
#             width='auto', #xs=7, sm=7, md=7, lg=7, xl=7,
        )),
], 
    fluid=True,
    style=CONTENT_STYLE 
)


# In[23]:


@app.callback(
    [Output('crude_drop', 'disabled'),
     Output('rate_drop', 'disabled')],
    [Input('radio', 'value')])
def update_drop(value):
    if value=='crude':
        return False, True
    elif value=='rates':
        return True, False


# In[24]:


@app.callback(
    [Output('map2', 'figure'),
     Output('crude2', 'figure'),
    Output('rates2', 'figure'),
    Output('table2','style_data_conditional'),
    Output('compare_incidence', 'figure'),
     Output('compare_mortality', 'figure')],
    [Input('crude_drop', 'value'),
     Input('rate_drop', 'value'),
    Input('crude_drop', 'disabled'),
     Input('rate_drop', 'disabled'),
    Input('location_list2', 'value'),
    Input('table2','selected_columns'),
    Input('locations', 'value'),
    Input('tabs','active_tab')])

def update_data(value1, value2, disabled1, disabled2, value, cols, nations, active):
    if not disabled1:
        fig = get_map(value1)
    if not disabled2:
        fig= get_map(value2)
        
    fig1 = plot_crude(value)
    fig2 = plot_rates(value)
    
    for col in cols:
        styles =data_bars(sub_df, col)
    
    if len(nations)>6:
        nations=nations[:6]
        
    daily, inci = compare_plot(nations, active)
    
    return fig, fig1, fig2, styles, daily, inci


# #### Main App Layout

# In[25]:


app.layout = html.Div(
    [
        navbar,
        content
    ]
)


# #### Running App

# In[26]:


if __name__=='__main__':
    app.run_server(debug=True, use_reloader=False) 

