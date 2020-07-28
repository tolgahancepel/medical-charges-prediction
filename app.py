# Import required libraries
import joblib
import copy
import pathlib
import dash
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
from dash import no_update
import plotly.graph_objs as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server


# Reading and preprocessing the dataset
# -----------------------------------------------------------------------------------------------
df = pd.read_csv("data/insurance.csv")



# Encoding features

ohe_smoker = OneHotEncoder(drop='first').fit(df["smoker"].values.reshape(-1,1))
ohe_smoker.get_feature_names(['smoker'])

ohe_sex = OneHotEncoder(drop='first').fit(df["sex"].values.reshape(-1,1))
ohe_sex.get_feature_names(['sex'])

ohe_region = OneHotEncoder(drop='first').fit(df["region"].values.reshape(-1,1))
ohe_region.get_feature_names(['region'])



df_2 = pd.concat(
    [
        df,
        pd.DataFrame(ohe_smoker.transform(df["smoker"].values.reshape(-1,1)).toarray(),
                    columns = ohe_smoker.get_feature_names(['smoker'])).astype(int)
    ], axis=1).drop("smoker", axis=1)



df_2 = pd.concat(
    [
        df_2,
        pd.DataFrame(ohe_sex.transform(df["sex"].values.reshape(-1,1)).toarray(),
                    columns = ohe_sex.get_feature_names(['sex'])).astype(int)
    ], axis=1).drop("sex", axis=1)

df_2 = pd.concat(
    [
        df_2,
        pd.DataFrame(ohe_region.transform(df["region"].values.reshape(-1,1)).toarray(),
                    columns = ohe_region.get_feature_names(['region'])).astype(int)
    ], axis=1).drop("region", axis=1)


X = df_2.drop('charges', axis = 1).values
y = df_2['charges'].values.reshape(-1,1)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(-1,1))


# df = pd.read_csv("data/insurance.csv")


# Random Forrest Model
# ------------------------------------------------------------------------------------------------

rf_path = 'data/random_forest_model.sav'
rf_model = joblib.load(rf_path)

lasso_path = 'data/lasso_model.sav'
lasso_model = joblib.load(lasso_path)

svr_path = 'data/svr_model.sav'
svr_model = joblib.load(svr_path)


layout = dict(
    autosize=True,
    # automargin=True,
    margin=dict(l=30, r=30, b=20, t=40),
    hovermode="closest",
    plot_bgcolor="#1a2229",
    paper_bgcolor="#2d353c",
    xaxis = dict(color="#9ba8b4", showgrid=False),
    yaxis = dict(color="#9ba8b4", showgrid=False),
    
    title="Satellite Overview",
    titlefont=dict(
        family='Open Sans',
        size=18,
        color='white'
    ),
    
    
    legend = dict(
            x=0.16,
            y=-0.12,
            traceorder="normal",
            font=dict(
                family="Open Sans",
                size=12,
                color="#9ba8b4"
            ),
            bgcolor="#1a2229",
            bordercolor="Black",
            borderwidth=1,
            orientation='h'
        )
    
    
)




def smoker_graph():
    
    layout_count = copy.deepcopy(layout)
    
    fig = go.Figure(layout=layout_count)


    fig.add_trace(go.Violin(x=df['smoker'][ df['smoker'] == 'yes' ],
                            y=df['charges'][ df['smoker'] == 'yes' ],
                            box_visible=True, opacity=1   ,
                            legendgroup='Smoker', scalegroup='M', name='Smoker',
                            fillcolor='#e8871a', line_color='black',))
    
    fig.add_trace(go.Violin(x=df['smoker'][ df['smoker'] == 'no' ],
                            y=df['charges'][ df['smoker'] == 'no' ],
                            box_visible=True, opacity=1,
                            legendgroup='Non-Smoker', scalegroup='M', name='Non-Smoker',
                            fillcolor='#00c0c7', line_color='black',))   
    
    fig.update_layout(
        title = "Smoker and Non-Smoker Charges",
        title_x=0.5,
        yaxis_title="Charges",
        
        titlefont=dict(
            family='Open Sans',
            size=18,
            color = "#ffffff"
        ),
        
        
        showlegend = False,
        
        yaxis_zeroline=False)
    
    return fig




def age_graph():
    
    layout_count = copy.deepcopy(layout)
    
    fig = go.Figure(layout=layout_count)
    
    fig.add_trace(go.Scatter(x=df["age"],
                             y=df["charges"],
                             mode='markers',
                             opacity=0.7,
                             marker_symbol = "hexagon",
                             marker=dict(
                                 color='#00c0c7',
                                 size=8,
                                  line=dict(
                                      color='black',
                                      width=0.4
                                  )
                             ),
                         )),  
    
    
    
    
    
    
    fig.update_layout(
        title = "Charges with respect to Age",
        title_x=0.5,
        xaxis_title="Age",
        # yaxis_title="Charges",
        
        
        titlefont=dict(
            family='Open Sans',
            size=18,
            color = "#ffffff"
        ),
        
        
        yaxis_zeroline=False)
    
    
    
    return fig




def rsquared_graph():
    
    layout_count = copy.deepcopy(layout)
    
    fig = go.Figure(layout=layout_count)
    
    fig.add_trace(
        go.Bar(
            name='test set',
            y=['Random Forest', 'Decision Tree', 'Lasso', 'Ridge', 'SVR', 'Linear'],
            x=[0.896686, 0.877656, 0.861600, 0.861564, 0.861047, 0.767264],
            orientation='h',
            marker=dict(
                color = '#006064'
            )
            
        ),    
    )
    
    fig.update_layout(
        title = "R2 Score (Test set)",
        title_x=0.5,
        yaxis_title="Model",
        titlefont=dict(
            family='Open Sans',
            size=18,
            color = "#ffffff"
        ),
        showlegend = False,
        yaxis_zeroline=False
    )
    
    return fig



def rmse_graph():
    
    layout_count = copy.deepcopy(layout)
    
    fig = go.Figure(layout=layout_count)
    
    fig.add_trace(
        go.Bar(
            name='test set',
            x=['Random Forest', 'Decision Tree', 'Lasso', 'Ridge', 'SVR', 'Linear'],
            y=[3948.307638, 4296.587786, 4569.831897, 4570.420248, 4578.942853, 5926.023602],
            orientation='v',
            marker=dict(
                color = '#AC5700'
            )
            
        ),    
    )
    
    fig.update_layout(
        title = "RMSE (Test set)",
        title_x=0.5,
        yaxis_title="Model",
        titlefont=dict(
            family='Open Sans',
            size=18,
            color = "#ffffff"
        ),
        showlegend = False,
        yaxis_zeroline=False
    )
    
    return fig






# Create app layout
app.layout = html.Div(
    [
        # dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        
        
        # Navbar
        # --------------------------------------------------------------------------------
        
        
        html.Div(
            
            [html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src=app.get_asset_url("dash-logo.png"),
                            id="cepel-logo",
                        )
                    ],
                    className="one-third column ", id = "cepel-col"
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3(
                                    "Medical Charges Prediction",
                                    style={"margin-bottom": "0px"},
                                    className = "white_input navbar_header_text"
                                ),
                                html.H5(
                                    "Machine Learning Dashboard", style={"margin-top": "0px"}
                                ),
                            ]
                        )
                    ],
                    className="two column",
                    id="title",
                ),
                
                
                
                
                
                html.Div(
                    [
                        html.Img(src=app.get_asset_url("GitHub-Mark-Light-64px.png"), id = "gh-logo"),
                        
                        html.A(
                            id="gh-link",
                            children=["View on GitHub"],
                            style={"color": "white", "border": "solid 1px white"},
                            href="https://github.com/tolgahancepel/medical-charges-prediction",
                        )
                    ],
                    className="one-third column",
                    id="button",
                ),
   
                
            ],
            id="header",
            className="row flex-display dark_header",
        )],
            
            
            
            
            
            
        
            ),
        
        
        # First Row
        # --------------------------------------------------------------------------------
        
        html.Div(
            [
                
                # BMI Calculator
                # --------------------------------------------------------------------------------
                
                html.Div(
                    [
                        
                        # Header of BMI
                        html.Div([
                            html.H6("Body Mass Index Calculator", className = "bmi_card_header_text"),
                            html.H4("Your BMI: 0", id = "bmi_value", className = "bmi_card_value_text")
                        ], className = "bmi_card_header"),
                        
                        # Height Input
                        
                        html.P("Height (cm)", className="control_label white_input"),
                        html.Div(
                            dcc.Input(
                                className = "bmi_input",
                            id="input_height", type="number", placeholder="Enter your height",
                            min=0, max=250),
                         className="dcc_control",                            
                         ),
                        
                        # Weight Input
                        
                        html.P("Weight (kg)", className="control_label white_input"),
                        html.Div(
                            dcc.Input(
                                className = "bmi_input",
                                id="input_weight", type="number", placeholder="Enter your weight",
                                min=0, max=250),
                         className="dcc_control ",                            
                         ),
                        
                        # Age Input
                        
                        html.P("Age", className="control_label white_input"),
                        html.Div(
                            dcc.Input(
                                className = "bmi_input",
                                id="input_age", type="number", placeholder="Enter your age",
                                min=2, max=122),
                         className="dcc_control ",                            
                         ),
                            
                        # Calculate and Reset Buttons
                        
                        html.Div(
                            [
                                html.Button('Reset', id='btn_reset', n_clicks=0, className= "btn_reset"),
                                html.Button('Calculate', id='btn_calculate', n_clicks=0, className = "btn_calculate"),
                            ],
                         className="dcc_control", id = "bmi-buttons"                         
                         ),

                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
                 
                
                # Prediction
                # --------------------------------------------------------------------------------
                
                html.Div(
                    [
                        
                        
                        html.Div(
                            [
                                # Header of Prediction Card
                                html.Div([
                                    html.H5("Prediction", className = "bmi_card_header_text"),
                                ], className = "prediction_card_header"),
                                
                                
                                html.Div([
                                    
                                    # Age
                                                                               
                                    html.Div(
                                        [
                                            html.P("Age", className="control_label white_input"),
                                            dcc.Input(
                                                id="predict_age", type="number", placeholder="Enter your age",
                                                min=2, max=122),
                                                
                                        ], className = "three columns predict_input",
                                    ),
                                    
                                    # BMI
                                    
                                    html.Div(
                                        [
                                            html.P("BMI", className="control_label white_input"),
                                            dcc.Input(
                                                id="predict_bmi", type="number", placeholder="Enter your BMI",
                                                min=2, max=122),
                                                
                                        ], className = "three columns predict_input",
                                    ),
                                    
                                    
                                    # Children
                                    
                                    html.Div(
                                        [
                                            html.P("Children", className="control_label white_input"),
                                            dcc.Dropdown(
                                                id='predict_children',
                                                options=[
                                                    {'label': '0', 'value': '0'},
                                                    {'label': '1', 'value': '1'},
                                                    {'label': '2', 'value': '2'},
                                                    {'label': '3', 'value': '3'},
                                                    {'label': '4', 'value': '4'},
                                                    {'label': '5', 'value': '5'},
                                                    {'label': '6', 'value': '6'},
                                                    {'label': '7', 'value': '7'},
                                                    {'label': '8', 'value': '8'},
                                                    {'label': '9', 'value': '9'},
                                                    {'label': '10', 'value': '10'},
                                                ], value='0'
                                            ),
                                                
                                        ], className = "three columns predict-input-last",
                                    ),
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                ], className = "row"),
                                
                                
                                
                                
                                
                                
                                
                                
                                html.Div([
                                    
                                    # Region
                                    
                                    html.Div(
                                            [
                                                  html.P("Region", className="control_label white_input"),
                                                  dcc.Dropdown(
                                                      id='predict_region',
                                                      options=[
                                                          {'label': 'Southwest', 'value': 'southwest'},
                                                          {'label': 'Southeast', 'value': 'southeast'},
                                                          {'label': 'Northwest', 'value': 'northwest'},
                                                          {'label': 'Northeast', 'value': 'northeast'}
                                                      ],
                                                      value="southwest"
                                                  ),
                                                
                                            ], className = "three columns predict_input",
                                        ),
                                    
                                    
                                    
                                    
                                       # Sex
                                        
                                        html.Div(
                                            [
                                                  dcc.RadioItems(
                                                    id = "predict_sex",
                                                    options=[
                                                        {"label": "Male ", "value": "male"},
                                                        {"label": "Female ", "value": "female"},
                                                    ],
                                                    value="male",
                                                    labelStyle={"display": "inline-block"},
                                                    className="dcc_control",
                                                ),
                                                
                                            ], className = "three columns predict_input", style = {"margin-top": "5%"}
                                        ),
                                    
                                    
                                    
                                    html.Div(
                                            [
                                                  dcc.Checklist(
                                                    id = "predict_smoker",
                                                    options=[
                                                        {'label': 'Smoker', 'value': 1},
                                                    ],
                                                    labelStyle={'display': 'inline-block'}
                                                )  
                                                
                                            ], className = "three columns predict-input-last", style = {"margin-top": "5%"}
                                        ),
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                ], className = "row"),
                                

                                
                                html.Div(
                                    [
                                        
                                        html.Button('Predict', id='btn_predict', n_clicks=0, className = "twelve columns btn-predict"),
                                    ],
                                 className="row container-display", style = {"padding-top": "20px"}                           
                                 ),
                                
                                
                            ],
                            className="pretty_container"
                        ),
                        
                        
                        # Prediction results
                        
                        html.Div(
                            [
                                html.Div(
                                    [html.H6(id="rf_result", children = "$0000.00", className = "predict_result"), html.P("Random Forest")],
                                    id="wells",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="svr_result", children = "$0000.00", className = "predict_result"), html.P("SVM")],
                                    id="gas",
                                    className="mini_container",
                                ),
                                html.Div(
                                    [html.H6(id="lasso_result", children = "$0000.00", className = "predict_result"), html.P("Lasso")],
                                    id="oil",
                                    className="mini_container",
                                ),
                            ],
                            id="info-container",
                            className="row container-display",
                        ),
                        
                    ],
                    id="right-column",
                    className="eight columns",
                ),
                
                
                
                
                
                
            ],
            className="row flex-display",
        ),
        
        
        
        html.Div(
            
            [
            
            
            
            
                
            
                html.Div(
                    [
                        dcc.Tabs(
                        [
                            
                            dcc.Tab(label = "Data Analysis", children = [
                                
                                html.Div([
                                    
                                    dcc.Graph(figure = smoker_graph(), className = "four columns", style = {"height": "400px" }),
                                    
                                    dcc.Graph(figure = age_graph(), className = "eight columns", style = {"height": "424px" })
                                    
                                ], className = "tab_content"),
                                
                                
                                
                            ]),
                            
                            
                            dcc.Tab(label = "Data Distribution", children = [
                                
                                html.Div([
                                    html.H4("Data Distribution"),
                                
                                    html.P("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec ornare nulla eget purus rhoncus, at posuere nibh sollicitudin. Quisque at consequat massa, a porttitor sapien. Duis eu diam risus. Cras tortor dui, luctus ac porttitor eget, interdum ut arcu. Sed vel massa faucibus, volutpat velit vel, fermentum dolor. Integer eu pharetra purus. Pellentesque in eros lacus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nullam tempor malesuada lacinia. Aenean aliquam tincidunt nibh, et pulvinar odio aliquam eget. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut consequat dolor magna, vel laoreet libero viverra vitae. Vivamus vestibulum luctus nibh et euismod. "),
                                ], className = "tab_content"),
                                
                                
                                
                            ]),
                            
                            dcc.Tab(label = "Models Performance", children = [
                                
                                html.Div([
                                    
                                    dcc.Graph(figure = rsquared_graph(), className = "twelve columns", style = {"height": "360px" }),
                                        
                                    
                                    dcc.Graph(figure = rmse_graph(), className = "twelve columns", style = {"height": "360px" }),
                                                                        
                                ], className = "tab_content"),
                                
                                
                                
                            ]),
                            
                            dcc.Tab(label = "About the App", children = [
                                
                                html.Div([
                                
                                    # html.P("""
                                    #        This dash application allows you to predict medical charges using machine learning alogirthms
                                    #        (Random Forest Regression, SVR and Lasso Regression). You can also:
                                    #        """
                                    #        ),
                                        
                                    dcc.Markdown('''
                                    #### **Predictive Analysis on Medical Charges**
                                    
                                    This dash application allows you to predict medical charges using machine learning alogirthms
                                    (Random Forest Regression, SVR and Lasso Regression). Developed with Python and all codes published
                                    on GitHub. Feel free to review and download repository. You can:
                                    * calculate body mass index,
                                    * predict medical costs billed by health insurance
                                    * review data analysis
                                    * explore data distribution.
                                    
                                    ##### **Inspiration**
                                    
                                    I have inspired one of my old Kaggle notebook. You can find details about different ML pipelines and hyperparameters tuning:  
                                    https://www.kaggle.com/tolgahancepel/medical-costs-regression-hypertuning-eda
                                    
                                    ##### **Dataset**
                                    https://www.kaggle.com/mirichoi0218/insurance
                                    
                                    
                                    
                                    
                                    
                                    '''
                                    )    
                                    
                                ], className = "tab_content"),
                                
                                
                                
                            ]),
                        
                        
                        ], className = ""               
                        ),
                    
                    
                    ], className = "tabs_pretty_container twelve columns")
                
            
                   
            ], className = "row flex-display",
            
            )
        
        
        
        
        

        
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)






# App Callbacks
# --------------------------------------------------------------------------------------------

# BMI Calculate Button

@app.callback(
    dash.dependencies.Output('bmi_value', 'children'),
    
    [dash.dependencies.Input('btn_calculate', 'n_clicks')],
    
    [dash.dependencies.State('input_weight', 'value'),
     dash.dependencies.State('input_height', 'value')],
)

def update_bmi(n_clicks, input_weight, input_height):
    if(n_clicks):
        return "Your BMI: " + ("%.2f" % (input_weight / (input_height * input_height) * 10000))
    
    else:
        return no_update


@app.callback(
    dash.dependencies.Output('predict_age', 'value'),
    
    [dash.dependencies.Input('btn_calculate', 'n_clicks')],
    
    [dash.dependencies.State('input_age', 'value')],
)

def copy_age(n_clicks, input_age):
    if(n_clicks):
        return input_age
    
    else:
        return no_update


@app.callback(
    dash.dependencies.Output('predict_bmi', 'value'),
    
    [dash.dependencies.Input('btn_calculate', 'n_clicks')],
    
    [dash.dependencies.State('input_weight', 'value'),
     dash.dependencies.State('input_height', 'value')],
)

def copy_bmi(n_clicks, input_weight, input_height):
    if(n_clicks):
        return ("%.2f" % (input_weight / (input_height * input_height) * 10000))
    
    else:
        return no_update




# BMI Reset Button

@app.callback(
    [dash.dependencies.Output('input_height', 'value'),
    dash.dependencies.Output('input_weight', 'value'),
    dash.dependencies.Output('input_age', 'value'),
    ],
    
    [dash.dependencies.Input('btn_reset', 'n_clicks')]
)

def reset_bmi(n_clicks):
    if(n_clicks):
        return (None, None, None)
    else:
        return no_update

# Prediction

@app.callback(
    [dash.dependencies.Output('rf_result', 'children'),
     dash.dependencies.Output('lasso_result', 'children'),
     dash.dependencies.Output('svr_result', 'children')],
    
    [dash.dependencies.Input('btn_predict', 'n_clicks')],
    
    [dash.dependencies.State('predict_age', 'value'),
     dash.dependencies.State('predict_bmi', 'value'),
     dash.dependencies.State('predict_children', 'value'),
     dash.dependencies.State('predict_region', 'value'),
     dash.dependencies.State('predict_sex', 'value'),
     dash.dependencies.State('predict_smoker', 'value')],
)


def predict_result(n_clicks, input_age, input_bmi, input_children, input_region, input_sex, input_smoker):
    if(n_clicks):
        
        if (not input_smoker):
            isSmoker = "no"
        elif (len(input_smoker) == 1):
            isSmoker = "yes"
        
        print("smoker: ", isSmoker)
        
        
        sample = [input_age, input_sex, input_bmi, input_children, isSmoker, input_region]
        
        # sample = [19, "female", 27.900, 0, "yes", "southwest"]
        
        
        sample = pd.DataFrame([sample], columns = ["age", "sex", "bmi", "children", "smoker", "region"])
        
        
        sample = pd.concat(
            [
                sample,
                pd.DataFrame(ohe_smoker.transform(sample["smoker"].values.reshape(-1,1)).toarray(),
                            columns = ohe_smoker.get_feature_names(['smoker'])).astype(int)
            ], axis=1).drop("smoker", axis=1)
        
        sample = pd.concat(
            [
                sample,
                pd.DataFrame(ohe_sex.transform(sample["sex"].values.reshape(-1,1)).toarray(),
                            columns = ohe_sex.get_feature_names(['sex'])).astype(int)
            ], axis=1).drop("sex", axis=1)
        
        sample = pd.concat(
            [
                sample,
                pd.DataFrame(ohe_region.transform(sample["region"].values.reshape(-1,1)).toarray(),
                            columns = ohe_region.get_feature_names(['region'])).astype(int)
            ], axis=1).drop("region", axis=1)
            
        
        
        
        
        
        
        rf_result = rf_model.predict(sample)
        lasso_result = lasso_model.predict(sample)
        svr_result = sc_y.inverse_transform(svr_model.predict(sc_X.transform(sample)))
        
        return ("$" + ("%.2f" % rf_result)), ("$" + ("%.2f" % lasso_result)), ("$" + ("%.2f" % svr_result))
    else:
        return no_update







# Main
if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=False)
