import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import datetime
import base64
import json
import pandas 
import plotly
import io
import numpy as np
import argparse
import os
import itertools
import urllib.request
from keras.models import load_model
from keras.applications import vgg16  
from keras.preprocessing.image import img_to_array   
from keras.utils.np_utils import to_categorical   
import math
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from tensorflow import get_default_graph
from io import BytesIO

### IMPORT MODEL AND DATA
path = '/Users/lizbaldo/Desktop/Insight_Project/' 
model = load_model(path + 'my_model_25.h5')
graph_top = get_default_graph()
model_vgg16 = vgg16.VGG16(include_top=False, weights='imagenet')
graph_vgg16 = get_default_graph()
text_directory = path + 'descriptions/'
data = pandas.read_csv(path + 'cleaned_wikiart_data_artist_app.csv',encoding='utf-8')

style = ['Abstract Art','Abstract Expressionism','Art Informel','Modern',\
         'Baroque','Color Field Painting','Cubism','Early Renaissance','Expressionism',\
         'High Renaissance','Impressionism','Magic Realism','Late Renaissance',\
         'Minimalism','Naive Art','Neoclassicism','Northern Renaissance','Pop Art',\
         'Post-Impressionism','Realism','Rococo','Romanticism','Surrealism','Symbolism','Ukiyo-e']

title_filename = 'find_your_art.png'
encoded_title = base64.b64encode(open(title_filename, 'rb').read())

Liz_Pic = 'Liz_Baldo_Linkedin.jpg' 
encoded_Liz = base64.b64encode(open(Liz_Pic, 'rb').read())

app = dash.Dash()

app.config.suppress_callback_exceptions = True

app.scripts.config.serve_locally = True

app.title='Find Your Art'

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
        html.H4(               
             children=dcc.Link('About', href='/about'), style={'margin-left': '20px'}
             ),
    
    html.Img(src='data:image/png;base64,{}'.format(encoded_title.decode()),style = {'display': 'block','textAlign': 'center','width': '70%','height': '70%','margin': 'auto'}),
    html.Div([
        html.Div([
            html.H3(
                children='Which style does this art piece belong to?',
                style={
                    'textAlign': 'center',
                    }),
              dcc.Upload(
                  id='upload-image',
                  children=html.Div([
                      'Drag and Drop or ',
                       html.A('Select Files')
                      ]),
                  style={
                      'width': '100%',
                      'height': '60px',
                      'lineHeight': '60px',
                      'borderWidth': '1px',
                      'borderStyle': 'dashed',
                      'borderRadius': '5px',
                      'textAlign': 'center',
                      'margin': 'auto'
                      },
                  multiple=False
                  ),
            html.Div(id='output-image-upload'),
            ]),
  
        ]),
    ])


def parse_contents(contents, filename, date):
    try:
        # Convert image url to the shape required by VGG16
        url_response = urllib.request.urlopen(contents)
        img = Image.open(BytesIO(url_response.read()))
        image = img.resize((224,224))
        image = img_to_array(image)  
        image = image / 255    
        image = np.expand_dims(image, axis=0)
        # Extract bottleneck features from the VGG16 network
        global graph_vgg16
        with graph_vgg16.as_default():
            bottleneck_prediction = model_vgg16.predict(image) 
        global graph_top
        with graph_top.as_default():
            class_predicted = model.predict_classes(bottleneck_prediction)
        inID = class_predicted[0]     
        label = style[inID] 
         
        #feat = np.array(image_small)
        #label = model.predict(feat.reshape(1, -1))
        title = str(label)
        f = open(text_directory + title + '_description.txt', 'r')
        file_contents = f.read()
        #print(title)
        data_label = data[data['style'].str.contains(title)]
        recommendation = data_label.sample(n=3)
        recommendation['contentId']=recommendation['contentId'].apply(str)  
     #   image_rec = list('/Users/lizbaldo/Desktop/wikiart-master/wikiart/images/' + recommendation['contentId'] + '.jpg')
        title_rec = list(recommendation['title']) 
        artist_rec = list('By '  + recommendation['artistName'])
        title_url = list(recommendation['url']) 
        artist_url = list(recommendation['artistUrl'])    
        return html.Div([

            html.Div([
            html.Div([
            html.Img(src=contents,style = {'display': 'block','width': '100%','height': '100%','margin-top': '25px'})
            ],className="six columns"),
            html.Div([
            html.H3(label,style = {'display': 'block','textAlign': 'center','margin-top': '25px'})
            ],className="six columns"),
            html.H6(str(file_contents),className="six columns",style = {'margin-left': '20px','textAlign': 'center'}),
            html.H6('Source: wikipedia.org',style = {'margin-top': '25px','display': 'block','textAlign': 'center'},className="six columns")
            ]),
            html.Div([
             html.Div([           
                html.H3('You might also like:',style = {'margin-top': '25px','display': 'block','textAlign': 'center'},className="twelve columns"),
                html.H6('Source: wikiart.org',style = {'margin-bottom': '25px','display': 'block','textAlign': 'center'},className="twelve columns")
                ]),
            html.Div([
                   html.Img(src='https://uploads5.wikiart.org/images/' + artist_url[0] + '/' + title_url[0] +'.jpg',
                            style = {'display': 'block-inline','margin-top': '20px','margin': 'auto' ,'width': '100%','height': '100%'}),
                  html.H6(title_rec[0],style = {'display': 'block','textAlign': 'center'}),
                  html.H6(artist_rec[0],style = {'display': 'block','textAlign': 'center'}),
                   ],className="four columns"),
            html.Div([
                   html.Img(src='https://uploads5.wikiart.org/images/' + artist_url[1] + '/' + title_url[1] +'.jpg',
                            style = {'display': 'block-inline','margin-top': '20px','margin': 'auto' ,'width': '100%','height': '100%'}),
                  html.H6(title_rec[1],style = {'display': 'block','textAlign': 'center'}),
                  html.H6(artist_rec[1],style = {'display': 'block','textAlign': 'center'}),
                   ],className="three columns"),
            html.Div([
                   html.Img(src='https://uploads5.wikiart.org/images/' + artist_url[2] + '/' + title_url[2] +'.jpg',
                            style = {'display': 'block-inline','margin-top': '20px','margin': 'auto' ,'width': '100%','height': '100%'}),
                   html.H6(title_rec[2],style = {'display': 'block','textAlign': 'center'}),
                   html.H6(artist_rec[2],style = {'display': 'block','textAlign': 'center'}),
                   ],className="four columns")
            ],className="twelve columns")
            ])
    except:
        return html.Div([
                   html.H6('Format not supported, or image larger than 1mb, please try again.',style = {'display': 'block','textAlign': 'center'},className="twelve columns")
                ])


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               Input('upload-image', 'filename'),
               Input('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:        
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

page_1_layout = html.Div([
               html.H4(               
             children=dcc.Link('Home', href='/'), style={'margin-left': '20px'}
             ),
    html.H1('About',style={
                    'textAlign': 'center','margin-bottom': '20px'
                    }),
    html.Div([
             html.H5(               
             children='Liz Baldo', style={'textAlign': 'center'}
             ),
           html.Img(src='data:image/png;base64,{}'.format(encoded_Liz.decode()),style = {'display': 'block','textAlign': 'center','margin': 'auto','width': '20%','height': '20%','margin-bottom': '10px'}),
           html.A("Find Liz on Linkedin", href='https://www.linkedin.com/in/lizbaldo/', target="_blank", style={'display': 'block','textAlign': 'center','margin': 'auto','margin-bottom': '30px'}),
            html.H6(               
            children='Liz is a Data Science Fellow at Insight Data Science in Boston, MA. She got her Ph.D. from UCLA in Civil Engineering and Water Resources Engineering. \
               Liz also grew up as the only space nerd in a family of painters, sculptors and musicians, and designed "Find Your Art" as a tool to help non-experts discover \
               styles they like without any prerequisite in art history!', style={'display': 'block','textAlign': 'center','margin': 'auto','width': '50%','height': '20%','margin-bottom': '30px'}),
             ]),

    html.Div([
            html.H6(children='Find more about the project!', style={'textAlign': 'center','margin-bottom': '10px'}), 
            html.Iframe(src="https://docs.google.com/presentation/d/e/2PACX-1vQQOn8v3BeIssWz2qzbBXc3u4RuBMLQSMWdV7klwNBgfbDIhTIKww-n3bAo-GAEQdVCcbPLXsjmzjNu/embed?start=false&loop=false&delayms=60000",\
                         style={'width': '960px','height': '569px','display': 'block','textAlign': 'center','margin': 'auto'})
       ]),
    html.Div(id='page-1-content'),
])

# Update the index
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/about':
        return page_1_layout
    else:
        return index_page
    

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

        
    
if __name__ == '__main__':
	app.run_server(debug=True)    
