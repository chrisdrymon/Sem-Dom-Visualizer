from nltk.corpus import wordnet as wn
from collections import Counter
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output
import random
import json


# A recursive function for climbing the synset hierarchy and gathering data along the way
def synset_counting(ss_list, ss_counter, pairs):
    next_list = []
    if len(ss_list) == 0:
        return ss_counter, pairs
    else:
        for ss in ss_list:
            for higherss in ss.hypernyms():
                if ss in pairs:
                    if higherss in pairs[ss]:
                        pass
                    else:
                        pairs[ss].append(higherss)
                else:
                    pairs[ss] = [higherss]
                ss_counter[higherss] += 1
                if higherss != wn.synset('entity.n.01'):
                    next_list.append(higherss)
        return synset_counting(next_list, ss_counter, pairs)


# Placed this here so it can be run to produce the initial graph
def make_dash(word):
    base_synsets = []
    basepaths = []
    base_defs = [html.H3('Definitions', style={'text-align': 'center'}), html.Br()]
    multiparents = []
    synset_counter = Counter()
    child_parent_pairs = {}

    if word is None:
        raise PreventUpdate

    if word is "":
        raise PreventUpdate

    # WordNet can handle some multiword phrases, but they need underscores instead of spaces
    word = word.replace(' ', '_')
    display_word = word.replace('_', ' ')

    # Create a list of all base synsets
    for i, synset in enumerate(wn.synsets(word, pos=wn.NOUN)):
        for path in synset.hypernym_paths():
            basepaths.append(path)
        base_synsets.append(synset)

    # Order the base synsets list, grab their definitions, and format them for display in the app
    for i, synset in enumerate(sorted(base_synsets)):
        ss_split = str(synset)[8:].split('.n.')
        base_defs.append(html.B(ss_split[0].replace('_', ' ')))
        base_defs.append(html.Span(ss_split[1][:2].lstrip('0'), className='ss'))
        base_defs.append(' ' + synset.definition())
        base_defs.append(html.Br())

    synset_counter, child_parent_pairs = synset_counting(base_synsets, synset_counter, child_parent_pairs)
    for child in child_parent_pairs:
        if len(child_parent_pairs[child]) > 1:
            multiparents.append(child)

    adjusted_paths = []
    max_len = 0
    for path in basepaths:
        # This number will be displayed in the app.
        if len(path) > max_len:
            max_len = len(path)
        # The tricky part here is converting the output of WordNet which sometimes assigns multiple hypernyms to a
        # single synset into the input for plotly which requires that each synset only have a single parent. So
        # renaming had to be done to synsets with multiple parents.
        multi = False
        revised_path = []
        multi_suffix = ''
        for k, item in enumerate(path):
            if multi:
                if item in multiparents:
                    multi_suffix = multi_suffix + '-' + str(path[k-1])
                    revised_path.append(str(item) + '-' + multi_suffix)
                else:
                    revised_path.append(str(item) + '-' + str(multi_suffix))
            else:
                if item in multiparents:
                    multi_suffix = str(path[k-1])
                    revised_path.append(str(item) + '-' + multi_suffix)
                    multi = True
                else:
                    revised_path.append(str(item))
        adjusted_paths.append(revised_path)

    ids = ["Synset('entity.n.01')"]
    labels = ['entity']
    parents = ['']
    for path in adjusted_paths:
        for j, node in enumerate(path):
            if node not in ids:
                ids.append(node)
                labels.append(node.split('.n')[0][8:].replace('_', ' '))
                parents.append(path[j-1])

    # This checks to see if WordNet recognizing the word as a noun. If not, it returns an error.
    if len(wn.synsets(word, pos=wn.NOUN)) == 0:
        err_title = f'Error: WordNet does not recognize "{display_word.capitalize()}" as a noun.'
        figure = {'data': [{'type': 'sunburst'}]}
    else:
        figure = {'data': [{'type': 'sunburst',
                            'ids': ids,
                            'labels': labels,
                            'parents': parents,
                            'hovertext': ids,
                            'hoverinfo': 'text'}],
                  'layout': {'font': {'family': 'Quicksand',
                                      'size': 24},
                             'margin': {'l': 10,
                                        'r': 10,
                                        'b': 10,
                                        't': 10},
                             'colorway': ['#457b9d', '#e63946']
                             }
                  }
    graph_title = f'Semantic Domains of "{display_word.capitalize()}"'
    base_ss_list = ['The noun "', html.B(f'{display_word}'), '" is a member of', html.H1(str(len(base_synsets))),
                    ' base synsets.']
    unique_paths = ['Unique paths from end nodes to root node:', html.H1(len(basepaths))]
    longest_path = ['Synsets along the longest path from end node to root node (including the end node and root '
                    'node):', html.H1(max_len)]

    return graph_title, figure, base_ss_list, base_defs, unique_paths, longest_path


# This will allow a layout of "impression" to be shown when the page is first loaded.
def initial_layout():
    init_title, init_fig, init_ss_list, init_defs, init_paths, init_longest_path = make_dash('impression')
    return html.Div(className='grid-container',
                    children=[html.Div(className='left-container',
                                       children=[html.Div(className='input-container',
                                                          children=[html.H3(className='input-label',
                                                                            children='Text Input'), html.Br(),
                                                                    dcc.Input(id='input-state',
                                                                              type='text',
                                                                              placeholder='Type a noun',
                                                                              debounce=True),
                                                                    html.Button(children='Go', id='start'), html.Br(),
                                                                    html.Button(id='random-button',
                                                                                children='Give Me a Random Word')]
                                                          ),
                                                 html.Div(className='info-container',
                                                          children=[html.H3(className='info-head',
                                                                            children='What is this?'),
                                                                    dcc.Markdown(what_string_1), html.Br(),
                                                                    dcc.Markdown(what_string_2), html.Br(),
                                                                    dcc.Markdown(what_string_3), html.Br(),
                                                                    html.H3(className='info-head',
                                                                            children='Why Do This?'),
                                                                    dcc.Markdown(why_string_1), html.Br(),
                                                                    html.H3(className='info-head',
                                                                            children='How I Made It'),
                                                                    dcc.Markdown(how_string_1)
                                                                    ]
                                                          )
                                                 ]
                                       ),
                              html.Div(className='center-container',
                                       children=[html.H3(id='graph-title',
                                                         className='graph-title',
                                                         children=init_title,
                                                         ),
                                                 html.Div(id='graph-box',
                                                          className='graph-box',
                                                          children=dcc.Graph(id='sem-dom-graph',
                                                                             figure=init_fig,
                                                                             config={'scrollZoom': True,
                                                                                     'responsive': True},
                                                                             style={'height': '100%',
                                                                                    'width': '100%'}
                                                                             )
                                                          )
                                                 ]
                                       ),
                              html.Div(className='right-container',
                                       children=[html.Div(id='base-synset-box',
                                                          children=init_ss_list,
                                                          className='right-box'),
                                                 html.Div(id='sense-def-box',
                                                          children=init_defs,
                                                          className='sense-box'),
                                                 html.Div(id='unique-paths',
                                                          children=init_paths,
                                                          className='right-box'),
                                                 html.Div(id='longest-path',
                                                          children=init_longest_path,
                                                          className='right-box'),
                                                 html.Div(className='mobile-info',
                                                          children=[html.H3(className='info-head',
                                                                            children='What is this?'),
                                                                    dcc.Markdown(what_string_1), html.Br(),
                                                                    dcc.Markdown(what_string_2), html.Br(),
                                                                    dcc.Markdown(what_string_3), html.Br(),
                                                                    html.H3(className='info-head',
                                                                            children='Why Do This?'),
                                                                    dcc.Markdown(why_string_1), html.Br(),
                                                                    html.H3(className='info-head',
                                                                            children='How I Made It'),
                                                                    dcc.Markdown(how_string_1)
                                                                    ]
                                                          )
                                                 ]
                                       )
                              ]
                    )


with open('data/all_nouns.json') as json_file:
    all_nouns = json.load(json_file)

# Construct a default sunburst graph. This prevents flickering when loading.
fig = go.Figure(go.Sunburst())

# Write out markdown text strings that will be used in the app
what_string_1 = '''This is an interactive semantic domains visualizer (click on it!). Given an English noun, this will 
display the 
hierarchy of semantic domains that word falls under according to [English WordNet](https://wordnet.princeton.edu/).'''
what_string_2 = '''Semantic domains are categories of meaning which are filled up by words 
which fit that meaning. This page, by default, displays the semantic domains for the word "impression." On the outer 
edges, 
you can see the end node domains that contain the various meanings of the word "impression."'''
what_string_3 = '''These domains are arranged in a hierarchy. An impression can be a depression which is a 
concave shape which is a solid which is a shape and so on until one works their way up to the root node "entity." All 
nouns are eventually entities.'''
why_string_1 = '''This is one step involved in a more complex semantic preferences project. I thought it was fun to 
look at in its own right so I shared it here. It also provided an opportunity to solve a deceptively tricky problem 
necessary for properly displaying the semantic domains of the semantic preferences project: WordNet will sometimes 
assign the same synset to multiple hypernyms. The input of this sunburst diagram, however, required that each synset 
have a unique ID (which is displayed upon hovering) with only a single parent. The problem is compounded as multiple 
synsets along a path from an end node to a top node may have multiple hypernyms. The number of nodes that require 
unique renaming grows exponentially with each multi-parent node along the same path.'''
how_string_1 = '''This project was written in Python utilizing Princeton's 
[English WordNet](https://wordnet.princeton.edu/) through the [Natural Language Toolkit](https://www.nltk.org/). The 
front end web app was made with [Dash](https://plotly.com/dash/) while the semantic domains visualizations were created 
using [Plotly](https://plotly.com/).'''

# Run the server
app = dash.Dash(__name__)
app.layout = initial_layout()


# This runs the randoms word input
@app.callback(
    Output('input-state', 'value'),
    [Input('random-button', 'n_clicks')]
)
def random_word(clicks):
    if clicks is None:
        raise PreventUpdate
    else:
        return random.choice(all_nouns)


# This is how the page is interactive and updated.
@app.callback(
    [Output('graph-title', 'children'),
     Output('sem-dom-graph', 'figure'),
     Output('base-synset-box', 'children'),
     Output('sense-def-box', 'children'),
     Output('unique-paths', 'children'),
     Output('longest-path', 'children')],
    [Input('input-state', 'value')]
)
def update_fig(word):
    return make_dash(word)


if __name__ == '__main__':
    app.run_server(debug=True)
