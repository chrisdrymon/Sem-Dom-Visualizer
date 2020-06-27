from nltk.corpus import wordnet as wn
from collections import Counter
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
import os
import random
import json
import pandas as pd
import string
import requests
from cltk.corpus.utils.formatter import cltk_normalize
from cltk.lemmatize.greek.backoff import BackoffGreekLemmatizer
from greek_accentuation.syllabify import *
from greek_accentuation.accentuation import *


def deaccent(dastring):
    """Returns an unaccented version of dastring."""
    aeinput = "άἀἁἂἃἄἅἆἇὰάᾀᾁᾂᾃᾄᾅᾆᾇᾰᾱᾲᾳᾴᾶᾷᾈᾉᾊᾋᾌᾍᾎᾏᾼἈἉΆἊἋἌἍἎἏᾸᾹᾺΆέἐἑἒἓἔἕὲέἘἙἚἛἜἝΈῈΈ"
    aeoutput = "αααααααααααᾳᾳᾳᾳᾳᾳᾳᾳααᾳᾳᾳαᾳᾼᾼᾼᾼᾼᾼᾼᾼᾼΑΑΑΑΑΑΑΑΑΑΑΑΑεεεεεεεεεΕΕΕΕΕΕΕΕΕ"
    hoinput = "ᾘᾙᾚᾛᾜᾝᾞᾟῌΉῊΉἨἩἪἫἬἭἮἯήἠἡἢἣἤἥἦἧὴήῆᾐᾑᾒᾓᾔᾕᾖᾗῂῃῄῇὀὁὂὃὄὅόὸόΌὈὉὊὋὌὍῸΌ"
    hooutput = "ῌῌῌῌῌῌῌῌῌΗΗΗΗΗΗΗΗΗΗΗηηηηηηηηηηηηῃῃῃῃῃῃῃῃῃῃῃῃοοοοοοοοοΟΟΟΟΟΟΟΟΟ"
    iuinput = "ΊῘῙῚΊἸἹἺἻἼἽἾἿΪϊίἰἱἲἳἴἵἶἷΐὶίῐῑῒΐῖῗΫΎὙὛὝὟϓϔῨῩῪΎὐὑὒὓὔὕὖὗΰϋύὺύῠῡῢΰῦῧ"
    iuoutput = "ΙΙΙΙΙΙΙΙΙΙΙΙΙΙιιιιιιιιιιιιιιιιιιιΥΥΥΥΥΥΥΥΥΥΥΥυυυυυυυυυυυυυυυυυυυ"
    wrinput = "ώὠὡὢὣὤὥὦὧὼῶώᾠᾡᾢᾣᾤᾥᾦᾧῲῳῴῷΏὨὩὪὫὬὭὮὯῺΏᾨᾩᾪᾫᾬᾭᾮᾯῼῤῥῬ"
    wroutput = "ωωωωωωωωωωωωῳῳῳῳῳῳῳῳῳῳῳῳΩΩΩΩΩΩΩΩΩΩΩῼῼῼῼῼῼῼῼῼρρΡ"
    # Strings to feed into translator tables to remove diacritics.

    aelphas = str.maketrans(aeinput, aeoutput, "⸀⸁⸂⸃·,.—")
    # This table also removes text critical markers and punctuation.

    hoes = str.maketrans(hoinput, hooutput, string.punctuation)
    # Removes other punctuation in case I forgot any.

    ius = str.maketrans(iuinput, iuoutput, '0123456789')
    # Also removes numbers (from verses).

    wros = str.maketrans(wrinput, wroutput, string.ascii_letters)
    # Also removes books names.

    return dastring.translate(aelphas).translate(hoes).translate(ius).translate(wros).lower()


# If a Greek word is given while in Greek mode and the orignal word is not in the list of Greek nouns, this will
# attempt to normalize the accentuation, try a series of accentuation patterns, and finally try to lemmatize the word.
# At each step, it will check if the resulting word is in the list of Greek nouns. Accents are easy to mess up in
# Ancient Greek, so this checks that. It should also allow for the entering of unaccented words, which is somewhat
# popular.
def greek_word_check(word):
    original_word = word
    if word.isascii():
        try:
            url = f'https://greekwordnet.chs.harvard.edu/translate/en/{word}/n/'
            trans_json = requests.get(url).json()
            word = random.choice(trans_json['results'])['lemma']
            if word in greek_nouns:
                return word
        except IndexError:
            return original_word
    if word in greek_nouns:
        return word
    word = cltk_normalize(word)
    if word in greek_nouns:
        return word
    word = deaccent(word)
    try:
        s = syllabify(word)
        for accentuation in possible_accentuations(s):
            if rebreath(add_accent(s, accentuation)) in greek_nouns:
                return add_accent(s, accentuation)
            if rebreath('h' + add_accent(s, accentuation)) in greek_nouns:
                return add_accent(s, accentuation)
    except TypeError:
        pass
    lemmatizer = BackoffGreekLemmatizer()
    word = lemmatizer.lemmatize([word])[0][1]
    if word in greek_nouns:
        return word
    try:
        s = syllabify(word)
        for accentuation in possible_accentuations(s):
            try_word = lemmatizer.lemmatize([rebreath(add_accent(s, accentuation))])[0][1]
            if try_word in greek_nouns:
                return try_word
            try_word = lemmatizer.lemmatize([rebreath('h' + add_accent(s, accentuation))])[0][1]
            if try_word in greek_nouns:
                return try_word
    except TypeError:
        pass
    else:
        return original_word


# A recursive function for climbing the synset hierarchy and gathering data along the way
def eng_synset_counting(ss_list, ss_counter, pairs):
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
        return eng_synset_counting(next_list, ss_counter, pairs)


# Placed this here so it can be run to produce the initial graph
def make_dash(word, lingua):
    base_synsets = []
    basepaths = []
    multiparents = []
    synset_counter = Counter()
    child_parent_pairs = {}
    glosses = []

    if word is None:
        raise PreventUpdate

    if word is "":
        raise PreventUpdate

    # Check for language
    if lingua == 'english':

        right_box_2 = [html.H3('Definitions', style={'text-align': 'center'}), html.Br()]

        # WordNet can handle some multiword phrases, but they need underscores instead of spaces
        word = word.replace(' ', '_')
        show_word = word.replace('_', ' ')

        # Create a list of all base synsets
        for i, synset in enumerate(wn.synsets(word, pos=wn.NOUN)):
            for path in synset.hypernym_paths():
                basepaths.append(path)
            base_synsets.append(synset)

        # Order the base synsets list, grab their definitions, and format them for display in the app
        for i, synset in enumerate(sorted(base_synsets)):
            ss_split = str(synset)[8:].split('.n.')
            right_box_2.append(html.B(ss_split[0].replace('_', ' ')))
            right_box_2.append(html.Span(ss_split[1][:2].lstrip('0'), className='ss'))
            right_box_2.append(' ' + synset.definition())
            right_box_2.append(html.Br())

        synset_counter, child_parent_pairs = eng_synset_counting(base_synsets, synset_counter, child_parent_pairs)
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
            graph_title = f'Error: WordNet does not recognize "{show_word.capitalize()}" as a noun.'
            figure = {'data': [{'type': 'sunburst'}]}
        else:
            graph_title = f'Semantic Domains of "{show_word.capitalize()}"'
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

        right_box_1 = ['The noun "', html.B(f'{show_word}'), '" is a member of', html.H1(str(len(base_synsets))),
                       ' base synsets.']
        right_box_3 = ['Unique paths from end nodes to root node:', html.H1(len(basepaths))]
        right_box_4 = ['Synsets along the longest path from end node to root node (including the end node and root '
                       'node):', html.H1(max_len)]
        box2c = 'sense-box'
        box3c = 'right-box'

    # If Greek is the language..
    else:
        # Find lemma id's from lemma, get synsets from lemma id's, get glosses from the synset dataframe
        # As it is, this prefers to only show glosses for synsets which are assigned a semfield. If none have a semfield
        # assigned, then it shows all glosses. This was done because some words have a huge number of glosses.
        ids = []
        labels = []
        codes = []
        parents = []
        right_box_3 = [html.H3('Definitions', style={'text-align': 'center'}), html.Br()]
        word = greek_word_check(word)

        # WordNet can handle some multiword phrases, but they need underscores instead of spaces
        show_word = word.replace('_', ' ')

        lillemma = lemma_df[lemma_df['lemma'] == word]
        for lemma_id in lillemma['id'].to_list():
            for synset_id in list(set(sense_df[sense_df['lemma'] == lemma_id]['synset'].to_list())):
                small_ss_df = synset_df[(synset_df['id'] == synset_id) & (synset_df['semfield'].notna())]
                for gloss in small_ss_df['gloss'].to_list():
                    glosses.append(gloss)
                for semfield in small_ss_df['semfield'].to_list():
                    # If there are multiple semantic fields, this separates those up. In this case, base_synsets are
                    # id numbers,not words.
                    if isinstance(semfield, str):
                        for item in semfield.split(','):
                            base_synsets.append(int(item))
                    else:
                        if semfield:
                            base_synsets.append(semfield)

        base_synsets = list(set(base_synsets))

        # In case no glosses are assigned a semantic field:
        if len(glosses) == 0:
            for lemma_id in lemma_df[lemma_df['lemma'] == word]['id'].to_list():
                for synset_id in list(set(sense_df[sense_df['lemma'] == lemma_id]['synset'].to_list())):
                    for gloss in synset_df[(synset_df['id'] == synset_id) &
                                           (synset_df['semfield'].isnull())]['gloss'].to_list():
                        glosses.append(gloss)

        if len(glosses) == 0:
            right_box_3.append(f'No definitions available for {show_word}')

        else:
            for i, definition in enumerate(glosses):
                right_box_3.append(str(i+1) + '. ' + definition)
                right_box_3.append(html.Br())

        # Convert synsets to ids, labels, parents, and codes (which will be mouse hover data)
        for ssid in base_synsets:
            next_id = ssid
            while pd.notna(semfield_df[semfield_df['id'] == next_id].iloc[0]['hypers']):
                lilsf_df = semfield_df[semfield_df['id'] == next_id]
                if next_id not in ids:
                    ids.append(next_id)
                    labels.append(lilsf_df.iloc[0]['english'])
                    codes.append(lilsf_df.iloc[0]['code'])
                    parents.append(int(lilsf_df.iloc[0]['hypers']))
                next_id = int(lilsf_df.iloc[0]['hypers'])
            if next_id not in ids:
                lilsf_df = semfield_df[semfield_df['id'] == next_id]
                ids.append(next_id)
                labels.append(lilsf_df.iloc[0]['english'])
                codes.append(lilsf_df.iloc[0]['code'])
                parents.append('')

        # This checks to see if WordNet recognizing the word as a noun. If not, it returns an error.
        if word not in greek_nouns:
            graph_title = f'Error: Ancient Greek WordNet does not recognize "{show_word.capitalize()}" as a noun.'
            figure = {'data': [{'type': 'sunburst'}]}
            right_box_1 = [f'No pronuncation data for {show_word}.']
        else:
            graph_title = f'Semantic Domains of "{show_word.capitalize()}"'
            figure = {'data': [{'type': 'sunburst',
                                'ids': ids,
                                'labels': labels,
                                'parents': parents,
                                'hovertext': codes,
                                'hoverinfo': 'text'}],
                      'layout': {'font': {'family': 'Quicksand',
                                          'size': 24},
                                 'margin': {'l': 10,
                                            'r': 10,
                                            'b': 10,
                                            't': 10},
                                 'colorway': ['#03045e', '#023e8a', '#0077b6', '#0096c7', '#00b4d8', '#48cae4',
                                              '#90e0ef', '#ade8f4', '#caf0f8', '#e63946']
                                 }
                      }
            if pd.notna(lillemma.iloc[0]['pronunciation']):
                right_box_1 = [f'{show_word} is pronounced', html.Br(), html.H1(lillemma.iloc[0]['pronunciation'])]
            else:
                right_box_1 = [f'No pronuncation data for {show_word}.']
        # Checks if word has been validated.
        if word not in validated_list:
            right_box_2 = ['The definitions of ', html.B(f'{show_word} '), html.Br(),
                           html.B('have not yet been manually validated.'), html.Br(),
                           'Thus the definitions and domains given will heavily rely on information from modern '
                           'English. ',
                           html.B('This is will likely produce some very inaccurate results. '),
                           'Currently, few Ancient Greek words have been validated.']
        else:
            right_box_2 = ['The definitions of ', html.B(f'{show_word}'), html.Br(),
                           html.B('have been validated.')]

        # Checks is word has semfield data.
        if len(base_synsets) == 0:
            right_box_4 = [f'There is no semantic field data on {show_word}.']
        else:
            right_box_4 = ['The noun "', html.B(f'{show_word}'), '" is a member of',
                           html.H1(str(len(base_synsets))),
                           ' outer semfields.']
        box2c = 'right-box'
        box3c = 'sense-box'

    return graph_title, figure, right_box_1, right_box_2, box2c, right_box_3, box3c, right_box_4


# This will allow a layout of "impression" to be shown when the page is first loaded.
def initial_layout():
    init_title, init_fig, init_ss_list, init_defs, b2c, init_paths, b3c, init_longest_path = \
        make_dash('impression', 'english')
    return html.Div(className='grid-container',
                    children=[html.Div(className='left-container',
                                       children=[html.Div(className='input-container',
                                                          children=[html.H3(className='input-label',
                                                                            children='Text Input'), html.Br(),
                                                                    dcc.RadioItems(id='language-sel',
                                                                                   className='radio-buttons',
                                                                                   options=[
                                                                                       {'label': 'Ancient Greek',
                                                                                        'value': 'greek'},
                                                                                       {'label': 'English',
                                                                                        'value': 'english'}
                                                                                   ],
                                                                                   value='english'),
                                                                    html.Br(),
                                                                    dcc.Input(id='input-state',
                                                                              type='text',
                                                                              placeholder='Type a noun',
                                                                              debounce=True),
                                                                    html.Button(children='Go', id='start'), html.Br(),
                                                                    html.Button(id='random-button',
                                                                                children='Give Me a Random Word')]
                                                          ),
                                                 html.Div(className='info-container',
                                                          id='info-container',
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
                                       children=[html.Div(id='right-box-1',
                                                          children=init_ss_list,
                                                          className='right-box'),
                                                 html.Div(id='right-box-2',
                                                          children=init_defs,
                                                          className=b2c),
                                                 html.Div(id='right-box-3',
                                                          children=init_paths,
                                                          className=b3c),
                                                 html.Div(id='right-box-4',
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


with open(os.path.join('data', 'english_nouns.json')) as json_file:
    english_nouns = json.load(json_file)

with open(os.path.join('data', 'greek_nouns_dict.json'), encoding='utf-8') as greek_dict:
    greek_nouns_dict = json.load(greek_dict)

with open(os.path.join('data', 'validated_list.json'), encoding='utf-8') as val_file:
    validated_list = json.load(val_file)

lemma_df = pd.read_csv(os.path.join('data', 'lemma.csv'))
greek_nouns = lemma_df['lemma'].to_numpy()
sense_df = pd.read_csv(os.path.join('data', 'literalsense.csv'))
synset_df = pd.read_csv(os.path.join('data', 'synset.csv'))
semfield_df = pd.read_csv(os.path.join('data', 'semfield.csv'))

info_panel_status = 'english'

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
    [Input('random-button', 'n_clicks')],
    [State('language-sel', 'value')]
)
def random_word(clicks, lang):
    if clicks is None:
        raise PreventUpdate
    else:
        if lang == 'english':
            return random.choice(english_nouns)
        else:
            return random.choice(greek_nouns)


# This is how the page is interactive and updated.
@app.callback(
    [Output('graph-title', 'children'),
     Output('sem-dom-graph', 'figure'),
     Output('right-box-1', 'children'),
     Output('right-box-2', 'children'),
     Output('right-box-2', 'className'),
     Output('right-box-3', 'children'),
     Output('right-box-3', 'className'),
     Output('right-box-4', 'children')],
    [Input('input-state', 'value')],
    [State('language-sel', 'value')]
)
def update_fig(word, lang):
    return make_dash(word, lang)


# This refreshes the left info panel
@app.callback(
    [Output('info-container', 'children')],
    [Input('language-sel', 'value')]
)
def change_language(language):
    if info_panel_status == language:
        raise PreventUpdate
    elif language == 'english':
        return [html.H3(className='info-head',
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
    else:
        return [html.H3(className='info-head',
                        children=html.H3("Don't know Greek? Type an English word and we'll try our best to translate "
                                         "it into an Ancient Greek noun."))]


if __name__ == '__main__':
    app.run_server(debug=True)
