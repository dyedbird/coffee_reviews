# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: gen
#     language: python
#     name: gen
# ---

import pandas as pd
import numpy as np
import re, sklearn
import random
import gensim
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import seaborn as sns
import json 
#from spellchecker import SpellChecker
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import defaultdict
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette
from fastcluster import linkage
from matplotlib.colors import rgb2hex, colorConverter
from sklearn.cluster import AgglomerativeClustering
from gensim import corpora, models, similarities, matutils
from gensim.models import CoherenceModel, TfidfModel, EnsembleLda
from gensim.test.utils import common_corpus, common_dictionary, datapath
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary
from scipy.spatial.distance import pdist, squareform
from matplotlib.pyplot import cm
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from gensim.models.ensemblelda import rank_masking
import matplotlib as mpl
#import pyLDAvis
#import pyLDAvis.gensim_models as gensimvis
import pprint as pp
import difflib
import pickle, glasbey
import networkx as nx
import  plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")

# #### EDA

coffee_df = pd.read_csv('simplified_coffee.csv', dtype = object)
# read in the csv file

coffee_df.head()

coffee_df.info()
#no NaN's

coffee_df['review'][random.choice(coffee_df.index)]
# rerun this cell to randomly read a row of reviews for a given coffee

coffee_df['review'][0]
#first row of reviews. 3 seperate reviews are concataneted into one in the original data set

# #### CLEANING 

corpus = pd.Series(data = coffee_df['review'], copy = True)
#convert dictionary to series

for i in corpus.index:
    corpus.iloc[i]=re.sub("[^A-Za-z0-9 -]+",' ',corpus.iloc[i]).lower()
    tokens = pd.Series(corpus.iloc[i].split()).unique()
    corpus.iloc[i]= ' '.join(tokens)
#Initial cleaning;remove all characters other than alphanumericals, space, , ', &, /. Also remove spaces with more than 1 character length

corpus[0]

# #### VECTORIZATION 

stopwords = list(ENGLISH_STOP_WORDS)
#convert frozenset to list

cv_tfidf = TfidfVectorizer(token_pattern = '(?u)[\w\-]{2,}', stop_words = stopwords)
# initialize vectorizer with custom token pattern 

doc_term= cv_tfidf.fit_transform(corpus).toarray()
# create document-term matrix

doc_term.shape
# n rows, m features

cos_sim= cosine_similarity(doc_term)
#Generate cosine similarity matrix

np.fill_diagonal(cos_sim, 1)

print(f"Sim Matrix Shape: {cos_sim.shape}")
print(f"Sim Matrix Symmetry Check: {np.allclose(cos_sim, cos_sim.T)}")
print(f"Sim Matrix All Ones Diagonal Check: {np.all(np.diag(cos_sim) == 1)}")

# #### DENDOGRAM PLOT

# +
#cmap = sns.color_palette(glasbey.create_palette(palette_size=50,lightness_bounds=(20, 40), chroma_bounds=(40, 50)), as_cmap = False)
#set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])
#  create color map and set color palette from it
# -

cmap = cm.tab10(np.linspace(0, 1, 10))
set_link_color_palette([mpl.colors.rgb2hex(rgb[:3]) for rgb in cmap])

sns.set_style('dark')
# set plot style

def get_cluster_classes(den, label='ivl'):
    cluster_idxs = defaultdict(list)
    for c, pi in zip(den['color_list'], den['icoord']):
        for leg in pi[1:3]:
            i = (leg - 5.0) / 10.0
            if abs(i - int(i)) < 1e-5:
                cluster_idxs[c].append(int(i))
    cluster_classes = Clusters()
    
    for c, l in cluster_idxs.items():

        # Extract unique labels for each cluster index
        unique_labels = pd.Series([den[label][i] for i in l]).unique()
        cluster_classes[c] = unique_labels.tolist()
    return cluster_classes
# custom function borrowed from https://www.nxn.se/valent/extract-cluster-elements-by-color-in-python


class Clusters(dict):
    def _repr_html_(self):
        html = '<table style="border: 0;">'
        for c in self:
            hx = rgb2hex(colorConverter.to_rgb(c))
            html += '<tr style="border: 0;">' \
            '<td style="background-color: {0}; ' \
                       'border: 0;">' \
            '<code style="background-color: {0};">'.format(hx)
            html += c + '</code></td>'
            html += '<td style="border: 0"><code>' 
            html += repr(self[c]) + '</code>'
            html += '</td></tr>'
        html += '</table>'
        return html
# custom function borrowed from https://www.nxn.se/valent/extract-cluster-elements-by-color-in-python


dist = lambda x:1-x
# function to convert similarity matrix to dissimilarity matrix

dist_mat = dist(cos_sim)
# convert similarity matrix to dissimilarity matrix

con_dissim = pdist(dist_mat)
# extract pairwise distances between observations in n=978 dimensional space in condensed 1-D array 

link = linkage(con_dissim, method='weighted')
del con_dissim
# initiate linkage function

c = np.linspace(0.2, 0.9, num=71)
#plt.figure(figsize=(48, 8))
for num in c:
    try: 
        den = dendrogram(link, labels=corpus.index, above_threshold_color = 'C0', color_threshold = num * max(link[:,2]), no_plot = True )
        if len(get_cluster_classes(den)['C0']) != 0:
            continue
    except: ValueError
    break
# custom function to determine threshold 

get_cluster_classes(den)
# call function to get clusters

len(get_cluster_classes(den))
# number of total cluster detected

G = nx.to_networkx_graph(cos_sim)
#create nx graph from sim matrix

def create_node_trace(G, size_threshold=1250, small_size=7, large_size=12):
    # collect node information from G to plot
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_sizes = []
    for i, node in enumerate(G.nodes(data=True)):
        # get node x,y position and store
        x, y = node[1]['pos']
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[1]['text'])
        node_color.append(node[1]['color'])
        if node[0] > size_threshold:
            node_sizes.append(large_size)
        else:
            node_sizes.append(small_size)

    # create node trace (i.e., scatter plot)
    # make it invisible by default
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_sizes,
            line_width=0.5,
        ),
        text=node_text,
        visible=False
    )
    return node_trace
#custom plot function as borrowed from https://towardsdatascience.com/visualising-similarity-clusters-with-interactive-graphs-20a4b2a18534


def create_edge_trace(G):
    # collect edges information from G to plot
    edge_weight = []
    edge_text = []
    edge_pos = []
    edge_color = []
    
    for edge in G.edges(data=True):
        
        # edge is line connecting two points
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_pos.append([[x0, x1, None], [y0, y1, None]])
        
        # edge line color when drawn
        edge_color.append("lightgrey")
    for u,v,d in G.edges(data=True):
        d['weight'] = cos_sim[u,v]*0.5
    
    # there is a trace for each edge
    edge_traces = []
    for i in range(len(edge_pos)):
        
        # edge line width
        line_width =  d['weight']

        # is scatter because it is line connecting two points
        trace = go.Scatter(
            x=edge_pos[i][0], y=edge_pos[i][1],
            line=dict(width=line_width, color=edge_color[i]),
            mode='lines',
            visible=False
        )
        edge_traces.append(trace)
    return edge_traces
# custom plot function as borrowed from https://towardsdatascience.com/visualising-similarity-clusters-with-interactive-graphs-20a4b2a18534


def filter_similarity_matrix_at_step(square_matrix, step_value):
    # copy matrix
    aux = square_matrix.copy()
    
    # set as NaN all values equal to or below threshold value
    aux[aux <= step_value] = np.nan
    
    # return filtered matrix
    return aux
# custom plot function as borrowed from https://towardsdatascience.com/visualising-similarity-clusters-with-interactive-graphs-20a4b2a18534


def get_interactive_slider_similarity_graph(square_matrix, slider_values, node_text=None, yaxisrange=None, xaxisrange=None):
    
    # Create figure with plotly
    fig = go.Figure()

    # key: slider value
    # value: list of traces to display for that slider value
    slider_dict = {}
    
    # total number of traces
    total_n_traces = 0
    
    # node positions on plot
    #node_pos = None

    # for each possible value in the slider, create and store traces (i.e., plots)
    for i, step_value in enumerate(slider_values):

        # update similarity matrix for the current step
        aux = filter_similarity_matrix_at_step(square_matrix, step_value)

        # create nx graph from sim matrix
        G = nx.to_networkx_graph(aux)
        
        # remove edges for 0 weight (NaN)
        G.remove_edges_from([(a, b) for a, b, attrs in G.edges(data=True) if np.isnan(attrs["weight"])])

        # assign node positions if None
        node_pos = nx.nx_agraph.graphviz_layout(G)

        # populate nodes with meta information
        for node in G.nodes(data=True):
            
            # node position
            node[1]['pos'] = node_pos[node[0]]

            # node text on hover if any is specified else is empty
            if node_text is not None:
                node[1]['text'] = node_text[node[0]]
            else:
                node[1]['text'] = ""
                
            # node color
            for color in list(get_cluster_classes(den)):
                if node[1]['text'] in get_cluster_classes(den)[color]:
                    node[1]['color'] = color
            #if node[1]['text'] in test_jcns:
                #node[1]['color'] = 'black'   
            
        # create edge taces (each edge is a trace, thus this is a list)
        edge_traces = create_edge_trace(G)
        
        # create node trace (a single trace for all nodes, thus it is not a list)
        node_trace = create_node_trace(G) 

        # store edge+node traces as single list for the current step value
        slider_dict[step_value] = edge_traces + [node_trace]
        
        # keep count of the total number of traces
        total_n_traces += len(slider_dict[step_value])

        # make sure that the first slider value is active for visualization
        if i == 0:
            for trace in slider_dict[step_value]:
                # make visible
                trace.visible = True
                
    # Create steps objects (one step per step_value)
    steps = []
    for step_value in slider_values:
        
        # count traces before adding new traces
        n_traces_before_adding_new = len(fig.data)
        
        # add new traces
        fig.add_traces(slider_dict[step_value])
        step = dict(
            # update figure when this step is active
            method="update",
            # make all traces invisible
            args=[{"visible": [False] * total_n_traces}],
            # label on the slider
            label=str(round(step_value, 3)),
        )

        # only toggle this step's traces visible, others remain invisible
        n_traces_for_step_value = len(slider_dict[step_value])
        for i in range(n_traces_before_adding_new, n_traces_before_adding_new + n_traces_for_step_value):
            step["args"][0]["visible"][i] = True
        
        # store step object in list of many steps
        steps.append(step)

    # create slider with list of step objects
    slider = [dict(
        active=0,
        steps=steps
    )]

    # add slider to figure and create layout
    fig.update_layout(
        sliders=slider,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(range=xaxisrange, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=yaxisrange, showgrid=False, zeroline=False, showticklabels=False),
        width=700, height=700,
    )
    return fig
# custom plot function as borrowed from https://towardsdatascience.com/visualising-similarity-clusters-with-interactive-graphs-20a4b2a18534


# +
#define slider steps (i.e., threshold values)
slider_steps = np.arange(0.25, 0.35, 0.05)
    
# get the slider figure
fig = get_interactive_slider_similarity_graph(
    cos_sim,
    slider_steps,
    node_text = corpus.index)
fig.show()
#custom plot function as borrowed from https://towardsdatascience.com/visualising-similarity-clusters-with-interactive-graphs-20a4b2a18534
# -

# #### TOPIC MODELING FOR CLUSTERS

clusters = dict(sorted(get_cluster_classes(den).items(), key=lambda x:len(x[1]), reverse = True))
#Create dict for clusters, sorted per longest values in keys

color_array = []
for i in range(len(clusters)):
    color_array.extend(np.repeat(list(clusters.keys())[i], len(pd.Series(clusters[list(clusters.keys())[i]]).unique())))
#Construct color array for each cluster corresponding to the number of jcns in each cluster

arrays = [color_array, pd.Series(list(chain.from_iterable(list(clusters.values())))).unique()]
#Map colors to each JCN   

index_new = pd.MultiIndex.from_arrays(arrays, names=('CLUSTER', 'COFFEE'))
# define multi_index

corpus_n = pd.Series(name="REVIEWS")
for i in list(chain.from_iterable(list(clusters.values()))):
    corpus_n[i] = coffee_df['review'][i] 
#Create re-ordered corpus for JCN's located in cluster

corpus_n[0]

df = pd.DataFrame(data={'REVIEWS':list(corpus_n)}, index = index_new)
#new data frame with multi_index

corpus_C = pd.Series()
cluster_values = index_new.get_level_values('CLUSTER')
for color in cluster_values.unique():
    corpus_C[color]=df.xs(color, level="CLUSTER")["REVIEWS"]
#Create corpus for each cluster

corpus_C['#9467bd'][82]

coffee_df['review'][82]

cv_tfidf_topic = TfidfVectorizer(analyzer='word', token_pattern = '(?u)[\w\-]{2,}', stop_words = stopwords)
#Define vectorizer
topic_dict = defaultdict(dict)
elda_model = defaultdict(dict)
corpus_gen = defaultdict(dict)
id2word = defaultdict(dict)
gen_dict = defaultdict(dict)
best_score = defaultdict(dict)
coherence= defaultdict(dict)
best_parameters = defaultdict(dict)
opt_model = defaultdict(dict)
corpus_g = defaultdict(list)
opt_topic = defaultdict(list)
dict1 = {}
prob = pd.Series()
terms = pd.Series()
topic = []
gensim_dictionary = Dictionary()
space = [ 
    Real(0.01, 0.99, name='epsilon'),
    Integer(1, 4, name='min_samples'),
    #Integer(1, 3, name='min_cores'),
    #Real(0.01, 0.99, name='masking_threshold')
        ]
@use_named_args(space)
def objective(epsilon, min_samples):
    print(f"Training model with epsilon={epsilon}, min_samples={min_samples}")
    elda_model[color] = EnsembleLda(corpus=corpus_gen[color],
                        id2word=id2word[color],
                        num_topics=10,
                        distance_workers=None,
                        num_models=8,
                        ensemble_workers=16,  
                        epsilon=epsilon,
                        min_samples=min_samples,
                        masking_method=rank_masking)
   
    
    topic = elda_model[color].print_topics(num_topics = -1, num_words=10)
    for i in range(len(topic)):
        terms[i] = [elem.split('*')[1].replace('"','').strip() for elem in topic[i][1].strip("'").split('+')]
    coh = CoherenceModel(topics = terms.tolist(), texts = corpus_g[color], dictionary = gen_dict[color], coherence='c_v')
    coherence[color] = coh.get_coherence()

    # We negate the coherence score because gp_minimize seeks to minimize the objective function
    return -coherence[color]

for color in cluster_values.unique():
    doc_term = cv_tfidf_topic.fit_transform(corpus_C[color])
    doc_term_trans = doc_term.transpose()
    corpus_gen[color] = matutils.Sparse2Corpus(doc_term_trans)
    id2word[color] = dict((v, k) for k, v in cv_tfidf_topic.vocabulary_.items()) 
    for jcn in corpus_C[color]:
        corpus_g[color].append(gensim.utils.simple_preprocess(jcn))
    corpus_g[color] = [[word for word in jcn if word not in stopwords] for jcn in corpus_g[color]]
    gen_dict[color] = corpora.Dictionary(corpus_g[color])
    #corpus_bow[color] = [gen_dict[color].doc2bow(text) for text in corpus_g[color]]
    # tfidf = TfidfModel(corpus_bow[color])
    # corpus_gen[color]= tfidf[corpus_bow[color]]
    try:
        res_gp= gp_minimize(objective, space, n_calls=10, random_state=0, n_jobs=-1)
        best_score[color] = (-res_gp.fun)
        best_parameters[color] = res_gp.x
        opt_model[color] = EnsembleLda(corpus=corpus_gen[color], id2word=id2word[color], num_topics=10, distance_workers = None, num_models = 8, 
                                        ensemble_workers = 16, epsilon = best_parameters[color][0], min_samples = best_parameters[color][1],
                                        masking_method=rank_masking )
        topics = opt_model[color].print_topics()
        opt_topic[color] = sorted(topics, key=lambda x: (x[1]), reverse=True)
    except: ValueError, NameError, KeyError

    try:
        for i in range(len(opt_topic[color])):
            prob[i] = [float(elem.split('*')[0].strip()) for elem in opt_topic[color][i][1].strip("'").split('+')]
            terms[i] = [elem.split('*')[1].replace('"','').strip() for elem in opt_topic[color][i][1].strip("'").split('+')]
            for key, val in zip(terms[i], prob[i]):
                dict1[key]=val
            topic_dict[color][i] = dict1
            dict1 = {}
    except: TypeError
    

cluster_idx = pd.Series(range(len(cluster_values.unique())), index = cluster_values.unique())
# create index for each cluster

coffee_titles = pd.Series(data = ['SWEET CHOCOLATE 1', 'SWEEET CHOCOLATE 2','ALMOND - COCOA', 'ESPRESSO', 'DARK CHOCOLATE',
                                'SWEET DARK CHOCOLATE', 'BITTERSWEET CHOCOLATE', 'FRUITY - SPICY'], index = cluster_values.unique())
# create index for each cluster

fig, axes = plt.subplots(2, 4, figsize = (30,10), sharex = True)
axes = axes.flatten()
for color in cluster_values.unique():
    try: 
        ax = axes[cluster_idx[color]]
        ax.barh(list(reversed(list(topic_dict[color][0].keys()))), list(reversed(list(topic_dict[color][0].values()))), color = color, height = 0.7)
        ax.set_title(f"{coffee_titles[color]}", fontdict = {"fontsize":24})
        ax.tick_params(axis = "both", which = "both", labelsize = 16)
    except: KeyError
    for i in "top right left".split():
        ax.spines[i].set_visible(False)
#plt.axes.Axes.tick_params
plt.subplots_adjust(top = 0.9, bottom = 0.2, wspace = 1, hspace = 0.3)
plt.savefig('Cluster_Topics_Orig')

# +
origin_dict = defaultdict(dict)
origins = list(coffee_df['origin'].value_counts().keys())
for color in cluster_values.unique():
    for origin in origins:
        origin_dict[color][origin]=len(coffee_df.iloc[clusters[color]][coffee_df['origin']==origin])


fig, ax = plt.subplots(figsize = (12,8))
values = np.zeros(len(origins))

for color in cluster_values.unique():
    p = ax.barh(y = list(reversed(origins)), width = list(reversed(list(origin_dict[color].values()))), color=color, left=values, height = 0.85)
    values = values + np.array(list(reversed(list(origin_dict[color].values()))))

ax.set_title("Flavor Profile Distribution over Origin")
ax.legend(loc='lower right', labels = coffee_titles)
plt.savefig('Flavor_Distro')
# -




