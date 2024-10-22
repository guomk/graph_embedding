# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Node representation learning with Deep Graph Infomax and HinSAGE
# 
# %% [markdown]
# This demo demonstrates how to perform unsupervised training of a GCN, GAT, APPNP, or GraphSAGE model using the Deep Graph Infomax algorithm (https://arxiv.org/pdf/1809.10341.pdf) on the GM12878_sample dataset. 
# 
# As with all StellarGraph workflows: first we load the dataset, next we create our data generators, and then we train our model. We then take the embeddings created through unsupervised training and predict the node classes using logistic regression.

# %%
import networkx as nx
import numpy as np

from stellargraph.mapper import (
    CorruptedGenerator,
    DirectedGraphSAGENodeGenerator,
)
from stellargraph import StellarDiGraph
from stellargraph.layer import DeepGraphInfomax, DirectedGraphSAGE, HinSAGE, MeanPoolingAggregator

from stellargraph import datasets
from stellargraph.utils import plot_history
from stellargraph.random import set_seed

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from IPython.display import display, HTML

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras import Model
if tf.test.gpu_device_name():
  print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))

from pathlib import Path
from helper import Preprocess, Feature

set_seed(0)
tf.random.set_seed(0)

# %% [markdown]
# ## Load graph and node features

# %%
DATA_DIR = Path("../../data/")
FEATURE_DIR = Path("../../data/features/")
FEATURE_NAME = 'adjacentTFs'
MODEL_NAME='renamed_genes'

data_processor = Preprocess()


# %%
df = data_processor.raw2train(DATA_DIR)

df

# %% [markdown]
# ## Feature extration
# ### Adjacent TFs
# 

# %%
try:
    feature_df = pd.read_csv(FEATURE_DIR / f'{FEATURE_NAME}.csv', index_col=0)
    print('Read features from existing feature file')
except:
    print('Generating features...')
    common_tf = set(data_processor.raw2tf(DATA_DIR, option='intersection')['tf'])
    feature_df = Feature().adjacentTFs(df, common_tf)

# rename some target genes 
tf_list = list(data_processor.raw2tf(DATA_DIR, option='intersection')['tf'])
gm_list = [name + '_gm' for name in tf_list]
k_list = [name + '_k' for name in tf_list]

gene_list = ['SDF4', 'EEF1A1P9', 'B3GALT6', 'SSU72', 'AL645728.1', 'CDK11A', 'NEDD8', 'UBC', 'VASP']

def rename(name):
    if name in gene_list:
        return name + '_2'
    else:
        return name

gene_rename_gm = df[df['source'].isin(gm_list) & df['target'].isin(gene_list)]
gene_rename_k = df[df['source'].isin(k_list) & df['target'].isin(gene_list)]

gene_rename_gm['target'] = gene_rename_gm['target'].map(lambda name: name + '_gm')
gene_rename_k['target'] = gene_rename_k['target'].map(lambda name: name + '_k')

df = pd.concat([df, gene_rename_gm, gene_rename_k], axis=0).reset_index()[['cell_type', 'source', 'target', 'type', 'weight']]



# add features for renamed genes
feature_rename = feature_df.loc[gene_list]
feature_rename_gm = feature_rename.copy(deep=True)
feature_rename_k = feature_rename.copy(deep=True)

feature_rename_gm.index = feature_rename_gm.index.map(lambda name: name + '_gm')
feature_rename_k.index = feature_rename_k.index.map(lambda name: name + '_k')
feature_df = pd.concat([feature_df, feature_rename_gm, feature_rename_k], axis=0)

# %% [markdown]
# ## Read graph

# %%
G = StellarDiGraph(edges=df[['source', 'target']], nodes=feature_df)
print(G.info())

# %% [markdown]
# ## Data Generators
# 
# Now we create the data generators using `CorruptedGenerator`. `CorruptedGenerator` returns shuffled node features along with the regular node features and we train our model to discriminate between the two. 
# 
# Note that:
# 
# - We typically pass all nodes to `corrupted_generator.flow` because this is an unsupervised task
# - We don't pass `targets` to `corrupted_generator.flow` because these are binary labels (true nodes, false nodes) that are created by `CorruptedGenerator`

# %%
# HinSAGE model 
graphsage_generator = DirectedGraphSAGENodeGenerator(
    G, batch_size=50, in_samples=[30, 5], out_samples=[30, 5], seed=0
)

graphsage_model = DirectedGraphSAGE(
    layer_sizes=[128, 16], activations=["relu", "relu"], generator=graphsage_generator, aggregator=MeanPoolingAggregator
)


corrupted_generator = CorruptedGenerator(graphsage_generator)
gen = corrupted_generator.flow(G.nodes())

# %% [markdown]
# ## Model Creation and Training
# 
# We create and train our `DeepGraphInfomax` model. Note that the loss used here must always be `tf.nn.sigmoid_cross_entropy_with_logits`.

# %%
import tensorflow as tf


# %%
infomax = DeepGraphInfomax(graphsage_model, corrupted_generator)
x_in, x_out = infomax.in_out_tensors()

model = Model(inputs=x_in, outputs=x_out)
model.compile(loss=tf.nn.sigmoid_cross_entropy_with_logits, 
              optimizer=Adam(lr=1e-4),
              )


# %%
epochs = 300


# %%
es = EarlyStopping(monitor="loss", min_delta=0, patience=20)

history = model.fit(gen, epochs=epochs, verbose=1, callbacks=[])
# model.save(f'./cps/{MODEL_NAME}.h5')


# %%
plot_history(history)

# %% [markdown]
# ## Extracting Embeddings
# 
# Since we've already trained the weights of our base model - HinSAGE in this example - we can simply use `base_model.in_out_tensors` to obtain the trained node embedding model. 

# %%
x_emb_in, x_emb_out = graphsage_model.in_out_tensors()

# for full batch models, squeeze out the batch dim (which is 1)
# x_out = tf.squeeze(x_emb_out, axis=0)
# emb_model = Model(inputs=x_emb_in, outputs=x_out)

# not using full batch models
emb_model = Model(inputs=x_emb_in, outputs=x_emb_out)

# %% [markdown]
# ## Visualisation with TSNE
# 
# Here we visualize the node embeddings with TSNE. As you can see below, the Deep Graph Infomax model produces well separated embeddings using unsupervised training.

# %%
all_embeddings = emb_model.predict(graphsage_generator.flow(G.nodes()))

trans = TSNE(n_components=2, random_state=0)
emb_transformed = pd.DataFrame(trans.fit_transform(all_embeddings), index=G.nodes())


# %%
def geneType(name):
    if name[-2:] == '_k':
        return 1
    elif name[-3:] == '_gm':
        return 2
    else:
        return 0

emb_transformed['type'] = emb_transformed.index.map(geneType)


# %%
emb = pd.DataFrame(all_embeddings, index=G.nodes())
emb['type'] = emb.index.map(geneType)
emb[emb.index == 'ATF3_gm']
emb.to_csv(f'./emb/{MODEL_NAME}.csv', index=True, header=True)


# %%
alpha = 0.7

fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(
    emb_transformed[0],
    emb_transformed[1],
    c=emb_transformed["type"],
#     cmap="Paired",
    alpha=alpha,
    s=5
)
ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
plt.title(f"TSNE visualization of HinSAGE embeddings for {MODEL_NAME}")

plt.savefig(f'./img/full/{MODEL_NAME}.png', dpi=150)
# plt.show()



