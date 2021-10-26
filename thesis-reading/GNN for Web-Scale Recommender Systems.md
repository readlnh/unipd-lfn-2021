# GNN for Web-Scale Recommender Systems

 This paper is talking more about the application of the gnn algorithm for recommender systems.

## Introduction

- **Low dimensional embeddings** of items and individual users. These embeddings can be used for item-item recommendation or themed collections recommendation.
- **Graph Convolutional Networks(GCNS)**: learn how to iteratively aggregate feature information from local graph neighborhoods using neural networks.
- The **main challenge** is to scale both the training as well as inference of GCN-based node embeddings to graphs with billions of nodes and tens of billions of edges. For example,
  all existing GCN-based recommender systems require operating on the full graph Laplacian during training.

 ## Problem Setup

**Pinterest** is a content discovery application where users interact with pins, which are visual bookmarks to online content (e.g., recipes they want to cook, or clothes they want to purchase).

**Pinsage**: a random-walk-based GCN framework, operates on a massive graph with 3 billion nodes and 18 billion edges—a graph that is 10, 000× larger than typical applications of GCNs.



## PingSage Model Architectural

Localized convolutional modules to generate embeddings for nodes.

Generating an embedding for a node, which depends on node's **input features** and the **graph structure around the node**.

### Brief description of the algorithm

1.  Transform the representations of node u’s **neighbors** through a **dense neural network**.
2.  Apply a **aggregator/pooling fuction** (e.g., a element-wise mean or weighted sum, denoted as γ ) on the resulting set of vectors.
3.  Concatenate the aggregated neighborhood vector nu with u’s current representation and transform the concatenated vector through another **dense neural network layer**.



### Importance-based Neighborhoods

How to define node neighborhoods?

- Previous GCN: simply examine k-hop graph neighborhoods.
- PingSage: importance-based neighborhoods, **random walk**



### Stacking Convolutions

1. Compute the neighborhoods of each node 
2. Apply K convolutional iterations to generate the layer-K representations of the target nodes. 
3. The output of the final convolutional layer is then fed through a fullyconnected neural network to generate the final output embeddings.

## Model Training

- Loss function: max-margin-based loss function
- Speed the training:
  - Multi-GPU training with large minibatches
  - Producer-consumer minibatch construction
  - Node Embeddings via MapReduce
- Sampling negative items:
  - Random Negative
  - Hard Negative
  - Curriculum training scheme: In n epoch, use n - 1 hard negative items.

## Experiment