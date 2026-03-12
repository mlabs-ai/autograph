# Autograph
This project presents an algorithm for identifying clusters (also known as
communities) within large graphs.

## Running on WikiData
1. Obtain a copy of the latest WikiData dump [here](https://dumps.wikimedia.org/wikidatawiki/entities/) (download the file `latest-all.json.bz2` or `latest-all.json.gz`).
2. Decompress the file and move it to `/data/wikidata`.
3. Set up a Python venv and run `pip install -r requirements.txt`.
4. Run `python python/find_wiki_subgraph.py <path to wikidata json> <relationship> <output dot file>`
to create a graph for the relationship you are interested in analyzing. For
instance, if you are interested in the [P171](https://www.wikidata.org/wiki/Property:P171)
relationship, you might run `python python/find_wiki_subgraph.py ./data/wikidata/latest-all.json P171 ./data/wikidata/P171.dot`.
5. Run `python python/cluster_wiki.py <input dot file> <output json file>` to run
the Autograph clustering algorithm on the graph contained in the given dot file and
place the output in the given JSON file. For instance, to run Autograph on the dot
file generated in the last step, you might run `python python/cluster_wiki.py ./data/wikidata/P171.dot ./data/clusters/P171.json` 

## Evaluating Against Other Algorithms
1. Set up a Python venv and run `pip install -r requirements.txt`.
2. Run `python python/evaluate_other_algorithms.py`