# Official Pytorch Implementation of Our Paper "Towards adaptive information propagation and aggregation in hypergraph model for node classification"



## Enviromment

```
python 3.7  
cuda 11.0  
torch 1.11.0
torch-gemotric 2.0.4
networkx 2.8.4
```




## Run command
Reproduce the results of the paper

```
#Cora-coauthorship
python main.py --hyper_type=coauthorship --hyper_name=cora
```

```
#cora-coauthorship
python main.py --hyper_type=cocitation --hyper_name=cora
```
```
#citeseer-cocitation
python main.py --hyper_type=cocitation --hyper_name=citeseer
```

```
#pubmed
python main.py --hyper_type=citeseer --hyper_name=pubmed
```