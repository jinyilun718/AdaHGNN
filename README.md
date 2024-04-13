# Official Pytorch Implementation of Our Paper "Towards adaptive information propagation and aggregation in hypergraph model for node classification"



## Enviromment

```
python 3.8  
torch 2.0.0
networkx 3.0
```




## Run command
Reproduce the results of the paper

```
#Cora-coauthorship
python main.py --hyper_type=coauthorship --hyper_name=cora
```

```
#cora-cocitation
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
