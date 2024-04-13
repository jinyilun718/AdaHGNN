from numbers import Number
from typing import Union
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import networkx as nx
from ppnp.data.simple.sparsegraph import SparseGraph
import pickle as pkl
# from torch_geometric.nn import MessagePassing
data_dir = Path(__file__).parent

# read graphs from files and then transform to sparseGraph format
def load_from_npz(file_name: str, name: str) -> SparseGraph:
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name
        Name of the file to load.

    Returns
    -------
    SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name, allow_pickle=True) as loader:
        loader = dict(loader)
        dataset = SparseGraph.from_flat_dict(loader, name) # data format: SparseGraph
    return dataset


def load_dataset(name: str,
                 directory: Union[Path, str] = data_dir
                 ) -> SparseGraph:
    """Load a dataset.

    Parameters
    ----------
    name
        Name of the dataset to load.
    directory
        Path to the directory where the datasets are stored.

    Returns
    -------
    SparseGraph
        The requested dataset in sparse format.

    """
    if isinstance(directory, str):
        directory = Path(directory)
    if not name.endswith('.npz'):
        name += '.npz'
    path_to_file = directory/ 'simple' / name
    if path_to_file.exists():
        return load_from_npz(path_to_file, name)
    else:
        raise ValueError("{} doesn't exist.".format(path_to_file))

def load_idx(dataset_str, data_type, num):
    root_path = 'ppnp/data/hyper'
    path = '/'.join([root_path, data_type, dataset_str])
    with open(path + '/' + 'splits/'+ num + '.pickle', 'rb') as f:
        idx_dict = pkl.load(f)
    return idx_dict

def load_hyper(dataset_str, data_type, add_noisy = None):
    root_path = 'ppnp/data/hyper'
    names = ['features', 'hypergraph', 'labels']
    path = '/'.join([root_path, data_type, dataset_str])
    objects = []
    for str in names:
        with open(path+'/'+str+'.pickle', 'rb') as f:
            objects.append(pkl.load(f))

    features, hypergraph, labels = tuple(objects)
    incidence_matrix = np.zeros([len(labels), len(hypergraph)], dtype=float)

    hyperedge_index = [[],[]]
    index = 0

    #计算节点对应的超边id-》2维数组
    # for idx, value in hypergraph.items():
    #     # for j in value:
    #     value = list(value)
    #     hyperedge_index[1] = hyperedge_index[1] + value
    #     hyperedge_index[0] = hyperedge_index[0] + [index]*len(value)
    #     index +=1

    hyperedge_index = hypergraph.copy()

    id = 0
    for idx, value in hypergraph.items():
        # hyperedge_index[id] = value
        for j in value:
            incidence_matrix[j, id]=1
        id +=1

    incidence_matrix = sp.csr_matrix(incidence_matrix)
    graph = SparseGraph(incidence_matrix)
    graph.attr_matrix = features
    graph.adj_matrix = incidence_matrix
    graph.hyperedge_index = hyperedge_index
    graph.labels = np.array(labels)
    return graph

def networkx_to_sparsegraph(
        nx_graph: Union['nx.Graph', 'nx.DiGraph'],
        label_name: str = None,
        sparse_node_attrs: bool = True,
        sparse_edge_attrs: bool = True
        ) -> 'SparseGraph':
    """Convert NetworkX graph to SparseGraph.

    Node attributes need to be numeric.
    Missing entries are interpreted as 0.
    Labels can be any object. If non-numeric they are interpreted as
    categorical and enumerated.

    This ignores all edge attributes except the edge weights.

    Parameters
    ----------
    nx_graph
        Graph to convert.

    Returns
    -------
    SparseGraph
        Converted graph.

    """
    import networkx as nx

    # Extract node names
    int_names = True
    for node in nx_graph.nodes:
        int_names &= isinstance(node, int)
    if int_names:
        node_names = None
    else:
        node_names = np.array(nx_graph.nodes)
        nx_graph = nx.convert_node_labels_to_integers(nx_graph)

    # Extract adjacency matrix
    adj_matrix = nx.adjacency_matrix(nx_graph)

    # Collect all node attribute names
    attrs = set()
    for _, node_data in nx_graph.nodes().data():
        attrs.update(node_data.keys())

    # Initialize labels and remove them from the attribute names
    if label_name is None:
        labels = None
    else:
        if label_name not in attrs:
            raise ValueError("No attribute with label name '{}' found.".format(label_name))
        attrs.remove(label_name)
        labels = [0 for _ in range(nx_graph.number_of_nodes())]

    if len(attrs) > 0:
        # Save attribute names if not integer
        all_integer = all((isinstance(attr, int) for attr in attrs))
        if all_integer:
            attr_names = None
            attr_mapping = None
        else:
            attr_names = np.array(list(attrs))
            attr_mapping = {k: i for i, k in enumerate(attr_names)}

        # Initialize attribute matrix
        if sparse_node_attrs:
            attr_matrix = sp.lil_matrix((nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32)
        else:
            attr_matrix = np.zeros((nx_graph.number_of_nodes(), len(attr_names)), dtype=np.float32)
    else:
        attr_matrix = None
        attr_names = None

    # Fill label and attribute matrices
    for inode, node_attrs in nx_graph.nodes.data():
        for key, val in node_attrs.items():
            if key == label_name:
                labels[inode] = val
            else:
                if not isinstance(val, Number):
                    if node_names is None:
                        raise ValueError("Node {} has attribute '{}' with value '{}', which is not a number."
                                         .format(inode, key, val))
                    else:
                        raise ValueError("Node '{}' has attribute '{}' with value '{}', which is not a number."
                                         .format(node_names[inode], key, val))
                if attr_mapping is None:
                    attr_matrix[inode, key] = val
                else:
                    attr_matrix[inode, attr_mapping[key]] = val
    if attr_matrix is not None and sparse_node_attrs:
        attr_matrix = attr_matrix.tocsr()

    # Convert labels to integers
    if labels is None:
        class_names = None
    else:
        try:
            labels = np.array(labels, dtype=np.float32)
            class_names = None
        except ValueError:
            class_names = np.unique(labels)
            class_mapping = {k: i for i, k in enumerate(class_names)}
            labels_int = np.empty(nx_graph.number_of_nodes(), dtype=np.float32)
            for inode, label in enumerate(labels):
                labels_int[inode] = class_mapping[label]
            labels = labels_int

    return SparseGraph(
            adj_matrix=adj_matrix, attr_matrix=attr_matrix, labels=labels,
            node_names=node_names, attr_names=attr_names, class_names=class_names,
            metadata=None)