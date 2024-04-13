import os
import pickle
import pandas as pd
import pandas as pd

root_path = os.path.dirname(os.path.abspath('__file__'))
path = root_path + '/data/coauthorship/dblp'
'''获取文件夹下所有文件名'''
def get_file_name(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(file)
    return file_list
name = get_file_name(path)[0:3]
feature = pickle.load(open(root_path + '/data/coauthorship/dblp/' + name[0], 'rb'))
graph = pickle.load(open(root_path + '/data/coauthorship/dblp/' + name[1], 'rb'))
labels = pickle.load(open(root_path + '/data/coauthorship/dblp/' + name[2], 'rb'))
temp = pd.DataFrame(feature)