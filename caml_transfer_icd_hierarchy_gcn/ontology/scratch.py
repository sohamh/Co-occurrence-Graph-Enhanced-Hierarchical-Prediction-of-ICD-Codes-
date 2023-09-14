# from icdcodex import hierarchy
# icd_10_cm_hierarchy, icd_10_cm_codes = hierarchy.icd10cm("2020")
# import networkx as nx
# from networkx.algorithms.traversal.breadth_first_search import bfs_tree
# import matplotlib.pyplot as plt
# import pydot
# from networkx.drawing.nx_pydot import graphviz_layout
#
#
# G = nx.relabel_nodes(icd_10_cm_hierarchy, {"root": "ICD-10-CM"})
# G_chapters = bfs_tree(G, "ICD-10-CM", depth_limit=2)
# plt.figure(figsize=(8,8))
# # nx.draw(G_chapters, with_labels=True)
# # # plt.savefig('myfig.png')
# # plt.show()
#
# # T = nx.balanced_tree(2, 5)
# #
# pos = graphviz_layout(G_chapters, prog="dot")
# nx.draw(G_chapters, pos)
# # plt.show()
# plt.savefig('myfig.png')


# import pandas as pd
# from sklearn.model_selection import train_test_split
# df = pd.read_csv("/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mimicdata/mimic3/train_full_hadm_ids.csv").rename(columns={
#     "los": "length_of_stay",
#     "dob": "date_of_birth",
#     "dod": "date_of_death",
#     "admittime": "date_of_admission"
# })
# df = pd.read_csv("/mnt/data2/soha/codes/caml-mimic-master_Soha_01/mimicdata/mimic3/train_50_hadm_ids.csv")


# df["date_of_birth"] = pd.to_datetime(df["date_of_birth"]).dt.date
# df["date_of_death"] = pd.to_datetime(df["date_of_death"]).dt.date
# df["date_of_admission"] = pd.to_datetime(df["date_of_admission"]).dt.date
# df["age"] = df.apply(lambda e: (e['date_of_admission'] - e['date_of_birth']).days/365, axis=1)
# df = df[df.seq_num == 1]  # we limit ourselves to the primary diagnosis code for simplicity
# df.gender = LabelEncoder().fit_transform(df.gender)
# G, icd_codes = hierarchy.icd9()
# df = df[df.icd9_code.isin(G.nodes())]
# features = ["length_of_stay", "gender", "age"]
# X = df[features].values
# y = df[["icd9_code"]].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


###### main
import networkx as nx
G=hierarchy.icd9()
tree=G[0]
leaf_nodes=codes_of_interest
subgraph = tree.subgraph(leaf_nodes)
dist_matrix = [[0] * len(leaf_nodes) for _ in range(len(leaf_nodes))]
for i, node1 in enumerate(leaf_nodes):
    for j, node2 in enumerate(leaf_nodes):
        if i != j:
            dist = nx.shortest_path_length(subgraph, node1, node2)
            # dist_matrix[i][j] = dist
            print(dist)
# nx.shortest_path_length(tree, '7422', '4550')