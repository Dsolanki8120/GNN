import sys
sys.path.append('/home/sdeepak/GNN/graph_nets/Granular_GNN')
import numpy as np
import pandas as pd
import os
import utils_np


CURRENT_DIR = os.getcwd()
DATA_PATH = os.path.join(CURRENT_DIR, "csv data files for 20 stages")
TIMESTEPS = 21  # t = 0 to 20
print("Notebook started")

def load_scalar_feat(name, t):
    # For files like 0_ball_rad.tab, which are space-separated and have 2 header lines
    print(f"load_scalar_feat called for {name}, t={t}")
    return pd.read_csv(
        f"{DATA_PATH}/{t}_{name}.tab",
        header=None,
        names=["ball_id", name],
        delim_whitespace=True,
        skiprows=2
    )

def load_all_node_data(t):
    
    print(f"load_all_node_data called for t={t}")  # Add this
    df = load_scalar_feat("ball_rad", t)
    df = df.merge(load_scalar_feat("ball_cn", t), on="ball_id")
    for name in ["ball_disp_x", "ball_disp_y", "ball_disp_z",
                 "ball_pos_x", "ball_pos_y", "ball_pos_z"]:
        df = df.merge(load_scalar_feat(name, t), on="ball_id")
    df.sort_values("ball_id", inplace=True)
    return df

def load_edges(t, id_to_idx):
    e1 = pd.read_csv(f"{DATA_PATH}/{t}_contact_end1.tab", delim_whitespace=True, header=None, skiprows=2)
    e2 = pd.read_csv(f"{DATA_PATH}/{t}_contact_end2.tab", delim_whitespace=True, header=None, skiprows=2)
    print(f"First few rows of {t}_contact_end1.tab:")
    print(e1.head())
    print(f"First few rows of {t}_contact_end2.tab:")
    print(e2.head())
    # Use the first column only
    e1_ids = e1.iloc[:, 0].astype(int)
    e2_ids = e2.iloc[:, 0].astype(int)
    # Only keep edges where both endpoints exist in id_to_idx
    valid_edges = []
    for a, b in zip(e1_ids, e2_ids):
        if a in id_to_idx and b in id_to_idx:
            valid_edges.append((id_to_idx[a], id_to_idx[b]))
    return set(valid_edges)

def compute_displacement_features(pos, disp, i, j, max_r):
    rel_pos = pos[i] - pos[j]
    rel_disp = disp[i] - disp[j]

    if np.linalg.norm(rel_pos) < 1e-6:
        return [0.0, 0.0]

    contact_dir = rel_pos / np.linalg.norm(rel_pos)
    disp_along = np.dot(rel_disp, contact_dir)
    disp_perp = np.linalg.norm(rel_disp - disp_along * contact_dir)

    return [disp_along / max_r, disp_perp / max_r]

def create_all_graphs():
    print("create_all_graphs called")

    graphs = []

    node_data_0 = load_all_node_data(0)
    # Ensure IDs are int for consistent mapping
    id_to_idx = {int(bid): i for i, bid in enumerate(node_data_0["ball_id"])}
    max_r = node_data_0["ball_rad"].max()

    print("First 10 node IDs:", list(id_to_idx.keys())[:10])

    ### Graph 0 ###
    print("Building Graph 0...")
    nodes_0 = np.stack([
        node_data_0["ball_rad"] / max_r,
        node_data_0["ball_cn"]
    ], axis=1).astype(np.float32)

    edges_0 = load_edges(0, id_to_idx)
    senders, receivers = zip(*edges_0) if edges_0 else ([], [])
    edge_features = [[0.0, 0.0, 0.0]] * len(senders)

    graph_0 = {
        "globals": np.zeros([1], dtype=np.float32),
        "nodes": nodes_0,
        "edges": np.array(edge_features, dtype=np.float32),
        "senders": np.array(senders, dtype=np.int32),
        "receivers": np.array(receivers, dtype=np.int32)
    }
    graphs.append(graph_0)

    ### Graphs 1 to 20 ###
    for t in range(1, TIMESTEPS):
        print(f"Building Graph {t}...")
        node_data_t = load_all_node_data(t)
        disp_t = node_data_t[["ball_disp_x", "ball_disp_y", "ball_disp_z"]].values
        pos_t = node_data_t[["ball_pos_x", "ball_pos_y", "ball_pos_z"]].values

        nodes = np.stack([
            node_data_t["ball_rad"] / max_r,
            node_data_t["ball_cn"]
        ], axis=1).astype(np.float32)

        edges_prev = load_edges(t-1, id_to_idx)
        edges_curr = load_edges(t, id_to_idx)
        all_edges = list(edges_prev.union(edges_curr))

        edge_features = []
        for (i, j) in all_edges:
            # Contact evolution status
            if (i, j) in edges_prev and (i, j) in edges_curr:
                status = 0.0
            elif (i, j) in edges_prev:
                status = -1.0
            else:
                status = 1.0
            # Relative displacement
            disp_feat = compute_displacement_features(pos_t, disp_t, i, j, max_r)
            edge_features.append([status] + disp_feat)

        senders, receivers = zip(*all_edges) if all_edges else ([], [])
        graph = {
            "globals": np.zeros([1], dtype=np.float32),
            "nodes": nodes,
            "edges": np.array(edge_features, dtype=np.float32),
            "senders": np.array(senders, dtype=np.int32),
            "receivers": np.array(receivers, dtype=np.int32)
        }
        graphs.append(graph)



    return [utils_np.data_dicts_to_graphs_tuple([g]) for g in graphs]
