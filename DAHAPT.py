import random
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score
import pandas as pd
import argparse
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


import csv

threshold = 0


def universal_hash(x, a=123456789, b=987654321, p=2**31-1, m= 4):
    return (a * x + b) % p % m

def str_to_int(s):
    return int.from_bytes(s.encode(), 'little')

def update_vector(v, s,hash_size ,N=1):
    
    idx = universal_hash(str_to_int(s[:hash_size]),m=N)
    v[idx] += 1

def load_data(data_path):
    return pd.read_csv(data_path, sep='\t', header=None, names=['src_id', 'src_type', 'dst_id', 'dst_type', 'e_type', 'graph_id'])

def get_train_test_ids(train_id_file,test_id_file):
    benign_ranges = [(0, 299), (400, 599)]
    anomalous_ranges = [(300, 399), (600, 699)]

    benign_graph_ids = []
    anomalous_graph_ids = []

    for start, end in benign_ranges:
        benign_graph_ids.extend(range(start, end+1))

    for start, end in anomalous_ranges:
        anomalous_graph_ids.extend(range(start, end+1))

    #train_id_file = "D:\\AD Survey\\datasets\\streamspot\\all1_train.txt"
    #test_id_file ="D:\\AD Survey\\datasets\\streamspot\\all1.txt" 

    train_graph_ids = pd.read_csv(train_id_file, header=None).iloc[:, 0].tolist()
    all_graph_ids = pd.read_csv(test_id_file, header=None).iloc[:, 0].tolist()

    test_graph_ids = list(set(all_graph_ids) - set(train_graph_ids))

    return train_graph_ids, test_graph_ids, benign_graph_ids, anomalous_graph_ids

def train(train_graph_ids, df, vector_size,string_size):
    train_vectors = np.zeros((len(train_graph_ids), vector_size))
    all_train_vector = np.array([]).reshape(0,vector_size)

    edge_count = 0

    edges_dict = {gid: df[df['graph_id'] == gid].to_dict('records') for gid in train_graph_ids}
    edge_offset = {graph_id: 0 for graph_id in train_graph_ids}
    clfs = []
    prev_rows = {}
    group_copy = train_graph_ids.copy()
    graph_edge_strings = {gid: "" for gid in train_graph_ids}
    while group_copy:
        gid = random.choice(group_copy)
        row = edges_dict[gid][edge_offset[gid]]
        prev_row = prev_rows.get(gid)
        if prev_row is None:
            edge_string = ''.join('0'+row['src_type'] + row['dst_type'] + row['e_type'])
        else:
            temporal_encoding = 0
            if row['src_id'] != prev_row['src_id'] and row['dst_id'] != prev_row['dst_id']:
                temporal_encoding = 2
            elif row['src_id'] != prev_row['src_id'] or row['dst_id'] != prev_row['dst_id']:
                temporal_encoding = 1
            edge_string = str(temporal_encoding) + ''.join(row['src_type'] + row['dst_type'] + row['e_type'])

        i = train_graph_ids.index(gid)
        graph_edge_strings[gid] += edge_string
        if len(graph_edge_strings[gid]) >= string_size:
            update_vector(train_vectors[i], graph_edge_strings[gid][:string_size],hash_size=string_size,N=vector_size)
            graph_edge_strings[gid] = ""  # Reset the string for the graph
        #update_vector(train_vectors[i], edge_string)    
        edge_count += 1
        prev_rows[gid] = row
        edge_offset[gid] += 1
        if edge_offset[gid] >= len(edges_dict[gid]):
            group_copy.remove(gid)
            all_train_vector = np.vstack((all_train_vector, train_vectors))
        #if edge_count % 3000 == 0:
            #all_train_vector = np.vstack((all_train_vector, train_vectors))

    all_train_vector = np.vstack((all_train_vector, train_vectors))
    train_vectors, val_vectors = train_test_split(train_vectors, test_size=0.2, random_state=42)

    # Initialize the IsolationForest model
    clf = IsolationForest(contamination=0.2)

    # Fit the model to the training data
    clf.fit(train_vectors)
    n_samples = val_vectors.shape[0]

    # Create an array of ones
    y_true_val = np.ones(n_samples, dtype=int)
    # Get the anomaly scores on the validation data
    best_threshold = 0
    best_f1 = 0
    for threshold in np.linspace(-1.0, 1.0, 100):
        y_pred = (clf.decision_function(val_vectors) >= threshold).astype(int)
        current_f1 = f1_score(y_true_val, y_pred)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = threshold



    '''anomaly_scores = clf.decision_function(val_vectors)

    # Compute the threshold on the validation data
    threshold = np.percentile(anomaly_scores, 1)  # 1% quantile'''


    '''clf = IsolationForest(contamination=0.002)
    clf.fit(train_vectors)




    anomaly_scores = clf.decision_function(all_train_vector)
    threshold = np.percentile(anomaly_scores, 1)  # 1% quantile'''
    return clf,best_threshold

def test(clf, test_graph_ids, df, benign_graph_ids, vector_size,string_size,threshold):
    test_vectors = np.zeros((len(test_graph_ids), vector_size))
    edge_count = 0
    results = []
    nb_vectors = 0
    
    edges_dict = {gid: df[df['graph_id'] == gid].to_dict('records') for gid in test_graph_ids}
    edge_offset = {graph_id: 0 for graph_id in test_graph_ids}

    prev_rows = {}
    group_copy = test_graph_ids.copy()
    graph_edge_strings = {gid: "" for gid in test_graph_ids}
    buffer_size = 10000  # or whatever size you deem appropriate
    data_buffer = []
    while group_copy:
        gid = random.choice(group_copy)
        row = edges_dict[gid][edge_offset[gid]]
        prev_row = prev_rows.get(gid)
        if prev_row is None:
            edge_string = ''.join('0'+row['src_type'] + row['dst_type'] + row['e_type'])
        else:
            temporal_encoding = 0
            if row['src_id'] != prev_row['src_id'] and row['dst_id'] != prev_row['dst_id']:
                temporal_encoding = 2
            elif row['src_id'] != prev_row['src_id'] or row['dst_id'] != prev_row['dst_id']:
                temporal_encoding = 1
            edge_string = str(temporal_encoding) + ''.join(row['src_type'] + row['dst_type'] + row['e_type'])

        graph_edge_strings[gid] += edge_string
        i = test_graph_ids.index(gid)
        if len(graph_edge_strings[gid]) >= string_size:
            update_vector(test_vectors[i], graph_edge_strings[gid][:string_size],hash_size=string_size,N=vector_size)
            graph_edge_strings[gid] = ""  # Reset the string for the graph
        #update_vector(test_vectors[i], edge_string) 
        edge_count += 1
        prev_rows[gid] = row
        edge_offset[gid] += 1
        if edge_offset[gid] >= len(edges_dict[gid]):
            group_copy.remove(gid)

        if edge_count % 10000 == 0:
            anomalous_scores = -clf.decision_function(test_vectors)
            true_labels = [0 if graph_id in benign_graph_ids else 1 for graph_id in test_graph_ids]
            auc_score = roc_auc_score(true_labels, anomalous_scores)
            optimal_threshold = threshold
            binary_predictions = [1 if score >= optimal_threshold else 0 for score in anomalous_scores]
            balanced_acc = balanced_accuracy_score(true_labels, binary_predictions)
            avg_precision = average_precision_score(true_labels, anomalous_scores)
            # Using the optimal threshold to classify the scores
            predicted_labels = (anomalous_scores >= optimal_threshold).astype(int)
            # Compute the confusion matrix
            tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
            # Compute FPR and FNR
            fpr_optimal = fp / (fp + tn)
            fnr_optimal = fn / (fn + tp)
            
            print(auc_score, balanced_acc, avg_precision,fpr_optimal,fnr_optimal)
            #print(auc_score,fpr_optimal,fnr_optimal,balanced_acc,avg_precision,pr)
            results.append((auc_score, balanced_acc, avg_precision,fpr_optimal,fnr_optimal))

    return results

def main(data_path,vector_size,train_path,test_path, output,string_size):
    df = load_data(data_path)
    train_graph_ids, test_graph_ids, benign_graph_ids, anomalous_graph_ids = get_train_test_ids(train_path,test_path)
    clf,th = train(train_graph_ids, df, vector_size,string_size)
    results = test(clf, test_graph_ids, df, benign_graph_ids, vector_size,string_size,th)

    with open(output, "w", newline='') as file:
        writer = csv.writer(file,delimiter=';')
        writer.writerow(["AUC", "Balanced Accuracy", "Average Precision"])  # Writing the header
        for auc_score, balanced_acc, avg_precision,fpr_optimal,fnr_optimal in results:
            writer.writerow([auc_score, balanced_acc, avg_precision,fpr_optimal,fnr_optimal])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamspot Anomaly Detection")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--train_ids', type=str, required=True, help="Path to the train graph ids")
    parser.add_argument('--test_ids', type=str, required=True, help="Path to the test graph ids")
    parser.add_argument('--vector_size', type=int, default=2**7, help="Size of the hash vector")
    parser.add_argument('--string_size', type=int, default=4, help="Size of the hash vector")
    parser.add_argument('--output', type=str, required=True, help="Path to the results")

    
    args = parser.parse_args()

    main(args.data_path, args.vector_size,args.train_ids,args.test_ids,args.output,args.string_size)
