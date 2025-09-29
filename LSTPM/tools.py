import pickle
import numpy as np
from collections import defaultdict
from collections import deque

from utils import geodistance


def caculate_time_sim(data_neural):
    time_checkin_set = defaultdict(set)
    for uid in data_neural:
        uid_sessions = data_neural[uid]
        for sid in uid_sessions['sessions']:
            session_current = uid_sessions['sessions'][sid]
            for checkin in session_current:
                timid = checkin[1]
                locid = checkin[0]
                if timid not in time_checkin_set:
                    time_checkin_set[timid] = set()
                time_checkin_set[timid].add(locid)
    sim_matrix = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            set_i = time_checkin_set[i]
            set_j = time_checkin_set[j]
            if len(set_i | set_j) > 0:
                jaccard_ij = len(set_i & set_j) / len(set_i | set_j)
            else:
                jaccard_ij = 0.0

            #jaccard_ij = len(set_i & set_j)/len(set_i | set_j)
            sim_matrix[i][j] = jaccard_ij
    return sim_matrix


def caculate_poi_distance(poi_coors, dataset_name):
    print("distance matrix")
    sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
    for i in range(len(poi_coors)):
        for j in range(i , len(poi_coors)):
            poi_current = i
            poi_target = j
            #poi_current = i + 1            
            #poi_target = j + 1
            poi_current_coor = poi_coors[poi_current]
            poi_target_coor = poi_coors[poi_target]
            distance_between = geodistance(poi_current_coor[1], poi_current_coor[0], poi_target_coor[1], poi_target_coor[0])
            if distance_between<1:
                distance_between = 1
            sim_matrix[poi_current][poi_target] = distance_between
            sim_matrix[poi_target][poi_current] = distance_between
    pickle.dump(sim_matrix, open(f'data/{dataset_name}_distance.pkl', 'wb'))
    return sim_matrix


def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())
    train_queue = list()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def create_dilated_rnn_input(session_sequence_current, poi_distance_matrix):
    sequence_length = len(session_sequence_current)
    session_sequence_current.reverse()
    session_dilated_rnn_input_index = [0] * sequence_length
    for i in range(sequence_length - 1):
        current_poi = [session_sequence_current[i]]
        poi_before = session_sequence_current[i + 1 :]
        distance_row = poi_distance_matrix[current_poi]
        distance_row_explicit = distance_row[:, poi_before][0]
        index_closet = np.argmin(distance_row_explicit)
        session_dilated_rnn_input_index[sequence_length - i - 1] = sequence_length-2-index_closet-i
    session_sequence_current.reverse()
    return session_dilated_rnn_input_index


def entropy_spatial(sessions):
    locations = {}
    days = sorted(sessions.keys())
    for d in days:
        session = sessions[d]
        for s in session:
            if s[0] not in locations:
                locations[s[0]] = 1
            else:
                locations[s[0]] += 1
    frequency = np.array([locations[loc] for loc in locations])
    frequency = frequency / np.sum(frequency)
    entropy = - np.sum(frequency * np.log(frequency))
    return entropy