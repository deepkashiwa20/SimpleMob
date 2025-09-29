import torch.nn as nn
from torch.autograd import Variable
import pickle
import numpy as np
import torch
import gc
import argparse


from model import Model
from metrics import get_acc
from utils import generate_input_history, generate_input_long_history, minibatch, pad_batch_of_lists_masks
from tools import generate_queue, caculate_poi_distance, caculate_time_sim, create_dilated_rnn_input


def generate_detailed_batch_data(one_train_batch):
    session_id_batch = []
    user_id_batch = []
    sequence_batch = []
    sequences_lens_batch = []
    sequences_tim_batch = []
    sequences_dilated_input_batch = []
    for sample in one_train_batch:
        user_id_batch.append(sample[0])
        session_id_batch.append(sample[1])
        session_sequence_current = [s[0] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_tim_current = [s[1] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_dilated_input = create_dilated_rnn_input(session_sequence_current, poi_distance_matrix)
        sequence_batch.append(session_sequence_current)
        sequences_lens_batch.append(len(session_sequence_current))
        sequences_tim_batch.append(session_sequence_tim_current)
        sequences_dilated_input_batch.append(session_sequence_dilated_input)
    return user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequences_tim_batch, sequences_dilated_input_batch


def train_network(network, num_epoch=40 ,batch_size = 32, criterion = None):
    candidate = data_neural.keys()
    data_train, train_idx = generate_input_history(data_neural, 'train', candidate=candidate)
    for epoch in range(num_epoch):
        network.train(True)
        i = 0
        run_queue = generate_queue(train_idx, 'random', 'train')
        for one_train_batch in minibatch(run_queue, batch_size = batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(one_train_batch)
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network(user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, True, poi_distance_matrix, sequence_dilated_rnn_index_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()
            # train with backprop
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            opt.step()
            if (i + 1) % 20 == 0:
                print("epoch" + str(epoch) + ": loss: " + str(loss))
            i += 1
        results = evaluate(network, 1)
        print("Scores: ", results)


def evaluate(network, batch_size = 2):
    network.train(False)
    candidate = data_neural.keys()
    data_test, test_idx = generate_input_long_history(data_neural, 'test', candidate=candidate)
    users_acc = {}
    with torch.no_grad():
        run_queue = generate_queue(test_idx, 'normal', 'test')
        for one_test_batch in minibatch(run_queue, batch_size=batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(
                one_test_batch)
            user_id_batch_test = user_id_batch
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = Variable(torch.LongTensor(np.array(padded_sequence_batch))).to(device)
            mask_batch_ix = Variable(torch.FloatTensor(np.array(mask_batch_ix))).to(device)
            mask_batch_ix_non_local = Variable(torch.FloatTensor(np.array(mask_batch_ix_non_local))).to(device)
            user_id_batch = Variable(torch.LongTensor(np.array(user_id_batch))).to(device)
            logp_seq = network(user_id_batch, padded_sequence_batch, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, False, poi_distance_matrix, sequence_dilated_rnn_index_batch)
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            for ii, u_current in enumerate(user_id_batch_test):
                if u_current not in users_acc:
                    users_acc[u_current] = [0, 0, 0, 0, 0, 0, 0, 0] # total, acc@1,5,10, ndcg@1,5,10, mrr
                acc, ndcg, mrr = get_acc(actual_next_tokens[ii], predictions_logp[ii])
                ###acc
                users_acc[u_current][1] += acc[2][0]#@1
                users_acc[u_current][2] += acc[1][0]#@5
                users_acc[u_current][3] += acc[0][0]#@10
                ###ndcg
                users_acc[u_current][4] += ndcg[2][0]  # @1
                users_acc[u_current][5] += ndcg[1][0]  # @5
                users_acc[u_current][6] += ndcg[0][0]  # @10
                ###mrr
                users_acc[u_current][7] += mrr
                ###total
                users_acc[u_current][0] += (sequences_lens_batch[ii]-1)
        tmp_acc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # acc@1,5,10, ndcg@1,5,10, mrr
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
            tmp_acc[2] = users_acc[u][3] + tmp_acc[2]

            tmp_acc[3] = users_acc[u][4] + tmp_acc[3]
            tmp_acc[4] = users_acc[u][5] + tmp_acc[4]
            tmp_acc[5] = users_acc[u][6] + tmp_acc[5]
            tmp_acc[6] = users_acc[u][7] + tmp_acc[6]
            sum_test_samples = sum_test_samples + users_acc[u][0]
        avg_acc = (np.array(tmp_acc)/sum_test_samples).tolist()
        return avg_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='nyc', help="dataset name: nyc, tky, ca")
    return parser.parse_args()


if __name__ == '__main__':

    np.random.seed(1)
    torch.manual_seed(1)

    args = parse_args()

    data = pickle.load(open(f'dataset/{args.dataset_name}_cut_one_day.pkl', 'rb'), encoding='iso-8859-1')

    data_neural = data['data_neural']
    poi_coordinate = data['vid_lookup']
    n_users = data['uid_list']
    n_items = data['vid_list']    

    time_sim_matrix = caculate_time_sim(data_neural)
    poi_distance_matrix = caculate_poi_distance(poi_coordinate, args.dataset_name)
    #poi_distance_matrix = pickle.load(open(f'data/{args.dataset_name}_distance.pkl', 'rb'), encoding='iso-8859-1')

    torch.cuda.empty_cache()
    gc.collect()
    device = torch.device("cuda")

    session_id_sequences = None
    user_id_session = None

    network = Model(n_users=n_users, n_items=n_items, data_neural=data_neural, tim_sim_matrix=time_sim_matrix).to(
        device)
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=0.0001,
                               weight_decay=1 * 1e-6)
    
    criterion = nn.NLLLoss().cuda()

    train_network(network,criterion=criterion)
