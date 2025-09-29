import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch


class Model(nn.Module):
    def __init__(self, n_users, n_items, emb_size=500, hidden_units=500, dropout=0.8, user_dropout=0.5, data_neural = None, tim_sim_matrix = None):
        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_units = hidden_units
        if emb_size == None:
            emb_size = hidden_units
        self.emb_size = emb_size
        ## todo why embeding?
        self.item_emb = nn.Embedding(n_items, emb_size)
        self.emb_tim = nn.Embedding(48, 10)
        self.lstmcell = nn.LSTM(input_size=emb_size, hidden_size=hidden_units)
        self.lstmcell_history = nn.LSTM(input_size=emb_size, hidden_size=hidden_units)
        self.linear = nn.Linear(hidden_units*2 , n_items)
        self.dropout = nn.Dropout(0.0)
        self.user_dropout = nn.Dropout(user_dropout)
        self.data_neural = data_neural
        self.tim_sim_matrix = tim_sim_matrix
        self.dilated_rnn = nn.LSTMCell(input_size=emb_size, hidden_size=hidden_units)# could be the same as self.lstmcell
        self.linear1 = nn.Linear(hidden_units, hidden_units)
        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, user_vectors, item_vectors, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, is_train, poi_distance_matrix, sequence_dilated_rnn_index_batch):
        batch_size = item_vectors.size()[0]
        sequence_size = item_vectors.size()[1]
        items = self.item_emb(item_vectors)
        item_vectors = item_vectors.cpu()
        x = items
        x = x.transpose(0, 1)
        h1 = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        c1 = Variable(torch.zeros(1, batch_size, self.hidden_units)).cuda()
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        out = out.transpose(0, 1)#batch_size * sequence_length * embedding_dim
        x1 = items
        # ###########################################################
        user_batch = np.array(user_vectors.cpu())
        y_list = []
        out_hie = []
        for ii in range(batch_size):
            ##########################################
            current_session_input_dilated_rnn_index = sequence_dilated_rnn_index_batch[ii]
            hiddens_current = x1[ii]
            dilated_lstm_outs_h = []
            dilated_lstm_outs_c = []
            for index_dilated in range(len(current_session_input_dilated_rnn_index)):
                index_dilated_explicit = current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0)
                if index_dilated == 0:
                    h = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    c = Variable(torch.zeros(1, self.hidden_units)).cuda()
                    (h, c) = self.dilated_rnn(hidden_current, (h, c))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dilated_rnn(hidden_current, (dilated_lstm_outs_h[index_dilated_explicit], dilated_lstm_outs_c[index_dilated_explicit]))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
            dilated_lstm_outs_h.append(hiddens_current[len(current_session_input_dilated_rnn_index):])
            dilated_out = torch.cat(dilated_lstm_outs_h, dim = 0).unsqueeze(0)
            out_hie.append(dilated_out)
            user_id_current = user_batch[ii]
            current_session_timid = sequence_tim_batch[ii][:-1]
            current_session_poiid = item_vectors[ii][:len(current_session_timid)]
            session_id_current = session_id_batch[ii]
            current_session_embed = out[ii]
            current_session_mask = mask_batch_ix_non_local[ii].unsqueeze(1)
            sequence_length = int(sum(np.array(current_session_mask.cpu()))[0])
            current_session_represent_list = []
            if is_train:
                for iii in range(sequence_length-1):
                    current_session_represent = torch.sum(current_session_embed * current_session_mask, dim=0).unsqueeze(0)/sum(current_session_mask)
                    current_session_represent_list.append(current_session_represent)
            else:
                for iii in range(sequence_length-1):
                    current_session_represent_rep_item = current_session_embed[0:iii+1]
                    current_session_represent_rep_item = torch.sum(current_session_represent_rep_item, dim = 0).unsqueeze(0)/(iii + 1)
                    current_session_represent_list.append(current_session_represent_rep_item)

            current_session_represent = torch.cat(current_session_represent_list, dim = 0)
            list_for_sessions = []
            list_for_avg_distance = []
            h2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()###whole sequence
            c2 = Variable(torch.zeros(1, 1, self.hidden_units)).cuda()
            for jj in range(session_id_current):
                sequence = [s[0] for s in self.data_neural[user_id_current]['sessions'][jj]]
                sequence = Variable(torch.LongTensor(np.array(sequence))).cuda()
                sequence_emb = self.item_emb(sequence).unsqueeze(1)
                sequence = sequence.cpu()
                sequence_emb, (h2, c2) = self.lstmcell_history(sequence_emb, (h2, c2))
                sequence_tim_id = [s[1] for s in self.data_neural[user_id_current]['sessions'][jj]]
                jaccard_sim_row = Variable(torch.FloatTensor(self.tim_sim_matrix[current_session_timid]),requires_grad=False).cuda()
                jaccard_sim_expicit = jaccard_sim_row[:,sequence_tim_id]
                distance_row = poi_distance_matrix[current_session_poiid.cpu().numpy(), :]
                distance_row_expicit = Variable(torch.FloatTensor(distance_row[:,sequence]),requires_grad=False).cuda()
                distance_row_expicit_avg = torch.mean(distance_row_expicit, dim = 1)
                jaccard_sim_expicit_last = F.softmax(jaccard_sim_expicit)
                hidden_sequence_for_current1 = torch.mm(jaccard_sim_expicit_last, sequence_emb.squeeze(1))
                hidden_sequence_for_current =  hidden_sequence_for_current1
                list_for_sessions.append(hidden_sequence_for_current.unsqueeze(0))
                list_for_avg_distance.append(distance_row_expicit_avg.unsqueeze(0))

            if list_for_avg_distance == []:
                out_layer_2 = current_session_represent
            else:
                avg_distance = torch.cat(list_for_avg_distance, dim = 0).transpose(0,1)
                sessions_represent = torch.cat(list_for_sessions, dim=0).transpose(0,1) ##current_items * history_session_length * embedding_size
                current_session_represent = current_session_represent.unsqueeze(2) ### current_items * embedding_size * 1
                sims = F.softmax(sessions_represent.bmm(current_session_represent).squeeze(2), dim = 1).unsqueeze(1) ##==> current_items * 1 * history_session_length
                #out_y_current = sims.bmm(sessions_represent).squeeze(1)
                out_y_current =torch.selu(self.linear1(sims.bmm(sessions_represent).squeeze(1)))
                ##############layer_2
                #layer_2_current = (lambda*out_y_current + (1-lambda)*current_session_embed[:sequence_length-1]).unsqueeze(2) #lambda from [0.1-0.9] better performance
                # layer_2_current = (out_y_current + current_session_embed[:sequence_length-1]).unsqueeze(2)##==>current_items * embedding_size * 1
                layer_2_current = (0.5 *out_y_current + 0.5 * current_session_embed[:sequence_length - 1]).unsqueeze(2)
                layer_2_sims =  F.softmax(sessions_represent.bmm(layer_2_current).squeeze(2) * 1.0/avg_distance, dim = 1).unsqueeze(1)##==>>current_items * 1 * history_session_length
                out_layer_2 = layer_2_sims.bmm(sessions_represent).squeeze(1)
            
            out_y_current_padd = Variable(torch.FloatTensor(sequence_size - sequence_length + 1, self.emb_size).zero_(),requires_grad=False).cuda()
            out_layer_2_list = []
            out_layer_2_list.append(out_layer_2)
            out_layer_2_list.append(out_y_current_padd)
            out_layer_2 = torch.cat(out_layer_2_list,dim = 0).unsqueeze(0)
            y_list.append(out_layer_2)
        y = torch.selu(torch.cat(y_list,dim=0))
        out_hie = F.selu(torch.cat(out_hie, dim = 0))
        out = F.selu(out)
        out = (out + out_hie) * 0.5
        out_put_emb_v1 = torch.cat([y, out], dim=2)
        output_ln = self.linear(out_put_emb_v1)
        output = F.log_softmax(output_ln, dim=-1)
        return output
