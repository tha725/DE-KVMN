import torch
from torch import nn
from einops import rearrange, repeat, reduce
import torch.nn.functional as F
from models.gat_layer import GATLayer
import argparse
import yaml
from models.bi_pooling import CompactBilinearPooling

torch.autograd.set_detect_anomaly(True)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def read_yaml(file_path):
    with open(file_path, "r") as f:
        config =  yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    return AttrDict(config)
    
class Memory(nn.Module):

    def __init__(self, c) -> None:
        super().__init__()

        assert c is not None, "config file needs to be given"
        assert c.slots > c.read_k and c.slots > c.compose_k, "total number of slots needs to be higher than selected k memory slots"

        self.c = c

        self.key_store = nn.Parameter(torch.empty((c.slots,c.k_dim)).to(self.c.device),requires_grad=False)
        self.value_store = nn.Parameter(torch.empty((c.slots,c.v_dim)).to(self.c.device),requires_grad=False)
        self.idx_store = nn.Parameter(torch.empty((c.slots, 1)).to(self.c.device),requires_grad=False)

        nn.init.xavier_normal_(self.key_store, gain=1.0)
        nn.init.xavier_normal_(self.value_store, gain=1.0)

        self.stacked_projector = nn.Linear(c.v_dim, 1)

        self.key_projector = nn.Linear(c.k_dim, c.k_dim)
        
        if c.comp_norm:
            self.composer_norm = nn.BatchNorm1d(1)
            self.gat_norm = nn.BatchNorm1d(c.v_dim)
            self.enc_norm = nn.BatchNorm1d(c.v_dim)

        self.gat = GATLayer(c.v_dim, c.v_dim, c.gah)
        
        self.A = torch.zeros((c.read_k+c.compose_k, c.read_k+c.compose_k)).to(self.c.device)
        
        for i in range(c.compose_k+c.read_k):
            self.A[i][i] = 1
        self.A[:c.compose_k,c.compose_k:] = 1
        # for i in range(c.compose_k, c.compose_k+c.read_k,1):
        #     self.A[i][i-c.compose_k] = 1
        
        print(self.A)
        self.gat_att = nn.Linear(c.v_dim, 1)
        self.enc_att = nn.Linear(c.v_dim, 1)

        self.cb_pool = CompactBilinearPooling(c.v_dim, c.v_dim, c.v_dim)

        self.temp_key_store = nn.Parameter(torch.empty((c.slots,c.k_dim)).to(self.c.device),requires_grad=False)
        self.temp_value_store = nn.Parameter(torch.empty((c.slots,c.v_dim)).to(self.c.device),requires_grad=False)
        self.temp_idx_store = nn.Parameter(torch.empty((c.slots,1)).to(self.c.device),requires_grad=False)
        
        nn.init.xavier_normal_(self.temp_key_store, gain=1.0)
        nn.init.xavier_normal_(self.temp_value_store, gain=1.0)

        

        self.repopulate = False
    
    def populate(self, key_store, value_store, j, idx_store):
        
        key_store = key_store.detach()
        value_store = value_store.detach()
        idx_store = idx_store.detach()

        self.key_store.data[j*self.c.population_batch:(j+1)*self.c.population_batch] = key_store
        self.value_store.data[j*self.c.population_batch:(j+1)*self.c.population_batch] = value_store
        self.idx_store.data[j*self.c.population_batch:(j+1)*self.c.population_batch] = idx_store

        self.temp_value_store.data[j*self.c.population_batch:(j+1)*self.c.population_batch] = value_store
        self.temp_key_store.data[j*self.c.population_batch:(j+1)*self.c.population_batch] = key_store
        self.temp_idx_store.data[j*self.c.population_batch:(j+1)*self.c.population_batch] = idx_store

        print("Population and Key Calculation Completed For Block "+str(j)+" ....")
    
    def pw_cosine_similarity(self, a):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = a / a.norm(dim=1)[:, None]
        res = 1-torch.mm(a_norm, b_norm.transpose(0,1))
        tot_dist = torch.sum(res,1)
        return tot_dist
    
    def read(self, key):
        
        key = rearrange(key, 'b d -> b 1 d')
        current_key_store = rearrange(self.key_store, 'm d -> 1 m d')
        current_value_store = rearrange(self.value_store, 'm d -> 1 m d')
        current_idx_store = rearrange(self.idx_store, 'm d -> 1 m d')

        # current_key_store = rearrange(self.key_store, 'm d -> 1 m d')
        # current_value_store = rearrange(self.value_store, 'm d -> 1 m d')

        key = repeat(key, 'b 1 d -> b m d', m=self.key_store.shape[0])
        
        current_key_store = repeat(current_key_store, '1 m d -> b m d', b = key.shape[0])
        current_value_store = repeat(current_value_store, '1 m d -> b m d', b = key.shape[0])
        current_idx_store = repeat(current_idx_store, '1 m d -> b m d', b = key.shape[0])

        sim_score = F.cosine_similarity(self.key_projector(key), self.key_projector(current_key_store), dim=-1)

        sim_values, indices = torch.topk(sim_score, k = self.c.read_k, dim=-1)
        
        # print('read', indices)

        batch_indices = torch.arange(sim_score.shape[0]).view(sim_score.shape[0], 1).expand_as(indices)
        selected_memories = current_value_store[batch_indices, indices]
        selected_idx = current_idx_store[batch_indices, indices]

        return selected_memories, indices, sim_values, selected_idx

    
    def compose(self, enc_out, selecetd_memory):
        
        # assert self.c.compose_k > self.c.read_k, 'K for Composition Should be Higher than K for Read'
        
        all_features = torch.cat([enc_out, selecetd_memory], dim=1)
        adj_matrix = repeat(self.A, 'k1 k2 -> b k1 k2', b = enc_out.shape[0])
        # print(all_features.shape, adj_matrix.shape, enc_out.shape, selecetd_memory.shape)
        gat_out, g_att = self.gat(all_features, adj_matrix)

        gat_att = F.softmax(torch.squeeze(self.gat_att(gat_out),dim=-1),dim=-1)
        gat_att = repeat(gat_att, 'b k -> b k d', d = self.c.v_dim)
        gat_out = reduce(torch.mul(gat_out, gat_att), 'b k d -> b d', 'sum')

        # gat_out = rearrange(gat_out, 'b 1 d -> b d')
        enc_out = rearrange(enc_out, 'b 1 d -> b d')

        # if self.c.comp_norm:
        #     gat_out = self.gat_norm(gat_out)
        #     enc_out = self.enc_norm(enc_out)

        gat_out = rearrange(gat_out, 'b d -> b d 1 1')
        enc_out = rearrange(enc_out, 'b d -> b d 1 1')

        composed_out = self.cb_pool(enc_out, gat_out)
        enc_out = rearrange(enc_out, 'b d 1 1-> b d')

        composed_out = torch.cat([composed_out, enc_out], dim=-1)

        return composed_out, [0], g_att
    
    #def write(self, stacked_features, input_features, indices_r, sim_values, input_idx):
    def write(self, stacked_features, input_features, indices_r, sim_values, input_idx, enc_out):
        
        slot_sims = [[] for _ in range(self.key_store.shape[0])]

        for i in range(stacked_features.shape[0]):
            best_feature = stacked_features[i][0]
            second_best_indices = indices_r[i][1]
            second_best_feature = self.value_store[second_best_indices]
            crnt_enc_out = enc_out[i]
            best_read_index = indices_r[i][0]
            best_sim = sim_values[i][0]
            input_feature = input_features[i]
            best_idx = input_idx[i][0]
            slot_sims[best_read_index].append([best_sim, best_feature, input_feature, best_idx, second_best_feature, crnt_enc_out])
        
        for idx, slots in enumerate(slot_sims):
            if len(slots) > 0:
                slots = sorted(slots, key=lambda x: x[0], reverse=True)
                if self.c.all_replace:
                    for slot in slots:
                        self.temp_key_store.data[idx] = slot[2]
                        self.temp_value_store.data[idx] = slot[1]
                        self.temp_idx_store.data[idx] = slot[3]
                else:
                    similarity_q_and_second_best = F.cosine_similarity(torch.squeeze(slots[0][5]), slots[0][4], dim=0)
                    similarity_best_and_second_best = F.cosine_similarity(slots[0][1], slots[0][4], dim=0)

                    # do theupdate if the update improves the diversity 
                    if similarity_best_and_second_best > similarity_q_and_second_best:
                        self.temp_key_store.data[idx] = slots[0][2]
                        self.temp_value_store.data[idx] = slots[0][1]
                        self.temp_idx_store.data[idx] = slots[0][3]
                    else:
                        continue
            
        if self.temp_key_store.shape[0] > self.c.max_slots:
            distance_metric = self.pw_cosine_similarity(self.temp_key_store)
            _, indices = torch.topk(distance_metric, k=self.c.revert_slots, dim=-1)
            self.temp_key_store = self.temp_key_store[indices]
            self.temp_value_store = self.temp_value_store[indices]
            self.temp_idx_store = self.temp_idx_store[indices]
            
            assert self.temp_key_store.shape[0] == self.c.revert_slots and self.temp_value_store.shape[0] == self.c.revert_slots, "Memory Flushing is not Successful"

        return self.temp_key_store.shape[0]
    

    def forward(self,out, key, idx):

        self.key_store.data = self.temp_key_store.data.detach()
        self.value_store.data = self.temp_value_store.data.detach()
        self.idx_store.data = self.temp_idx_store.data.detach()

        input_features = key
        stacked_features = rearrange(out, 'b d -> b 1 d')
        idx = idx.detach()
        
        # for i in range(1,self.c.encoders,1):
        #     stacked_features = torch.cat([stacked_features, rearrange(intermediate_activations[self.c.feature_layer+'_'+str(i)].detach(), 'b d -> b 1 d')], dim=1)
        
        # assert stacked_features.shape[1] == self.c.encoders, "Number of Encoders and Hidden Feature Dimensions Does Not Match."

        # if self.c.comp_norm:
        #     stacked_features = self.composer_norm(stacked_features)

        selecetd_memory, indices_r, sim_values, selected_idx = self.read(input_features)

        composed_out, indices_c, g_att = self.compose(stacked_features, selecetd_memory)

        if self.training:
            updated_slots = self.write(stacked_features, input_features, indices_r, sim_values, idx, stacked_features)
            return composed_out, indices_c, indices_r, g_att, updated_slots, selected_idx
        
        torch.cuda.empty_cache()
        
        return composed_out, indices_c, indices_r, g_att, selected_idx

