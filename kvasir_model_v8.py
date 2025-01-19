import torch
from torch import nn
from einops import reduce
from models.memory_v8 import Memory
import torch.nn as nn
from transformers import ViTForImageClassification
import random
from typing import Optional, Tuple, Union

class KVASIR_NM(nn.Module):

    def __init__(self, c) -> None:
        super().__init__()

        assert c is not None, "config file needs to be given"

        self.c = c

        self.encoder_1 = ViTForImageClassification.from_pretrained(c.vit_weight)
        self.encoder_1.classifier = nn.Identity()

        if c.kvasir_encoder_freeze:
            for param in self.encoder_1.parameters():
                param.requires_grad = False
            print("Encoder Frozen during Training")

        self.memory = Memory(c)

        self.downstream = nn.ModuleList()


        current_dim = c.v_dim

        for hdim in c.iemocap_downstream:
            self.downstream.append(nn.Linear(current_dim, hdim))
            self.downstream.append(nn.ReLU())
            current_dim = hdim

        if c.as_binary:
            self.downstream.append(nn.Linear(current_dim,1,bias=False))
        else:
            self.downstream.append(nn.Linear(current_dim,c.num_classes,bias=False))

    
    def populate_memory(self,x,j):
        
        self.encoder_1.eval()

        hx = self.encoder_1(**x['image'], output_hidden_states=False).logits
  
        key_store = x['key'].detach()
        idx_store = x['idx'].detach()
        value_store = hx.detach()

        self.memory.populate(key_store, value_store, j, idx_store)

        self.encoder_1.train()

    def forward(self,audio, key, idx):

        out = self.encoder_1(**audio, output_hidden_states=False).logits

        if self.training:
            composed_out, indices_c, indices_r, g_att, updated_slots, selected_idx = self.memory(out, key, idx)
        else:
            composed_out, indices_c, indices_r, g_att, selected_idx = self.memory(out, key, idx)
        
        out = composed_out

        for layer in self.downstream:
            out = layer(out)

        if self.training:
            return out, indices_c, indices_r, g_att, updated_slots, selected_idx
        
        return out, indices_c, indices_r, g_att, selected_idx
