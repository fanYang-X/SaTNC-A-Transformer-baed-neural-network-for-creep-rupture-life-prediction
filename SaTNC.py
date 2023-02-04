import math
import torch
import torch.nn as nn
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, encode_layer, N):
        super(Encoder, self).__init__()
        self.layer_list = clones(encode_layer, N)
        
    def forward(self, x, mask):
        for layer in self.layer_list:
            x = layer(x, mask)
        return x

class SublayeResConnection(nn.Module):
    def __init__(self, size, dropout=0):
        super(SublayeResConnection, self).__init__()
        self.norm = nn.LayerNorm(size) 
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer, mask):
        if mask is not None:
            return self.norm(x + self.Dropout(sub_layer(x, mask)))
        else:
            return self.norm(x + self.Dropout(sub_layer(x)))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, multi_attention, feed_forward):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.attention = multi_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayeResConnection(self.d_model, 0), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, self.attention, mask)
        x = self.sublayer[1](x, self.feed_forward, None)
        return x

def SelfAttention(q, k, v, mask):
    k_t = torch.transpose(k, dim0=-2, dim1=-1)
    q_k_t = torch.matmul(q, k_t)
    scale_factor = math.sqrt(q.size(-1))
    q_k_t = q_k_t.masked_fill(q_k_t == 0, -1e9)
    soft_value = nn.Softmax(dim=-1)(q_k_t/scale_factor)
    soft_value = soft_value*mask
    z = torch.matmul(soft_value, v)
    return z, soft_value

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.head = head
        self.atten_score = None
        self.d_k = self.d_model//self.head
        self.LinearList0 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.LinearList1 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.LinearList2 = nn.ModuleList([nn.Linear(self.d_model, self.d_k, bias=False) for _ in range(self.head)])
        self.Linear = nn.Linear(self.d_model, self.d_model)

    def forward(self, x, mask):
        n_batch = x.size(0)
        q = torch.zeros((n_batch, self.head, x.size(1), self.d_k))
        k = torch.zeros((n_batch, self.head, x.size(1), self.d_k))
        v = torch.zeros((n_batch, self.head, x.size(1), self.d_k))
        for i in range(self.head):
            q[:, i, :, :] = self.LinearList0[i](x)
            k[:, i, :, :] = self.LinearList1[i](x)
            v[:, i, :, :] = self.LinearList2[i](x)
        z, self.atten_score = SelfAttention(q, k, v, mask)
        z = z.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.Linear(z)

class ElemFeedForward(nn.Module):
    def __init__(self, d_forward):
        super(ElemFeedForward, self).__init__()
        self.Dropout = nn.Dropout(0.2)
        self.d_input = 18
        self.Linear1 = nn.Linear(self.d_input, d_forward)
        self.Linear2 = nn.Linear(d_forward, self.d_input)
        self.ReLu = nn.ReLU()
        
    def forward(self, x):
        r = self.Dropout(x)
        r = r.transpose(1, 2)
        r = self.Linear1(r)
        r = self.ReLu(r)
        r = self.Linear2(r)
        r = r.transpose(1, 2)
        return r

class LpBlock(nn.Module):
    def __init__(self, input_size, output_size):
        super(LpBlock, self).__init__()
        self.Linear = nn.Linear(input_size, output_size)
        self.BatchNormal = nn.BatchNorm1d(output_size)
        self.ReLu = nn.ReLU()

    def forward(self, x): 
        x = self.Linear(x)
        x = self.BatchNormal(x)
        x = self.ReLu(x)
        return x

def Mask(x):
    bool_tensor = x[:, :, 0]
    bool_tensor = bool_tensor.masked_fill(bool_tensor != 0, 1)
    mask_mat = bool_tensor.reshape(-1, x.size(1), 1).repeat(1, 1, 18)
    mask_mat = mask_mat.reshape(-1, 1, mask_mat.size(-2), mask_mat.size(-1)).repeat(1, 4, 1, 1)
    return mask_mat
    

class SaTNC(nn.Module):
    def __init__(self, d_model, head, attention_num, d_forward):
        super(SaTNC, self).__init__()
        self.d_model = d_model
        self.Linear0 = nn.Linear(9, self.d_model - 2, bias=False)
        self.head = head
        self.attention_num = attention_num
        self.d_forward = d_forward
        self.cls = nn.Embedding(19, d_model - 2, padding_idx=18)
        c = copy.deepcopy
        self.encoder_layer = EncoderLayer(self.d_model, c(MultiHeadAttention(self.d_model, self.head)), 
                                                            c(ElemFeedForward(self.d_forward)))
        self.TransFormerEncoder = Encoder(self.encoder_layer, self.attention_num)
        self.mlp_layer = nn.ModuleList([LpBlock(self.d_model + 6, 256), LpBlock(256, 512), LpBlock(512, 512)])
        self.Linear = nn.Linear(512, 1)

    def forward(self, x):
        # Feature Fusion
        # encode input
        r = self.Linear0(x[0][:, : -1, : -2])
        ## cls token type embedding
        r = torch.cat([r, torch.ones(r.size(0), 1, self.d_model - 2)], dim=1)
        ## add element embedding
        r += self.cls(x[1])
        # condition transformer
        r = torch.cat([r, x[0][:, :, -2: ]], dim=-1)
        # calac mask
        mask = Mask(x[0])
        r = self.TransFormerEncoder(r, mask)
        # get cls vector
        r = r[:, -1, :]
        # add domain features
        r = torch.cat([r, x[2]], dim=-1)
        for mlp in self.mlp_layer:
            r = mlp(r)
        r = self.Linear(r)
        return r