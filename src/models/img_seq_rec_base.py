import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class SublayerConnection_single(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection_single, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, x_out):
        "Apply residual connection to any sublayer with the same size."
        return x_in + self.dropout(self.norm(x_out))


class DisentangleSublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(DisentangleSublayerConnection, self).__init__()
        self.norm_a = LayerNorm(hidden_size)
        self.norm_b = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in_a, x_out_a, x_in_b, x_out_b):
        "Apply residual connection to any sublayer with the same size."
        return x_in_a + self.dropout(self.norm_a(x_out_a)), x_in_b + self.dropout(self.norm_b(x_out_b))


class AuxiliarySublayerConnection(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(AuxiliarySublayerConnection, self).__init__()
        # self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_in , x_out):
        return x_in + self.dropout(x_out)
    

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class PositionwiseFeedForward_light(nn.Module):
    def __init__(self, hidden_size, length, dropout=0.1) -> None:
        super(PositionwiseFeedForward_light, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size)
        self.scale_w = nn.Parameter(torch.FloatTensor(length, 1))
        # self.scale_w = nn.Parameter(torch.FloatTensor(length, hidden_size))
        # self.scale_w = nn.Parameter(torch.FloatTensor(1, hidden_size)) 
        # self.bias_w = nn.Parameter(torch.FloatTensor(1, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.scale_w)  ## can not be should
    
    def forward(self, hidden):    
        hidden = self.scale_w.unsqueeze(0) * self.w_1(hidden)  
        # hidden = self.w_1(hidden)  ## no scale w, ablation
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        # activation = torch.relu(hidden)
        return self.dropout(activation)
        

class PositionwiseFeedForward_switch(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward_switch, self).__init__()
        # self.w_0 = nn.Linear(hidden_size, hidden_size)
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        # nn.init.xavier_normal_(self.w_0.weight)
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, input):
        # hidden = self.w_1(self.w_0(input))
        hidden = self.w_1(input)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class DisentanglePositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(DisentanglePositionwiseFeedForward, self).__init__()
        self.w_1_a = nn.Linear(hidden_size, hidden_size*4)
        self.w_2_a = nn.Linear(hidden_size*4, hidden_size)
        self.w_1_b = nn.Linear(hidden_size, hidden_size*4)
        self.w_2_b = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1_a.weight)
        nn.init.xavier_normal_(self.w_2_a.weight)
        nn.init.xavier_normal_(self.w_1_b.weight)
        nn.init.xavier_normal_(self.w_2_b.weight)

    def forward(self, hidden_a, hidden_b):
        hidden_a = self.w_1_a(hidden_a)
        activation_a = 0.5 * hidden_a * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden_a + 0.044715 * torch.pow(hidden_a, 3))))
        hidden_b = self.w_1_b(hidden_b)
        activation_b = 0.5 * hidden_b * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden_b + 0.044715 * torch.pow(hidden_b, 3))))
        return self.w_2_a(self.dropout(activation_a)), self.w_2_b(self.dropout(activation_b))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            # mask = mask.unsqueeze(1).unsqueeze(1).repeat(1,corr.shape[1],1,1)
            mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            corr = corr.masked_fill(mask == 0, -1e9)
            
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class CrossMultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.linearw_layer = nn.Linear(hidden_size, hidden_size)
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, k_b, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1))
        k_b = self.linearw_layer(k_b)
        k_b = k_b.view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
        corr_ab = torch.matmul(q, k_b.transpose(-2, -1))
        corr += corr_ab
        corr = corr / math.sqrt(q.size(-1))
        if mask is not None:
            # mask = mask.unsqueeze(1).unsqueeze(1).repeat(1,corr.shape[1],1,1)
            mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            corr = corr.masked_fill(mask == 0, -1e9)
            
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class DisentangleMultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers_a = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.linear_layers_b = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer_a = nn.Linear(hidden_size, hidden_size)
        self.w_layer_b = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer_a.weight)
        nn.init.xavier_normal_(self.w_layer_b.weight)

    def forward(self, q_a, k_a, v_a, q_b, k_b, v_b, mask=None):
        batch_size = q_a.shape[0]
        q_a, k_a, v_a = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers_a, (q_a, k_a, v_a))]
        q_b, k_b, v_b = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers_b, (q_b, k_b, v_b))]
        corr_a = torch.matmul(q_a, k_a.transpose(-2, -1)) / math.sqrt(2*q_a.size(-1))
        corr_ab = torch.matmul(q_a, k_b.transpose(-2, -1)) / math.sqrt(2*q_a.size(-1))
        corr_a = corr_a + corr_ab
        corr_b = torch.matmul(q_b, k_b.transpose(-2, -1)) / math.sqrt(2*q_b.size(-1))
        corr_ba = torch.matmul(q_b, k_a.transpose(-2, -1)) / math.sqrt(2*q_a.size(-1))
        corr_b = corr_b + corr_ba
        
        if mask is not None:
            # mask = mask.unsqueeze(1).repeat(1,corr_a.shape[1], 1, 1)
            mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1).repeat(1, corr_a.shape[1], 1, 1)
            corr_a = corr_a.masked_fill(mask == 0, -1e9)
            corr_b = corr_b.masked_fill(mask == 0, -1e9)
            
        prob_attn_a = F.softmax(corr_a, dim=-1)
        prob_attn_b = F.softmax(corr_b, dim=-1)
        if self.dropout is not None:
            prob_attn_a = self.dropout(prob_attn_a)
            prob_attn_b = self.dropout(prob_attn_b)
        hidden_a = torch.matmul(prob_attn_a, v_a)
        hidden_b = torch.matmul(prob_attn_b, v_b)
        hidden_a = self.w_layer_a(hidden_a.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        hidden_b = self.w_layer_b(hidden_b.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden_a, hidden_b


class AuxiliaryMultiHeadedAttention(nn.Module):
    def __init__(self,  heads, hidden_size, dropout):
        super(AuxiliaryMultiHeadedAttention, self).__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.linear_layer_b = nn.Linear(hidden_size, hidden_size)
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, k_b, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        k_b = self.linear_layer_b(k_b).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
        corr = torch.matmul(q, k.transpose(-2, -1))
        corr_ab = torch.matmul(q, k_b.transpose(-2, -1)) 
        # corr += torch.normal(mean=torch.full(corr.shape, 0.0), std=torch.full(corr.shape, 0.001)).to(corr_ab.device) * corr_ab
        corr += corr_ab
        corr = corr / math.sqrt(3*q.size(-1))
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            # mask = mask.unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            corr = corr.masked_fill(mask == 0, -1e9)
            
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class AuxiliaryMultiHeadedAttention_light(nn.Module):
    def __init__(self,  heads, hidden_size, length, dropout):
        super(AuxiliaryMultiHeadedAttention_light, self).__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.scale_w = nn.Parameter(torch.FloatTensor(heads, length)) 
        # self.scale_w = nn.Parameter(torch.FloatTensor(heads, length, length))  
        # self.scale_w = nn.Parameter(torch.FloatTensor(1, length, length))  
        # self.bias_w = nn.Parameter(torch.FloatTensor(heads, length))
        # self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.linear_layer_b = nn.Linear(hidden_size, hidden_size)  
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, k_b, mask=None):
        batch_size = q.shape[0]
        # q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        q, k, v = [x.view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for x in (q, k, v)]
        k_b = self.linear_layer_b(k_b)
        k_b = k_b.view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2)
        corr = torch.matmul(q, k.transpose(-2, -1))
        corr_ab = torch.matmul(q, k_b.transpose(-2, -1)) 
        # corr += torch.normal(mean=torch.full(corr.shape, 0.0), std=torch.full(corr.shape, 0.001)).to(corr_ab.device) * corr_ab
        corr += corr_ab
        
        corr = corr / math.sqrt(q.size(-1))
        
        corr = corr * self.scale_w.unsqueeze(-1).unsqueeze(0)   ## no scale_w ablation  
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            # mask = mask.unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            # corr = torch.clamp(corr, -1e9, 1e9)
            corr = corr.masked_fill(mask == 0, -1e9) 
            prob_attn = F.softmax(corr, dim=-1)
            # corr = corr.masked_fill(mask == 0, 0)
            # prob_attn = corr
            
            
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class SimAttention(nn.Module):
    def __init__(self, heads, hidden_size, length, dropout):
        super(SimAttention, self).__init__()
        assert hidden_size % heads == 0
        self.num_heads = heads
        self.size_head = hidden_size // heads
        # self.scale_w = nn.Parameter(torch.zeros([heads, length]).float()) 
        self.scale_w = nn.Parameter(torch.FloatTensor(heads, length))
        # self.scale_q = nn.Parameter(torch.FloatTensor(heads, length))
        # self.scale_k = nn.Parameter(torch.FloatTensor(heads, length))
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        # self.v_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_wights()
            
    def init_wights(self):
        nn.init.xavier_normal_(self.w_layer.weight)
        # nn.init.xavier_normal_(self.v_layer.weight)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        # v = self.v_layer(v)
        q, k, v = [x.view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for x in (q, k, v)]
        
        # q = q * self.scale_q.unsqueeze(-1).unsqueeze(0)
        # k = k * self.scale_q.unsqueeze(-1).unsqueeze(0)
        
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        corr = corr * self.scale_w.unsqueeze(-1).unsqueeze(0)
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, mask.shape[1], 1).unsqueeze(1).repeat(1, corr.shape[1], 1, 1)
            corr = corr.masked_fill(mask == 0, -1e9)
        
        prob_attn = F.softmax(corr, dim=-1)
        # prob_attn = corr
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class CrossAttention(nn.Module):
    def __init__(self, heads, hidden_size, length_patch, length_token, dropout):
        super(CrossAttention, self).__init__()
        assert hidden_size % heads == 0
        self.num_heads = heads
        self.size_head = hidden_size // heads
        self.scale_img = nn.Parameter(torch.FloatTensor(heads, length_patch))
        self.scale_title = nn.Parameter(torch.FloatTensor(heads, length_token))
        self.dropout = nn.Dropout(p=dropout)
        self.img_layer = nn.Linear(hidden_size, hidden_size)
        self.title_layer = nn.Linear(hidden_size, hidden_size)
        self.init_wights()
            
    def init_wights(self):
        nn.init.xavier_normal_(self.img_layer.weight)
        nn.init.xavier_normal_(self.title_layer.weight)
    
    def forward(self, img_seqitem_rep, title_seqitem_rep, mask_token):
        """_summary_
        Args:
            img_seqitem_rep: bsz x len x patch x dim
            title_seqitem_rep: bsz x len x title_len x dim
            mask_token: bsz x len x title_padding_mask
        Returns:
            img_seq_rep: bsz x len x dim
            title_seq_rep: bsz x len x dim
        """
        img_seqitem_rep_heads = img_seqitem_rep.view(img_seqitem_rep.shape[0], img_seqitem_rep.shape[1], -1, self.num_heads, self.size_head).transpose(2, 3)
        title_seqitem_rep_heads = title_seqitem_rep.view(title_seqitem_rep.shape[0], title_seqitem_rep.shape[1], -1, self.num_heads, self.size_head).transpose(2, 3)
        corr_img = torch.matmul(img_seqitem_rep_heads, title_seqitem_rep_heads.transpose(-2, -1)) / math.sqrt(img_seqitem_rep_heads.size(-1))
        corr_title = corr_img.transpose(-2, -1)
        
        if corr_img.shape[-2] == self.scale_img.shape[1]:
            corr_img = corr_img * self.scale_img.unsqueeze(-1).unsqueeze(0).unsqueeze(0) 
            corr_title = corr_title * self.scale_title.unsqueeze(-1).unsqueeze(0).unsqueeze(0) 
        else:
            corr_img = corr_img * self.scale_title.unsqueeze(-1).unsqueeze(0).unsqueeze(0) 
            corr_title = corr_title * self.scale_img.unsqueeze(-1).unsqueeze(0).unsqueeze(0) 
        if mask_token != None:
            mask_token = mask_token.unsqueeze(2).repeat(1, 1, corr_img.shape[2], 1).unsqueeze(-2).repeat(1, 1, 1, corr_img.shape[-2], 1)
            corr_img = corr_img.masked_fill(mask_token == 0, -1e9)
            corr_title = corr_title.masked_fill(mask_token.transpose(-1, -2) == 0, -1e9)
        
        prob_attn_img = F.softmax(corr_img, dim=-1)
        prob_attn_title = F.softmax(corr_title, dim=-1)
        if self.dropout is not None:
            prob_attn_img = self.dropout(prob_attn_img)
            prob_attn_title = self.dropout(prob_attn_title)
        if prob_attn_img.shape[-1] == img_seqitem_rep_heads.shape[-2]:
            hidden_img = torch.matmul(prob_attn_img, img_seqitem_rep_heads)
            hidden_title = torch.matmul(prob_attn_title, title_seqitem_rep_heads)
        else:
            hidden_img = torch.matmul(prob_attn_title, img_seqitem_rep_heads)
            hidden_title = torch.matmul(prob_attn_img, title_seqitem_rep_heads)
        hidden_img = self.img_layer(hidden_img.transpose(2, 3).contiguous().view(img_seqitem_rep_heads.shape[0], img_seqitem_rep_heads.shape[1], -1, self.num_heads * self.size_head))
        hidden_title = self.img_layer(hidden_title.transpose(2, 3).contiguous().view(title_seqitem_rep_heads.shape[0], title_seqitem_rep_heads.shape[1], -1, self.num_heads * self.size_head))
        return hidden_img, hidden_title
        

class CrossTransBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, patch_len, token_len, dropout):
        super(CrossTransBlock, self).__init__()
        self.attention = CrossAttention(heads=attn_heads, hidden_size=hidden_size, length_patch=patch_len, length_token=token_len, dropout=dropout)
        self.feed_forward_img = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.feed_forward_title = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer_img = SublayerConnection_single(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer_title = SublayerConnection_single(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer_img = SublayerConnection_single(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer_title = SublayerConnection_single(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden_img_seqbatch, hidden_title_seq_token, mask_seq_token):
        hidden_img_seqbatch_att, hidden_title_seq_token_att = self.attention(hidden_img_seqbatch, hidden_title_seq_token, mask_seq_token)
        # hidden_img_seqbatch = self.input_sublayer_img(hidden_img_seqbatch, hidden_img_seqbatch_att)
        # hidden_title_seq_token = self.input_sublayer_title(hidden_title_seq_token, hidden_title_seq_token_att)
        hidden_img_seqbatch = hidden_img_seqbatch_att
        hidden_title_seq_token = hidden_title_seq_token_att
        hidden_img_seqbatch_ffn = self.feed_forward_img(hidden_img_seqbatch) 
        hidden_title_seq_token_ffn = self.feed_forward_title(hidden_title_seq_token)
        hidden_img_seqbatch = self.output_sublayer_img(hidden_img_seqbatch, hidden_img_seqbatch_ffn)
        hidden_title_seq_token = self.output_sublayer_title(hidden_title_seq_token, hidden_title_seq_token_ffn) 
        return self.dropout(hidden_img_seqbatch), self.dropout(hidden_title_seq_token)
        

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, seq_len, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        # self.attention = SimAttention(heads=attn_heads, hidden_size=hidden_size, length=seq_len, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        # self.feed_forward = PositionwiseFeedForward_light(hidden_size=hidden_size, length=seq_len, dropout=dropout)
        # self.feed_forward = PositionwiseFeedForward_switch(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class LightTransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, seq_len, dropout):
        super(LightTransformerBlock, self).__init__()
        # self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.attention = SimAttention(heads=attn_heads, hidden_size=hidden_size, length=seq_len, dropout=dropout)
        # self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward_light(hidden_size=hidden_size, length=seq_len, dropout=dropout)
        # self.feed_forward = PositionwiseFeedForward_switch(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class CrossTransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, seq_len, dropout):
        super(CrossTransformerBlock, self).__init__()
        self.corss_attention = CrossMultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection_single(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        
        self.dropout = nn.Dropout(p=dropout)
        self.norm_img = LayerNorm(hidden_size)
        self.norm_title = LayerNorm(hidden_size)

    def forward(self, hidden_img, hidden_title, mask):
        hidden_img_norm = self.norm_img(hidden_img)
        hidden_title_norm = self.norm_title(hidden_title)
        hidden = self.corss_attention(hidden_img_norm, hidden_img_norm, hidden_img_norm, hidden_title_norm, mask)
        hidden = self.input_sublayer(hidden_img, hidden)
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Vanilla_TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(Vanilla_TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class DisentangelTransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(DisentangelTransformerBlock, self).__init__()
        self.input_sublayer = DisentangleSublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.disentangle_attention = DisentangleMultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = DisentanglePositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = DisentangleSublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, hidden_img, hidden_title, mask):
        hidden_img_att, hidden_title_att = self.disentangle_attention(hidden_img, hidden_img, hidden_img, hidden_title, hidden_title, hidden_title, mask)
        hidden_img, hidden_title = self.input_sublayer(hidden_img, hidden_img_att, hidden_title, hidden_title_att)
        hidden_img_ffn, hidden_title_ffn = self.feed_forward(hidden_img, hidden_title)
        hidden_img, hidden_title = self.output_sublayer(hidden_img, hidden_img_ffn, hidden_title, hidden_title_ffn)
        return self.dropout(hidden_img), self.dropout(hidden_title)
                

class AuxiliaryTransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, seq_len, dropout):
        super(AuxiliaryTransformerBlock, self).__init__()
        # self.auxiliary_attention = AuxiliaryMultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.auxiliary_attention = AuxiliaryMultiHeadedAttention_light(heads=attn_heads, hidden_size=hidden_size, length=seq_len, dropout=dropout)
        # self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward_light(hidden_size=hidden_size, length=seq_len, dropout=dropout)
        # self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = AuxiliarySublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_img = LayerNorm(hidden_size)
        self.norm_title = LayerNorm(hidden_size)

    def forward(self, hidden_img, hidden_title, mask):
        hidden_img_norm = self.norm_img(hidden_img)
        hidden_title_norm = self.norm_title(hidden_title)
        hidden = self.auxiliary_attention(hidden_img_norm, hidden_img_norm, hidden_img_norm, hidden_title_norm, mask)
        hidden = self.input_sublayer(hidden_img, hidden)
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)
   

class DTAUTransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, seq_len, dropout) -> None:
        super(DTAUTransformerBlock, self).__init__()
        self.img_auxTransBlock = AuxiliaryTransformerBlock(hidden_size=hidden_size, attn_heads=attn_heads, seq_len=seq_len, dropout=dropout)
        self.title_auxTransBlock = AuxiliaryTransformerBlock(hidden_size=hidden_size, attn_heads=attn_heads, seq_len=seq_len, dropout=dropout)
        
    def forward(self, hidden_img, hidden_title, mask):
        hidden_img = self.img_auxTransBlock(hidden_img, hidden_title, mask)
        hidden_title = self.title_auxTransBlock(hidden_title, hidden_img, mask)
        return hidden_img, hidden_title


class Globallocafuse(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear_one = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_two = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear_three = nn.Linear(hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.linear_one.weight)
        nn.init.xavier_normal_(self.linear_two.weight)
        nn.init.xavier_normal_(self.linear_three.weight)
        nn.init.xavier_normal_(self.linear_transform.weight)
    
    def forward(self, seq_rep, mask):
        ht = seq_rep[:, -1, :]  # batch_size x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(seq_rep)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_rep * mask.unsqueeze(-1), 1)
        a = self.linear_transform(torch.cat([a, ht], 1))
        return a


class Img_Seq_Rep_Rec(nn.Module):
    def __init__(self, args):
        super(Img_Seq_Rep_Rec, self).__init__()
        self.Trans_seq_rep = nn.ModuleList(
            [TransformerBlock(args.hidden_size, args.attn_head, args.seq_len, args.dropout) for _ in range(args.n_blocks)])
        self.fuse_gl = Globallocafuse(args.hidden_size)
        self.fuse = args.fuse_gl_flag
        
    def forward(self, hidden, mask):
        for transformer_temp in self.Trans_seq_rep:
            hidden = transformer_temp.forward(hidden, mask)
        if self.fuse: 
            hidden = self.fuse_gl(hidden, mask)
        else:
            hidden = hidden[:,-1,:]
        return hidden


class Img_Seq_Rep_Rec_Light(nn.Module):
    def __init__(self, args):
        super(Img_Seq_Rep_Rec_Light, self).__init__()
        self.Trans_seq_rep = nn.ModuleList(
            [LightTransformerBlock(args.hidden_size, args.attn_head, args.seq_len, args.dropout) for _ in range(args.n_blocks)])
        
    def forward(self, hidden, mask):
        for transformer_temp in self.Trans_seq_rep:
            hidden = transformer_temp.forward(hidden, mask)
            hidden = hidden[:,-1,:]
        return hidden


class Img_title_tower_Rec(nn.Module):
    def __init__(self, args):
        super(Img_title_tower_Rec, self).__init__()
        # self.AuxiTrans = nn.ModuleList([AuxiliaryTransformerBlock(args.hidden_size, args.attn_head, args.seq_len, args.dropout) for _ in range(args.n_blocks)])
        self.DTAuxiTrans = nn.ModuleList([DTAUTransformerBlock(args.hidden_size, args.attn_head, args.seq_len, args.dropout) for _ in range(args.n_blocks)])
        self.linear_img = nn.Linear(args.hidden_size, args.hidden_size)
        self.linear_title = nn.Linear(args.hidden_size, args.hidden_size)
        self.nums_block = args.n_blocks
        
    def forward(self, img_rep, title_rep, mask):
        for i, trans_temp in enumerate(self.DTAuxiTrans):
            img_rep, title_rep = trans_temp(img_rep, title_rep, mask)
        img_rep = img_rep[:, -1, :]
        title_rep = title_rep [:, -1, :]
        last_rep = self.linear_img(img_rep) + self.linear_title(title_rep)
        
        """
        for i, trans_temp in enumerate(self.AuxiTrans):
            if i % 2 == 0:
                img_rep = trans_temp(img_rep, title_rep, mask)
            else:
                title_rep = trans_temp(title_rep, img_rep, mask)
        if self.nums_block % 2 == 0:
            last_rep = img_rep
        else:
            last_rep = title_rep
        """
        return last_rep


class Img_title_fuse_Rec(nn.Module):
    def __init__(self, args):
        super(Img_title_fuse_Rec, self).__init__()
        self.AuxiTrans = nn.ModuleList([AuxiliaryTransformerBlock(args.hidden_size, args.attn_head, args.seq_len, args.dropout) for _ in range(args.n_blocks)])
        self.nums_block = args.n_blocks
        
    def forward(self, img_rep, title_rep, mask):
        
        for i, trans_temp in enumerate(self.AuxiTrans):
            img_rep = trans_temp.forward(img_rep, title_rep, mask)
            
            
        """
        for i, trans_temp in enumerate(self.AuxiTrans):
            if i % 2 == 0:
                img_rep = trans_temp(img_rep, title_rep, mask)
            else:
                title_rep = trans_temp(title_rep, img_rep, mask)
        if self.nums_block % 2 == 0:
            last_rep = img_rep
        else:
            last_rep = title_rep
        """
        
        return img_rep[:, -1, :]


class Img_title_fuse_Rec_VanillTrans(nn.Module):
    def __init__(self, args):
        super(Img_title_fuse_Rec_VanillTrans, self).__init__()
        self.AuxiTrans = nn.ModuleList([TransformerBlock(args.hidden_size, args.attn_head, args.seq_len, args.dropout) for _ in range(args.n_blocks)])
        self.nums_block = args.n_blocks
        
    def forward(self, img_rep, title_rep, mask):
        
        for i, trans_temp in enumerate(self.AuxiTrans):
            img_rep = trans_temp.forward(img_rep, mask)
            
            
        """
        for i, trans_temp in enumerate(self.AuxiTrans):
            if i % 2 == 0:
                img_rep = trans_temp(img_rep, title_rep, mask)
            else:
                title_rep = trans_temp(title_rep, img_rep, mask)
        if self.nums_block % 2 == 0:
            last_rep = img_rep
        else:
            last_rep = title_rep
        """
        
        return img_rep[:, -1, :]


class Img_title_fuse_Rec_VanillCrossTrans(nn.Module):
    def __init__(self, args):
        super(Img_title_fuse_Rec_VanillCrossTrans, self).__init__()
        self.Trans = nn.ModuleList([CrossTransformerBlock(args.hidden_size, args.attn_head, args.seq_len, args.dropout) for _ in range(args.n_blocks)])
        self.nums_block = args.n_blocks
        
    def forward(self, img_rep, title_rep, mask):
        
        for i, trans_temp in enumerate(self.Trans):
            img_rep = trans_temp.forward(img_rep, title_rep, mask)
            
            
        """
        for i, trans_temp in enumerate(self.AuxiTrans):
            if i % 2 == 0:
                img_rep = trans_temp(img_rep, title_rep, mask)
            else:
                title_rep = trans_temp(title_rep, img_rep, mask)
        if self.nums_block % 2 == 0:
            last_rep = img_rep
        else:
            last_rep = title_rep
        """
        
        return img_rep[:, -1, :]


class Img_title_ID_Vanilla_Rec(nn.Module):
    def __init__(self, args):
        super(Img_title_ID_Vanilla_Rec, self).__init__()
        self.Trans = nn.ModuleList([Vanilla_TransformerBlock(args.hidden_size, args.attn_head, args.dropout) for _ in range(args.n_blocks)])
        
    def forward(self, reps, mask):
        for i, trans_temp in enumerate(self.Trans):
            reps = trans_temp.forward(reps, mask)
        return reps[:, -1, :]


class Cross_img_title_rep(nn.Module):
    def __init__(self, args):
        super(Cross_img_title_rep, self).__init__()
        self.CrossTrans = nn.ModuleList([CrossTransBlock(args.hidden_size, args.attn_head, args.patch_len, args.token_len, args.dropout) for _ in range(args.cross_blocks)])
        self.nums_block = args.cross_blocks
        
    def forward(self, img_patch_seq_rep, title_token_seq_rep, mask):
        for i,  trans_temp in enumerate(self.CrossTrans):
            if i == 0:
                img_patch_seq_rep, title_token_seq_rep = trans_temp(img_patch_seq_rep, title_token_seq_rep, mask_seq_token=mask)
            else:
                img_patch_seq_rep, title_token_seq_rep = trans_temp(img_patch_seq_rep, title_token_seq_rep, mask_seq_token=None)
        return img_patch_seq_rep[:, :, 0, :], title_token_seq_rep[:, :, -1, :]
        

class Linear_pred(nn.Module):
    def __init__(self, args, nums_item):
        super(Linear_pred, self).__init__()
        self.pred_linear = nn.Linear(args.hidden_size, nums_item, bias=False)
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.pred_linear.weight)
    
    def forward(self, rep_seq):
        return self.pred_linear(rep_seq)
    

class NoLinear_pred(nn.Module):
    def __init__(self, args, nums_item):
        super(NoLinear_pred, self).__init__()
        self.pred_linear = nn.Linear(args.hidden_size, args.hidden_size, bias=False)
        self.pred_linear_2 = nn.Linear(args.hidden_size, nums_item, bias=False)
        self.actf = torch.nn.ReLU()
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.pred_linear.weight)
        nn.init.xavier_uniform_(self.pred_linear_2.weight)
    
    def forward(self, rep_seq):
        return self.pred_linear_2(self.actf(self.pred_linear(rep_seq)))
    

class Linear_pred_head_tail(nn.Module):
    def __init__(self, args, num_head, num_tail, head_ids, tail_ids):
        super(Linear_pred_head_tail, self).__init__()
        self.pred_head_flag = nn.Linear(args.hidden_size, 1, bias=False)
        self.pred_head = nn.Linear(args.hidden_size, num_head, bias=False) 
        self.pred_tail = nn.Linear(args.hidden_size, num_tail, bias=False) 
        self.nums = num_head + num_tail 
        self.head_ids = head_ids
        self.tail_ids = tail_ids
        self.init_weights()
    
    def init_weights(self):
        nn.init.xavier_uniform_(self.pred_head_flag.weight)
        nn.init.xavier_uniform_(self.pred_head.weight)
        nn.init.xavier_uniform_(self.pred_tail.weight)
    
    def forward(self, rep_seq):
        
        head_flag_prob = torch.sigmoid(self.pred_head_flag(rep_seq))
        head_prob = head_flag_prob * torch.softmax(self.pred_head(rep_seq), dim=-1)
        tail_prob = (1 - head_flag_prob) * torch.softmax(self.pred_tail(rep_seq), dim=-1)
        prob = torch.randn(rep_seq.shape[0], self.nums+1).to(head_prob.device)
        prob[:, self.head_ids] = head_prob
        prob[:, self.tail_ids] = tail_prob
        return prob, head_flag_prob


class Img_title_cross_emb(nn.Module):
    def __init__(self, seq_len, title_token_len, img_token_len):
        super(Img_title_cross_emb, self).__init__()
        self.corr_adapter = nn.Parameter(torch.ones(seq_len, title_token_len, img_token_len)) 
        
    def forward(self, img_seq_embs, title_seq_embs, title_seq_masks):
        """_summary_

        Args:
            img_seq_embs (_type_): Batch_size, seq_length, length_imgtoken, dimension
            title_seq_embs (_type_): Batch_size, seq_length, length_titletoken, dimension
            title_seq_masks (_type_): Batch_size, seq_length, length_titletoken,
        """
        
        emb_temp_img = img_seq_embs
        corrs = torch.matmul(title_seq_embs, emb_temp_img.transpose(-1, -2))
        corrs = corrs * title_seq_masks.unsqueeze(-1)
        corrs = self.corr_adapter * corrs
        corrs = corrs.masked_fill(corrs==0, -1e9)
        img_rep_cross = torch.matmul(F.softmax(corrs, dim=-1), emb_temp_img)
        title_rep_cross = torch.matmul(F.softmax(corrs.transpose(-1, -2), dim=-1), title_seq_embs)
        img_seq_rep = img_rep_cross[:, :, -1, :]
        title_seq_rep = title_rep_cross[:, :, -1, :]
        return img_seq_rep, title_seq_rep
    
