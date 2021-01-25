# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    ''' AttentionPooling used to weighted aggregate news/tokens vectors
    Arg:
        d_h: the last dimension of input
    '''

    def __init__(self, d_h, hidden_size=200):
        super(AdditiveAttention, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size)
        self.att_fc2 = nn.Linear(hidden_size, 1)


    def forward(self, x, attn_mask=None):
        """
        Args:
            x: batch_size, seq_length, vector_dim
            attn_mask: batch_size, seq_length
        Returns:
            (shape) batch_size, vector_dim
        """
        bz = x.shape[0]
        e = self.att_fc1(x)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * (attn_mask.unsqueeze(1).unsqueeze(1))
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, d_v):
        '''
        Args:
            d_model(int): the last dim of input
            n_heads(int): the number of heads
            d_k(int): the dim of heads for key
            d_v(int): the dim of heads for value
        '''
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K, V, mask=None):
        _, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads,
                               self.d_v).transpose(1, 2)

        context, attn = ScaledDotProductAttention(self.d_k)(
            q_s, k_s, v_s, mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.d_v)
        return context  # self.layer_norm(output + residual)



class TextEncoder(torch.nn.Module):
    def __init__(self,
                 bert_model,
                 bus_num,
                 seg_num,
                 word_embedding_dim,
                 num_attention_heads,
                 query_vector_dim,
                 dropout_rate,
                 bert_layer_hidden=None,
                 enable_gpu=True):
        super(TextEncoder, self).__init__()
        self.bert_model = bert_model
        self.bus_num = bus_num
        self.seg_num = seg_num
        self.dropout_rate = dropout_rate
        self.bert_layer_hidden = bert_layer_hidden
        self.multihead_attention = MultiHeadAttention(word_embedding_dim,
                                                      num_attention_heads, 20,
                                                      20)
        self.additive_attention = AdditiveAttention(num_attention_heads * 20,
                                                    query_vector_dim)


    def forward(self,segments,token_masks,seg_masks,key_position=None,fre_cnt=None):
        """
        Returns:
            (shape) (batch_size*segment_num, word_embedding_dim)
        """
        #For each segment, concatenate it and all [CLS](including its [CLS]), so need to mask one of its [CLS]
        batch_seg_num,seq_length = segments.shape
        batch_size,seg_num = seg_masks.shape
        if self.bus_num>0:
            token_masks = token_masks.view(batch_size,seg_num,seq_length)
            token_masks = torch.cat([seg_masks.unsqueeze(1).expand(-1,seg_num,-1),token_masks],dim=-1) #B S S+L
            for seg_idx in range(seg_num):
                token_masks[:,seg_idx,seg_idx] = 0
            token_masks = token_masks.view(batch_seg_num,seq_length+seg_num)

        last_hidden_states = self.bert_model(segments, token_masks, position_ids=key_position, fre_cnt=fre_cnt, seg_num=seg_num)[0] #BS S+L D
        # print(last_hidden_states)
        if self.bus_num>0:
            station_emb = last_hidden_states[:, seg_num]  # BS D
            station_emb = station_emb.view(-1, seg_num, last_hidden_states.size(-1)) #B S D
            station_emb = station_emb.unsqueeze(1).repeat(1,seg_num, 1, 1)  # B S S D
            station_emb = station_emb.view(batch_seg_num,seg_num,last_hidden_states.size(-1)) #BS S D
            last_hidden_states[:, :seg_num] = station_emb

        last_hidden_states = F.dropout(last_hidden_states,p=self.dropout_rate,training=self.training)
        propogation = self.multihead_attention(
            last_hidden_states,last_hidden_states,last_hidden_states, token_masks)
        text_vec = F.dropout(propogation, p=self.dropout_rate, training=self.training) #BS S+L D

        if self.bus_num > 0:
            text_vec = self.additive_attention(text_vec[:, self.bus_num:], token_masks[:, self.bus_num:]) #BS D
        else:
            text_vec = self.additive_attention(text_vec, token_masks) #BS D

        text_vectors = text_vec.view(batch_size,seg_num,-1) #B S D
        return text_vectors


class ElementEncoder(torch.nn.Module):
    def __init__(self, num_elements, embedding_dim, enable_gpu=True):
        super(ElementEncoder, self).__init__()
        self.enable_gpu = enable_gpu
        self.embedding = nn.Embedding(num_elements,
                                      embedding_dim,
                                      padding_idx=0)

    def forward(self, element):
        element_vector = self.embedding(element)
        return element_vector


class NewsEncoder(torch.nn.Module):
    def __init__(self, args, bert_model, category_dict_size,
                 subcategory_dict_size):
        super(NewsEncoder, self).__init__()
        self.args = args

        self.text_encoders = TextEncoder(bert_model,
                            args.bus_num,
                            args.seg_num,
                            args.word_embedding_dim,
                            args.num_attention_heads, args.news_query_vector_dim,
                            args.drop_rate, args.bert_layer_hidden, args.enable_gpu)

        element_encoders_candidates = ['category', 'subcategory']
        element_encoders = set(args.news_attributes) & set(element_encoders_candidates)

        name2num = {
            "category": category_dict_size + 1,
            "subcategory": subcategory_dict_size + 1
        }
        self.element_encoders = nn.ModuleDict({
            name: ElementEncoder(name2num[name],
                                 args.num_attention_heads * 20,
                                 args.enable_gpu)
            for name in (element_encoders)
        })

        if len(args.news_attributes) > 1:
            self.final_attention = AdditiveAttention(
                args.num_attention_heads * 20, args.news_query_vector_dim)

        self.reduce_dim_linear = nn.Linear(args.num_attention_heads * 20,
                                           args.news_dim)


    def forward(self, segments, token_masks, seg_masks, key_position=None, fre_cnt=None, elements=None):
        """
        Returns:
            (shape) batch_size, news_dim
        """

        all_vectors = self.text_encoders(segments,token_masks,seg_masks,key_position,fre_cnt) #B S D

        if 'body' in self.args.news_attributes and self.args.body_seg_num>1:
            body_vec = torch.stack(all_vectors[-self.args.body_seg_num:],dim=1)
            body_vec = torch.mean(body_vec,dim=1) #B 1 D
            all_vectors = torch.cat([all_vectors[:,:self.args.seg_num-self.args.body_seg_num],body_vec],dim=1)


        for idx, name in enumerate(self.element_encoders):
            ele_vec = self.element_encoders[name](elements[:,idx])
            all_vectors = all_vectors + (ele_vec,)

        if all_vectors.size(1) == 1:
            final_news_vector = all_vectors.squeeze(1)
        else:
            final_news_vector = self.final_attention(all_vectors,attn_mask=seg_masks) #B D

        final_news_vector = self.reduce_dim_linear(final_news_vector)
        return final_news_vector


class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.news_additive_attention = AdditiveAttention(
            args.news_dim, args.user_query_vector_dim)

        self.att_fc1 = nn.Linear(self.args.news_dim, args.user_query_vector_dim)
        self.att_fc2 = nn.Linear(args.user_query_vector_dim, 1)


    def _autoreg_user_modeling(self, vec, mask, pad_doc,
                      min_log_length=1):
        '''
        We propose autoregressive user modeling for more efficient utilization of user history,
        where all of the news clicks about a user can be predicted at one-shot of news encoding.
        For a click history of length L, this function generates L-1 user vectors for predicting the last L-1 clicks.
        '''

        if self.args.add_pad:
            padding_doc = pad_doc.expand(vec.shape[0], self.args.news_dim).unsqueeze(1)
            vec = torch.cat([padding_doc,vec],1)
            min_log_length += 1

        # batch_size:B, log_length:L+1, news_dim:D, predicte item num: G
        vec = vec[:,:-1,:]
        B,L,D = vec.shape      #B L D
        min_log_length = min(min_log_length,L)
        G = L+1 - min_log_length

        autor_mask = torch.ones((L, G), dtype=torch.float).triu(1 - min_log_length).transpose(0, 1).to(vec.device) # G L
        auto_mask = autor_mask.unsqueeze(0).repeat(B, 1, 1)  #B G L

        if not self.args.use_pad:
            auto_mask = torch.mul(autor_mask,mask[:,:-1].unsqueeze(2).repeat(1,1,L)) #B G L

        vec_repeat = vec.unsqueeze(1).repeat(1, G, 1, 1) #B G L L

        weights = self.att_fc1(vec_repeat)
        weights = nn.Tanh()(weights)
        weights = self.att_fc2(weights).squeeze(3)
        weights = weights.masked_fill(auto_mask == 0, -1e8)
        weights = torch.softmax(weights, dim=-1) #B G L
        user_vec = torch.matmul(weights, vec)  #B G D

        return user_vec


    def infer_user_vec(self,log_vec, log_mask):
        weights = self.att_fc1(log_vec)
        weights = nn.Tanh()(weights)
        weights = self.att_fc2(weights).squeeze(2)
        weights = weights.masked_fill(log_mask == 0, -1e8)
        weights = torch.softmax(weights, dim=-1)
        user_vec = torch.matmul(weights.unsqueeze(1),log_vec).squeeze(1)
        return user_vec


    def forward(self, log_vec, log_mask, pad_embedding):
        """Autoregressive Framework. Generate H-1 user vecs for each user (H is the length of clicked history)
        Args:
            vec: the vecs of clicked history
            mask: the mask of user history
            pad_doc: learnable padding vec
            min_log_length: use at least "min_log_length" clicked news to predict the next one
        Returns:
            (shape) batch_size, history_length - 1, news_dim
        """
        log_vec = self._autoreg_user_modeling(log_vec, log_mask, pad_embedding)

        return log_vec



class SpeedyFeed(torch.nn.Module):
    """SpeedyFeed
    """

    def __init__(self,
                 args,
                 bert_model,
                 category_dict_size=0,
                 subcategory_dict_size=0):
        '''
        Args:
            args: hyper-parameters
            bert_model: large transformers model such as bert, tnlr
            category_dict_size: size of category
            subcategory_dict_size: size of subcategory
        '''
        super(SpeedyFeed, self).__init__()
        self.args = args
        self.news_encoder = NewsEncoder(args,
                                        bert_model,
                                        category_dict_size,
                                        subcategory_dict_size)
        self.user_encoder = UserEncoder(args)
        self.pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)  #nn.Parameter default requires_grad=True
        #
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                segments,
                token_masks,
                seg_masks,
                elements,
                cache_vec,  # the index of embedding from cache
                batch_hist,  #user num  history lenght
                batch_mask,  #user num  history lenght
                batch_negs,  #user num  history lenght-1  npratio
                key_position=None,
                fre_cnt=None,
                ):
        """
        Args:
            segments: input_ids of segments
            token_masks: attention mask for tokens in segments
            seg_masks: mask the empty segments
            key_position: position_ids of keywords, required if use content refinement
            fre_cnt: frequence of keywords, required if use content refinement

            cache_vec: the news vecs get from shared cache
            batch_hist: user history
            batch_mask: the mask of user history
            batch_negs: negative news

        Returns:
            loss: training loss
            encode_vecs: encoded news vecs, which will be storaged into cache
        """
        encode_vecs = self.news_encoder(segments, token_masks, seg_masks,key_position, fre_cnt, elements)
        if cache_vec is not None:
            news_vecs = torch.cat([self.pad_doc,cache_vec,encode_vecs],0)
        else:
            news_vecs = torch.cat([self.pad_doc, encode_vecs], 0)

        reshape_negs = batch_negs.view(-1,)
        neg_vec = news_vecs.index_select(0,reshape_negs)
        neg_vec = neg_vec.view(batch_negs.size(0),batch_negs.size(1),batch_negs.size(2),-1) #B G N D

        reshape_hist = batch_hist.view(-1,)
        log_vec = news_vecs.index_select(0,reshape_hist)
        log_vec = log_vec.view(batch_hist.size(0),batch_hist.size(1),-1) # batch_size, log_length, news_dim

        user_vector = self.user_encoder(log_vec, batch_mask, self.pad_doc) #B G D

        candidate = torch.cat([log_vec[:,1:,:].unsqueeze(2),neg_vec],2) #B G 1+N D

        score = torch.matmul(candidate, user_vector.unsqueeze(-1)).squeeze(dim=-1) #B G 1+N
        logits = F.softmax(score, -1)
        loss = -torch.log(logits[:, :, 0] + 1e-9)
        loss = torch.mul(loss, batch_mask[:, 1:])
        loss = torch.sum(loss)/(torch.sum(batch_mask[:, 1:])+1e-9)
        return loss, encode_vecs



