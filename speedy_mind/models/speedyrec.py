import torch
from torch import nn

from utility.utils import MODEL_CLASSES


class AttentionPooling(nn.Module):
    def __init__(self, d_h, hidden_size, drop_rate):
        super(AttentionPooling, self).__init__()
        self.att_fc1 = nn.Linear(d_h, hidden_size // 2)
        self.att_fc2 = nn.Linear(hidden_size // 2, 1)
        self.drop_layer = nn.Dropout(p=drop_rate)

    def forward(self, x, attn_mask=None):
        bz = x.shape[0]
        e = self.att_fc1(x)  # (bz, seq_len, 200)
        e = nn.Tanh()(e)
        alpha = self.att_fc2(e)  # (bz, seq_len, 1)

        alpha = torch.exp(alpha)
        if attn_mask is not None:
            alpha = alpha * attn_mask.unsqueeze(2)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-8)

        x = torch.bmm(x.permute(0, 2, 1), alpha)
        x = torch.reshape(x, (bz, -1))  # (bz, 400)
        return x


class TextEncoder(nn.Module):
    def __init__(self, args):
        super(TextEncoder, self).__init__()
        self.args = args

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.pretreained_model]
        self.config = config_class.from_pretrained(args.pretrained_model_path, output_hidden_states=True)
        if args.num_hidden_layers != -1: self.config.num_hidden_layers = args.num_hidden_layers
        if 'speedymind_ckpts' in args.pretrained_model_path:
            self.unicoder = model_class(config=self.config)
        else:
            self.unicoder = model_class.from_pretrained(
                args.pretrained_model_path,
                config=self.config)

        self.drop_layer = nn.Dropout(p=args.drop_rate)
        self.fc = nn.Linear(
            self.config.hidden_size,
            args.news_dim)

        if 'abstract' in self.args.news_attributes:
            self.text_att = AttentionPooling(
                args.news_dim, args.news_dim,
                drop_rate=args.drop_rate)
            self.sent_att = AttentionPooling(
                self.config.hidden_size, self.config.hidden_size,
                drop_rate=args.drop_rate)

    def sent_encode(self, inputs):
        batch_size, num_words = inputs.shape
        num_words = num_words // 2
        text_ids = torch.narrow(inputs, 1, 0, num_words)
        text_attmask = torch.narrow(inputs, 1, num_words, num_words)

        sent_vec = self.unicoder(text_ids, text_attmask)[0] #B L D
        if 'abstract' in self.args.news_attributes:
            sent_vec = self.sent_att(sent_vec, text_attmask)
        else:
            sent_vec = torch.mean(sent_vec, dim=1)

        news_vec = self.fc(sent_vec)
        return news_vec


    def forward(self, inputs):
        vecs = []
        title = torch.narrow(inputs, 1, 0, self.args.num_words_title*2)
        title_vec = self.sent_encode(title)
        vecs.append(title_vec)

        if 'abstract' in self.args.news_attributes:
            abs = torch.narrow(inputs, 1, self.args.num_words_title*2, self.args.num_words_abstract*2)
            abs_vec = self.sent_encode(abs)
            vecs.append(abs_vec)
        if len(vecs) == 1:
            return vecs[0]
        else:
            vecs = torch.cat(vecs, dim=-1).view(-1, len(vecs), self.args.news_dim) #B 2 D
            final_news_vector = self.text_att(vecs)
            return final_news_vector


class UserEncoder(nn.Module):
    def __init__(self, args, text_encoder=None):
        super(UserEncoder, self).__init__()
        self.args = args
        self.news_pad_doc = nn.Parameter(torch.empty(1, args.news_dim).uniform_(-1, 1)).type(torch.FloatTensor)
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.news_attn_pool = AttentionPooling(
            args.news_dim, args.news_dim,
            drop_rate=args.drop_rate)

    def get_user_log_vec(
            self,
            sent_vecs,
            log_mask,
            log_length,
            attn_pool,
            pad_doc,
            use_mask=True
    ):
        bz = sent_vecs.shape[0]
        if use_mask:
            user_log_vecs = attn_pool(sent_vecs, log_mask)
        else:
            padding_doc = pad_doc.expand(bz, self.args.news_dim).unsqueeze(1).expand(
                bz, sent_vecs.size(1), self.args.news_dim)
            sent_vecs = sent_vecs * log_mask.unsqueeze(2) + padding_doc * (1 - log_mask.unsqueeze(2))
            user_log_vecs = attn_pool(sent_vecs)
        return user_log_vecs

    def forward(self, user_news_vecs, log_mask,
                user_log_mask=False):
        user_vec = self.get_user_log_vec(
            user_news_vecs, log_mask,
            self.args.user_log_length,
            self.news_attn_pool, self.news_pad_doc,
            user_log_mask
        )
        return user_vec


class MLNR(nn.Module):
    def __init__(self, args):
        super(MLNR, self).__init__()

        self.args = args
        self.news_encoder = TextEncoder(args)
        self.user_encoder = UserEncoder(args, self.news_encoder if self.args.title_share_encoder else None)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self,
                news_vecs,  # the index of embedding from cache
                hist_sequence,  #user num  history lenght
                hist_sequence_mask,  #user num  history lenght
                candidate_inx,  #user num  history lenght-1  npratio
                labels,
                compute_loss=True
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
        reshape_candidate = candidate_inx.view(-1,)
        candidate_vec = news_vecs.index_select(0, reshape_candidate)
        candidate_vec = candidate_vec.view(candidate_inx.size(0),candidate_inx.size(1), -1) #B N D

        reshape_hist = hist_sequence.view(-1,)
        log_vec = news_vecs.index_select(0,reshape_hist)
        log_vec = log_vec.view(hist_sequence.size(0),hist_sequence.size(1),-1) # batch_size, log_length, news_dim

        user_vec = self.user_encoder(
            log_vec, hist_sequence_mask
        ).unsqueeze(-1)

        score = torch.bmm(candidate_vec, user_vec).squeeze(-1)
        if compute_loss:
            loss = self.loss_fn(score, labels)
            return loss, score
        else:
            return score


