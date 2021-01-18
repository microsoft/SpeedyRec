# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from LanguageModels.BaseModel import *

class TuringNLRv3PreTrainedModel(BertPreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = TuringNLRv3ForSeq2SeqConfig
    supported_convert_pretrained_model_archive_map = {
        "tnlrv3": TuringNLRv3_PRETRAINED_MODEL_ARCHIVE_MAP,
    }
    base_model_prefix = "TuringNLRv3_for_seq2seq"
    pretrained_model_archive_map = {
        **TuringNLRv3_PRETRAINED_MODEL_ARCHIVE_MAP,
    }

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
            cls, pretrained_model_name_or_path, reuse_position_embedding=None,
            replace_prefix=None, *model_args, **kwargs,
    ):
        model_type = kwargs.pop('model_type', 'tnlrv3')
        if model_type is not None and "state_dict" not in kwargs:
            if model_type in cls.supported_convert_pretrained_model_archive_map:
                pretrained_model_archive_map = cls.supported_convert_pretrained_model_archive_map[model_type]
                if pretrained_model_name_or_path in pretrained_model_archive_map:
                    state_dict = get_checkpoint_from_transformer_cache(
                        archive_file=pretrained_model_archive_map[pretrained_model_name_or_path],
                        pretrained_model_name_or_path=pretrained_model_name_or_path,
                        pretrained_model_archive_map=pretrained_model_archive_map,
                        cache_dir=kwargs.get("cache_dir", None), force_download=kwargs.get("force_download", None),
                        proxies=kwargs.get("proxies", None), resume_download=kwargs.get("resume_download", None),
                    )
                    state_dict = state_dict_convert[model_type](state_dict)
                    kwargs["state_dict"] = state_dict
                    logger.info("Load HF ckpts")
                elif os.path.isfile(pretrained_model_name_or_path):
                    state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                elif os.path.isdir(pretrained_model_name_or_path):
                    state_dict = torch.load(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME),
                                            map_location='cpu')
                    kwargs["state_dict"] = state_dict_convert[model_type](state_dict)
                    logger.info("Load local ckpts")
                else:
                    raise RuntimeError("Not fined the pre-trained checkpoint !")

        if kwargs["state_dict"] is None:
            logger.info("TNLRv3 does't support the model !")
            raise NotImplementedError()

        config = kwargs["config"]
        state_dict = kwargs["state_dict"]
        # initialize new position embeddings (From Microsoft/UniLM)
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                max_range = config.max_position_embeddings if reuse_position_embedding else old_vocab_size
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[shift: shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " % (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(torch.ones(
                    size=(config.max_position_embeddings, state_dict[_k].shape[1])), dtype=torch.float)
                new_postion_embedding = nn.Parameter(data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding

        _k2 = 'bert.rel_pos_bias.weight'
        if _k2 in state_dict and state_dict[_k2].shape[1] != (config.bus_num + config.rel_pos_bins):
            logger.info(
                f"rel_pos_bias.weight.shape[1]:{state_dict[_k2].shape[1]} != config.bus_num+config.rel_pos_bins:{config.bus_num + config.rel_pos_bins}")
            old_rel_pos_bias = state_dict[_k2]
            new_rel_pos_bias = torch.cat(
                [old_rel_pos_bias, old_rel_pos_bias[:, -1:].expand(old_rel_pos_bias.size(0), config.bus_num)], -1)
            new_rel_pos_bias = nn.Parameter(data=new_rel_pos_bias, requires_grad=True)
            state_dict[_k2] = new_rel_pos_bias.data
            del new_rel_pos_bias

        if replace_prefix is not None:
            new_state_dict = {}
            for key in state_dict:
                if key.startswith(replace_prefix):
                    new_state_dict[key[len(replace_prefix):]] = state_dict[key]
                else:
                    new_state_dict[key] = state_dict[key]
            kwargs["state_dict"] = new_state_dict
            del state_dict

        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position, frequence and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        fix_word_embedding = getattr(config, "fix_word_embedding", None)
        if fix_word_embedding:
            self.word_embeddings.weight.requires_grad = False
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        else:
            self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.fre_embedding = nn.Embedding(100, config.hidden_size)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, fre_cnt=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if fre_cnt is None:
            embeddings = inputs_embeds + position_embeddings
        else:
            fre_embedding = self.fre_embedding(fre_cnt)
            embeddings = inputs_embeds + position_embeddings + fre_embedding

        if self.token_type_embeddings:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_ids



class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.bus_num = config.bus_num
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, rel_pos=None, seg_num=0):
        all_hidden_states = ()
        all_attentions = ()

        batch_seg_num, _, emb_dim = hidden_states.shape
        batch_size = batch_seg_num // seg_num

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.bus_num > 0 and i > 0:
                station_emb = hidden_states[:, seg_num]  # BS D
                station_emb = station_emb.view(batch_size, seg_num, emb_dim)  # B S D
                station_emb = station_emb.unsqueeze(1).repeat(1, seg_num, 1, 1)  # B S S D
                station_emb = station_emb.view(batch_seg_num, seg_num, emb_dim)  # BS S D
                hidden_states[:, :seg_num] = station_emb

            if i == 0:
                temp_attention_mask = attention_mask.clone()
                temp_attention_mask[:, :, :, :seg_num] = -10000.0
                layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask, rel_pos=rel_pos)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, rel_pos=rel_pos)

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class SpeedyModel(TuringNLRv3PreTrainedModel):
    r"""
    Args:
        input_ids: input_ids of segments
        attention_mask: attention mask for tokens in segments
        key_position: position_ids of keywords, required if use content refinement
        fre_cnt: frequence of keywords, required if use content refinement
        seg_num: the number of segments

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, bus_num+sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, bus_num+sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, bus_num+sequence_length, bus_num+sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """

    def __init__(self, config):
        super(SpeedyModel, self).__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        if not isinstance(config, TuringNLRv3ForSeq2SeqConfig):
            self.pooler = BertPooler(config)
        else:
            self.pooler = None

        if self.config.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.config.rel_pos_bins + config.bus_num, config.num_attention_heads,
                                          bias=False)
        else:
            self.rel_pos_bias = None

    def forward(self, input_ids, attention_mask, key_position, fre_cnt, seg_num):
        '''
        Args:
            input_ids: BS L
            attention_mask: BS S+L
        '''
        embedding_output, position_ids = self.embeddings(input_ids=input_ids,fre_cnt=fre_cnt)  # BS L D, BS L
        attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0  # BS 1 1 S+L

        rel_pos = None
        if self.config.rel_pos_bins > 0:
            if key_position is not None:
                position_ids = key_position
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)  # BS L L
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.config.rel_pos_bins,
                                               max_distance=self.config.max_rel_pos)

            if self.config.bus_num > 0:
                # BS S+L L
                rel_pos = torch.cat(
                    [torch.zeros(rel_pos.size(0), self.config.bus_num, rel_pos.size(2)).to(rel_pos.device).long(),
                     rel_pos], dim=1)
                other_seg_relpos = torch.arange(self.config.rel_pos_bins,
                                                self.config.rel_pos_bins + self.config.bus_num).to(
                    rel_pos.device).long()
                other_seg_relpos = other_seg_relpos.unsqueeze(0).unsqueeze(0).expand(rel_pos.size(0), rel_pos.size(1),
                                                                                     -1)
                # BS S+L S+L
                rel_pos = torch.cat([other_seg_relpos, rel_pos], dim=-1)

            rel_pos = F.one_hot(rel_pos, num_classes=self.config.rel_pos_bins + self.config.bus_num).type_as(
                embedding_output[0])
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)  # BS H S_L S+L

        if self.config.bus_num > 0:
            # Add station_placeholder
            station_placeholder = torch.zeros(embedding_output.size(0), seg_num, embedding_output.size(-1)).type(
                embedding_output.dtype).to(embedding_output.device)  # BS S D
            embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # BS S+L D

        encoder_outputs = self.encoder(
            embedding_output, attention_mask=attention_mask,
            rel_pos=rel_pos, seg_num=seg_num)

        return encoder_outputs  # last-layer hidden state, (all hidden states), (all attentions)


class SpeedyModelForRec(TuringNLRv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = SpeedyModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask, position_ids, fre_cnt, seg_num):
        return self.bert(input_ids, attention_mask, position_ids, fre_cnt, seg_num)

