import numpy as np
import torch.nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader, RandomSampler
from transformers import BertModel, BertForTokenClassification, BertTokenizer
from transformers.models.bert.modeling_bert import BertPooler, BertSelfAttention
import InputFeature


DEVICE = "cuda:0"
MAX_SEQUENCE_LENGTH = 80

class SelfAttention(torch.nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, MAX_SEQUENCE_LENGTH))
        zero_tensor = torch.tensor(zero_vec).float().to(DEVICE)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])

class Model(BertForTokenClassification):
    def __init__(self, bert_base_model):
        config = bert_base_model.config
        super(Model, self).__init__(config=config)
        self.bert_for_global_context = bert_base_model      # BERT编码器
        self.bert_for_local_context = bert_base_model
        self.pooler = BertPooler(config)        # 池化层
        self.dense = torch.nn.Linear(768, 3)    # 全连接层
        self.bert_global_focus = self.bert_for_global_context
        self.dropout = torch.nn.Dropout(0.1)    # dropout层
        self.SA1 = SelfAttention(config)        # 自注意力机制
        self.SA2 = SelfAttention(config)
        self.linear_double = torch.nn.Linear(768 * 2, 768)  # 全连接层
        self.linear_triple = torch.nn.Linear(768 * 3, 768)

    def get_ids_for_local_context_extractor(self, text_indices):
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(DEVICE)

    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(DEVICE)

    def get_batch_polarities(self, b_polarities):
        b_polarities = b_polarities.detach().cpu().numpy()
        shape = b_polarities.shape
        polarities = np.zeros((shape[0]))
        i = 0
        for polarity in b_polarities:
            polarity_idx = np.flatnonzero(polarity + 1)
            polarities[i] = polarity[polarity_idx[0]]
            i += 1
        polarities = torch.from_numpy(polarities).long().to(DEVICE)
        return polarities

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None, polarities=None, valid_ids=None, attention_mask_label=None):
        input_ids_spc = self.get_ids_for_local_context_extractor(input_ids_spc)
        labels = self.get_batch_token_labels_bert_base_indices(labels)
        global_context_out = self.bert_for_global_context(input_ids_spc, token_type_ids, attention_mask)
        polarity_labels = self.get_batch_polarities(polarities)
        batch_size, max_len, feat_dim = global_context_out[0].shape
        '''
        bert的输出是tuple类型的，包括4个：
        last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态。
        pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的，这个输出不是对输入的语义内容的一个很好的总结，对于整个输入序列的隐藏状态序列的平均化或池化通常更好。
        hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)。
        attentions：这也是输出的一个可选项，如果输出，需要指定config.output_attentions=True,它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值。
        '''
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(DEVICE)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[0][i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)
        pooled_out = self.pooler(global_context_out)
        pooled_out = self.dropout(pooled_out)
        apc_logits = self.dense(pooled_out)
        if labels is not None:
            # 训练过程计算损失
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss_sen = torch.nn.CrossEntropyLoss()
            loss_ate = loss_fct(ate_logits.view(-1, self.num_labels), labels.view(-1))
            loss_apc = loss_sen(apc_logits, polarity_labels)
            return loss_ate, loss_apc
        else:
            return ate_logits, apc_logits


# 设置训练的输入
def get_train_dataloader(train_load, label_list, max_seq_length, batch_size):
    train_features = InputFeature.get_train_features(train_load, label_list, max_seq_length)

    all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_polarities = torch.tensor([f.polarities for f in train_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_polarities,
                               all_valid_ids, all_lmask_ids)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader


# 设置测试的输入
def get_eval_dataloader(test_load, label_list, max_seq_length, batch_size):
    test_features = InputFeature.get_test_features(test_load, label_list, max_seq_length)

    all_spc_input_ids = torch.tensor([f.input_ids_spc for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_polarities = torch.tensor([f.polarities for f in test_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in test_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in test_features], dtype=torch.long)
    eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_polarities,
                              all_valid_ids, all_lmask_ids)
    eval_sample = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sample, batch_size=batch_size)
    return eval_dataloader









