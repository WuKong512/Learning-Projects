import json
import os
import sys
import logging
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, classification_report
from transformers import BertModel, BertTokenizer, AdamW
import ModelConstruct
import warnings


EPOCH = 5
EVAL_STEP = 10     # 每10个step执行一个评估
MAX_SEQUENCE_LENGTH = 80
PRETRAINED_BERT_MODEL = "bert-base-chinese"
LEARNING_RATE = 3e-5
BATCH_SIZE = 16
DEVICE = "cuda:0"
LABEL_LIST = ["O", "B-ASP", "I-ASP", "[CLS]", "[SEP]"]
NUM_LABELS = len(LABEL_LIST) + 1

model_path = r"C:\Users\51268\.cache\huggingface\hub\models--bert-base-chinese"
bert_base_model = BertModel.from_pretrained(model_path)
bert_base_model.config.num_labels = NUM_LABELS
tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def evaluate(dataloader, label_list):
    apc_result = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}
    ate_result = 0
    y_true = []
    y_pred = []
    n_test_correct, n_test_total = 0, 0
    test_apc_logits_all, test_polarities_all = None, None
    model.eval()    # 将网络设置为评估状态
    label_map = {i: label for i, label in enumerate(label_list, 1)}
    for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask in dataloader:
        input_ids_spc = input_ids_spc.to(DEVICE)
        input_mask = input_mask.to(DEVICE)
        segment_ids = segment_ids.to(DEVICE)
        valid_ids = valid_ids.to(DEVICE)
        label_ids = label_ids.to(DEVICE)
        polarities = polarities.to(DEVICE)
        l_mask = l_mask.to(DEVICE)
        with torch.no_grad():
            ate_logits, apc_logits = model(
                input_ids_spc, segment_ids, input_mask,
                valid_ids=valid_ids, polarities=polarities, attention_mask_label=l_mask)
        polarities = model.get_batch_polarities(polarities)
        n_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
        n_test_total += len(polarities)
        if test_polarities_all is None:
            test_polarities_all = polarities
            test_apc_logits_all = apc_logits
        else:
            test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
            test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)
        label_ids = model.get_batch_token_labels_bert_base_indices(label_ids)
        ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
        ate_logits = ate_logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        for i, label in enumerate(label_ids):
            temp_1 = []
            temp_2 = []
            for j, m in enumerate(label):
                if j == 0:
                    continue
                elif label_ids[i][j] == len(label_list):
                    y_true += temp_1
                    y_pred += temp_2
                    break
                else:
                    temp_1.append(label_map.get(label_ids[i][j], 'O'))
                    temp_2.append(label_map.get(ate_logits[i][j], 'O'))
    test_acc = n_test_correct / n_test_total
    test_f1 = f1_score(torch.argmax(test_apc_logits_all, -1).cpu(), test_polarities_all.cpu(), labels=[0, 1],
                       average='macro')
    test_acc = round(test_acc * 100, 2)
    test_f1 = round(test_f1 * 100, 2)
    apc_result = {'max_apc_test_acc': test_acc, 'max_apc_test_f1': test_f1}
    report = classification_report(y_true, y_pred, digits=4)
    tmps = report.split()
    ate_result = round(float(tmps[7]) * 100, 2)
    return apc_result, ate_result


model = ModelConstruct.Model(bert_base_model)
_ = model.to(DEVICE)

param_optimizer = list(model.named_parameters())    # 模型中的所有参数
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001},
    {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00001}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, weight_decay=0.00001)

max_apc_test_acc = 0
max_apc_test_f1 = 0
max_ate_test_f1 = 0
global_step = 0
train_load = "./datasets/notebook/notebook.atepc.train.dat"
train_dataloader = ModelConstruct.get_train_dataloader(train_load, LABEL_LIST, MAX_SEQUENCE_LENGTH, BATCH_SIZE)
test_load = "./datasets/notebook/notebook.atepc.test.dat"
eval_dataloader = ModelConstruct.get_eval_dataloader(test_load, LABEL_LIST, MAX_SEQUENCE_LENGTH, BATCH_SIZE)

for epoch in range(EPOCH):
    # 每个epoch
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # 一个step
        model.train()   # 将网络设置为train的模式
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask = batch    # 取一个batch的数据
        loss_ate, loss_apc = model(
            input_ids_spc, segment_ids, input_mask, label_ids, polarities, valid_ids, l_mask)       # 向前传播，计算损失
        loss = loss_ate + loss_apc
        loss.backward()  # 反向传播计算梯度
        nb_tr_examples += input_ids_spc.size(0)
        nb_tr_steps += 1
        optimizer.step()
        optimizer.zero_grad()
        global_step += 1
        if global_step % EVAL_STEP == 0:    # 评估
            warnings.filterwarnings("ignore")
            apc_result, ate_result = evaluate(eval_dataloader, LABEL_LIST)
            if apc_result['max_apc_test_acc'] > max_apc_test_acc:
                max_apc_test_acc = apc_result['max_apc_test_acc']
            if apc_result['max_apc_test_f1'] > max_apc_test_f1:
                max_apc_test_f1 = apc_result['max_apc_test_f1']
            if ate_result > max_ate_test_f1:
                max_ate_test_f1 = ate_result
            current_apc_test_acc = apc_result['max_apc_test_acc']
            current_apc_test_f1 = apc_result['max_apc_test_f1']
            current_ate_test_f1 = round(ate_result, 2)      # 四舍五入到小数点两位
            logger.info('*' * 10)
            logger.info('Epoch %s' % epoch)
            logger.info(f'APC_test_acc: {current_apc_test_acc}(max: {max_apc_test_acc}) '
                        f'APC_test_f1: {current_apc_test_f1}(max: {max_apc_test_f1})')
            logger.info(f'ATE_test_f1: {current_ate_test_f1}(max: {max_ate_test_f1})')
            logger.info('*' * 10)



# # 保存模型
# SAVE_PATH = "./temp/"
# os.makedirs(SAVE_PATH, exist_ok=True)
# model.save_pretrained(SAVE_PATH)
# tokenizer.save_pretrained(SAVE_PATH)
# label_map = {i : label for i, label in enumerate(LABEL_LIST, 1)}
# model_config = {
#     "bert_model": PRETRAINED_BERT_MODEL,
#     "do_lower": True,
#     "max_seq_length": MAX_SEQUENCE_LENGTH,
#     "num_labels": len(LABEL_LIST) + 1,
#     "label_map": label_map
# }
# json.dump(model_config, open(os.path.join(SAVE_PATH, "config.json"), "w"))








