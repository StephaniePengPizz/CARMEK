import torch
from torch import nn
from transformers import RobertaModel
from knowledge_cache import get_knowledgescore
import torch.nn.functional as F
import torch.nn.init as init
from nlp_rake import rake_text
from transformers import RobertaTokenizer
import numpy as np
from sklearn.preprocessing import MinMaxScaler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class BiDirectionalAttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiDirectionalAttentionLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 定义查询、键、值的线性变换层
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.key_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)

        # 定义输出的线性变换层
        self.output_layer = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x, y):
        # 进行线性变换得到查询、键、值
        query = self.query_layer(x)
        key = self.key_layer(y)
        value = self.value_layer(y)

        # 计算注意力得分
        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.hidden_size)

        # 进行softmax操作得到注意力权重
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        #print(attention_probs.size())
        #print(value.size())
        # 对值进行加权平均得到输出
        context_x = torch.matmul(attention_probs, value)
        #print(x.size())
        #print(context_x.size())
        # 进行线性变换得到输出
        output = self.output_layer(torch.cat([query, context_x], dim=-1))
        #print(output.size())
        return output
def RankingLoss(score, common_knowledge_score, common_knowledge_score2, external_knowledge_score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False, no_common_knowledge=False, common_knowledge_weight=1, external_knowledge_weight=1):
    # score: 每个候选摘要的得分，形状为 (batch_size, num_candidates)
    # summary_score: 真实摘要的得分，形状为 (batch_size,)
    # margin: 边际值，用于计算 MarginRankingLoss，用于惩罚负得分
    # gold_margin: 金摘要的边际值，用于计算 MarginRankingLoss
    # gold_weight: 金摘要的损失权重
    # no_gold: 是否使用金摘要
    # no_cand: 是否使用候选摘要

    # 初始化一个与 score 相同形状的张量，每个元素的值都是 1
    ones = torch.ones_like(score)

    # 创建一个 MarginRankingLoss 对象，边际值为 0.0
    loss_func = torch.nn.MarginRankingLoss(0.0)

    # 计算总损失，将 score 与自身比较
    TotalLoss = loss_func(score, score, ones)
    # candidate loss
    # 如果使用候选摘要，则遍历所有可能的组合
    n = score.size(1)

    # 如果不使用候选摘要，则返回总损失
    if not no_cand:
        for i in range(1, n):
            # 将 score 按列切片，得到前 i 个和后 n-i 个候选摘要的得分v
            pos_score = score[:, :-i]
            neg_score = score[:, i:]
            # 将 pos_score 和 neg_score 摊平为一维张量
            pos_score = pos_score.contiguous().view(-1)
            neg_score = neg_score.contiguous().view(-1)
            # 初始化一个与 pos_score 相同形状的张量，每个元素的值都是 1
            ones = torch.ones_like(pos_score)
            # 创建一个 MarginRankingLoss 对象，边际值为 margin * i
            loss_func = torch.nn.MarginRankingLoss(margin * i)
            # 计算损失
            loss = loss_func(pos_score, neg_score, ones)
            # 加入总损失
            TotalLoss += loss
    # 如果不适用金摘要，则返回总损失
    if no_gold:
        return TotalLoss

    pos_score = summary_score.unsqueeze(-1).expand_as(score)
    neg_score = score
    # 将 pos_score 和 neg_score 摊平为一维张量
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    # 初始化一个与 pos_score 相同形状的张量，每个元
    ones = torch.ones_like(pos_score)
    # 创建一个 MarginRankingLoss 对象，边际值为 gold_margin
    loss_func = torch.nn.MarginRankingLoss(gold_margin)
    # 计算损失，并乘以 gold_weight，加入总损失
    TotalLoss += gold_weight * loss_func(pos_score, neg_score, ones)

    # common knowledge loss
    # 如果不使用共同知识，则返回总损失
    if no_common_knowledge:
        return TotalLoss

    # 将 common_knowledge_score 扩展为与 score 相同形状的张量，以便于和 score 进行比较
    pos_score = common_knowledge_score
    neg_score = score
    # 将 pos_score 和 neg_score 摊平为一维张量
    pos_score = pos_score.contiguous().view(-1)
    neg_score = neg_score.contiguous().view(-1)
    # 初始化一个与 pos_score 相同形状的张量，每个元
    ones = torch.ones_like(pos_score)
    # 创建一个 MarginRankingLoss 对象，边际值为 0.0（或其他合适的值）
    loss_func = torch.nn.MarginRankingLoss(0.0)
    # 计算损失，并乘以 common_knowledge_weight，加入总损失
    TotalLoss += common_knowledge_weight * loss_func(pos_score, neg_score, ones)

    if common_knowledge_score2 != None:
        # 将 common_knowledge_score 扩展为与 score 相同形状的张量，以便于和 score 进行比较
        pos_score = common_knowledge_score2
        neg_score = score
        # 将 pos_score 和 neg_score 摊平为一维张量
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        # 初始化一个与 pos_score 相同形状的张量，每个元
        ones = torch.ones_like(pos_score)
        # 创建一个 MarginRankingLoss 对象，边际值为 0.0（或其他合适的值）
        loss_func = torch.nn.MarginRankingLoss(0.0)
        # 计算损失，并乘以 common_knowledge_weight，加入总损失
        TotalLoss += common_knowledge_weight * loss_func(pos_score, neg_score, ones)

    if external_knowledge_score != None:
        # 将 common_knowledge_score 扩展为与 score 相同形状的张量，以便于和 score 进行比较
        pos_score = external_knowledge_score
        neg_score = score
        # 将 pos_score 和 neg_score 摊平为一维张量
        pos_score = pos_score.contiguous().view(-1)
        neg_score = neg_score.contiguous().view(-1)
        # 初始化一个与 pos_score 相同形状的张量，每个元
        ones = torch.ones_like(pos_score)
        # 创建一个 MarginRankingLoss 对象，边际值为 0.0（或其他合适的值）
        loss_func = torch.nn.MarginRankingLoss(0.0)
        # 计算损失，并乘以 common_knowledge_weight，加入总损失
        TotalLoss += external_knowledge_weight * loss_func(pos_score, neg_score, ones)

    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id, knowledge_graph):
        super(ReRanker, self).__init__()
        self.encoder = RobertaModel.from_pretrained(encoder)
        self.pad_token_id = pad_token_id
        # 添加更多隐藏层
        self.hidden_layer = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.hidden_size = 128
        # 添加注意力机制
        self.attention_layer = nn.Linear(256, 1)

        # 添加多任务学习
        self.task_layer = nn.Linear(256, 2)  # 假设有2个任务

        # 添加正则化
        self.dropout = nn.Dropout(0.2)
        # 定义权重作为可学习参数
        self.w_context = nn.Parameter(torch.ones(1, 768))
        self.w_candidate = nn.Parameter(torch.ones(5, 768))
        self.w_keyword = nn.Parameter(torch.ones(1, 768))

        # 初始化权重
        init.xavier_uniform_(self.w_context)
        self.tok = RobertaTokenizer.from_pretrained("roberta-base", verbose=False)
        self.maxlen = 512
        self.pad_token_id = self.tok.pad_token_id
        self.total_len = 512
        self.cls_token_id = self.tok.cls_token_id
        self.sep_token_id = self.tok.sep_token_id
        # 定义双向注意力层
        self.bi_directional_attention_layer = BiDirectionalAttentionLayer(self.encoder.config.hidden_size,
                                                                          self.hidden_size)
        self.knowledge_graph = knowledge_graph

    def bert_encode(self, x, max_len=-1):
        _ids = self.tok.encode(x, add_special_tokens=False)
        ids = [self.cls_token_id]
        if max_len > 0:
            ids.extend(_ids[:max_len - 2])
        else:
            ids.extend(_ids[:self.total_len - 2])
        ids.append(self.sep_token_id)
        return ids

    def forward(self, question_id, context_id, candidate_id, answer_id=None, require_gold=True):

        batch_size = question_id.size(0)

        # 获取文本对应的嵌入向量
        input_mask = question_id != self.pad_token_id
        out = self.encoder(question_id, attention_mask=input_mask)[0]
        question_emb = out[:, 0, :]

        input_mask = context_id != self.pad_token_id
        out = self.encoder(context_id, attention_mask=input_mask)[0]
        context_emb = out[:, 0, :]
        #print(out)
        #print(context_emb)
        #print(input_mask)

        if require_gold:
            # get reference score
            # 获取gold reference的嵌入向量以及相似度得分
            input_mask = answer_id != self.pad_token_id
            out = self.encoder(answer_id, attention_mask=input_mask)[0]
            answer_emb = out[:, 0, :]
            answer_score = torch.cosine_similarity(answer_emb, question_emb, dim=-1)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))
        #print("cc", candidate_id)
        # 获取候选摘要对应的嵌入向量
        input_mask = candidate_id != self.pad_token_id
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)
        #print(out)
        # get candidate score
        # 计算候选摘要与文本之间的相似度得分
        question_emb = self.bi_directional_attention_layer(question_emb, context_emb)
        context_emb = self.bi_directional_attention_layer(context_emb, question_emb)

        question_emb = question_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, question_emb, dim=-1)
        #print("question_emb", question_emb)
        #print("question_emb_size", question_emb.size())
        #print("score", score)
        """
        context_emb = context_emb.unsqueeze(1).expand_as(candidate_emb)
        common_knowledge_score = torch.cosine_similarity(candidate_emb, context_emb, dim=-1)
        """
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        #print("cd", context_id)

        #print(candidate_id.size())
        context = tokenizer.decode(torch.flatten(context_id), skip_special_tokens=True)
        keywords_tuple = rake_text(context)
        keywords_list = [item[0] for item in keywords_tuple]

        keyword2 = []
        extra_scores = None
        for i, keyword in enumerate(keywords_list):
            if i > 10:
                break
            """
            for single_word in keyword.split(' '):
                if single_word in self.knowledge_graph:
                    for item in self.knowledge_graph[single_word]:
                        # keyword2.append(item)
                        keyword2.append(item)
            """
            keyword_ids = self.bert_encode(keyword)
            keyword_tensor = torch.tensor([keyword_ids]).to(device)
            input_mask = keyword_tensor != self.pad_token_id
            out = self.encoder(keyword_tensor, attention_mask=input_mask)[0]
            keyword_emb = out[:, 0, :]
            keyword_emb = keyword_emb.unsqueeze(1).expand_as(candidate_emb)

            normalized_w_keyword = self.w_keyword / torch.sum(self.w_keyword)
            normalized_w_candidate = self.w_candidate / torch.sum(self.w_candidate)

            weighted_candidate_emb = candidate_emb * normalized_w_candidate
            weighted_keyword_emb = keyword_emb * normalized_w_keyword

            new_extra_score = torch.cosine_similarity(weighted_keyword_emb, weighted_candidate_emb, dim=-1).unsqueeze(0)

            if extra_scores is None:
                extra_scores = new_extra_score
            else:
                extra_scores = torch.cat([extra_scores, new_extra_score], dim=0)
        extra_score = extra_scores.mean(dim=0)
        """
        external_knowledge_score = torch.zeros(batch_size, candidate_num, dtype=torch.float32).to(device)
        # print(len(candidate_id))
        # print(candidate_num)
        
        for k in range(len(candidate_id)):
            answers = tokenizer.decode(candidate_id[k], skip_special_tokens=True)
            for candidate in answers.split(' '):
                if candidate in keyword2:
                    external_knowledge_score[k // candidate_num][k % candidate_num] += 1

        for batch_num in range(batch_size):
            sample = external_knowledge_score[batch_num]
            all_equal = torch.all(sample == sample[0])
            if not all_equal:
                # print("yes")
                mean = sample.mean(dim=0)
                std = sample.std(dim=0)

                # 进行标准化并映射到 [0, 1] 范围
                normalized_scores = (sample - mean) / std
                sample = (normalized_scores - normalized_scores.min()) / (
                        normalized_scores.max() - normalized_scores.min())
                external_knowledge_score[batch_num] = sample
            else:
                external_knowledge_score[batch_num] = torch.zeros(candidate_num, dtype=torch.float32).to(device)
            # print("ee", external_knowledge_score)
        """

        normalized_w_context = self.w_context / torch.sum(self.w_context)
        normalized_w_candidate = self.w_candidate / torch.sum(self.w_candidate)

        # 对词向量应用权重，得到加权后的向量
        #print(context_emb.size())
        context_emb = context_emb.unsqueeze(1).expand_as(candidate_emb)
        weighted_candidate_emb = candidate_emb * normalized_w_candidate
        weighted_context_emb = context_emb * normalized_w_context
        #print(context_emb.size())
        #print(weighted_context_emb.size())
        # 计算加权的余弦相似度
        # 添加更多隐藏层和注意力机制
        common_knowledge_score = torch.cosine_similarity(weighted_candidate_emb, weighted_context_emb, dim=-1)

        #print(common_knowledge_score.size())
        output = {'score': score, 'knowledge_score': common_knowledge_score, 'extra_score': extra_score, 'external_knowledge_score': None}
        if require_gold:
            output['answer_score'] = answer_score
        return output
