# modified from https://github.com/maszhongming/MatchSum
import torch
from torch import nn
from transformers import RobertaModel
from knowledge_cache import get_knowledgescore
import numpy as np
import torch.nn.init as init

def RankingLoss(score, common_knowledge_score, summary_score=None, margin=0, gold_margin=0, gold_weight=1, no_gold=False, no_cand=False, no_common_knowledge=False, common_knowledge_weight=1):
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
    # gold summary loss
    # 将 summary_score 扩展为与 score 相同形状的张量，以便于和 score 进行比较
    #print(score)
    #print(summary_score)
    #print(common_knowledge_score)
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

    return TotalLoss


class ReRanker(nn.Module):
    def __init__(self, encoder, pad_token_id):
        super(ReRanker, self).__init__()
        self.encoder = RobertaModel.from_pretrained(encoder)
        self.pad_token_id = pad_token_id
        # 添加更多隐藏层
        self.hidden_layer = nn.Linear(768, 256)
        self.relu = nn.ReLU()

        # 添加注意力机制
        self.attention_layer = nn.Linear(256, 1)

        # 添加多任务学习
        self.task_layer = nn.Linear(256, 2)  # 假设有2个任务

        # 添加正则化
        self.dropout = nn.Dropout(0.2)
        # 定义权重作为可学习参数
        self.w_context = nn.Parameter(torch.ones(1, 768))
        self.w_candidate = nn.Parameter(torch.ones(5, 768))

        # 初始化权重
        init.xavier_uniform_(self.w_context)

    def forward(self, question_id, context_id, candidate_id, answer_id=None, require_gold=True):

        batch_size = question_id.size(0)

        # 获取文本对应的嵌入向量
        input_mask = question_id != self.pad_token_id
        out = self.encoder(question_id, attention_mask=input_mask)[0]
        question_emb = out[:, 0, :]

        input_mask = context_id != self.pad_token_id
        out = self.encoder(context_id, attention_mask=input_mask)[0]
        context_emb = out[:, 0, :]

        if require_gold:
            # get reference score
            # 获取gold reference的嵌入向量以及相似度得分
            input_mask = answer_id != self.pad_token_id
            out = self.encoder(answer_id, attention_mask=input_mask)[0]
            answer_emb = out[:, 0, :]
            answer_score = torch.cosine_similarity(answer_emb, question_emb, dim=-1)

        candidate_num = candidate_id.size(1)
        candidate_id = candidate_id.view(-1, candidate_id.size(-1))

        # 获取候选摘要对应的嵌入向量
        input_mask = candidate_id != self.pad_token_id
        out = self.encoder(candidate_id, attention_mask=input_mask)[0]
        candidate_emb = out[:, 0, :].view(batch_size, candidate_num, -1)

        # get candidate score
        # 计算候选摘要与文本之间的相似度得分
        question_emb = question_emb.unsqueeze(1).expand_as(candidate_emb)
        score = torch.cosine_similarity(candidate_emb, question_emb, dim=-1)
        #print("question_emb", question_emb)
        #print("score", score)
        """
        context_emb = context_emb.unsqueeze(1).expand_as(candidate_emb)
        common_knowledge_score = torch.cosine_similarity(candidate_emb, context_emb, dim=-1)
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
        #weighted_context_emb = self.dropout(weighted_context_emb)
        weighted_context_emb = self.hidden_layer(weighted_context_emb)
        #weighted_candidate_emb = self.dropout(weighted_candidate_emb)
        weighted_candidate_emb = self.hidden_layer(weighted_candidate_emb)
        common_knowledge_score = torch.cosine_similarity(weighted_candidate_emb, weighted_context_emb, dim=-1)
        #print(common_knowledge_score.size())
        output = {'score': score, 'knowledge_score': common_knowledge_score}
        if require_gold:
            output['answer_score'] = answer_score
        return output
