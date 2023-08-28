import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import model_external
import pickle
import time
import numpy as np
import os
import json
import random
from compare_mt.rouge.rouge_scorer import RougeScorer
from transformers import RobertaModel, RobertaTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp, ReRankingDataset
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from model_external import RankingLoss
import logging
from base_setting import base_setting
torch.cuda.empty_cache()
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)

"""
函数 evaluation 在进行文本摘要评价，将文本摘要结果与参考摘要进行 Rouge 评价，
并输出 Rouge-1，Rouge-2，Rouge-L 指标，以及准确率。

具体操作流程为：
1.加载数据集，包括 tokenizer，数据集，dataloader 等。
2.加载预训练模型，将其设为评价器 scorer，并载入预训练权重。
3.遍历数据集中的每一个摘要，使用 scorer 评估其生成的摘要结果与参考摘要的 Rouge 指标，并输出评价指标，同时将结果写入磁盘中。

函数 test 与 evaluation 类似，但不会将评价结果写入磁盘，而是将评价指标直接输出。该函数是在测试集上进行模型评价时使用的。
"""

def evaluation(args):
    # load data
    base_setting(args)
    tok = RobertaTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = ReRankingDataset(f"/root/autodl-tmp/{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, maxlen=512,
                                is_sorted=False, maxnum=args.max_num, is_untok=True)
    dataloader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    with open("temp2.json", "r") as json_file:
        knowledge_graph = json.load(json_file)
    scorer = model_external.ReRanker(model_path, tok.pad_token_id, knowledge_graph)
    if args.cuda:
        scorer = scorer.cuda()
    scorer.load_state_dict(torch.load(
        os.path.join("/tmp/optimization/cache/{0}--{1}--{2}--{3}".format(args.improve_type, args.model_name, args.dataname, args.real_num),
                     args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    scorer.eval()
    model_name = args.model_pt.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)
    print(args.improve_type)
    print(model_name)
    print(args.dataname)
    mkdir("/tmp/optimization/result/{}".format(args.improve_type))
    mkdir("/tmp/optimization/result/{0}/{1}".format(args.improve_type, model_name))
    mkdir("/tmp/optimization/result/{0}/{1}/reference".format(args.improve_type, model_name))
    mkdir("/tmp/optimization/result/{0}/{1}/candidate".format(args.improve_type, model_name))
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    cnt = 0
    acc = 0
    scores = []
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, args.gpuid[0])
            samples = batch["data"]
            output = scorer(batch["qtn_input_ids"], batch["ctx_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['answer_score']
            similarity = similarity.cpu().numpy()

            if i % 100 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            scores.extend(similarity.tolist())
            acc += (max_ids == batch["scores"].cpu().numpy().argmax(1)).sum()
            for j in range(similarity.shape[0]):
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["answer"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
                with open("/tmp/optimization/result/%s/%s/candidate/%d.dec" % (args.improve_type, model_name, cnt), "w") as f:
                    for s in sents:
                        print(s, file=f)
                with open("/tmp/optimization/result/%s/%s/reference/%d.ref" % (args.improve_type, model_name, cnt), "w") as f:
                    for s in sample["answer"]:
                        print(s, file=f)

                # Perform "whether it conforms to knowledge" evaluation here
                #knowledge_score = knowledge_score[j]  # Get the knowledge score for the current sample
                #knowledge_threshold = 0.5  # Set a threshold for knowledge score
                #conforms_to_knowledge = knowledge_score >= knowledge_threshold

                # Print whether the current output conforms to knowledge
                #print(
                #    f"Sample {cnt}: {'Conforms to knowledge' if conforms_to_knowledge else 'Does not conform to knowledge'}")

                cnt += 1
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    print(f"accuracy: {acc / cnt}")
    print("rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (rouge1, rouge2, rougeLsum))


def test(dataloader, scorer, args, gpuid):
    scorer.eval()
    loss = 0
    cnt = 0
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    with torch.no_grad():
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            samples = batch["data"]
            output = scorer(batch["qtn_input_ids"], batch["ctx_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            similarity, gold_similarity = output['score'], output['answer_score']
            similarity = similarity.cpu().numpy()
            if i % 1000 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["answer"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    scorer.train()
    loss = 1 - ((rouge1 + rouge2 + rougeLsum) / 3)
    print(f"rouge-1: {rouge1}, rouge-2: {rouge2}, rouge-L: {rougeLsum}")

    if len(args.gpuid) > 1:
        loss = torch.FloatTensor([loss]).to(gpuid)
        dist.all_reduce(loss, op=dist.reduce_op.SUM)
        loss = loss.item() / len(args.gpuid)
    return loss


def run(rank, args):
    base_setting(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank]
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)

    if is_master:
        # id = len(os.listdir("/tmp/optimization/cache/{}".format(args.improve_type)))
        recorder = Recorder(args.improve_type, args.model_name, args.dataname, args.real_num, args.log)

    tok = RobertaTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp, pad_token_id=tok.pad_token_id, is_test=True)
    train_set = ReRankingDataset(f"/root/autodl-tmp/{args.dataset}/{args.datatype}/train", args.model_type, maxlen=args.max_len,
                                 maxnum=args.max_num)
    val_set = ReRankingDataset(f"/root/autodl-tmp/{args.dataset}/{args.datatype}/val", args.model_type, is_test=True, maxlen=512,
                               is_sorted=False, maxnum=args.max_num)
    print("dataset done")
    """
    这段代码是为了创建用于训练和验证的数据加载器。
    如果 is_mp（即是否为分布式训练）为True，那么会创建分布式的数据加载器，使用 DistributedSampler 来对数据进行分布式采样，以确保在分布式训练期间每个进程获得不同的数据。
    如果 is_mp 为False，那么将创建普通的数据加载器。

    具体来说，对于训练数据，如果是分布式训练，将使用 DistributedSampler 对数据进行分布式采样，并使用 collate_fn 来处理不同大小的样本，确保它们都能够放入一个批次中。
    而对于验证数据，则不进行数据的shuffle操作，因为在验证中我们不需要打乱数据的顺序，只需要按照给定的顺序处理即可。

    num_workers参数指定了在加载数据时使用的进程数量，batch_size指定了每个批次的大小。train_set 和 val_set 是训练集和验证集，collate_fn 和 collate_fn_val 是用于处理不同大小的样本的函数。
    """
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val,
                                    sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                collate_fn=collate_fn)
        val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
    # build models
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    with open("temp2.json", "r") as json_file:
        knowledge_graph = json.load(json_file)
    scorer = model_external.ReRanker(model_path, tok.pad_token_id, knowledge_graph)
    if len(args.model_pt) > 0:
        scorer.load_state_dict(torch.load(
            os.path.join(
                "/tmp/optimization/cache/{0}--{1}--{2}--{3}".format(args.improve_type, args.model_name, args.dataname,
                                                                    args.real_num),
                args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    if args.cuda:
        if len(args.gpuid) == 1:
            scorer = scorer.cuda()
        else:
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            scorer = nn.parallel.DistributedDataParallel(scorer.to(gpuid), [gpuid], find_unused_parameters=True)
    print("start training")
    scorer.train()
    init_lr = args.max_lr / args.warmup_steps
    s_optimizer = optim.Adam(scorer.parameters(), lr=init_lr)
    if is_master:
        recorder.write_config(args, [scorer], __file__)
    minimum_loss = 100
    all_step_cnt = 0
    # start training
    for epoch in range(args.epoch):
        print(epoch)
        s_optimizer.zero_grad()
        step_cnt = 0
        sim_step = 0
        avg_loss = 0
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            output = scorer(batch["qtn_input_ids"], batch["ctx_input_ids"], batch["candidate_ids"], batch["tgt_input_ids"])
            #print(batch["ctx_input_ids"])
            #print(batch["qtn_input_ids"])
            similarity, gold_similarity, knowledge_score, extra_score, external_knowledge_score = output['score'], output['answer_score'], output['knowledge_score'], output['extra_score'], output['external_knowledge_score']
            loss = args.scale * RankingLoss(similarity, knowledge_score, extra_score, external_knowledge_score, gold_similarity, args.margin, args.gold_margin,
                                            args.gold_weight)
            loss = loss / args.accumulate_step
            avg_loss += loss.item()
            loss.backward()
            if step_cnt == args.accumulate_step:
                # optimize step
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(scorer.parameters(), args.grad_norm)
                step_cnt = 0
                sim_step += 1
                all_step_cnt += 1
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            if sim_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                #print("id: %d" % id)
                print(f"similarity: {similarity[:, :10]}")
                if not args.no_gold:
                    print(f"gold similarity: {gold_similarity}")
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f" % (epoch + 1, sim_step,
                                                                         avg_loss / args.report_freq))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss / args.report_freq}, all_step_cnt)
                recorder.print()
                avg_loss = 0
            del similarity, gold_similarity, knowledge_score, loss

            if all_step_cnt % args.cnt_step == 0 and all_step_cnt != 0 and step_cnt == 0:
                loss = test(val_dataloader, scorer, args, gpuid)
                if loss < minimum_loss and is_master:
                    minimum_loss = loss
                    if is_mp:
                        recorder.save(scorer.module, "scorer.bin")
                    else:
                        recorder.save(scorer, "scorer.bin")
                    recorder.save(s_optimizer, "optimizer.bin")
                    recorder.print("best - epoch: %d, batch: %d" % (epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val rouge: %.6f" % (1 - loss))


def main(args):
    # set env
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Parameter')
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpuid", nargs='+', type=int, default=0)
    parser.add_argument("-e", "--evaluate", type=bool, default=True)
    parser.add_argument("-l", "--log", action="store_true")
    parser.add_argument("-p", "--port", type=int, default=12355)
    parser.add_argument("--model_pt", default="23-08-26-18:38/scorer.bin", type=str)
    #parser.add_argument("--model_pt", default="", type=str)
    parser.add_argument("--encode_mode", default=None, type=str)
    parser.add_argument("--improve_type", default="cra_km2_ba", type=str)
    args = parser.parse_args()
    print('--------------------------')
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
            print(1)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
            print(2)
        else:
            main(args)
            print(3)