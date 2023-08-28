import json
from compare_mt.rouge.rouge_scorer import RougeScorer
from multiprocessing import Pool
import os
import random
from itertools import combinations
from functools import partial
import re
import nltk
import shutil
import argparse

#nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

"""
collect_diverse_beam_data 函数接受一个参数 args，
里面包含了数据集的路径、候选项数目等信息。
函数的主要作用是从数据集中读取数据，并将读取到的数据处理成模型输入格式。
具体而言，它会读取经过分词后的 source 和 target 文件，并读取模型生成的 candidate 文件。
然后，它会将每个 candidate 分别加入候选项列表中。
当候选项列表达到指定的长度时，它会读取对应的 source 和 target 文本，并将它们与候选项列表一起返回。
"""


def collect_diverse_beam_data(args):
    split = os.path.join(args.split)
    src_dir = os.path.join(args.src_dir)
    tgt_dir = os.path.join(args.tgt_dir)
    cands = []
    cands_untok = []
    cnt = 0
    with open(os.path.join(src_dir, f"{split}.question.tokenized")) as qtn, open(
            os.path.join(src_dir, f"{split}.context.tokenized")) as ctx, open(
            os.path.join(src_dir, f"{split}.target.tokenized")) as tgt, open(
            os.path.join(src_dir, f"{split}.question")) as qtn_untok, open(
            os.path.join(src_dir, f"{split}.context")) as ctx_untok, open(
            os.path.join(src_dir, f"{split}.target")) as tgt_untok:
        with open(os.path.join(src_dir, f"{split}.out.tokenized")) as f_1, open(
                os.path.join(src_dir, f"{split}.out")) as f_2:
            for (x, y) in zip(f_1, f_2):
                x = x.strip().lower()
                cands.append(x)
                y = y.strip().lower()
                cands_untok.append(y)
                if len(cands) == args.cand_num:
                    qtn_line = qtn.readline()
                    qtn_line = qtn_line.strip().lower()
                    ctx_line = ctx.readline()
                    ctx_line = ctx_line.strip().lower()


                    tgt_line = tgt.readline()
                    tgt_line = tgt_line.strip().lower()
                    print(3)
                    print(tgt_line)
                    print(4)
                    print(tgt)
                    qtn_line_untok = qtn_untok.readline()
                    qtn_line_untok = qtn_line_untok.strip().lower()
                    ctx_line_untok = ctx_untok.readline()
                    ctx_line_untok = ctx_line_untok.strip().lower()
                    tgt_line_untok = tgt_untok.readline()
                    tgt_line_untok = tgt_line_untok.strip().lower()
                    yield (qtn_line, ctx_line, tgt_line, cands, qtn_line_untok, ctx_line_untok, tgt_line_untok, cands_untok,
                           os.path.join(tgt_dir, f"{cnt}.json"))
                    cands = []
                    cands_untok = []
                    cnt += 1


"""
build_diverse_beam 函数接受一个包含 source 和 target 文本以及候选项的字典作为输入，
并将其转换成模型需要的格式。具体而言，它会将 source、target 和候选项中的文本进行分句，
并计算每个候选项与 target 的 Rouge 分数。最后，它将处理好的数据保存到一个 JSON 文件中。
"""

def build_diverse_beam(input):
    qtn_line, ctx_line, tgt_line, cands, qtn_line_untok, ctx_line_untok, tgt_line_untok, cands_untok, tgt_dir = input
    cands = [sent_detector.tokenize(x) for x in cands]
    answer = sent_detector.tokenize(tgt_line)
    _answer = "\n".join(answer)
    question = sent_detector.tokenize(qtn_line)
    context = sent_detector.tokenize(ctx_line)

    def compute_rouge(hyp):
        score = all_scorer.score(_answer, "\n".join(hyp))
        return (score["rouge1"].fmeasure + score["rouge2"].fmeasure + score["rougeLsum"].fmeasure) / 3

    candidates = [(x, compute_rouge(x)) for x in cands]
    cands_untok = [sent_detector.tokenize(x) for x in cands_untok]
    answer_untok = sent_detector.tokenize(tgt_line_untok)
    question_untok = sent_detector.tokenize(qtn_line_untok)
    context_untok = sent_detector.tokenize(ctx_line_untok)
    candidates_untok = [(cands_untok[i], candidates[i][1]) for i in range(len(candidates))]
    output = {
        "question": question,
        "context": context,
        "answer": answer,
        "candidates": candidates,
        "question_untok": question_untok,
        "context_untok": context_untok,
        "answer_untok": answer_untok,
        "candidates_untok": candidates_untok,
    }
    with open(tgt_dir, "w") as f:
        json.dump(output, f)


def make_diverse_beam_data(args):
    data = collect_diverse_beam_data(args)
    with Pool(processes=8) as pool:
        list(pool.imap_unordered(build_diverse_beam, data, chunksize=64))
    print("finish")

def createdir(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
def deletedir(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

def pre(tgt_dir):
    deletedir("./{}".format(tgt_dir))

    createdir("./{}".format(tgt_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing Parameter')
    parser.add_argument("--cand_num", type=int, default=16)
    parser.add_argument("--src_dir", type=str)
    parser.add_argument("--tgt_dir", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    pre(args.tgt_dir)
    make_diverse_beam_data(args)






