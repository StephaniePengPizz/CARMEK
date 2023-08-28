from bioasq_preprocess import pro, sample
from datasets import Dataset, DatasetDict
import torch
import os
import json

def build_dataset_for_bio():
    torch.cuda.empty_cache()
    #print(1)
    raw1, raw2 = pro(1)
    raw1 = [item for item in raw1 if item["question"] is not None and item["context"] is not None and item["answer"] is not None]
    raw1 = [item for item in raw1 if item["question"][0].strip() != "" and item["context"][0]!= "" and item["answer"][0].strip()!= ""]

    questions = [item["question"] for item in raw1]
    contexts = [item["context"] for item in raw1]
    answers = [item["answer"] for item in raw1]
    for i in range(len(answers)):
        temp = ""
        if len(answers[i]) > 1:
            for sentence in answers[i]:
                temp += sentence
            answers[i] = [temp]
        #print(answers[i])
    print("bb", len(answers))
    #for i in range(len(answers)):
    #    if len(answers[i]) > 1:
            #print(answers[i])
    train_data = {
        "question": questions,
        "context": contexts,
        "answer": answers
    }

    raw2 = [item for item in raw2 if item["question"] is not None and item["context"] is not None and item["answer"] is not None]
    raw2 = [item for item in raw2 if item["question"][0].strip()!= "" and item["context"][0]!= "" and item["answer"][0].strip() != ""]

    questions = [item["question"] for item in raw2]
    contexts = [item["context"] for item in raw2]
    answers = [item["answer"] for item in raw2]

    for i in range(len(answers)):
        temp = ""
        if len(answers[i]) > 1:
            for sentence in answers[i]:
                temp += sentence
            answers[i] = [temp]

    test_data = {
        "question": questions,
        "context": contexts,
        "answer": answers
    }

    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)

    datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})
    #print(datasets["train"][0])
    return datasets

def build_dataset_for_pubmedqa():

    torch.cuda.empty_cache()

    questions_train, contexts_train, answers_train = [], [], []
    questions_val, contexts_val, answers_val = [], [], []
    questions_test, contexts_test, answers_test = [], [], []
    # 读取train_JSON文件
    for root, dirs, files in os.walk("/root/autodl-tmp/pubmedqa-master/data"):
        for directory in dirs:
            folder = os.path.join(root, directory)
            for file in os.listdir(folder):
                if 'train' in file:
                    file_path = os.path.join(folder, file)
                    with open(file_path, 'r') as f:
                        train_dataset = json.load(f)
                    questions, answers, contexts = load_data(train_dataset)
                    for item1, item2, item3 in zip(questions, answers, contexts):
                        questions_train.append(item1)
                        answers_train.append([item2])
                        contexts_train.append(item3)
                if 'dev' in file:
                    file_path = os.path.join(folder, file)
                    with open(file_path, 'r') as f:
                        val_dataset = json.load(f)
                    questions, answers, contexts = load_data(val_dataset)
                    for item1, item2, item3 in zip(questions, answers, contexts):
                        questions_val.append(item1)
                        answers_val.append([item2])
                        contexts_val.append(item3)



    # 读取test_JSON文件
    with open("/root/autodl-tmp/pubmedqa-master/data/test_set.json", "r") as f:
        test_dataset = json.load(f)
    questions, answers, contexts = load_data(test_dataset)
    for item1, item2, item3 in zip(questions, answers, contexts):
        questions_test.append(item1)
        answers_test.append([item2])
        contexts_test.append(item3)
    print("aa", answers_train[500])
    train_data = {
        "question": questions_train,
        "context": contexts_train,
        "answer": answers_train
    }
    val_data = {
        "question": questions_val,
        "context": contexts_val,
        "answer": answers_val
    }
    test_data = {
        "question": questions_test,
        "context": contexts_test,
        "answer": answers_test
    }
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)
    #print(train_dataset)
    datasets = DatasetDict({'train': train_dataset, 'val': val_dataset, 'test': test_dataset})
    print(datasets["train"][4300])
    return datasets

def load_data(dataset):
    j = 0
    questions = []
    answers = []
    contexts = []
    for data_id, data in dataset.items():
        j += 1
        context_original = data["CONTEXTS"]
        # 将所有上下文合并为一个字符串
        context = " ".join(context_original)
        questions.append(data["QUESTION"])
        answers.append(data["LONG_ANSWER"])
        contexts.append(context)
    # print(datasets["train2"][0])
    return questions, answers, contexts
