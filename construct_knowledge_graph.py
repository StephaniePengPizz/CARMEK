import re
from concept import get_related_concepts
from bioasq_preprocess import pro
import json
from pubmedqa_preprocess import bio2
dataset = "pubmedqa"
data1, data2, data3 = pro(2)
knowledge_graph = {}
json_filename = "knowledge_graph.json"
data4 = bio2()
def stage1():
    i = 0
    for item in data3:
        i = i + 1
        context = item['context']
        print(i)
        word_list = re.findall(r'\b\w+\b', context)
        single_letter_words = [word for word in word_list if len(word) > 1]
        # 循环查询每个单词的相关概念
        for word in single_letter_words:
            related_concepts = get_related_concepts(word, knowledge_graph)
            if related_concepts == [] or related_concepts == {}:
                continue
            knowledge_graph[word] = related_concepts

        if i % 5 == 0:
            with open(json_filename, "w") as json_file:
                json.dump(knowledge_graph, json_file, indent=4)
            print(knowledge_graph)

def stagetemp():
    with open(json_filename, "r") as json_file:
        loaded_data = json.load(json_file)

    # 利用读取到的数据
    for concept, info in loaded_data.items():
        knowledge_graph.update(info)
    with open("temp.json", "w") as json_file:
        json.dump(knowledge_graph, json_file, indent=4)
    print(knowledge_graph)

def stage2():
    with open("temp_bioasq.json", "r") as json_file:
        knowledge_graph = json.load(json_file)
    #knowledge_graph = {}
    #i = 555
    for (i, item) in enumerate(data3):
        context = item['context']
        print(i)
        word_list = re.findall(r'\b\w+\b', context)
        single_letter_words = [word for word in word_list if len(word) > 1]
        # 循环查询每个单词的相关概念
        for word in single_letter_words:
            knowledge_graph = get_related_concepts(word, knowledge_graph)
            #print(len(knowledge_graph))
            #print(knowledge_graph)

        if i % 5 == 0:
            with open("temp2.json", "w") as json_file:
                json.dump(knowledge_graph, json_file, indent=4)
        print(knowledge_graph)

def stage_pub():
    #with open("knowledge_graph_pubmedqa.json", "r") as json_file:
    #    knowledge_graph = json.load(json_file)
    knowledge_graph = {}
    for (i, j) in enumerate(data4):
        print(i)
        context = j[0]
        word_list = re.findall(r'\b\w+\b', context)
        single_letter_words = [word for word in word_list if len(word) > 1]
        # 循环查询每个单词的相关概念
        for word in single_letter_words:
            knowledge_graph = get_related_concepts(word, knowledge_graph)
            # print(len(knowledge_graph))
            # print(knowledge_graph)

        if i % 5 == 0:
            with open("pubmedqa_graph.json", "w") as json_file:
                json.dump(knowledge_graph, json_file, indent=4)
        print(knowledge_graph)

if dataset == "bioasq":
    stage2()
else:
    stage_pub()
