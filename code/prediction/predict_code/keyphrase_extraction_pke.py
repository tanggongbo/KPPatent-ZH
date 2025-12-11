import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM
from semf1_metrics import SemanticMatchingMetric
from f1_metrics import compute_macro_avg_f1_metrics
import numpy as np
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from pke_zh import TextRank, TfIdf, SingleRank, PositionRank, TopicRank, MultipartiteRank, Yake, KeyBert

'''
apply different methods to predict the keyphrases of the patent data

patent data origins has multiple classes, with different patent_text sections
'''
# patent text from different sections
CLASSES = {'TACD':['title', 'abstract', 'claims', 'description'],
           'TAC':['title', 'abstract', 'claims'],
           'TA':['title', 'abstract'],
           'TC':['title', 'claims'],
           'AC':['abstract', 'claims'],
           'T':['title'],
           'A':['abstract'],
           'C':['claims'],
           'D':['description']
           }

METHOD_LIST = ['tf_idf', 'text_rank']


def process_data(data, src, patent_text_class):
    # 处理关键词
    keywords = data['keywords']

    data = data['body']
    title, abstract = data['title'], data['abstract']
    if src == 'test':
        claims = "".join(data['claims'])
        description = "".join(list(data['description'].values())[:4]) + \
                      "".join(list(data['description'].values())[4])
    else:
        claims = "".join(data['claims']['items'])
        description = "".join(list(data['description'].values()))
    
    
    patent_text = ""
    for section in CLASSES[patent_text_class]:
        section_text = f'{locals()[section]}'
        patent_text += section_text

    patent_text = patent_text.replace("\n", "\n") 
    # file.write(patent_text)
    present_keys, absent_keys = [], []
    for word in keywords:
        if patent_text.__contains__(word):
            present_keys.append(word)
        else:
            absent_keys.append(word)
    instance = {'patent_text': patent_text, 'present_keys': present_keys, 'absent_keys': absent_keys}

    return instance

'''
get data from processed jsonl files, based on the sections of the patent text
'''
def get_data(input_file, patent_text_class):    
    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data_src = 'test'
    instances = []
    for i, line in enumerate(lines):
        data = json.loads(line)
        instance = process_data(data, data_src, patent_text_class)
        
        instances.append(instance)
        
    return instances


'''
get data from processed jsonl files, based on the sections of the patent text
'''
def get_test_data(input_file, patent_text_class):    
    # 读取JSONL文件
    with open(input_file, 'r', encoding='utf-8') as file:
        data_dict = json.load(file)
    
    data_src = 'test'
    instances = []
    for key, data in data_dict.items():
        # 处理关键词
        instance = process_data(data, data_src, patent_text_class)
        instances.append(instance)
        
    return instances
# from pke_zh import TextRank, TfIdf, SingleRank, PositionRank, TopicRank, MultipartiteRank, Yake, KeyBert
def tf_idf_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    tfidf_m = TfIdf()
    results = []
    for test_text in test_texts:
        keywords = tfidf_m.extract(test_text, n_best=50)
        res = [kw[0] for kw in keywords]
        results.append(res)    
    
    return results

def text_rank_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    text_rank_m = TextRank()
    results = []
    for test_text in test_texts:
        keywords = text_rank_m.extract(test_text, n_best=50)
        res = [kw[0] for kw in keywords]
        results.append(res)
    
    return results

def single_rank_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    single_rank_m = SingleRank()
    results = []
    for test_text in test_texts:
        keywords = single_rank_m.extract(test_text, n_best=50)
        res = [kw[0] for kw in keywords]
        results.append(res)
    
    return results

def position_rank_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    position_rank_m = PositionRank()
    results = []
    for test_text in test_texts:
        keywords = position_rank_m.extract(test_text, n_best=50)
        res = [kw[0] for kw in keywords]
        results.append(res)
    
    return results

def topic_rank_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    topic_rank_m = TopicRank()
    results = []
    for test_text in test_texts:
        keywords = topic_rank_m.extract(test_text, n_best=50)
        res = [kw[0] for kw in keywords]
        results.append(res)        
    
    return results



def multipartite_rank_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    multipartite_rank_m = MultipartiteRank()
    results = []
    for test_text in test_texts:
        keywords = multipartite_rank_m.extract(test_text, n_best=50)
        res = [kw[0] for kw in keywords]
        results.append(res)
    
    return results

def yake_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    yake_m = Yake()
    results = []
    for test_text in test_texts:
        keywords = yake_m.extract(test_text, n_best=50)
        res = [kw[0] for kw in keywords]
        results.append(res)
    
    return results

def keyBert_pke_zh(test_data):
    # 提取测试数据中的文本和关键词
    test_texts = [instance['patent_text'] for instance in test_data]

    KeyBert_m = KeyBert()
    results = []
    # return results
    for test_text in test_texts:
        keywords = KeyBert_m.extract(test_text, n_best=15)
        res = [kw[0] for kw in keywords]
        results.append(res)
    
    return results

def get_data_sets(section):
    test_data_file = r'../data/test.json'
    try:
        # 先尝试作为完整JSON文件读取
        with open(test_data_file, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        test_data = []
        for key, data in data_dict.items():
            instance = process_data(data, 'test', section)
            test_data.append(instance)
        # test_data = test_data[:1]
    except Exception as e:
        print(f"读取文件出错: {e}")
        test_data = []
    
    return test_data, test_data, test_data

def train_models(model_name, training_data, dev_data):
    if model_name.startswith("keyLLM_\n") :
        print(f"[Info] 模型 {model_name} 是基于 LLM 的方法，无需训练。\n") 
    return None

def predict_keyphrases(model_name, test_data):
    if model_name == 'tf_idf_pke_zh':
        results = tf_idf_pke_zh(test_data)
    elif model_name == 'text_rank_pke_zh':
        results = text_rank_pke_zh(test_data)
    elif model_name == 'single_rank_pke_zh':
        results = single_rank_pke_zh(test_data)
    elif model_name == 'position_rank_pke_zh':
        results = position_rank_pke_zh(test_data)
    elif model_name == 'topic_rank_pke_zh':
        results = topic_rank_pke_zh(test_data)
    elif model_name == 'multipartite_rank_pke_zh':
        results = multipartite_rank_pke_zh(test_data)
    elif model_name == 'yake_pke_zh':
        results = yake_pke_zh(test_data)
    elif model_name == 'keyBert_pke_zh':
        results = keyBert_pke_zh(test_data)
        
    return results   

UNSUPERVISED_EXTRACT_LIST = ['tf_idf_pke_zh', 'text_rank_pke_zh', 'single_rank_pke_zh', 'position_rank_pke_zh', 'topic_rank_pke_zh', 'multipartite_rank_pke_zh', 'yake_pke_zh', 'keyBert_pke_zh']
SUPERVISED_EXTRACT_LIST = []
UNSUPERVISED_PREDICT_LIST = []
SUPERVISED_PREDICT_LIST = []
METHOD_TYPES = {"unsupervised_extract": UNSUPERVISED_EXTRACT_LIST, 
                "supervised_extract": SUPERVISED_EXTRACT_LIST, 
                "unsupervised_predict": UNSUPERVISED_PREDICT_LIST, 
                "supervised_predict": SUPERVISED_PREDICT_LIST}

def main():
    sections = ['A', 'C', 'TAC']
    
    for section in sections:
        # get data
        training_data, dev_data, test_data = get_data_sets(section)

        for METHOD_TYPE, METHOD_LIST in METHOD_TYPES.items():
            for model_name in METHOD_LIST:
                # train the models
                train_models(model_name, training_data, dev_data)
            
                # predict the keyphrases of the test data
                keywords = predict_keyphrases(model_name, test_data)
                
                output_file = f"./evaluation/pke_zh/{section}_uke_log.{model_name}.jsonl"
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                print(f"[Info] {output_file}目录已建立。\n") 
                
                all_present_keys = []
                all_absent_keys = []
                all_predicted_keys = []
                
                
                with open(output_file, 'w', encoding='utf-8') as file:
                    for i, result in enumerate(keywords):
                        all_predicted_keys.append(result)
                        all_present_keys.append(test_data[i]['present_keys'])
                        all_absent_keys.append(test_data[i]['absent_keys'])
                        
                        json_str = json.dumps({
                            "predicted_keywords": result,
                            "present_keys": test_data[i]['present_keys'],
                            "absent_keys": test_data[i]['absent_keys'],
                        }, ensure_ascii=False)
                        file.write(json_str + '\n')
                        

if __name__ == '__main__':
    main()
