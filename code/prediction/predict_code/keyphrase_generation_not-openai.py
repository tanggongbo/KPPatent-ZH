import json
import os
import openai
from keybert.llm import OpenAI
from keybert import KeyLLM
from semf1_metrics import SemanticMatchingMetric
from f1_metrics import compute_macro_avg_f1_metrics
import numpy as np
import time
from tenacity import retry, stop_after_attempt, wait_exponential

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

def extract_keywords_from_document(client, model, generator_kwargs, document):
    try:
        prompt = f"""
        你是一个关键短语生成器。请从文档中生成50个最能描述文档主题的关键短语，必须严格遵守以下要求：
        1. 关键短语之间用逗号分隔
        2. 严格避免重复或相似的关键短语
        3. 按重要性从高到低排序
        4. 关键词不能为空格
    
        文档：{document}
        """
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **generator_kwargs
        )
        result = response.choices[0].message.content.strip()
        return [kw.strip() for kw in result.split(",") if kw.strip()]
    except Exception as e:
        print(f"文档处理失败，错误：{e}")
        return []

def keyLLM_pke_zh(test_data, model_name):
    
    model = model_name.replace('keyLLM_', '')
    
    test_texts = [instance['patent_text'] for instance in test_data]
    
    client = openai.OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='your_api_key_here'  # 替换为你的API密钥
    )
    generator_kwargs = {
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1500,
        "top_p": 1
    }

    prompt = """
        你是一个关键短语生成器。请从文档中生成50个最能描述文档主题的关键短语，必须严格遵守以下要求：
        1. 关键短语之间用逗号分隔
        2. 严格避免重复或相似的关键短语
        3. 按重要性从高到低排序
        4. 关键词不能为空格
    
        文档：document
        """
   
    filtered_keywords = []
    sensitive_texts = []

    for i, text in enumerate(test_texts):
        try:
            # 单独提取每一条文本的关键词
            keywords = extract_keywords_from_document(client, model, generator_kwargs, text)
            # keywords = KeyLLM_m.extract_keywords([text])[0]  # 返回的是列表中第一个元素

            # 过滤规则
            filtered_sublist = [
                kw.replace('.', '') for kw in keywords
                if not any(c.isdigit() for c in kw)
                and not any(c.isalpha() and c.isascii() for c in kw)
                and not any('\u0370' <= c <= '\u03FF' for c in kw)
                and not any(c in '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~' for c in kw)
                and "第一" not in kw
                and "第二" not in kw
                and "第三" not in kw
            ]
            filtered_keywords.append(filtered_sublist)

        except Exception as e:
            # 如果报错信息中包含“敏感词”相关提示
            if "sensitive" in str(e).lower():
                print(f"[!] 第 {i} 条文本触发敏感词，跳过")
                sensitive_texts.append((i, text))
            else:
                print(f"[!] 第 {i} 条文本发生其他错误：{e}")
                # 可以选择 raise e 或继续跳过
                continue
    return filtered_keywords, sensitive_texts, prompt, generator_kwargs

def get_data_sets(patent_text_class):
    test_data_file = r'../data/test.json'
    try:
        # 先尝试作为完整JSON文件读取
        with open(test_data_file, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        test_data = []
        for key, data in data_dict.items():
            instance = process_data(data, 'test', patent_text_class)
            test_data.append(instance)
        # test_data = test_data[:5]
        test_data = test_data
    except Exception as e:
        print(f"读取文件出错: {e}")
        test_data = []
    
    return test_data, test_data, test_data

def train_models(model_name, training_data, dev_data):
    if model_name.startswith("keyLLM_\n") :
        print(f"[Info] 模型 {model_name} 是基于 LLM 的方法，无需训练。\n") 
    return None

def predict_keyphrases(model_name, test_data):
    if model_name.startswith('keyLLM_'):
        # model = model_name.replace('keyLLM_', '')  # 提取真实模型名
        keywords, sensitive_texts, prompt, kwargs = keyLLM_pke_zh(test_data, model_name)
    return keywords, sensitive_texts, prompt, kwargs    

UNSUPERVISED_EXTRACT_LIST = ['keyLLM_deepseek-r1', 'keyLLM_deepseek-v3', 'keyLLM_gemini-2.0-pro-exp']

SUPERVISED_EXTRACT_LIST = []
UNSUPERVISED_PREDICT_LIST = []
SUPERVISED_PREDICT_LIST = []

METHOD_TYPES = {
    "unsupervised_extract": UNSUPERVISED_EXTRACT_LIST, 
    "supervised_extract": SUPERVISED_EXTRACT_LIST, 
    "unsupervised_predict": UNSUPERVISED_PREDICT_LIST, 
    "supervised_predict": SUPERVISED_PREDICT_LIST
}

def main():
    for patent_text_class in ['C', 'A', 'TAC']:
        print(f"开始处理专利类别: {patent_text_class}")
        
        training_data, dev_data, test_data = get_data_sets(patent_text_class)
        print(f"测试数据数量: {len(test_data)}")
        
        method_name = 'ukg'  # 注意这里改为ukg

        for METHOD_TYPE, METHOD_LIST in METHOD_TYPES.items():
            if not METHOD_LIST:
                continue
                
            print(f"处理方法类型: {METHOD_TYPE}")
            print(f"模型列表: {METHOD_LIST}")
            
            for model_name in METHOD_LIST:
                print(f"处理模型: {model_name}")
                
                try:
                    train_models(model_name, training_data, dev_data)
                    
                    keywords, sensitive_texts, prompt, kwargs = predict_keyphrases(model_name, test_data)
                    
                    output_dir = f"./evaluation/api/"
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    base_model_name = model_name.replace('keyLLM_', '')
                    output_file = output_dir / f"{patent_text_class}_{method_name}_log.keyLLM_{base_model_name}.jsonl"
                    
                    print(f"正在生成文件: {output_file}")
                    
                    records_written = 0
                    with open(output_file, 'w', encoding='utf-8') as file:
                        for i, result in enumerate(keywords):
                            if isinstance(result, str):
                                result = [kw.strip() for kw in result.split(',') if kw.strip()]
                            elif not isinstance(result, list):
                                result = list(result) if result else []
                            
                            present_keys = []
                            absent_keys = []
                            
                            if i < len(test_data):
                                if 'present_keys' in test_data[i]:
                                    present_keys = test_data[i]['present_keys']
                                elif 'present_keys' in test_data[i].get('keywords', {}):
                                    present_keys = test_data[i]['keywords']['present_keys']
                                
                                if 'absent_keys' in test_data[i]:
                                    absent_keys = test_data[i]['absent_keys']
                                elif 'absent_keys' in test_data[i].get('keywords', {}):
                                    absent_keys = test_data[i]['keywords']['absent_keys']
                            
                            json_line = {
                                "predicted_keywords": result,
                                "present_keys": present_keys,
                                "absent_keys": absent_keys
                            }
                            
                            json_str = json.dumps(json_line, ensure_ascii=False)
                            file.write(json_str + '\n')
                            records_written += 1
                    
                    print(f"成功写入 {records_written} 条记录到 {output_file}")
                    
                except Exception as e:
                    print(f"处理模型 {model_name} 时出现错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print("JSONL文件生成完成！")
    print("所有文件已保存到: ./evaluation/api/")

if __name__ == '__main__':
    main()