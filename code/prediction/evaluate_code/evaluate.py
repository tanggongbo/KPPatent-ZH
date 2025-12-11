import json
import os
import torch
import numpy as np
from semf1_metrics_realK import SemanticMatchingMetric
from f1_metrics_realK import compute_macro_avg_f1_metrics


def calculate_metrics(all_present, all_absent, all_predicted):
    """计算语义匹配和F1指标"""
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = SemanticMatchingMetric(device=device)
    
    sem_scores = metric.score_corpus(
        all_present,  # 保持原始格式
        all_absent,   # 保持原始格式
        all_predicted # 保持原始格式
    )
    
    # 计算F1指标
    f1_scores = compute_macro_avg_f1_metrics(
        all_present, all_absent, all_predicted,
        match_type='subword'
    )

    # 将结果转换为numpy数组，添加空值检查
    def safe_mean(arr):
        return np.mean(arr) if len(arr) > 0 else 0.0

    # 将结果转换为numpy数组
    return {
        'present': {
            'semP': safe_mean(sem_scores['present']['semantic_p']),
            'semR': safe_mean(sem_scores['present']['semantic_r']),
            'semF1': safe_mean(sem_scores['present']['semantic_f1']),
            'F1@5': f1_scores['present']['F1@5'],
            'F1@10': f1_scores['present']['F1@10'],
            'F1@15': f1_scores['present']['F1@15'],
            'F1@M': f1_scores['present']['F1@M']
        },
        'absent': {
            'semP': safe_mean(sem_scores['absent']['semantic_p']),
            'semR': safe_mean(sem_scores['absent']['semantic_r']),
            'semF1': safe_mean(sem_scores['absent']['semantic_f1']),
            'F1@5': f1_scores['absent']['F1@5'],
            'F1@10': f1_scores['absent']['F1@10'],
            'F1@15': f1_scores['absent']['F1@15'],
            'F1@M': f1_scores['absent']['F1@M']
        }
    }

def process_keyword_files(base_dir, max_lines=696):
    """处理关键词文件并输出结果，根据目录类型自动选择读取方式"""
    # 判断目录类型
    if 'api' in base_dir:
        # API模式的文件处理
        sections = ['A', 'C', 'TAC']
        methods = ['uke', 'ukg']
        models = ['deepseek-r1', 'deepseek-v3', 'gemini-2.0-pro-exp', 'gpt-4o-2024-11-20', 'gpt-4o-mini']
        
        output_file = os.path.join(base_dir, '_results_stepnull_realK_API.txt')
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for section in sections:
                for method in methods:
                    for model in models:
                        filename = f"{section}_{method}_log.keyLLM_{model}.jsonl"
                        filepath = os.path.join(base_dir, filename)
                        
                        if not os.path.exists(filepath):
                            print(f"文件 {filepath} 不存在，跳过")
                            continue
                        
                        all_present = []
                        all_absent = []
                        all_predicted = []
                        
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:max_lines]
                            
                            for line in lines:
                                if not line.strip() or line.startswith('---'):
                                    continue
                                try:
                                    data = json.loads(line)
                                    all_predicted.append(data['predicted_keywords'])
                                    all_present.append(data['present_keys'])
                                    all_absent.append(data['absent_keys'])
                                except Exception as e:
                                    print(f"解析文件 {filepath} 中的行 {line} 时出错: {e}")
                                    continue
                        
                        if not all_present:  # 跳过空文件
                            print(f"文件 {filepath} 内容为空，跳过")
                            continue
                            
                        model_results = calculate_metrics(all_present, all_absent, all_predicted)
                        
                        # 输出结果
                        out_f.write(f"\n=== {section} ===\n")
                        out_f.write(f"Method: {method}\n")
                        out_f.write(f"Model: {model}\n\n")
                        
                        out_f.write("Present Keys:\n")
                        out_f.write("semP\tsemR\tsemF1\tF1@5\tF1@10\tF1@15\tF1@M\n")
                        out_f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n\n".format(
                            model_results['present']['semP'],
                            model_results['present']['semR'],
                            model_results['present']['semF1'],
                            model_results['present']['F1@5'],
                            model_results['present']['F1@10'],
                            model_results['present']['F1@15'],
                            model_results['present']['F1@M']
                        ))
                        
                        out_f.write("Absent Keys:\n")
                        out_f.write("semP\tsemR\tsemF1\tF1@5\tF1@10\tF1@15\tF1@M\n")
                        out_f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(
                            model_results['absent']['semP'],
                            model_results['absent']['semR'],
                            model_results['absent']['semF1'],
                            model_results['absent']['F1@5'],
                            model_results['absent']['F1@10'],
                            model_results['absent']['F1@15'],
                            model_results['absent']['F1@M']
                        ))           

    elif 'pke_zh' in base_dir:
        # pke_zh模式的文件处理
        sections = ['A', 'C', 'TAC']
        methods = ['uke']
        models = ['tf_idf_pke_zh', 'text_rank_pke_zh', 'single_rank_pke_zh', 'position_rank_pke_zh', 'topic_rank_pke_zh', 'multipartite_rank_pke_zh', 'yake_pke_zh', 'keyBert_pke_zh']
        
        output_file = os.path.join(base_dir, '_results_stepnull_realK_pke.txt')
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for section in sections:
                for method in methods:
                    for model in models:
                        filename = f"{section}_{method}_log.{model}.jsonl"
                        filepath = os.path.join(base_dir, filename)
                        
                        if not os.path.exists(filepath):
                            continue
                        
                        all_present = []
                        all_absent = []
                        all_predicted = []
                        
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:max_lines]
                            
                            for line in lines:
                                if not line.strip() or line.startswith('---'):
                                    continue
                                try:
                                    data = json.loads(line)
                                    all_predicted.append(data['predicted_keywords'])
                                    all_present.append(data['present_keys'])
                                    all_absent.append(data['absent_keys'])
                                except:
                                    continue
                        
                        if not all_present:  # 跳过空文件
                            continue
                            
                        model_results = calculate_metrics(all_present, all_absent, all_predicted)
                        
                        # 输出结果
                        out_f.write(f"\n=== {section} ===\n")
                        out_f.write(f"Method: {method}\n")
                        out_f.write(f"Model: {model}\n\n")
                        
                        out_f.write("Present Keys:\n")
                        out_f.write("semP\tsemR\tsemF1\tF1@5\tF1@10\tF1@15\tF1@M\n")
                        out_f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n\n".format(
                            model_results['present']['semP'],
                            model_results['present']['semR'],
                            model_results['present']['semF1'],
                            model_results['present']['F1@5'],
                            model_results['present']['F1@10'],
                            model_results['present']['F1@15'],
                            model_results['present']['F1@M']
                        ))
                        
                        out_f.write("Absent Keys:\n")
                        out_f.write("semP\tsemR\tsemF1\tF1@5\tF1@10\tF1@15\tF1@M\n")
                        out_f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(
                            model_results['absent']['semP'],
                            model_results['absent']['semR'],
                            model_results['absent']['semF1'],
                            model_results['absent']['F1@5'],
                            model_results['absent']['F1@10'],
                            model_results['absent']['F1@15'],
                            model_results['absent']['F1@M']
                        ))
    
    elif 'kaiyuan' in base_dir:
        # 开源模式的文件处理
        sections = ['a', 'c', 'tac']
        types = ['base', 'lora']
        models = ['gemma-3-4b-it', 'gemma-3-12b-it', 'mozi', 'mozi3-7b', 'qwen3', 'Qwen3-8B']
        
        output_file = os.path.join(base_dir, 'all_results_Kaiyuan.txt')
        
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for section in sections:
                for model in models:
                    for type_ in types:
                        filename = f"prediction_test_{model}_{section}_{type_}_keywords.jsonl"
                        filepath = os.path.join(base_dir, filename)
                        
                        if not os.path.exists(filepath):
                            print(f"文件 {filepath} 不存在，跳过")
                            continue
                        
                        all_present = []
                        all_absent = []
                        all_predicted = []
                        
                        with open(filepath, 'r', encoding='utf-8') as f:
                            lines = f.readlines()[:max_lines]
                            
                            for line in lines:
                                if not line.strip() or line.startswith('---'):
                                    continue
                                try:
                                    data = json.loads(line)
                                    all_predicted.append(data['predicted_keywords'])
                                    all_present.append(data['present_keys'])
                                    all_absent.append(data['absent_keys'])
                                except Exception as e:
                                    print(f"解析文件 {filepath} 中的行时出错: {e}")
                                    continue
                        
                        if not all_present:  # 跳过空文件
                            print(f"文件 {filepath} 内容为空，跳过")
                            continue
                        
                        model_results = calculate_metrics(all_present, all_absent, all_predicted)
                        
                        # 输出结果
                        out_f.write(f"\n=== {section.upper()} ===\n")  # 转换为大写保持格式一致
                        out_f.write(f"Type: {type_}\n")
                        out_f.write(f"Model: {model}\n\n")
                        
                        out_f.write("Present Keys:\n")
                        out_f.write("semP\tsemR\tsemF1\tF1@5\tF1@10\tF1@15\tF1@M\n")
                        out_f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n\n".format(
                            model_results['present']['semP'],
                            model_results['present']['semR'],
                            model_results['present']['semF1'],
                            model_results['present']['F1@5'],
                            model_results['present']['F1@10'],
                            model_results['present']['F1@15'],
                            model_results['present']['F1@M']
                        ))
                        
                        out_f.write("Absent Keys:\n")
                        out_f.write("semP\tsemR\tsemF1\tF1@5\tF1@10\tF1@15\tF1@M\n")
                        out_f.write("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(
                            model_results['absent']['semP'],
                            model_results['absent']['semR'],
                            model_results['absent']['semF1'],
                            model_results['absent']['F1@5'],
                            model_results['absent']['F1@10'],
                            model_results['absent']['F1@15'],
                            model_results['absent']['F1@M']
                        ))
                        
    else:
        raise ValueError(f"未知的目录类型: {base_dir}")

if __name__ == '__main__':
    # 初始化CUDA
    if torch.cuda.is_available():
        torch.cuda.init()
    
    evaluation_dir = '../evaluation'
    
    # 明确指定要处理的两个文件夹
    target_dirs = ['api', 'pke_zh', 'kaiyuan']
    
    for dir_name in target_dirs:
        dir_path = os.path.join(evaluation_dir, dir_name)
        if os.path.isdir(dir_path):
            print(f"\n正在处理目录: {dir_path}")
            try:
                process_keyword_files(dir_path)
                print(f"成功处理: {dir_path}")
            except Exception as e:
                print(f"处理目录 {dir_path} 时出错: {str(e)}")
        else:
            print(f"警告: 目录 {dir_path} 不存在，跳过")