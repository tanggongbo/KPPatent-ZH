import json
from openai import OpenAI
import os
import time
import re
from collections import defaultdict

# ------------------ 多选题回答函数 ------------------
def answer_mcq(client, model, question, max_retries=3, retry_delay=1):
    """
    回答多选题（MCQ）
    question: 题干+专利文本+选项 (str)
    max_retries: 最大重试次数
    retry_delay: 重试延迟（秒）
    return: 模型选的选项组合 (如'AB', 'ACD'等)，失败返回None
    """
    formatted_question = f"题目：{question}"
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专利文本阅读理解助手。请回答以下多选题，从选项中选择所有正确的关键词，并严格只输出正确选项的字母组合（如'AB'、'ACD'），不要输出其他内容。"},
                    {"role": "user", "content": formatted_question}
                ],
                temperature=0,
                seed=42,
                top_p=1
            )
            result = completion.choices[0].message.content.strip().upper()
            
            # 检查结果是否有效（只包含A,B,C,D的字母组合）
            if result and all(char in "ABCD" for char in result) and len(result) <= 4:
                # 去重并排序
                sorted_result = ''.join(sorted(set(result)))
                return sorted_result
            else:
                print(f"第 {attempt+1} 次尝试返回无效答案: '{result}'")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                
        except Exception as e:
            print(f"第 {attempt+1} 次尝试失败: {e}")
            if attempt < max_retries - 1:
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
    
    print(f"经过 {max_retries} 次尝试后仍然失败")
    return None

# ------------------ LLM_mcq_predict 函数 ------------------
def LLM_mcq_predict(test_data, model_name, max_retries=3):
    """
    使用 LLM 对多选题数据集进行预测
    test_data: list of dict, 每个 dict 包含 "question" 字段
    model_name: str, 模型名称
    max_retries: 最大重试次数
    return: (predictions, failed_count)
    """
    
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='sk-KFGXGtYwV6ghZoVZNJvt9PdMlE72IaptRxsuQXt0pFRodxpa'
    )

    predictions = []
    failed_count = 0

    for i, item in enumerate(test_data):
        question_text = item['question']
        try:
            pred = answer_mcq(client, model_name, question_text, max_retries=max_retries)
            if pred is None:
                failed_count += 1
                print(f"[!] 第 {i+1} 条题目预测失败: 经过 {max_retries} 次尝试后模型返回无效答案")
            predictions.append(pred)
            print(f"第 {i+1}/{len(test_data)} 题预测完成: {pred}")
        except Exception as e:
            print(f"[!] 第 {i+1} 条题目预测失败: {e}")
            predictions.append(None)
            failed_count += 1

    return predictions, failed_count

# ------------------ 读取多选题数据 ------------------
def load_mcq_dataset(file_path):
    """
    读取多选题数据集
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"成功读取 {len(data)} 道多选题")
    return data

# ------------------ 计算多选题指标 ------------------
def calculate_mcq_metrics(test_data, predictions):
    """
    计算多选题的两种指标：
    1. Exact Match (严格匹配率)
    2. 宏平均的 Precision / Recall / F1
    """
    exact_match = 0  # 严格匹配的题目数
    total_valid = 0  # 有效预测数量
    
    # 用于宏平均计算的累加器
    macro_precision_sum = 0
    macro_recall_sum = 0
    macro_f1_sum = 0
    
    # 用于微平均计算的累加器
    micro_tp = 0  # 真正例总数
    micro_fp = 0  # 假正例总数
    micro_fn = 0  # 假反例总数
    
    # 错误分析
    error_types = {
        'correct': 0,           # 完全正确
        'partial_correct': 0,   # 部分正确（选了部分正确答案，没有选错误答案）
        'partial_wrong': 0,     # 部分错误（选了部分正确答案，但也选了错误答案）
        'completely_wrong': 0,  # 完全错误（一个正确答案都没选）
        'over_selected': 0      # 多选了错误答案
    }
    
    for i, (item, pred) in enumerate(zip(test_data, predictions)):
        if pred is not None:
            total_valid += 1
            
            standard_answer = set(item['answer'])
            predicted_answer = set(pred) if pred else set()
            
            # 1. 计算 Exact Match
            if standard_answer == predicted_answer:
                exact_match += 1
                error_types['correct'] += 1
            else:
                # 错误类型分析
                correct_selected = len(standard_answer & predicted_answer)  # 正确选中的数量
                wrong_selected = len(predicted_answer - standard_answer)    # 错误选中的数量
                missed_correct = len(standard_answer - predicted_answer)    # 漏选的数量
                
                if correct_selected > 0 and wrong_selected == 0:
                    error_types['partial_correct'] += 1
                elif correct_selected > 0 and wrong_selected > 0:
                    error_types['partial_wrong'] += 1
                elif correct_selected == 0 and wrong_selected > 0:
                    error_types['completely_wrong'] += 1
                elif correct_selected == 0 and wrong_selected == 0 and missed_correct > 0:
                    error_types['over_selected'] += 1
            
            # 2. 计算当前题目的 Precision, Recall, F1
            tp = len(standard_answer & predicted_answer)  # 真正例：预测正确且实际正确的选项
            fp = len(predicted_answer - standard_answer)  # 假正例：预测正确但实际错误的选项
            fn = len(standard_answer - predicted_answer)  # 假反例：预测错误但实际正确的选项
            
            # 当前题目的指标
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 累加到宏平均
            macro_precision_sum += precision
            macro_recall_sum += recall
            macro_f1_sum += f1
            
            # 累加到微平均
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            
            print(f"第{i+1}题: 标准答案={''.join(sorted(standard_answer))}, 预测={''.join(sorted(predicted_answer))}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, {'✓' if standard_answer == predicted_answer else '✗'}")
        else:
            print(f"第{i+1}题: 标准答案={item['answer']}, 预测=失败, ✗")
    
    # 计算最终指标
    exact_match_rate = exact_match / total_valid if total_valid > 0 else 0
    
    # 宏平均指标
    macro_precision = macro_precision_sum / total_valid if total_valid > 0 else 0
    macro_recall = macro_recall_sum / total_valid if total_valid > 0 else 0
    macro_f1 = macro_f1_sum / total_valid if total_valid > 0 else 0
    
    # 微平均指标（作为参考）
    micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0
    micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    metrics = {
        'total_questions': len(test_data),
        'valid_predictions': total_valid,
        'failed_predictions': len(test_data) - total_valid,
        
        # 指标1: Exact Match
        'exact_match': {
            'count': exact_match,
            'rate': exact_match_rate
        },
        
        # 指标2: 宏平均 Precision/Recall/F1
        'macro_average': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        
        # 微平均指标（作为参考）
        'micro_average': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1,
            'tp': micro_tp,
            'fp': micro_fp,
            'fn': micro_fn
        },
        
        # 错误分析
        'error_analysis': error_types,
        'error_distribution': {
            error_type: count / total_valid for error_type, count in error_types.items()
        }
    }
    
    return metrics

# ------------------ 保存预测结果 ------------------
def save_predictions(test_data, predictions, output_file, metrics=None):
    """保存预测结果到文件"""
    results = {
        'metadata': {
            'total_questions': len(test_data),
            'valid_predictions': metrics['valid_predictions'] if metrics else 0,
            'failed_predictions': metrics['failed_predictions'] if metrics else 0,
            'exact_match_rate': metrics['exact_match']['rate'] if metrics else 0,
            'macro_f1': metrics['macro_average']['f1'] if metrics else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'metrics': metrics,
        'predictions': []
    }
    
    for i, (item, pred) in enumerate(zip(test_data, predictions)):
        standard_answer_set = set(item['answer'])
        predicted_answer_set = set(pred) if pred else set()
        
        # 计算当前题目的指标
        tp = len(standard_answer_set & predicted_answer_set) if pred else 0
        fp = len(predicted_answer_set - standard_answer_set) if pred else 0
        fn = len(standard_answer_set - predicted_answer_set) if pred else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['predictions'].append({
            "question_id": i,
            "question": item['question'],
            "standard_answer": item['answer'],
            "correct_keywords": item.get('correct_keywords', []),
            "wrong_keywords": item.get('wrong_keywords', []),
            "predicted_answer": pred,
            "is_exact_match": standard_answer_set == predicted_answer_set if pred is not None else False,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "is_failed": pred is None
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"预测结果已保存到: {output_file}")

# ------------------ 打印详细统计报告 ------------------
def print_detailed_statistics(metrics, data_type, model_name):
    """打印详细的统计报告"""
    print(f"\n{'='*80}")
    print(f"           {model_name} - {data_type.upper()} 多选题详细统计报告")
    print(f"{'='*80}")
    
    print(f"\n总体统计:")
    print(f"  总题目数: {metrics['total_questions']}")
    print(f"  有效预测: {metrics['valid_predictions']}")
    print(f"  失败预测: {metrics['failed_predictions']}")
    
    print(f"\n指标1 - Exact Match (严格匹配率):")
    exact_match = metrics['exact_match']
    print(f"  完全匹配题目数: {exact_match['count']}")
    print(f"  严格匹配率: {exact_match['rate']:.3f} ({exact_match['count']}/{metrics['valid_predictions']})")
    
    print(f"\n指标2 - 宏平均 Precision/Recall/F1:")
    macro = metrics['macro_average']
    print(f"  宏平均精确率: {macro['precision']:.3f}")
    print(f"  宏平均召回率: {macro['recall']:.3f}")
    print(f"  宏平均F1值: {macro['f1']:.3f}")
    
    print(f"\n参考指标 - 微平均 Precision/Recall/F1:")
    micro = metrics['micro_average']
    print(f"  微平均精确率: {micro['precision']:.3f}")
    print(f"  微平均召回率: {micro['recall']:.3f}")
    print(f"  微平均F1值: {micro['f1']:.3f}")
    print(f"  TP: {micro['tp']}, FP: {micro['fp']}, FN: {micro['fn']}")
    
    print(f"\n错误类型分析:")
    for error_type, proportion in metrics['error_distribution'].items():
        count = metrics['error_analysis'][error_type]
        print(f"  {error_type}: {proportion:.3f} ({count}题)")
    
    print(f"{'='*80}")

if __name__ == '__main__':
    # 测试配置列表 - 增加了semantic数据
    test_configs = [
        # subword数据测试
        {
            'data_type': 'subword',
            'question_type': 'mcq',
            'model_name': 'gpt-4o-2024-11-20',
            'test_file': '../../data/questions/subword/mcq_dataset.json'
        },
        {
            'data_type': 'subword',
            'question_type': 'mcq',
            'model_name': 'gemini-2.5-pro-preview-06-05',
            'test_file': '../../data/questions/subword/mcq_dataset.json'
        },
        # semantic数据测试
        {
            'data_type': 'semantic',
            'question_type': 'mcq',
            'model_name': 'gpt-4o-2024-11-20',
            'test_file': '../../data/questions/semantic/mcq_dataset.json'
        },
        {
            'data_type': 'semantic',
            'question_type': 'mcq',
            'model_name': 'gemini-2.5-pro-preview-06-05',
            'test_file': '../../data/questions/semantic/mcq_dataset.json'
        }
    ]

    # 公共参数
    max_retries = 3
    test_size = None  # 测试所有题目

    # 确保结果目录存在
    os.makedirs('../../result', exist_ok=True)

    # 遍历所有测试配置
    for i, config in enumerate(test_configs, 1):
        data_type = config['data_type']
        model_name = config['model_name']
        question_type = config['question_type']
        test_file = config['test_file']

        print(f"\n{'='*80}")
        print(f"测试配置 {i}/{len(test_configs)}: {model_name} on {data_type} 多选题数据")
        print(f"{'='*80}")

        try:
            # 读取数据
            test_data = load_mcq_dataset(test_file)
            
            # 选择测试子集
            if test_size and test_size < len(test_data):
                test_subset = test_data[:test_size]
            else:
                test_subset = test_data
            
            print(f"测试 {len(test_subset)} 道多选题")
            print(f"模型: {model_name}, 数据类型: {data_type}")
            print(f"最大重试次数: {max_retries}")
            print("=" * 50)

            # 调用预测
            predictions, failed_count = LLM_mcq_predict(test_subset, model_name, max_retries=max_retries)

            # 计算指标
            metrics = calculate_mcq_metrics(test_subset, predictions)
            
            # 输出详细统计报告
            print_detailed_statistics(metrics, data_type, model_name)

            # 将统计信息追加到文件
            stats_file = f'../../result/mcq_stats.txt'
            with open(stats_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"配置: {model_name} - {data_type}\n")
                f.write(f"Exact Match: {metrics['exact_match']['rate']:.3f}\n")
                f.write(f"宏平均 F1: {metrics['macro_average']['f1']:.3f}\n")
                f.write(f"微平均 F1: {metrics['micro_average']['f1']:.3f}\n")
                f.write(f"详细指标: {json.dumps(metrics, ensure_ascii=False, indent=2)}\n")
                f.write(f"{'='*80}\n")
            print(f"统计信息已保存到: {stats_file}")

            # 保存预测结果
            output_file = f'../../result/{model_name}_{data_type}_{question_type}_predictions.json'
            save_predictions(test_subset, predictions, output_file, metrics)

        except Exception as e:
            print(f"[!] 配置 {i} 测试失败: {e}")
            continue

    print(f"\n所有多选题测试完成! 共测试了 {len(test_configs)} 个配置")