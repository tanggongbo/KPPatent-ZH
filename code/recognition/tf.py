import json
from openai import OpenAI
import os
import time
import random
from collections import defaultdict

# ------------------ 判断题回答函数 ------------------
def answer_true_false(client, model, question, max_retries=3, retry_delay=1):
    """
    回答判断题
    question: 判断题文本 (str)
    max_retries: 最大重试次数
    retry_delay: 重试延迟（秒）
    return: 模型选的答案 ('1'/'0')，失败返回None
    """
    formatted_question = f"题目：{question}"
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专利文本阅读理解助手。请判断以下陈述是否正确，并严格只输出'1'(正确)或'0'(错误)，不要输出其他内容。"},
                    {"role": "user", "content": formatted_question}
                ],
                temperature=0,
                seed=42,
                top_p=1
            )
            result = completion.choices[0].message.content.strip()
            
            # 检查结果是否有效
            if result in ["1", "0"]:
                return result
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

# ------------------ LLM_true_false_predict 函数 ------------------
def LLM_true_false_predict(test_data, model_name, max_retries=3):
    """
    使用 LLM 对判断题数据集进行预测
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
            pred = answer_true_false(client, model_name, question_text, max_retries=max_retries)
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

# ------------------ 读取判断题数据 ------------------
def load_true_false_dataset(file_path):
    """
    读取判断题数据集，并展平结构
    原始格式: [{"patent_id": "1", "questions": [{question1}, {question2}, ...]}]
    返回格式: [{"question": "...", "answer": "...", "keyword_type": "...", "query_design": "...", "patent_id": "..."}]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    flattened_data = []
    for patent in raw_data:
        patent_id = patent.get('patent_id', 'unknown')
        for question in patent.get('questions', []):
            question_data = question.copy()
            question_data['patent_id'] = patent_id
            flattened_data.append(question_data)
    
    print(f"从 {len(raw_data)} 个专利中提取了 {len(flattened_data)} 道判断题")
    return flattened_data

# ------------------ 随机抽样函数 ------------------
def sample_questions_by_patent(flattened_data, questions_per_patent=2):
    """
    从每个专利中随机抽取指定数量的题目
    """
    # 按专利ID分组
    patent_groups = defaultdict(list)
    for question in flattened_data:
        patent_id = question['patent_id']
        patent_groups[patent_id].append(question)
    
    # 从每个专利中随机抽取题目
    sampled_questions = []
    for patent_id, questions in patent_groups.items():
        if len(questions) <= questions_per_patent:
            sampled_questions.extend(questions)
        else:
            sampled_questions.extend(random.sample(questions, questions_per_patent))
    
    print(f"从 {len(patent_groups)} 个专利中抽取了 {len(sampled_questions)} 道题目")
    return sampled_questions

# ------------------ 计算准确率 ------------------
def calculate_accuracy(test_data, predictions):
    """计算预测准确率（排除预测失败的题目）"""
    correct = 0
    total_valid = 0
    
    for i, (item, pred) in enumerate(zip(test_data, predictions)):
        if pred is not None:
            total_valid += 1
            if pred == item['answer']:
                correct += 1
            print(f"第{i+1}题: 标准答案={item['answer']}, 预测={pred}, {'✓' if pred == item['answer'] else '✗'}")
        else:
            print(f"第{i+1}题: 标准答案={item['answer']}, 预测=失败, ✗")
    
    accuracy = correct / total_valid if total_valid > 0 else 0
    return accuracy, correct, total_valid

# ------------------ 计算多维度准确率 ------------------
def calculate_detailed_accuracy(test_data, predictions):
    """
    计算不同维度的准确率
    """
    stats = {
        'total': len(test_data),
        'valid_predictions': 0,
        'failed_predictions': 0,
        'correct_predictions': 0,
        'overall_accuracy': 0,
        
        # 按keyword_type统计
        'pos_correct': 0, 'pos_total': 0, 'pos_accuracy': 0,
        'neg_correct': 0, 'neg_total': 0, 'neg_accuracy': 0,
        
        # 按query_design统计
        'is_correct': 0, 'is_total': 0, 'is_accuracy': 0,
        'not_correct': 0, 'not_total': 0, 'not_accuracy': 0,
        
        # 组合维度统计
        'pos_is_correct': 0, 'pos_is_total': 0, 'pos_is_accuracy': 0,
        'pos_not_correct': 0, 'pos_not_total': 0, 'pos_not_accuracy': 0,
        'neg_is_correct': 0, 'neg_is_total': 0, 'neg_is_accuracy': 0,
        'neg_not_correct': 0, 'neg_not_total': 0, 'neg_not_accuracy': 0
    }
    
    for item, pred in zip(test_data, predictions):
        keyword_type = item.get('keyword_type', '')
        query_design = item.get('query_design', '')
        
        # 统计keyword_type
        if keyword_type == 'pos':
            stats['pos_total'] += 1
        elif keyword_type == 'neg':
            stats['neg_total'] += 1
            
        # 统计query_design
        if query_design == 'is':
            stats['is_total'] += 1
        elif query_design == 'not':
            stats['not_total'] += 1
            
        # 统计组合维度
        if keyword_type == 'pos' and query_design == 'is':
            stats['pos_is_total'] += 1
        elif keyword_type == 'pos' and query_design == 'not':
            stats['pos_not_total'] += 1
        elif keyword_type == 'neg' and query_design == 'is':
            stats['neg_is_total'] += 1
        elif keyword_type == 'neg' and query_design == 'not':
            stats['neg_not_total'] += 1
        
        # 统计预测结果
        if pred is not None:
            stats['valid_predictions'] += 1
            
            if pred == item['answer']:
                stats['correct_predictions'] += 1
                
                # 按维度统计正确预测
                if keyword_type == 'pos':
                    stats['pos_correct'] += 1
                elif keyword_type == 'neg':
                    stats['neg_correct'] += 1
                    
                if query_design == 'is':
                    stats['is_correct'] += 1
                elif query_design == 'not':
                    stats['not_correct'] += 1
                    
                if keyword_type == 'pos' and query_design == 'is':
                    stats['pos_is_correct'] += 1
                elif keyword_type == 'pos' and query_design == 'not':
                    stats['pos_not_correct'] += 1
                elif keyword_type == 'neg' and query_design == 'is':
                    stats['neg_is_correct'] += 1
                elif keyword_type == 'neg' and query_design == 'not':
                    stats['neg_not_correct'] += 1
        else:
            stats['failed_predictions'] += 1
    
    # 计算准确率
    if stats['valid_predictions'] > 0:
        stats['overall_accuracy'] = stats['correct_predictions'] / stats['valid_predictions']
    
    if stats['pos_total'] > 0:
        stats['pos_accuracy'] = stats['pos_correct'] / stats['pos_total']
    if stats['neg_total'] > 0:
        stats['neg_accuracy'] = stats['neg_correct'] / stats['neg_total']
    if stats['is_total'] > 0:
        stats['is_accuracy'] = stats['is_correct'] / stats['is_total']
    if stats['not_total'] > 0:
        stats['not_accuracy'] = stats['not_correct'] / stats['not_total']
    if stats['pos_is_total'] > 0:
        stats['pos_is_accuracy'] = stats['pos_is_correct'] / stats['pos_is_total']
    if stats['pos_not_total'] > 0:
        stats['pos_not_accuracy'] = stats['pos_not_correct'] / stats['pos_not_total']
    if stats['neg_is_total'] > 0:
        stats['neg_is_accuracy'] = stats['neg_is_correct'] / stats['neg_is_total']
    if stats['neg_not_total'] > 0:
        stats['neg_not_accuracy'] = stats['neg_not_correct'] / stats['neg_not_total']
    
    return stats

# ------------------ 保存预测结果 ------------------
def save_predictions(test_data, predictions, output_file, stats=None):
    """保存预测结果到文件"""
    results = {
        'metadata': {
            'total_questions': len(test_data),
            'valid_predictions': stats['valid_predictions'] if stats else 0,
            'failed_predictions': stats['failed_predictions'] if stats else 0,
            'overall_accuracy': stats['overall_accuracy'] if stats else 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'detailed_statistics': stats,
        'predictions': []
    }
    
    for i, (item, pred) in enumerate(zip(test_data, predictions)):
        results['predictions'].append({
            "question_id": i,
            "patent_id": item.get('patent_id', 'unknown'),
            "question": item['question'],
            "standard_answer": item['answer'],
            "keyword_type": item.get('keyword_type', ''),
            "query_design": item.get('query_design', ''),
            "predicted_answer": pred,
            "is_correct": pred == item['answer'] if pred is not None else False,
            "is_failed": pred is None
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"预测结果已保存到: {output_file}")

# ------------------ 打印详细统计报告 ------------------
def print_detailed_statistics(stats, data_type, model_name):
    """打印详细的统计报告"""
    print(f"\n{'='*80}")
    print(f"           {model_name} - {data_type.upper()} 数据详细统计报告")
    print(f"{'='*80}")
    
    print(f"\n总体统计:")
    print(f"  总题目数: {stats['total']}")
    print(f"  有效预测: {stats['valid_predictions']}")
    print(f"  失败预测: {stats['failed_predictions']}")
    print(f"  正确预测: {stats['correct_predictions']}")
    print(f"  总体准确率: {stats['overall_accuracy']:.3f} ({stats['correct_predictions']}/{stats['valid_predictions']})")
    
    print(f"\n按关键词类型统计:")
    print(f"  正例关键词 (pos): {stats['pos_accuracy']:.3f} ({stats['pos_correct']}/{stats['pos_total']})")
    print(f"  负例关键词 (neg): {stats['neg_accuracy']:.3f} ({stats['neg_correct']}/{stats['neg_total']})")
    
    print(f"\n按查询设计统计:")
    print(f"  正向查询 (is): {stats['is_accuracy']:.3f} ({stats['is_correct']}/{stats['is_total']})")
    print(f"  反向查询 (not): {stats['not_accuracy']:.3f} ({stats['not_correct']}/{stats['not_total']})")
    
    print(f"\n组合维度统计:")
    print(f"  正例-正向 (pos-is): {stats['pos_is_accuracy']:.3f} ({stats['pos_is_correct']}/{stats['pos_is_total']})")
    print(f"  正例-反向 (pos-not): {stats['pos_not_accuracy']:.3f} ({stats['pos_not_correct']}/{stats['pos_not_total']})")
    print(f"  负例-正向 (neg-is): {stats['neg_is_accuracy']:.3f} ({stats['neg_is_correct']}/{stats['neg_is_total']})")
    print(f"  负例-反向 (neg-not): {stats['neg_not_accuracy']:.3f} ({stats['neg_not_correct']}/{stats['neg_not_total']})")
    print(f"{'='*80}")

if __name__ == '__main__':
    # 测试配置列表
    test_configs = [
        {
            'data_type': 'subword',
            'question_type': 'true_false',
            'model_name': 'gpt-4o-2024-11-20',
            'test_file': '../../data/questions/subword/true_false_dataset.json'
        },
        {
            'data_type': 'subword',
            'question_type': 'true_false',
            'model_name': 'gemini-2.5-pro-preview-06-05',
            'test_file': '../../data/questions/subword/true_false_dataset.json'
        },
        {
            'data_type': 'semantic',
            'question_type': 'true_false',
            'model_name': 'gpt-4o-2024-11-20',
            'test_file': '../../data/questions/semantic/true_false_dataset.json'
        },
        {
            'data_type': 'semantic',
            'question_type': 'true_false',
            'model_name': 'gemini-2.5-pro-preview-06-05',
            'test_file': '../../data/questions/semantic/true_false_dataset.json'
        }
    ]

    # 公共参数
    max_retries = 3
    questions_per_patent = 2  # 每个专利抽取2道题

    # 确保结果目录存在
    os.makedirs('../../result', exist_ok=True)

    # 遍历所有测试配置
    for i, config in enumerate(test_configs, 1):
        data_type = config['data_type']
        model_name = config['model_name']
        question_type = config['question_type']
        test_file = config['test_file']

        print(f"\n{'='*80}")
        print(f"测试配置 {i}/{len(test_configs)}: {model_name} on {data_type} 判断题数据")
        print(f"{'='*80}")

        try:
            # 读取并处理数据
            flattened_data = load_true_false_dataset(test_file)
            
            # 从每个专利中随机抽取题目
            test_subset = sample_questions_by_patent(flattened_data, questions_per_patent)
            # test_subset = test_subset[:5] 
            
            print(f"测试 {len(test_subset)} 道判断题")
            print(f"模型: {model_name}, 数据类型: {data_type}")
            print(f"最大重试次数: {max_retries}")
            print("=" * 50)

            # 调用预测
            predictions, failed_count = LLM_true_false_predict(test_subset, model_name, max_retries=max_retries)

            # 计算基础准确率
            accuracy, correct_count, valid_count = calculate_accuracy(test_subset, predictions)
            
            # 计算详细统计信息
            detailed_stats = calculate_detailed_accuracy(test_subset, predictions)

            # 输出详细统计报告
            print_detailed_statistics(detailed_stats, data_type, model_name)

            # 将统计信息追加到文件
            stats_file = f'../../result/tf_stats.txt'
            with open(stats_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"配置: {model_name} - {data_type}\n")
                f.write(f"总体准确率: {accuracy:.3f} ({correct_count}/{valid_count})\n")
                f.write(f"详细统计: {json.dumps(detailed_stats, ensure_ascii=False, indent=2)}\n")
                f.write(f"{'='*80}\n")
            print(f"统计信息已保存到: {stats_file}")

            # 保存预测结果
            output_file = f'../../result/{model_name}_{data_type}_{question_type}_predictions.json'
            save_predictions(test_subset, predictions, output_file, detailed_stats)

        except Exception as e:
            print(f"[!] 配置 {i} 测试失败: {e}")
            continue

    print(f"\n所有判断题测试完成! 共测试了 {len(test_configs)} 个配置")