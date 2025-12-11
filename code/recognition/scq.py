import json
from openai import OpenAI
import os
import time

# ------------------ answer_scq 函数 ------------------
def answer_scq(client, model, question, max_retries=3, retry_delay=1):
    """
    回答单选题（SCQ）
    question: 题干+专利文本+选项 (str)
    max_retries: 最大重试次数
    retry_delay: 重试延迟（秒）
    return: 模型选的选项 key ('A'/'B'...)，失败返回None
    """
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "你是一个专利文本阅读理解助手。请回答以下单选题，并严格只输出正确选项的字母，不要输出关键词，也不要输出其他内容。"},
                    {"role": "user", "content": question}
                ],
                temperature=0,           # 设置为0确保确定性
                seed=42,                 # 设置随机种子
                top_p=1                  # 设置为1，配合temperature=0
            )
            result = completion.choices[0].message.content.strip()
            
            # 检查结果是否有效
            if result and result[0] in ["A","B","C","D"]:
                return result[0]
            else:
                print(f"第 {attempt+1} 次尝试返回无效答案: '{result}'")
                if attempt < max_retries - 1:  # 如果不是最后一次尝试
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                
        except Exception as e:
            print(f"第 {attempt+1} 次尝试失败: {e}")
            if attempt < max_retries - 1:  # 如果不是最后一次尝试
                print(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
    
    print(f"经过 {max_retries} 次尝试后仍然失败")
    return None

# ------------------ LLM_scq_predict 函数 ------------------
def LLM_scq_predict(test_data, model_name, max_retries=3):
    """
    使用 LLM 对单选题数据集进行预测
    test_data: list of dict, 每个 dict 包含 "question" 字段
    model_name: str, 模型名称
    max_retries: 最大重试次数
    return: (predictions, failed_count)
    """
    
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='sk-KFGXGtYwV6ghZoVZNJvt9PdMlE72IaptRxsuQXt0pFRodxpa'  # 替换成你的 API Key
    )

    predictions = []
    failed_count = 0

    for i, item in enumerate(test_data):
        question_text = item['question']
        try:
            pred = answer_scq(client, model_name, question_text, max_retries=max_retries)
            if pred is None:  # 如果返回None，也算作失败
                failed_count += 1
                print(f"[!] 第 {i+1} 条题目预测失败: 经过 {max_retries} 次尝试后模型返回无效答案")
            predictions.append(pred)
            print(f"第 {i+1}/{len(test_data)} 题预测完成: {pred}")
        except Exception as e:
            print(f"[!] 第 {i+1} 条题目预测失败: {e}")
            predictions.append(None)
            failed_count += 1

    return predictions, failed_count

# ------------------ 读取 SCQ 测试数据 ------------------
def load_scq_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# ------------------ 计算准确率 ------------------
def calculate_accuracy(test_data, predictions):
    """计算预测准确率（排除预测失败的题目）"""
    correct = 0
    total_valid = 0  # 有效预测数量
    
    for i, (item, pred) in enumerate(zip(test_data, predictions)):
        if pred is not None:  # 只统计有效预测
            total_valid += 1
            if pred == item['answer']:
                correct += 1
            print(f"第{i+1}题: 标准答案={item['answer']}, 预测={pred}, {'✓' if pred == item['answer'] else '✗'}")
        else:
            print(f"第{i+1}题: 标准答案={item['answer']}, 预测=失败, ✗")
    
    accuracy = correct / total_valid if total_valid > 0 else 0
    return accuracy, correct, total_valid

# ------------------ 保存预测结果 ------------------
def save_predictions(test_data, predictions, output_file):
    """保存预测结果到文件"""
    results = []
    for i, (item, pred) in enumerate(zip(test_data, predictions)):
        results.append({
            "question_id": i,
            "question": item['question'],
            "standard_answer": item['answer'],
            "predicted_answer": pred,
            "is_correct": pred == item['answer'] if pred is not None else False,
            "is_failed": pred is None
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"预测结果已保存到: {output_file}")



if __name__ == '__main__':
    # 测试配置列表：每个配置包含 data_type, model_name, test_file
    test_configs = [
        # 1. subword/scq_dataset.json 测试
        {
            'data_type': 'subword',
            'question_type': 'scq',
            'model_name': 'gpt-4o-2024-11-20',
            'test_file': '../../data/questions/subword/scq_dataset.json'
        },
        {
            'data_type': 'subword',
            'question_type': 'scq',
            'model_name': 'gemini-2.5-pro-preview-06-05',
            'test_file': '../../data/questions/subword/scq_dataset.json'
        },
        # 2. semantic/scq_dataset.json 测试
        {
            'data_type': 'semantic',
            'question_type': 'scq',
            'model_name': 'gpt-4o-2024-11-20',
            'test_file': '../../data/questions/semantic/scq_dataset.json'
        },
        {
            'data_type': 'semantic',
            'question_type': 'scq',
            'model_name': 'gemini-2.5-pro-preview-06-05',
            'test_file': '../../data/questions/semantic/scq_dataset.json'
        }
    ]

    # 公共参数
    max_retries = 3  # 最大重试次数
    test_size = 10   # 测试数量

    # 确保结果目录存在
    os.makedirs('../../result', exist_ok=True)

    # 遍历所有测试配置
    for i, config in enumerate(test_configs, 1):
        data_type = config['data_type']
        model_name = config['model_name']
        question_type = config['question_type']
        test_file = config['test_file']

        print(f"\n{'='*80}")
        print(f"测试配置 {i}/{len(test_configs)}: {model_name} on {data_type} data")
        print(f"{'='*80}")

        try:
            # 读取数据
            test_data = load_scq_dataset(test_file)
            
            # 选择测试子集
            # test_subset = test_data[:test_size]
            # 测试所有题目
            test_subset = test_data
            print(f"测试 {len(test_subset)} 条题目")

            print(f"正在测试模型: {model_name}")
            print(f"数据类型: {data_type}, 题目类型: {question_type}")
            print(f"最大重试次数: {max_retries}")
            print("=" * 50)

            # 调用预测
            predictions, failed_count = LLM_scq_predict(test_subset, model_name, max_retries=max_retries)

            # 计算准确率（排除失败题目）
            accuracy, correct_count, valid_count = calculate_accuracy(test_subset, predictions)

            # 输出统计信息
            stats = f"""\n统计结果 (配置 {i}: {model_name} - {data_type}):
                总题目数: {len(test_subset)}
                有效回答: {valid_count}
                无效回答: {failed_count}
                回答正确: {correct_count}
                准确率: {accuracy:.3f} ({correct_count}/{valid_count})"""
            print(stats)

            # 将统计信息追加到文件
            stats_file = f'../../result/scq_stats.txt'
            with open(stats_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"配置: {model_name} - {data_type}\n")
                f.write(stats)
                f.write(f"\n{'='*50}\n")
            print(f"统计信息已保存到: {stats_file}")

            # 保存预测结果
            output_file = f'../../result/{model_name}_{data_type}_{question_type}_predictions.json'
            save_predictions(test_subset, predictions, output_file)

        except Exception as e:
            print(f"[!] 配置 {i} 测试失败: {e}")
            continue

    print(f"\n所有测试完成! 共测试了 {len(test_configs)} 个配置")