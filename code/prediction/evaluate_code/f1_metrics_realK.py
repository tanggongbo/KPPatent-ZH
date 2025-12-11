import numpy as np
import json
import jieba
jieba.setLogLevel(20)  # 设置日志级别为ERROR，不输出INFO信息
 
def compute_macro_avg_f1_metrics(all_present_keys, all_absent_keys, all_predicted_keys, match_type, k_list=[5, 10, 15, 'M']):
    """
    计算所有example的宏平均F1指标(F1@5, F1@10, F1@15, F1@M)
    :param all_present_keys: list[list[str]] - 所有example的present keys列表，每个example的关键词为一个子列表
    :param all_absent_keys: list[list[str]] - 所有example的absent keys列表，格式同上
    :param all_predicted_keys: list[list[str]] - 所有example的预测关键词列表，格式同上
    :param match_type: str - 匹配模式，可选'exact'(精确匹配)|'substr'(子串匹配)|'subword'(子词匹配)
    :param k_list: list - 要计算的k值列表，如[5,10,15,'M']，'M'表示所有预测词
    :return: dict - 包含两个子字典(present/absent)，每个子字典包含各k值的宏平均F1分数
    """
    
    # 初始化结果字典
    macro_results = {
        'present': {f'F1@{k}': [] for k in k_list},
        'absent': {f'F1@{k}': [] for k in k_list}
    }
    
    # 计算每个example的F1分数
    for present_keys, absent_keys, predicted_keys in zip(all_present_keys, all_absent_keys, all_predicted_keys):
        if not predicted_keys:  # 确保跳过空预测样本
            continue
        present_f1 = compute_f1_metrics(present_keys, predicted_keys, match_type, k_list)
        absent_f1 = compute_f1_metrics(absent_keys, predicted_keys, match_type, k_list)
        
        # 收集各k值的F1分数
        for k in k_list:
            macro_results['present'][f'F1@{k}'].append(present_f1[f'F1@{k}'])
            macro_results['absent'][f'F1@{k}'].append(absent_f1[f'F1@{k}'])
    
    # 计算宏平均(对所有example的F1分数取平均)
    for k in k_list:
        macro_results['present'][f'F1@{k}'] = float(np.mean(macro_results['present'][f'F1@{k}'])) if macro_results['present'][f'F1@{k}'] else 0.0
        macro_results['absent'][f'F1@{k}'] = float(np.mean(macro_results['absent'][f'F1@{k}'])) if macro_results['absent'][f'F1@{k}'] else 0.0
    
    return macro_results


def compute_f1_metrics(trg_keyphrases, pred_keyphrases, match_type, k_list=[5, 10, 15, 'M']):
    """
    计算单个example在多个k值下的F1指标
    :param trg_keyphrases: list[str] - 目标关键词列表(ground truth)
    :param pred_keyphrases: list[str] - 预测关键词列表
    :param match_type: str - 匹配模式，同compute_macro_avg_f1_metrics
    :param k_list: list - 要计算的k值列表
    :return: dict - 包含各k值F1分数的字典，键为'F1@{k}'，值为对应分数
    """
    
    # 将关键词列表转换为分词形式
    trg_tokenized = [list(jieba.cut(keyphrase)) for keyphrase in trg_keyphrases]
    pred_tokenized = [list(jieba.cut(keyphrase)) for keyphrase in pred_keyphrases]
    
    # 计算匹配矩阵
    is_match = compute_match_result(trg_tokenized, pred_tokenized, match_type)
    
    # 计算各k值的指标
    _, _, f1_scores, _, _, _ = compute_classification_metrics_at_ks(
        is_match,
        len(pred_keyphrases),
        len(trg_keyphrases),
        k_list=k_list
    )
    
    # 构建结果字典
    results = {}
    for k, score in zip(k_list, f1_scores):
        results[f'F1@{k}'] = score
    
    return results


def is_subword_overlap(pred_word_list, trg_word_list):
    """
    使用jieba分词判断两个词是否存在子词重叠
    :param pred_word_list: list[str] - 单个预测关键词的分词列表
    :param trg_word_list: list[str] - 单个预测关键词的分词列表  
    :return: bool - True表示存在至少一个共同子词
    """
    # 将分词列表转换为集合
    pred_tokens = set(pred_word_list)
    trg_tokens = set(trg_word_list)
    
    # 检查集合中是否有共同子词
    return len(pred_tokens & trg_tokens) > 0

def compute_match_result(trg_str_list, pred_str_list, type):
    """
    计算单个example的预测关键词与目标关键词的匹配结果
    :param trg_str_list: list[list[str]] - 目标关键词列表，每个元素是一个分词后的关键词(token列表)
    示例: [['人工', '智能'], ['机器', '学习']]
    :param pred_str_list: list[list[str]] - 预测关键词列表，每个元素是一个分词后的关键词(token列表)
    示例: [['深度', '学习'], ['人工', '智能']]
    :param type: str - 匹配类型，'exact'表示精确匹配，'sub'表示子串匹配
    
    内部参数说明:
    :param pred_word_list: list[str] - 单个预测关键词的分词列表
    :param joined_pred_word_list: str - 将pred_word_list用空格连接后的字符串
    :param trg_word_list: list[str] - 单个预测关键词的分词列表  
    :param joined_trg_word_list: str - 将trg_word_list用空格连接后的字符串
    
    :return: np.ndarray - 匹配结果矩阵，形状为[n_trgs, n_preds]
    """   
    assert type in ['exact', 'substr', 'subword'], "支持类型: exact, substr, subword"

    num_pred_str = len(pred_str_list)  # 预测关键词数量
    num_trg_str = len(trg_str_list)  # 目标关键词数量

    # 二维匹配矩阵 [n_preds, n_trgs]
    is_match = np.zeros((num_pred_str, num_trg_str), dtype=bool)
    for pred_idx, pred_word_list in enumerate(pred_str_list):
        joined_pred_word_list = ' '.join(pred_word_list)
        for trg_idx, trg_word_list in enumerate(trg_str_list):
            joined_trg_word_list = ' '.join(trg_word_list)
            if type == 'exact':
                if joined_pred_word_list == joined_trg_word_list:
                    is_match[pred_idx][trg_idx] = True
                    break
            elif type == 'substr':
                if joined_pred_word_list in joined_trg_word_list:
                    is_match[pred_idx][trg_idx] = True
                    break
            elif type == 'subword':
                if is_subword_overlap(pred_word_list, trg_word_list):
                    is_match[pred_idx][trg_idx] = True
                    break
        
    return is_match

            

def compute_classification_metrics_at_ks(
    is_match, num_pred, num_trg, k_list,
    meng_rui_precision=False,
    match_mode='original'  # 可选: 'original', 'recall_like'
): 
    """
    使用二维匹配矩阵[n_preds, n_trgs]计算单个example在多个k值下的分类指标(精确率、召回率、F1)
    :param is_match: np.ndarray - 匹配结果矩阵，当dimension=1时形状为[n_preds]，当dimension=2时形状为[n_preds, n_trgs]
    :param num_pred: int - 预测词总数
    :param num_trg: int - 目标词总数
    :param k_list: list - 要计算的k值列表，支持数字或特殊字符('M','G','O')
    :param meng_rui_precision: bool - 是否使用特殊精度计算方式
        - True: 当预测数不足k时，分母使用实际预测数
        - False: 分母始终使用k
    :param match_mode: str - 匹配数计算方式: 
        - 'original': 仅统计前k个预测中的匹配
        - 'recall_like': 统计所有匹配但不超过k
    
    内部参数说明：
    :param pred_matches: np.ndarray[bool] - 每个预测词是否匹配任何目标词 [n_preds]
    :param pred_matches_cum: np.ndarray[int] - 预测词匹配数的累计和 [n_preds]
    :param total_matched_preds: int - 匹配的预测词总数
    :param trg_matches: np.ndarray[bool] - 每个目标词是否匹配任何预测词 [n_trgs]
    :param total_matched_trgs: int - 匹配的目标词总数
    :param k_resolved: int - 解析后的实际k值(处理特殊字符后的数值)
    :param curr_matches_preds: int - 当前k值下匹配的预测词数
    :param curr_matches_trgs: int - 当前k值下匹配的目标词数
    :param curr_preds: int - 当前k值下的预测数(考虑meng_rui_precision设置)
    
    :return: tuple - 包含6个列表的元组，顺序为:
        (精确率列表, 召回率列表, F1列表, 
         匹配预测词数列表, 匹配目标词数列表, 预测数列表)
        每个列表对应不同k值的结果
    """
    
    assert match_mode in ['original', 'recall_like']
    # 检查输入矩阵维度是否正确
    assert is_match.ndim == 2 and is_match.shape == (num_pred, num_trg)
    
    # 处理空预测或空目标的情况
    if num_pred == 0 or num_trg == 0:
        return [0]*len(k_list), [0]*len(k_list), [0]*len(k_list), [0]*len(k_list), [0]*len(k_list), [0]*len(k_list)
    
    # 计算预测词的匹配情况
    pred_matches = is_match.any(axis=1) # 每个预测词是否匹配至少一个目标词 [n_preds]
    pred_matches_cum = np.cumsum(pred_matches)  # 预测词匹配数的累计和 [n_preds]
    total_matched_preds = pred_matches.sum()  # 总匹配预测词数
    
    trg_matches = is_match.any(axis=0)  # 每个目标词是否匹配至少一个预测词 [n_trgs]，按列计算
    total_matched_trgs = trg_matches.sum()  # 总匹配目标词数
    
    results = ([], [], [], [], [], [])  # precision, recall, f1, matches_preds, matches_trgs, preds
    
    for k in k_list:
        # 1.解析k值，此处除了M返回num_pred，其余都返回k
        k_resolved = _resolve_k(k, num_pred, num_trg)
        
        # 2.根据匹配模式计算当前k值下的匹配数
        if match_mode == 'original':
            # 模式1: 仅统计前k个预测中的匹配
            curr_matches_preds = pred_matches_cum[k_resolved-1] if num_pred > k_resolved else pred_matches_cum[-1] # 前k个预测中的匹配预测词数
            curr_matches_trgs = (is_match[:k_resolved].any(axis=0)).sum() if num_pred > 0 else 0 # 前k个预测中匹配的去重后目标词数
        else:
            # 模式2: 统计所有预测，但匹配数不超过k
            curr_matches_preds = min(k_resolved, total_matched_preds)
            # (1)获取前curr_matches_preds个匹配预测对应的目标词数
            matched_pred_indices = np.where(pred_matches)[0][:curr_matches_preds]
            curr_matches_trgs = (is_match[matched_pred_indices].any(axis=0)).sum() if len(matched_pred_indices) > 0 else 0

            
        # 3. 计算当前k值下的预测数
        curr_preds = min(k_resolved, num_pred) if meng_rui_precision else k_resolved
        
        # 4. 计算指标
        precision = curr_matches_preds / curr_preds if curr_preds > 0 else 0.0      # precision = 匹配预测词数 / 预测数
        recall = curr_matches_trgs / num_trg if num_trg > 0 else 0.0        # recall = 匹配目标词数 / 目标总数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 5. 存储结果
        for lst, val in zip(results, [precision, recall, f1, curr_matches_preds, curr_matches_trgs, curr_preds]):
            lst.append(val)

    return results


def _resolve_k(k, num_pred, num_trg):
    """
    解析特殊k值('M','G','O')为具体数值
    :param k: int|str - 原始k值，可以是数字或特殊字符
    :param num_pred: int - 预测词数量
    :param num_trg: int - 目标词数量
    :return: int - 解析后的实际k值
    """
    if k == 'M': return num_pred # 如果k为'M'，则返回预测数
    elif k == 'G': return num_trg if num_pred < num_trg else num_pred # 如果k为'G'且预测数小于目标数，则返回目标数
    elif k == 'O': return num_trg # 如果k为'O'，则返回目标数
    return k


if __name__ == '__main__':
    all_present_keys = []
    all_absent_keys = []
    all_predicted_keys = []
    
    len_examples = 696  # 修改为len_examples
    input_file = '/root/siton-tmp/UKE-keyllm/keywords/code/evaluation/api_quchong/A_uke_log.keyLLM_gpt-4o-2024-11-20.jsonl'
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= len_examples:  # 修改为len_examples
                break
            line = line.strip()
            if not line:  # 跳过空行
                continue
            try:
                data = json.loads(line)
                all_predicted_keys.append(data['predicted_keywords'])
                all_present_keys.append(data['present_keys'])
                all_absent_keys.append(data['absent_keys'])

            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}, 跳过该行内容: {line[:50]}...")
            except KeyError as e:
                print(f"缺少必要字段: {e}, 跳过该文档")
    

    match_type = 'subword'  # 可改为'exact'或'substr'
    print(f"当前匹配模式: {match_type}")
    
    # 计算宏平均F1分数
    macro_f1_scores = compute_macro_avg_f1_metrics(
        all_present_keys, 
        all_absent_keys, 
        all_predicted_keys,
        match_type=match_type
    )
    print("数据评测结果:")
    print(f"example数量: {len(all_predicted_keys)}")
    print("Present Keys:")
    print(f"macro-F1: {macro_f1_scores['present']}")
    print("Absent Keys:")
    print(f"macro-F1: {macro_f1_scores['absent']}")
    
    
    
    

    
