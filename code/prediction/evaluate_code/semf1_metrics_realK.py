import torch
from FlagEmbedding import FlagModel
import numpy as np
import json
 
class SemanticMatchingMetric:  # 不再继承KeyphraseMetric
    def __init__(self, 
                 model_name_or_path="/root/siton-tmp/UKE-keyllm/keywords/models/bge-small-zh-v1.5",  # 修改为本地路径
                 similarity_threshold=0.65,
                 pooling_across_phrases='mean',
                 use_fp16=True,
                 local_files_only=True,
                 device='cpu'):  # 新增device参数
        
        self.device = device
        self.model = FlagModel(
            model_name_or_path,
            use_fp16=use_fp16,
            local_files_only=local_files_only,
            device=device  # 将device传递给FlagModel
        )
        self.similarity_threshold = similarity_threshold
        self.pooling_across_phrases = pooling_across_phrases

    def to(self, device):
        """支持设备转移"""
        self.device = device
        if hasattr(self.model, 'to'):
            self.model.to(device)
        return self

    def encode_phrases(self, phrases):
        """BGE专用编码方法（自动处理归一化）"""
        if not phrases:
            return np.zeros((0, 768))
        
        result = self.model.encode(
            phrases,
            batch_size=32,
            max_length=512
        )
        
        embeddings = result
            
        return np.array(embeddings).reshape(len(phrases), -1)
    
    # def encode_phrases(self, phrases):
    #     """优化后的BGE编码方法"""
    #     if not phrases:
    #         return np.zeros((0, 768))
        
    #     # 使用encode方法并添加批处理
    #     embeddings = self.model.encode(
    #         phrases,
    #         batch_size=32,
    #         max_length=512,
    #         normalize_embeddings=True  # 添加归一化
    #     )
        
    #     return np.array(embeddings).reshape(len(phrases), -1)
    
    def evaluate_single_example(self, preds, labels):
        n_labels, n_preds = len(labels), len(preds)
        
        # BGE编码（合并处理提升效率）
        all_phrases = labels + preds
        embeddings = self.encode_phrases(all_phrases)
        
        # 确保embeddings是2维数组 [n_phrases, embedding_dim]
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim == 0:
            embeddings = np.zeros((0, 768))
            
        label_embeds = embeddings[:n_labels]
        pred_embeds = embeddings[n_labels:]
        
        # 计算precision
        if n_labels == 0 or n_preds == 0:
            cur_p = 0
        else:
            # 对每个预测词，计算它与所有真实关键词的相似度
            sim_matrix = torch.matmul(
                torch.tensor(pred_embeds, dtype=torch.float32).to(self.device),
                torch.tensor(label_embeds, dtype=torch.float32).to(self.device).T
            )
            # print("\n预测词与真实关键词的相似度矩阵:")
            # print("预测词:", preds)
            # print("真实关键词:", labels)
            # print("相似度矩阵:")
            # print(sim_matrix.cpu().numpy().round(3))
            
            # 新增二值矩阵输出
            # binary_matrix = (sim_matrix >= self.similarity_threshold).float()
            # print(f"\n阈值 {self.similarity_threshold} 的二值矩阵(>=阈值为1,否则为0):")
            # print(binary_matrix.cpu().numpy())
            
            max_sim_values, _ = torch.max(sim_matrix, dim=1)
            cur_p = torch.mean((max_sim_values >= self.similarity_threshold).float()).item()
            
        # 计算recall
        if n_labels == 0 or n_preds == 0:
            cur_r = 0
        else:
            # 对每个真实关键词，计算其与所有预测关键词的最大相似度
            sim_matrix = torch.matmul(
                torch.tensor(label_embeds, dtype=torch.float32).to(self.device),
                torch.tensor(pred_embeds, dtype=torch.float32).to(self.device).T
            )
            max_sim_values, _ = torch.max(sim_matrix, dim=1)
            cur_r = torch.mean((max_sim_values >= self.similarity_threshold).float()).item()


        # 计算f1
        cur_f1 = 0 if (cur_p + cur_r) == 0 else 2 * cur_p * cur_r / (cur_p + cur_r)
        
        return {'p': cur_p, 'r': cur_r, 'f1': cur_f1}

    def score_corpus(self, all_present_keys, all_absent_keys, all_predicted_keys):
        semf1_results = {
            'present': {'semantic_p': [], 'semantic_r': [], 'semantic_f1': []},
            'absent': {'semantic_p': [], 'semantic_r': [], 'semantic_f1': []}
        }
        
        for present_keys, absent_keys, predicted_keys in zip(all_present_keys, all_absent_keys, all_predicted_keys):  
            if not predicted_keys:  # 跳过空预测样本
                continue          
            
            present_str_list = present_keys if not present_keys or isinstance(present_keys[0], str) else [' '.join(tokens) for tokens in present_keys]
            absent_str_list = absent_keys if not absent_keys or isinstance(absent_keys[0], str) else [' '.join(tokens) for tokens in absent_keys]
            predicted_str_list = predicted_keys if not predicted_keys or isinstance(predicted_keys[0], str) else [' '.join(tokens) for tokens in predicted_keys]
            
            present_semf1 = self.evaluate_single_example(
                predicted_str_list,
                present_str_list
            )
            absent_semf1 = self.evaluate_single_example(
                predicted_str_list,
                absent_str_list
            ) if absent_keys else {'p': 0, 'r': 0, 'f1': 0}
            
            for m in ['p', 'r', 'f1']:
                semf1_results['present'][f'semantic_{m}'].append(present_semf1[m])
                semf1_results['absent'][f'semantic_{m}'].append(absent_semf1[m])
                
        return semf1_results
    
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
    
    # # 初始化并运行评测
    metric = SemanticMatchingMetric()
    scores = metric.score_corpus(all_present_keys, all_absent_keys, all_predicted_keys)
    
    # 测试不同阈值
    # for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85]:
    #     print(f"\n===== 测试阈值: {threshold} =====")
    #     metric = SemanticMatchingMetric(similarity_threshold=threshold)
    #     scores = metric.score_corpus(all_present_keys, all_absent_keys, all_predicted_keys)
    
    
    # 输出阈值
    print(f"使用的语义匹配阈值: {metric.similarity_threshold}")
    print("文件数据评测结果:")
    print(f"example数量: {len(all_predicted_keys)}")
    print("Present Keys:")
    print(f"平均 Semantic Precision: {np.mean(scores['present']['semantic_p']):.4f}")
    print(f"平均 Semantic Recall: {np.mean(scores['present']['semantic_r']):.4f}")
    print(f"平均 Semantic F1: {np.mean(scores['present']['semantic_f1']):.4f}")
    
    print("Absent Keys:")
    print(f"平均 Semantic Precision: {np.mean(scores['absent']['semantic_p']):.4f}")
    print(f"平均 Semantic Recall: {np.mean(scores['absent']['semantic_r']):.4f}")
    print(f"平均 Semantic F1: {np.mean(scores['absent']['semantic_f1']):.4f}")