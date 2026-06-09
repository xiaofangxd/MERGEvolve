import numpy as np
from typing import List
from openai import OpenAI
import os
from typing import List
from loguru import logger
import time
from pathlib import Path
import sqlite3
import hashlib
import json

class SimpleBERTScore:
    def __init__(self, model_name: str = "text-embedding-3-small", cache_dir: str = ".cache"):
        # Initialize tokenizer and model
        self.model = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        self.model_name = model_name
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_path / "embeddings.db"
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    text TEXT,
                    model TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    def get_single_embedding(self, text: str, max_retries: int = 5, initial_delay: float = 1.0) -> List:
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT embedding FROM embeddings WHERE text_hash = ? AND model = ?",
                    (text_hash, self.model_name)
                ).fetchone()
                
                if result:
                    return json.loads(result[0])
                
            # 如果缓存未命中，从API获取
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    encoded = self.model.embeddings.create(
                        input=text,
                        model=self.model_name
                    )
                    embedding = encoded.data[0].embedding
                    
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute(
                            "INSERT INTO embeddings (text_hash, text, model, embedding) VALUES (?, ?, ?, ?)",
                            (text_hash, text, self.model_name, json.dumps(embedding))
                        )
                        conn.commit()
                    
                    return embedding
                    
                except Exception as e:
                    if attempt == max_retries - 1:  # Last attempt
                        logger.error(f"Final attempt failed. Error getting embeddings: {e}")
                        return None  # 改为返回None而不是抛出异常
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {delay:.1f}s... Error: {e}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
        except Exception as e:
            logger.error(f"Database error: {e}")
            return None

    def get_bert_embeddings(self, texts: List[str], max_retries: int = 5, initial_delay: float = 1.0) -> List:
        embeddings = []
        for text in texts:
            embedding = self.get_single_embedding(text, max_retries, initial_delay)
            if embedding is None:
                logger.error(f"Failed to get embedding for text: {text}")
                return []  # 保持与原函数一致的错误处理行为
            embeddings.append(embedding)
        return embeddings

    def compute_pairwise_cosine_scores(self, ref_embeddings: List, cand_embeddings: List) -> List:
        # Convert lists to numpy arrays
        ref_embeddings = np.array(ref_embeddings)
        cand_embeddings = np.array(cand_embeddings)
        
        # Reshape arrays if needed (assuming embeddings are 1D)
        if len(ref_embeddings.shape) == 1:
            ref_embeddings = ref_embeddings.reshape(1, -1)
        if len(cand_embeddings.shape) == 1:
            cand_embeddings = cand_embeddings.reshape(1, -1)
        
        # 计算范数（L2范数）
        ref_norm = np.linalg.norm(ref_embeddings, axis=1, keepdims=True)
        cand_norm = np.linalg.norm(cand_embeddings, axis=1, keepdims=True)
    
        # 归一化向量
        ref_embeddings_normalized = ref_embeddings / ref_norm
        cand_embeddings_normalized = cand_embeddings / cand_norm
        
        # 计算余弦相似度
        cosine_sim = np.dot(ref_embeddings_normalized, cand_embeddings_normalized.T)
        
        # 确保所有值都在[-1, 1]范围内（处理数值误差）
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        return cosine_sim

    def compute_bertscore(self, references: List[str], candidates: List[str]):
        # 获取embeddings
        ref_embeddings = self.get_bert_embeddings(references)
        cand_embeddings = self.get_bert_embeddings(candidates)

        # 计算相似度矩阵
        pairwise_scores = self.compute_pairwise_cosine_scores(ref_embeddings, cand_embeddings)

        # 计算precision (candidate -> reference)
        precision = np.mean(np.max(pairwise_scores, axis=1))
        
        # 计算recall (reference -> candidate)
        recall = np.mean(np.max(pairwise_scores, axis=0))
        
        # 计算F1
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def get_best_sentence(sentence_list: List[str]) -> tuple[str, float]:
    """
    Calculates BERTScore for each sentence compared with all other sentences in the list.
    
    Args:
        sentence_list: List of sentences to compare
    
    Returns:
        dict: {
            'scores': 2D matrix of F1 scores,
            'sentence_pairs': List of (sentence1, sentence2, score) tuples sorted by score,
            'average_scores': Dict mapping each sentence to its average F1 score with others
        }
    """
    if len(sentence_list) == 1:
        return {
            'scores': np.zeros((1, 1)),
            'sentence_pairs': [],
            'average_scores': {},
            'best_sentence': (sentence_list[0], 1.0)
        }
    
    bert_scorer = SimpleBERTScore(model_name="text-embedding-3-small")
    valid_sentences = [s for s in sentence_list if s and s != "None" and len(s) > 0]
    if len(valid_sentences) != len(sentence_list):
        logger.warning(f"Length of input sentence list is {len(sentence_list)}, length of valid_sentences is {len(valid_sentences)}")
    n = len(valid_sentences)
    if n == 1:
        return {
            'scores': np.zeros((1, 1)),
            'sentence_pairs': [],
            'average_scores': {},
            'best_sentence': (valid_sentences[0], 1.0)
        }
    elif n == 0:
        return {
            'scores': np.zeros((1, 1)),
            'sentence_pairs': [],
            'average_scores': {},
            'best_sentence': ("None", 0.0)
        }
        
    # if there is only one sentence, return the sentence and 1.0
    all_embeddings = bert_scorer.get_bert_embeddings(valid_sentences)
    all_embeddings = np.array(all_embeddings)

    score_matrix = bert_scorer.compute_pairwise_cosine_scores(all_embeddings, all_embeddings)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            similarity = float(score_matrix[i][j])
            pairs.append((valid_sentences[i], valid_sentences[j], similarity))

    avg_scores = dict()
    for i, sentence in enumerate(valid_sentences):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        scores = score_matrix[i][mask]
        avg_scores[sentence] = float(np.mean(scores))

    best_sentence = max(avg_scores.items(), key=lambda x: x[1])
    
    return {
        'scores': score_matrix,
        'sentence_pairs': sorted(pairs, key=lambda x: x[2], reverse=True),
        'average_scores': avg_scores,
        'best_sentence': best_sentence
    }

def main():
    # 示例文本
    sentences = ["i want to control my weight.", "this is a test", "The cat sits on the mat.", "The weather is nice today."]
    
    # 获取最佳句子
    result = get_best_sentence(sentences)
    print(f"scores: {result['scores']}")
    print(f"sentence_pairs: {result['sentence_pairs']}")
    print(f"average_scores: {result['average_scores']}")
    print(f"best_sentence: {result['best_sentence']}")
    
if __name__ == "__main__":
    main()