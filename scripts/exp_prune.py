import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = 'cl-nagoya/ruri-v3-reranker-310m'
print('Loading tokenizer and base model...')
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

test_cases = [
    {
        'query': '日本の地方自治体におけるDX推進の現状と課題',
        'docs': [
            '自治体DX推進計画では、情報システムの標準化・共通化や行政手続きのオンライン化が進められています。課題として専門人材の不足が挙げられます。',
            '地方都市における人口減少対策と産業振興の取り組みについて。地域資源を活用した観光振興が各自治体で実施されています。',
            'クラウドサービスの導入により業務効率化を図る民間企業の事例紹介。コスト削減とセキュリティ対策が重要です。',
            '明日の天気は全国的に晴れのち曇りとなる見込みです。',
            'ピタゴラスの定理は直角三角形の斜辺の長さを求める幾何学の定理です。'
        ]
    },
    {
        'query': '量子コンピュータの動作原理と実用化に向けた課題',
        'docs': [
            '量子コンピュータは重ね合わせと量子もつれを利用して超並列計算を行います。量子誤り訂正が実用化への最大の障壁です。',
            '従来の半導体コンピュータは微細化の物理的限界に直面しており、新しい計算アーキテクチャが模索されています。',
            '機械学習の最適化アルゴリズムには勾配降下法やAdamなどが広く使用されています。',
            '東京から京都への新幹線での移動時間は約2時間15分です。'
        ]
    }
]

pairs = []
for tc in test_cases:
    q = tc['query']
    for d in tc['docs']:
        pairs.append([q, d])

inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    base_scores = torch.sigmoid(base_model(**inputs).logits.squeeze(-1)).numpy()

def rank_order(scores):
    return list(np.argsort(scores)[::-1])

print('\n=== 基準モデル (全25層) ===')
print('DXクエリ スコア:', np.round(base_scores[:5], 4), '順位:', rank_order(base_scores[:5]))
print('量子クエリ スコア:', np.round(base_scores[5:], 4), '順位:', rank_order(base_scores[5:]))

configs = [
    ('均等20層 (25->20)', np.linspace(0, 24, 20, dtype=int)),
    ('均等16層 (25->16)', np.linspace(0, 24, 16, dtype=int)),
    ('均等12層 (25->12)', np.linspace(0, 24, 12, dtype=int)),
    ('末尾重視16層 (0-3 + 13-24)', np.array([0, 1, 2, 3] + list(range(13, 25)))),
    ('前半12層 (0-11)', np.arange(12)),
    ('後半12層 (13-24)', np.arange(13, 25)),
]

for name, idx in configs:
    m_pruned = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    m_pruned.model.layers = torch.nn.ModuleList([m_pruned.model.layers[i] for i in idx])
    m_pruned.config.num_hidden_layers = len(idx)
    with torch.no_grad():
        pruned_scores = torch.sigmoid(m_pruned(**inputs).logits.squeeze(-1)).numpy()
    
    dx_rank = rank_order(pruned_scores[:5])
    quantum_rank = rank_order(pruned_scores[5:])
    dx_match = (dx_rank == rank_order(base_scores[:5]))
    quantum_match = (quantum_rank == rank_order(base_scores[5:]))
    
    print(f'\n--- {name} (層数: {len(idx)}) ---')
    print(f'DXクエリ: {np.round(pruned_scores[:5], 4)} | 順位一致: {dx_match} (Top-1正解: {dx_rank[0] == 0})')
    print(f'量子クエリ: {np.round(pruned_scores[5:], 4)} | 順位一致: {quantum_match} (Top-1正解: {quantum_rank[0] == 0})')
