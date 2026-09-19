"""LangChain / LlamaIndex 互換性自動検証テストスクリプト"""
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Any, List

# dist_assets を python path に追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dist_assets"))

from ruri_v3_lite import RuriV3Lite
from ruri_v3_reranker_lite import RuriV3RerankerLite

@dataclass
class DummyDocument:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

def test_ruri_v3_lite_langchain_interface():
    print("=== Testing RuriV3Lite LangChain Interface ===")
    model = RuriV3Lite(repo_id="Chottokun/ruri-v3-30m-lite")

    # 1. embed_query
    query = "日本の首都はどこですか？"
    query_emb = model.embed_query(query)
    assert isinstance(query_emb, list), "embed_query must return a list"
    assert len(query_emb) == 256, f"Expected dimension 256, got {len(query_emb)}"
    assert isinstance(query_emb[0], float), "Elements must be floats"
    print(f"✓ embed_query passed (dimension: {len(query_emb)})")

    # 2. embed_documents
    docs = ["日本の首都は東京都です。", "大阪は関西の主要都市です。"]
    docs_emb = model.embed_documents(docs)
    assert isinstance(docs_emb, list), "embed_documents must return a list"
    assert len(docs_emb) == 2, f"Expected 2 embeddings, got {len(docs_emb)}"
    assert len(docs_emb[0]) == 256, f"Expected dimension 256, got {len(docs_emb[0])}"
    print(f"✓ embed_documents passed (count: {len(docs_emb)})")

def test_ruri_v3_lite_llamaindex_interface():
    print("=== Testing RuriV3Lite LlamaIndex Interface ===")
    model = RuriV3Lite(repo_id="Chottokun/ruri-v3-30m-lite")

    # 1. get_query_embedding
    q_emb = model.get_query_embedding("テスト")
    assert isinstance(q_emb, list) and len(q_emb) == 256
    print("✓ get_query_embedding passed")

    # 2. get_text_embedding
    t_emb = model.get_text_embedding("テストテキスト")
    assert isinstance(t_emb, list) and len(t_emb) == 256
    print("✓ get_text_embedding passed")

    # 3. get_text_embeddings
    ts_emb = model.get_text_embeddings(["テキスト1", "テキスト2"])
    assert isinstance(ts_emb, list) and len(ts_emb) == 2
    print("✓ get_text_embeddings passed")

def test_ruri_v3_reranker_langchain_interface():
    print("=== Testing RuriV3RerankerLite LangChain Interface ===")
    reranker = RuriV3RerankerLite(
        repo_id="Chottokun/ruri-v3-reranker-310m-lite",
        precision="int8_full",
        device="cpu"
    )

    query = "日本の首都はどこですか？"
    documents = [
        DummyDocument(page_content="明日の天気は晴れです。", metadata={"id": 1}),
        DummyDocument(page_content="日本の首都は東京都です。", metadata={"id": 2}),
        DummyDocument(page_content="大阪は関西地方です。", metadata={"id": 3}),
    ]

    compressed = reranker.compress_documents(documents, query)
    assert len(compressed) == 3
    # 首都(id: 2) が Top 1 に来ているはず
    assert compressed[0].metadata["id"] == 2
    assert "rerank_score" in compressed[0].metadata
    assert compressed[0].metadata["rerank_score"] > compressed[1].metadata["rerank_score"]
    print(f"✓ compress_documents passed (Top 1 ID: {compressed[0].metadata['id']}, Score: {compressed[0].metadata['rerank_score']:.4f})")

def main():
    try:
        test_ruri_v3_lite_langchain_interface()
        test_ruri_v3_lite_llamaindex_interface()
        test_ruri_v3_reranker_langchain_interface()
        print("\n🎉 ALL LLM LIBRARY COMPATIBILITY TESTS PASSED!")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
