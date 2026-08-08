import pytest
from app.retrieval.retriever import get_retriever
from app.retrieval.advanced_retriever import AdvancedRetriever, _doc_key
from langchain_core.documents import Document

def test_doc_key_uniqueness():
    doc1 = Document(page_content="LangChain agent setup")
    doc2 = Document(page_content="LangChain agent setup")
    doc3 = Document(page_content="LangGraph state graph")

    assert _doc_key(doc1) == _doc_key(doc2)
    assert _doc_key(doc1) != _doc_key(doc3)

def test_advanced_retriever_initialization():
    retriever = AdvancedRetriever(use_multi_query=False, use_hybrid=True, use_reranker=False)
    assert retriever.use_multi_query is False
    assert retriever.use_hybrid is True
    assert retriever.use_reranker is False
