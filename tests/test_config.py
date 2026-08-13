from pathlib import Path
import pytest
from app.core.config import AppConfig, get_settings, settings


def test_config_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    assert settings is s1


def test_config_structure():
    assert isinstance(settings, AppConfig)
    assert isinstance(settings.paths.base_dir, Path)
    assert isinstance(settings.models.chat_model_name, str)
    assert isinstance(settings.rag.top_k_results, int)
    assert settings.rag.chunk_size > 0
    assert ".pdf" in settings.rag.supported_extensions
    assert "faiss_index" in settings.vector_db.faiss_path
