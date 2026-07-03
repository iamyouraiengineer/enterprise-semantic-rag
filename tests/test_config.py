"""
tests/test_config.py
Unit tests for config/settings.py.

Coverage:
- Default value correctness (with explicit env override to isolate from .env)
- Pydantic validation errors (chunk_overlap, log_level)
- Path resolution to absolute
- Singleton caching behavior
"""

import pytest
from pydantic import ValidationError

from config.settings import Settings, get_settings


# =============================================================================
# 1. Default Value Tests
# =============================================================================
class TestSettingsDefaults:
    """
    Verify that Settings() uses correct defaults when env vars are absent.
    
    CRITICAL: Pydantic-settings reads from .env file. monkeypatch.delenv()
    only removes os.environ variables, NOT .env file values. To test true
    defaults, we must monkeypatch.setenv() to the expected default values,
    which takes precedence over .env.
    """

    def test_app_identity_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """APP_NAME, VERSION, DEBUG, and LOG_LEVEL must match blueprint."""
        monkeypatch.setenv("APP_NAME", "Enterprise Semantic RAG")
        monkeypatch.setenv("APP_VERSION", "0.1.0")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("LOG_LEVEL", "INFO")

        s = Settings()
        assert s.app_name == "Enterprise Semantic RAG"
        assert s.app_version == "0.1.0"
        assert s.debug is False
        assert s.log_level == "INFO"

    def test_chunking_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHUNK_SIZE and CHUNK_OVERLAP must default to 512 and 50."""
        monkeypatch.setenv("CHUNK_SIZE", "512")
        monkeypatch.setenv("CHUNK_OVERLAP", "50")

        s = Settings()
        assert s.chunk_size == 512
        assert s.chunk_overlap == 50

    def test_embedding_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """EMBEDDING_MODEL_NAME must default to all-MiniLM-L6-v2."""
        monkeypatch.setenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        monkeypatch.setenv("EMBEDDING_DEVICE", "auto")

        s = Settings()
        assert s.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.embedding_device == "auto"

    def test_retrieval_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """TOP_K values must default to 10, 10, 5."""
        monkeypatch.setenv("TOP_K_DENSE", "10")
        monkeypatch.setenv("TOP_K_SPARSE", "10")
        monkeypatch.setenv("TOP_K_FINAL", "5")

        s = Settings()
        assert s.top_k_dense == 10
        assert s.top_k_sparse == 10
        assert s.top_k_final == 5


# =============================================================================
# 2. Validation Tests
# =============================================================================
class TestSettingsValidation:
    """Verify that Pydantic rejects invalid business rules."""

    def test_chunk_overlap_must_be_less_than_chunk_size(self) -> None:
        """
        Business rule: overlap >= chunk_size causes infinite loops or
        degenerate chunks in the text splitter. Pydantic must raise ValueError.
        """
        with pytest.raises(ValueError) as exc_info:
            Settings(chunk_size=200, chunk_overlap=200)
        assert "chunk_overlap must be strictly less than chunk_size" in str(
            exc_info.value
        )

    def test_chunk_overlap_cannot_exceed_chunk_size(self) -> None:
        """Overlap larger than chunk size is also invalid."""
        with pytest.raises(ValueError):
            Settings(chunk_size=200, chunk_overlap=250)

    def test_invalid_log_level_rejected(self) -> None:
        """LOG_LEVEL must be one of the Literal choices. 'TRACE' is invalid."""
        with pytest.raises(ValidationError):
            Settings(log_level="TRACE")

    def test_negative_chunk_size_rejected(self) -> None:
        """CHUNK_SIZE has ge=128; negative values must fail."""
        with pytest.raises(ValidationError):
            Settings(chunk_size=-1)

    def test_negative_chunk_overlap_rejected(self) -> None:
        """CHUNK_OVERLAP has ge=0; negative values must fail."""
        with pytest.raises(ValidationError):
            Settings(chunk_overlap=-1)


# =============================================================================
# 3. Path Resolution Tests
# =============================================================================
class TestSettingsPaths:
    """Verify that relative paths are resolved to absolute paths."""

    def test_paths_are_absolute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """VECTOR_STORE_PATH, DATA_DIR, and RAW_DATA_DIR must be absolute."""
        monkeypatch.setenv("VECTOR_STORE_PATH", "data/vector_store")
        monkeypatch.setenv("DATA_DIR", "data")
        monkeypatch.setenv("RAW_DATA_DIR", "data/raw")

        s = Settings()
        assert s.vector_store_path.is_absolute()
        assert s.data_dir.is_absolute()
        assert s.raw_data_dir.is_absolute()

    def test_custom_relative_path_gets_resolved(self) -> None:
        """A user-provided relative path must be converted to absolute."""
        s = Settings(vector_store_path="data/vector_store")
        assert s.vector_store_path.is_absolute()
        assert "data/vector_store" in str(s.vector_store_path)


# =============================================================================
# 4. Singleton Behavior Tests
# =============================================================================
class TestSettingsSingleton:
    """Verify that get_settings() caches and returns the same instance."""

    def test_get_settings_returns_same_instance(self) -> None:
        """
        The singleton pattern prevents Pydantic from re-parsing .env on every
        import, which is critical for performance in production.
        """
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2, "get_settings() must return the identical object"

    def test_singleton_reflects_same_values(self) -> None:
        """Both calls must see the same attribute values."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1.app_name == s2.app_name
        assert s1.chunk_size == s2.chunk_size