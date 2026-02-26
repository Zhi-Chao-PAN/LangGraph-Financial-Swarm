"""Tests for multi-LLM backend support."""

from unittest.mock import MagicMock, patch

from src.core.llm_backends import (
    PRESETS,
    LLMBackend,
    LLMConfig,
    check_backend_available,
    list_backends,
    select_best_backend,
)


class TestLLMConfig:
    def test_basic(self):
        cfg = LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="deepseek-r1:8b",
            base_url="http://localhost:11434",
        )
        assert cfg.display_name == "ollama:deepseek-r1:8b"
        assert cfg.api_key is None

    def test_openai(self):
        cfg = LLMConfig(
            backend=LLMBackend.OPENAI,
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
        )
        assert cfg.display_name == "openai:gpt-4o"
        assert cfg.api_key == "sk-test"

    def test_defaults(self):
        cfg = LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="test",
            base_url="http://localhost",
        )
        assert cfg.temperature == 0.1
        assert cfg.timeout == 120
        assert cfg.max_tokens == 4096


class TestPresets:
    def test_presets_exist(self):
        assert "ollama-deepseek" in PRESETS
        assert "ollama-qwen" in PRESETS
        assert "openai-gpt4o" in PRESETS
        assert "deepseek-chat" in PRESETS

    def test_preset_types(self):
        assert PRESETS["ollama-deepseek"].backend == LLMBackend.OLLAMA
        assert PRESETS["openai-gpt4o"].backend == LLMBackend.OPENAI
        assert PRESETS["deepseek-chat"].backend == LLMBackend.DEEPSEEK


class TestCheckBackend:
    def test_openai_no_key(self):
        cfg = LLMConfig(
            backend=LLMBackend.OPENAI,
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="",
        )
        assert not check_backend_available(cfg)

    def test_openai_with_key(self):
        cfg = LLMConfig(
            backend=LLMBackend.OPENAI,
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
            api_key="sk-test123",
        )
        assert check_backend_available(cfg)

    def test_deepseek_no_key(self):
        cfg = LLMConfig(
            backend=LLMBackend.DEEPSEEK,
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
        )
        assert not check_backend_available(cfg)

    @patch("src.core.llm_backends.requests")
    def test_ollama_available(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"models": [{"name": "deepseek-r1:8b"}]}
        mock_requests.get.return_value = mock_resp

        cfg = LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="deepseek-r1:8b",
            base_url="http://localhost:11434",
        )
        assert check_backend_available(cfg)

    @patch("src.core.llm_backends.requests")
    def test_ollama_unavailable(self, mock_requests):
        mock_requests.get.side_effect = Exception("Connection refused")

        cfg = LLMConfig(
            backend=LLMBackend.OLLAMA,
            model="deepseek-r1:8b",
            base_url="http://localhost:11434",
        )
        assert not check_backend_available(cfg)

    @patch("src.core.llm_backends.requests")
    def test_custom_available(self, mock_requests):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_requests.get.return_value = mock_resp

        cfg = LLMConfig(
            backend=LLMBackend.CUSTOM,
            model="custom-model",
            base_url="http://my-server:8080",
        )
        assert check_backend_available(cfg)


class TestSelectBestBackend:
    @patch("src.core.llm_backends.check_backend_available")
    def test_preferred(self, mock_check):
        mock_check.return_value = True
        result = select_best_backend("ollama-deepseek")
        assert result is not None
        assert result.model == "deepseek-r1:8b"

    @patch("src.core.llm_backends.check_backend_available")
    def test_fallback(self, mock_check):
        # First call (preferred) fails, second succeeds
        mock_check.side_effect = [False, True, True, True]
        configs = {
            "primary": LLMConfig(
                backend=LLMBackend.OLLAMA,
                model="primary",
                base_url="http://localhost:11434",
            ),
            "fallback": LLMConfig(
                backend=LLMBackend.OLLAMA,
                model="fallback",
                base_url="http://localhost:11434",
            ),
        }
        result = select_best_backend("primary", configs)
        assert result is not None

    @patch("src.core.llm_backends.check_backend_available")
    def test_none_available(self, mock_check):
        mock_check.return_value = False
        result = select_best_backend("ollama-deepseek")
        assert result is None


class TestListBackends:
    @patch("src.core.llm_backends.check_backend_available")
    def test_list(self, mock_check):
        mock_check.return_value = False
        result = list_backends()
        assert len(result) == len(PRESETS)
        assert all("name" in r for r in result)
        assert all("available" in r for r in result)

    @patch("src.core.llm_backends.check_backend_available")
    def test_custom_configs(self, mock_check):
        mock_check.return_value = True
        configs = {
            "test": LLMConfig(
                backend=LLMBackend.OLLAMA,
                model="test",
                base_url="http://localhost",
            ),
        }
        result = list_backends(configs)
        assert len(result) == 1
        assert result[0]["available"]
