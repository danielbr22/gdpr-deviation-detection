import asyncio
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _make_httpx_response(json_body: dict, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.raise_for_status = MagicMock()
    return resp


class TestOllamaBackend(unittest.TestCase):
    def setUp(self):
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"

    def test_call_async_ollama_returns_content(self):
        response_body = {"message": {"content": '{"match": 1, "reasoning": "yes"}'}}
        mock_resp = _make_httpx_response(response_body)

        async def fake_post(*args, **kwargs):
            return mock_resp

        import src.utils.llm_client as client
        # Reload to pick up env vars
        import importlib
        importlib.reload(client)

        with patch("httpx.AsyncClient") as mock_cls:
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_ctx

            result = asyncio.run(client.call_async("sys", "user", json_mode=True))

        self.assertEqual(result, '{"match": 1, "reasoning": "yes"}')

    def test_call_async_ollama_adds_no_think_prefix(self):
        captured = {}
        response_body = {"message": {"content": "yes"}}
        mock_resp = _make_httpx_response(response_body)

        import src.utils.llm_client as client
        import importlib
        importlib.reload(client)

        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                async def capture_post(url, **kwargs):
                    captured["payload"] = kwargs.get("json", {})
                    return mock_resp
                mock_ctx.post = capture_post
                mock_cls.return_value = mock_ctx
                return await client.call_async("sys", "user")

        asyncio.run(run())
        messages = captured["payload"]["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        self.assertTrue(user_content.startswith("/no_think\n"))

    def test_concurrency_ollama_default(self):
        import src.utils.llm_client as client
        import importlib
        importlib.reload(client)
        os.environ.pop("LLM_CONCURRENCY", None)
        self.assertEqual(client.concurrency(), 1)


class TestOpenAIBackend(unittest.TestCase):
    def setUp(self):
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

    def tearDown(self):
        os.environ["LLM_PROVIDER"] = "ollama"

    def test_call_async_openai_returns_content(self):
        response_body = {"choices": [{"message": {"content": '{"match": 2}'}}]}
        mock_resp = _make_httpx_response(response_body)

        import src.utils.llm_client as client
        import importlib
        importlib.reload(client)

        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                mock_ctx.post = AsyncMock(return_value=mock_resp)
                mock_cls.return_value = mock_ctx
                return await client.call_async("sys", "user", json_mode=True)

        result = asyncio.run(run())
        self.assertEqual(result, '{"match": 2}')

    def test_json_mode_sets_response_format(self):
        captured = {}
        response_body = {"choices": [{"message": {"content": "{}"}}]}
        mock_resp = _make_httpx_response(response_body)

        import src.utils.llm_client as client
        import importlib
        importlib.reload(client)

        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                async def capture_post(url, **kwargs):
                    captured["payload"] = kwargs.get("json", {})
                    return mock_resp
                mock_ctx.post = capture_post
                mock_cls.return_value = mock_ctx
                return await client.call_async("My system prompt", "user", json_mode=True)

        asyncio.run(run())
        self.assertEqual(captured["payload"].get("response_format"), {"type": "json_object"})

    def test_openai_no_think_prefix_not_added(self):
        captured = {}
        response_body = {"choices": [{"message": {"content": "yes"}}]}
        mock_resp = _make_httpx_response(response_body)

        import src.utils.llm_client as client
        import importlib
        importlib.reload(client)

        async def run():
            with patch("httpx.AsyncClient") as mock_cls:
                mock_ctx = MagicMock()
                mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
                mock_ctx.__aexit__ = AsyncMock(return_value=False)
                async def capture_post(url, **kwargs):
                    captured["payload"] = kwargs.get("json", {})
                    return mock_resp
                mock_ctx.post = capture_post
                mock_cls.return_value = mock_ctx
                return await client.call_async("sys", "user input")

        asyncio.run(run())
        messages = captured["payload"]["messages"]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        self.assertFalse(user_content.startswith("/no_think"))

    def test_concurrency_openai_default(self):
        import src.utils.llm_client as client
        import importlib
        importlib.reload(client)
        os.environ.pop("LLM_CONCURRENCY", None)
        self.assertEqual(client.concurrency(), 20)


if __name__ == "__main__":
    unittest.main()
