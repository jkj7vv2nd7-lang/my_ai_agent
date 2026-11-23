import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

sys.path.insert(0, "/working_dir/c_cdf7402806b7f807/artifacts/file_generation/ttl=63d/output")
import local_ai_agent as agent


class _FakeResponse:
    def __init__(self, lines: list[dict], status: int = 200) -> None:
        self._raw = [json.dumps(x).encode("utf-8") + b"\n" for x in lines]
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._raw)

    def read(self) -> bytes:
        return b"".join(self._raw)


class TestBrushedUpAgent(unittest.TestCase):
    def test_smart_route_models_prefers_light_and_coder(self):
        models = [
            {"name": "qwen3:32b", "details": {"parameter_size": "32B"}},
            {"name": "qwen3.5:9b", "details": {"parameter_size": "9B"}},
            {"name": "qwen2.5-coder:7b", "details": {"parameter_size": "7B"}},
        ]
        assignments, reasons = agent.smart_route_models(models)
        self.assertEqual(assignments["requirements"], "qwen3.5:9b")
        self.assertEqual(assignments["implementation"], "qwen2.5-coder:7b")
        self.assertNotEqual(assignments["qa"], "qwen3:32b")
        self.assertIn("コード", reasons["implementation"])

    def test_call_ollama_think_400_fallback_and_thinking_stream(self):
        seen_payloads = []

        def fake_urlopen(req, timeout=None):
            payload = json.loads(req.data.decode("utf-8"))
            seen_payloads.append(payload)
            if len(seen_payloads) == 1:
                raise urllib.error.HTTPError(
                    req.full_url,
                    400,
                    "Bad Request",
                    {},
                    fp=io.BytesIO(b'{"error":"think parameter not supported"}'),
                )
            return _FakeResponse([
                {"thinking": "step1", "response": ""},
                {"response": "最終回答", "done": True},
            ])

        with patch("urllib.request.urlopen", side_effect=fake_urlopen), patch("time.sleep", return_value=None):
            res = agent.call_ollama(
                endpoint=agent.DEFAULT_ENDPOINT,
                model="legacy-model",
                prompt="test",
                system="sys",
                options={"num_predict": 100},
                disable_thinking=True,
                max_retries=2,
            )
        self.assertEqual(res, "最終回答")
        self.assertIn("think", seen_payloads[0])
        self.assertNotIn("think", seen_payloads[1])

    def test_extract_and_validate_html(self):
        raw = "```html\n<!DOCTYPE html><html><head><title>App</title></head><body><main>OK</main></body></html>\n```"
        html = agent.extract_html(raw)
        self.assertIsNotNone(html)
        res = agent.validate_html(html)
        self.assertTrue(res["passed"], res)

    def test_pipeline_resume_skips_completed(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            (run_dir / "01_requirements.md").write_text("# req", encoding="utf-8")
            (run_dir / "02_design.md").write_text("# design", encoding="utf-8")
            (run_dir / "run.json").write_text(
                json.dumps({"request": "todo", "completed_stages": ["requirements", "design"], "status": "failed"}),
                encoding="utf-8",
            )
            args = agent.build_parser().parse_args(["todo", "--routing-mode", "single", "--no-stream"])
            stages_called = []

            def fake_call(**kwargs):
                stages_called.append(kwargs["stage"])
                if kwargs["stage"] == "implementation":
                    return "<!DOCTYPE html><html><head><title>A</title></head><body><p>App</p></body></html>"
                return "STATUS: APPROVED\nOK"

            with patch.object(agent, "call_ollama", side_effect=fake_call), \
                 patch.object(agent, "check_render", return_value={"available": False, "ok": None}):
                code = agent.run_pipeline("todo", args, run_dir, resume=True)
            self.assertEqual(code, 0)
            self.assertEqual(stages_called, ["implementation", "qa"])

    def test_run_revision_non_destructive(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            src_html = base / "orig.html"
            src_html.write_text("<!DOCTYPE html><html><head><title>Old</title></head><body><p>Old</p></body></html>", encoding="utf-8")
            out_root = base / "runs"
            args = agent.build_parser().parse_args(["ボタンを追加して", "--routing-mode", "single", "--no-stream"])

            def fake_call(**kwargs):
                if kwargs["stage"] == "design":
                    return "# 修正方針\nボタンを追加"
                if kwargs["stage"] == "implementation":
                    return "<!DOCTYPE html><html><head><title>New</title></head><body><button>Click</button></body></html>"
                return "STATUS: APPROVED\nOK"

            with patch.object(agent, "call_ollama", side_effect=fake_call), \
                 patch.object(agent, "check_render", return_value={"available": False, "ok": None}):
                code, new_dir = agent.run_revision(src_html, "ボタンを追加して", args, output_root=out_root)

            self.assertEqual(code, 0)
            self.assertTrue((new_dir / "00_original_backup.html").exists())
            self.assertTrue((new_dir / "index.html").exists())
            self.assertIn("Old", src_html.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
