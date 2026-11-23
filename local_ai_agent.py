#!/usr/bin/env python3
"""
Local AI App Builder Agent — Polished & Complete Edition
========================================================
4段階パイプライン（要件定義 → デザイン設計 → 実装 → QAレビュー）で
外部CDN不要の単一HTMLアプリをローカルOllamaで自動生成・改修するエージェント。

主な改善・完成ポイント:
1. 途中停止対策:
   - keep_alive="5m" でステージ間のモデル再ロード・メモリ断片化を防止
   - 思考モデル (qwen3 / qwen3.5 等) 向けに think=False を既定送信（非対応時は自動フォールバック）
   - thinking ストリーム受信時も進捗通知を行い無応答フリーズを防止
   - 通信/タイムアウト/空応答に対する指数バックオフ付き自動リトライ (最大3回)
   - HTML途中切れ時の num_predict 自動引き上げ (10240 -> 14336) と修復ループ
2. 機能・仕様の完成:
   - 16GB環境向けスマートモデル自動ルーティング (20B以下優先・工程別スコアリング)
   - 中断/エラー地点からのワンクリック再開 (resume)
   - 完成アプリ・外部HTMLファイルの非破壊修正フロー (run_revision)
   - Ollama接続確認・導入済みモデル取得・ブラウザ即時プレビュー
3. UI/UXのブラッシュアップ:
   - 接続ステータスバッジ・プリセットプロンプト・4工程進捗カード・成果物ビューアタブ
"""

from __future__ import annotations

import argparse
import datetime as dt
from html.parser import HTMLParser
import http.client
import json
import os
from pathlib import Path
import queue
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

DEFAULT_ENDPOINT = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3.5:9b"
STAGES = ("requirements", "design", "implementation", "qa")
STAGE_LABELS = {
    "requirements": "1/4 要件定義",
    "design": "2/4 デザイン設計",
    "implementation": "3/4 実装",
    "qa": "4/4 QAレビュー",
}

PROMPT_PRESETS: dict[str, str] = {
    "ポモドーロタイマー": (
        "集中25分・休憩5分のポモドーロタイマーを作ってください。"
        "開始/一時停止/リセット、作業タスク名のメモ、完了セッション数の記録（localStorage保存）、"
        "終了時のWeb Audioビープ通知を備えてください。"
    ),
    "収支・家計メモ": (
        "シンプルで見やすい家計簿・収支メモアプリを作ってください。"
        "日付・カテゴリ・金額・メモの入力、収入/支出の合計と残高カード、カテゴリ別集計バー、"
        "CSVエクスポートとlocalStorage永続化に対応してください。"
    ),
    "Markdownメモ帳": (
        "左右2ペインのリアルタイムMarkdownメモ帳を作ってください。"
        "複数ノートの作成・検索・削除、見出し/太字/リスト/コードブロックのプレビュー、"
        "文字数カウント、localStorage自動保存と.mdダウンロード機能を入れてください。"
    ),
    "4択クイズアプリ": (
        "プログラミング基礎の4択クイズアプリを作ってください。"
        "全10問、選択肢クリックで即時正誤判定と解説表示、進捗バー、最終スコアと間違えた問題の復習リストを表示してください。"
    ),
}


class StageError(RuntimeError):
    def __init__(self, stage: str, message: str, cancelled: bool = False) -> None:
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.cancelled = cancelled


def validate_local_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint.strip())
    if parsed.scheme != "http":
        raise StageError("connection", "ローカル環境保護のため http:// のみ許可されています。")
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise StageError("connection", "ローカルホスト (localhost / 127.0.0.1 / ::1) のみ接続できます。")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise StageError("connection", "エンドポイントURLに認証情報やクエリは含められません。")
    return endpoint.strip()


def tags_endpoint_from_generate(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/api/tags", "", "", ""))


def fetch_ollama_models(endpoint: str, timeout: float = 15.0) -> list[dict[str, Any]]:
    endpoint = validate_local_endpoint(endpoint)
    tags_url = tags_endpoint_from_generate(endpoint)
    req = urllib.request.Request(tags_url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=min(timeout, 30.0)) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        models = data.get("models")
        if not isinstance(models, list):
            raise StageError("connection", "Ollama /api/tags の応答形式が不正です。")
        return models
    except StageError:
        raise
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        socket.timeout,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        raise StageError("connection", f"Ollama に接続できません ({tags_url}): {exc}") from exc


def _parse_param_billions(model_info: dict[str, Any]) -> float | None:
    name = str(model_info.get("name", ""))
    details = model_info.get("details") or {}
    param_str = str(details.get("parameter_size") or "")
    for source in (param_str, name):
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", source)
        if m:
            return float(m.group(1))
        m2 = re.search(r":(\d+(?:\.\d+)?)b", source, flags=re.IGNORECASE)
        if m2:
            return float(m2.group(1))
    return None


def _model_score(model_info: dict[str, Any], stage: str) -> tuple[int, str]:
    name = str(model_info.get("name", "")).lower()
    params = _parse_param_billions(model_info)
    is_coder = any(k in name for k in ("coder", "codellama", "starcoder", "deepseek-coder"))
    is_reasoning = any(k in name for k in ("qwen3", "deepseek-r1", "gemma3", "phi4"))
    is_ja_good = any(k in name for k in ("qwen", "gemma", "elyza", "swallow"))
    is_vision_or_embed = any(k in name for k in ("-vl", "vl:", "llava", "minicpm-v", "moondream", "embed"))

    score = 0
    notes: list[str] = []

    if stage in {"implementation", "qa"}:
        if params is not None and 7.0 <= params <= 10.5:
            score += 50
            notes.append("9B級")
        if "qwen3.5" in name:
            score += 20
        if is_coder:
            score += 60
            notes.append("コード欠陥検出向き" if stage == "qa" else "コード特化モデル")
        if is_reasoning:
            score += 10
    else:
        if params is not None and 6.5 <= params <= 9.5:
            score += 40
            notes.append("軽量バランス")
        if not is_coder:
            score += 70
            notes.append("汎用・推論モデル")
        else:
            score += 25
        if is_reasoning:
            score += 25
            notes.append("推論が強い系統")
        if is_ja_good:
            score += 15
            notes.append("日本語が得意")

    if is_vision_or_embed:
        score -= 200
        notes.append("画像/埋込特化(テキスト工程非推奨)")

    if params is not None and params > 20.0:
        score -= 40
        notes.append("20B超の大型モデル")

    return score, ("・".join(notes) if notes else "標準")


def smart_route_models(model_details_list: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    """16GB環境向けに20B以下のモデルを優先しつつ各工程の最適モデルを選定する。"""
    if not model_details_list:
        raise StageError("connection", "Ollama にインストール済みのモデルがありません。`ollama pull qwen3.5:9b` 等を実行してください。")

    light_models = [
        m for m in model_details_list
        if (_parse_param_billions(m) is None or (_parse_param_billions(m) or 0) <= 20.0)
    ]
    pool = light_models if light_models else model_details_list
    fallback_note = "" if light_models else "・他に軽量モデルがなく大型で代用"

    assignments: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for stage in STAGES:
        ranked = sorted(pool, key=lambda m: _model_score(m, stage)[0], reverse=True)
        best = ranked[0]
        _, reason = _model_score(best, stage)
        assignments[stage] = str(best.get("name", DEFAULT_MODEL))
        reasons[stage] = f"{reason}{fallback_note}"

    return assignments, reasons


def generation_options(args: argparse.Namespace, stage: str, kind: str | None = None, attempt: int = 0) -> dict[str, Any]:
    base_ctx = max(2048, int(getattr(args, "num_ctx", 4096)))
    base_predict = max(2048, int(getattr(args, "num_predict", 4096)))
    base_temp = float(getattr(args, "temperature", 0.3))
    think_bonus = 4096 if getattr(args, "think", False) else 0

    if stage in {"requirements", "design"}:
        return {
            "temperature": 0.3,
            "num_predict": min(max(base_predict, 1536), 2048) + think_bonus,
            "num_ctx": base_ctx,
        }
    if stage == "implementation" or kind == "repair":
        predict_target = max(base_predict, 10240 if attempt == 0 else 14336) + think_bonus
        return {
            "temperature": 0.2 if kind != "repair" else 0.15,
            "num_predict": predict_target,
            "num_ctx": min(max(base_ctx, 8192), 16384),
        }
    if stage == "qa":
        return {
            "temperature": 0.1,
            "num_predict": 1536 + think_bonus,
            "num_ctx": min(max(base_ctx, 8192), 16384),
        }
    return {
        "temperature": base_temp,
        "num_predict": base_predict,
        "num_ctx": base_ctx,
    }


def call_ollama(
    *,
    endpoint: str,
    model: str,
    prompt: str,
    system: str,
    options: dict[str, Any],
    timeout: float = 600.0,
    stream: bool = True,
    keep_alive: str | int = "5m",
    disable_thinking: bool = True,
    max_retries: int = 3,
    stage: str = "generation",
    token_callback: Callable[[int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    endpoint = validate_local_endpoint(endpoint)
    use_stream = stream or (token_callback is not None)
    include_think_flag = disable_thinking
    last_error: Exception | None = None

    for retry in range(max_retries):
        if cancel_event and cancel_event.is_set():
            raise StageError(stage, "ユーザーにより中止されました。", cancelled=True)

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": use_stream,
            "keep_alive": keep_alive,
            "options": options,
        }
        if include_think_flag:
            payload["think"] = False

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if not use_stream:
                    raw = resp.read().decode("utf-8")
                    data = json.loads(raw)
                    if data.get("error"):
                        raise StageError(stage, f"Ollama エラー: {data['error']}")
                    text = data.get("response", "")
                    if not isinstance(text, str) or not text.strip():
                        raise StageError(stage, "Ollama から空の応答が返されました。")
                    return text

                chunks: list[str] = []
                total_chars = 0
                last_notified = 0
                saw_line = False
                for raw_line in resp:
                    if cancel_event and cancel_event.is_set():
                        raise StageError(stage, "ユーザーにより中止されました。", cancelled=True)
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    saw_line = True
                    obj = json.loads(line)
                    if obj.get("error"):
                        raise StageError(stage, f"Ollama エラー: {obj['error']}")
                    piece = obj.get("response") or ""
                    thinking_piece = obj.get("thinking") or ""
                    if piece:
                        chunks.append(piece)
                        total_chars += len(piece)
                    elif thinking_piece:
                        total_chars += len(thinking_piece)
                    if token_callback and (total_chars - last_notified >= 300):
                        last_notified = total_chars
                        token_callback(total_chars, stage)
                    if obj.get("done"):
                        break

                if not saw_line:
                    raise StageError(stage, "Ollama ストリームが空でした。")
                result = "".join(chunks).strip()
                if not result:
                    if not include_think_flag:
                        include_think_flag = True
                        last_error = StageError(stage, "思考トークン上限のため思考モードOFFで再試行します。")
                        continue
                    raise StageError(stage, "Ollama の応答テキストが空でした（思考トークン上限の可能性があります）。")
                if token_callback:
                    token_callback(len(result), stage)
                return result

        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            if exc.code == 400 and include_think_flag and "think" in err_body.lower():
                include_think_flag = False
                last_error = exc
                continue
            last_error = StageError(stage, f"HTTP {exc.code}: {err_body or exc.reason}")
        except StageError as exc:
            if exc.cancelled:
                raise
            last_error = exc
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            http.client.IncompleteRead,
            json.JSONDecodeError,
            UnicodeDecodeError,
            OSError,
        ) as exc:
            last_error = StageError(stage, f"通信/タイムアウトエラー: {exc}")

        if retry + 1 < max_retries:
            wait_sec = min(2 ** (retry + 1), 8)
            for _ in range(wait_sec * 4):
                if cancel_event and cancel_event.is_set():
                    raise StageError(stage, "ユーザーにより中止されました。", cancelled=True)
                time.sleep(0.25)

    if isinstance(last_error, StageError):
        raise last_error
    raise StageError(stage, f"Ollama 呼び出しに失敗しました: {last_error}")


def extract_html(text: str) -> str | None:
    cleaned = text.strip().lstrip("\ufeff")
    fences = re.findall(r"```(?:html)?\s*\n(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    candidates = fences + [cleaned]
    for cand in candidates:
        m = re.search(r"<!doctype\s+html|<html\b", cand, flags=re.IGNORECASE)
        if m:
            return cand[m.start() :].strip()
    return None


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.html_count = 0
        self.head_count = 0
        self.body_count = 0
        self.in_body = False
        self.in_style = False
        self.in_script = False
        self.body_has_visible = False
        self.unclosed_struct: list[str] = []
        self.errors: list[str] = []
        self.inline_scripts: list[str] = []
        self._current_script: list[str] = []
        self._current_script_has_src = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        attr_dict = {k.lower(): v for k, v in attrs}
        if t == "html":
            self.html_count += 1
            self.unclosed_struct.append("html")
        elif t == "head":
            self.head_count += 1
            self.unclosed_struct.append("head")
        elif t == "body":
            self.body_count += 1
            self.in_body = True
            self.unclosed_struct.append("body")
        elif t == "style":
            self.in_style = True
        elif t == "script":
            self.in_script = True
            self._current_script = []
            self._current_script_has_src = bool(attr_dict.get("src"))
            if self.in_body and self._current_script_has_src:
                self.body_has_visible = True
        elif self.in_body and not self.in_style and not self.in_script:
            self.body_has_visible = True

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in {"html", "head", "body"}:
            if t in self.unclosed_struct:
                if self.unclosed_struct[-1] != t:
                    self.errors.append(f"閉じタグの順序不正: </{t}>")
                self.unclosed_struct.remove(t)
            else:
                self.errors.append(f"対応する開始タグのない </{t}>")
            if t == "body":
                self.in_body = False
        elif t == "style":
            self.in_style = False
        elif t == "script":
            if self.in_script and not self._current_script_has_src:
                code = "".join(self._current_script).strip()
                if code:
                    self.inline_scripts.append(code)
            self.in_script = False
            self._current_script = []

    def handle_data(self, data: str) -> None:
        if self.in_script and not self._current_script_has_src:
            self._current_script.append(data)
        elif self.in_body and not self.in_style and not self.in_script and data.strip():
            self.body_has_visible = True


def validate_html(html: str) -> dict[str, Any]:
    parser = _DocumentParser()
    errors: list[str] = []
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:
        errors.append(f"HTML パースエラー: {exc}")

    errors.extend(parser.errors)
    if parser.html_count != 1:
        errors.append(f"<html> タグ数が不正です ({parser.html_count})")
    if parser.head_count != 1:
        errors.append(f"<head> タグ数が不正です ({parser.head_count})")
    if parser.body_count != 1:
        errors.append(f"<body> タグ数が不正です ({parser.body_count})")
    if parser.unclosed_struct:
        errors.append(f"未閉じの構造タグがあります: {', '.join(parser.unclosed_struct)}")
    if not parser.body_has_visible:
        errors.append("<body> 内に表示可能なコンテンツがありません。")

    js_status = "skipped"
    js_detail = "インラインJSなし"
    if parser.inline_scripts:
        node = shutil.which("node")
        if not node:
            js_status = "unavailable"
            js_detail = "node 未導入のため構文チェックをスキップ"
        else:
            combined_js = "\n;\n".join(parser.inline_scripts)
            with tempfile.NamedTemporaryFile("w", suffix=".js", encoding="utf-8", delete=False) as tf:
                tf.write(combined_js)
                tmp_js = tf.name
            try:
                proc = subprocess.run([node, "--check", tmp_js], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
                if proc.returncode == 0:
                    js_status = "passed"
                    js_detail = "JS 構文チェック合格"
                else:
                    js_status = "failed"
                    js_detail = (proc.stderr or proc.stdout).strip()[:500]
                    errors.append(f"JavaScript 構文エラー: {js_detail}")
            except Exception as exc:
                js_status = "unavailable"
                js_detail = f"node 実行エラー: {exc}"
            finally:
                try:
                    os.unlink(tmp_js)
                except OSError:
                    pass

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "javascript": {"status": js_status, "detail": js_detail},
    }


def find_browser() -> str | None:
    candidates = [
        "google-chrome",
        "chromium",
        "chromium-browser",
        "msedge",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    for cand in candidates:
        found = shutil.which(cand) if not os.path.isabs(cand) else (cand if os.path.exists(cand) else None)
        if found:
            return found
    return None


def check_render(html: str, work_dir: Path | None = None) -> dict[str, Any]:
    browser = find_browser()
    if not browser:
        return {"available": False, "ok": None, "reason": "ヘッドレスブラウザ未検出"}

    target_dir = work_dir or Path(tempfile.gettempdir())
    html_path = target_dir / "_render_check.html"
    html_path.write_text(html, encoding="utf-8")
    try:
        proc = subprocess.run(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                "--virtual-time-budget=3000",
                "--dump-dom",
                html_path.as_uri(),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        dom = proc.stdout or ""
        if "<html" not in dom.lower():
            return {"available": False, "ok": None, "reason": "DOM 出力が空でした"}
        body_match = re.search(r"<body[^>]*>(.*)</body>", dom, flags=re.IGNORECASE | re.DOTALL)
        body_inner = body_match.group(1) if body_match else dom
        text_only = re.sub(r"<(script|style|noscript|template)\b.*?</\1>", "", body_inner, flags=re.IGNORECASE | re.DOTALL)
        text_only = re.sub(r"<[^>]+>", " ", text_only).strip()
        return {
            "available": True,
            "ok": len(text_only) >= 4,
            "visible_text_chars": len(text_only),
        }
    except Exception as exc:
        return {"available": False, "ok": None, "reason": f"レンダーチェック失敗: {exc}"}
    finally:
        try:
            html_path.unlink(missing_ok=True)
        except OSError:
            pass


REQ_SYSTEM = "あなたは経験豊富なプロダクトマネージャーです。ユーザーの要望から、単一HTMLアプリとして実装可能な要件定義書を日本語の箇条書き（400〜700文字程度）で簡潔に出力してください。"
DESIGN_SYSTEM = "あなたはUI/UXデザイナー兼フロントエンド設計者です。要件定義に基づき、画面構成・配色・状態管理・操作フローを日本語の箇条書き（400〜800文字程度）で簡潔に設計してください。"
IMPL_SYSTEM = (
    "あなたは熟練のフロントエンドエンジニアです。設計に基づき、外部CDNに依存せずブラウザ単体で完結して動作する"
    "高品質な単一HTMLファイル（<!DOCTYPE html> から </html> まで）を完全に出力してください。途中で省略しないでください。"
)
REPAIR_SYSTEM = (
    "あなたはフロントエンドのデバッグ専門家です。以下のHTMLコードとエラー/欠陥リストを修正し、"
    "必ず <!DOCTYPE html> から </html> まで完結した単一HTMLファイルのみを出力してください。"
)
REVISION_PLAN_SYSTEM = (
    "あなたはフロントエンド開発のリードエンジニアです。既存HTMLとユーザーの修正依頼を読み、"
    "既存機能を壊さずに修正を反映するための具体的な改修方針を簡潔な日本語Markdownで出力してください。"
)
QA_SYSTEM = (
    "あなたはQAエンジニアです。要件定義と実装HTMLを比較し、致命的な欠陥や動作しない機能がないか検査してください。\n"
    "問題がなければ先頭行に `STATUS: APPROVED` と出力してください。\n"
    "修正が必要な場合は先頭行に `STATUS: NEEDS_REVISION` と出力し、箇条書きで具体的な修正点を挙げてください。"
)


def create_run_directory(base_dir: Path | None = None) -> Path:
    root = base_dir or (Path.cwd() / "runs")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"run_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"run_{stamp}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    (run_dir / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_stage_models(args: argparse.Namespace, log: Callable[[str], None]) -> dict[str, str]:
    routing_mode = getattr(args, "routing_mode", "single")
    if routing_mode == "smart":
        try:
            installed = fetch_ollama_models(args.endpoint, timeout=float(args.timeout))
            assignments, reasons = smart_route_models(installed)
            for st in STAGES:
                log(f"  [モデル自動選定] {STAGE_LABELS[st]}: {assignments[st]} ({reasons[st]})")
            return assignments
        except StageError as exc:
            log(f"  [注意] スマート自動割当に失敗したため単一モデル ({args.model}) を使用します: {exc.message}")
    return {st: args.model for st in STAGES}


def run_pipeline(
    request_text: str,
    args: argparse.Namespace,
    run_dir: Path,
    resume: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    token_callback: Callable[[int, str], None] | None = None,
    stage_callback: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> int:
    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def set_stage(st: str, status: str) -> None:
        if stage_callback:
            stage_callback(st, status)

    manifest_path = run_dir / "run.json"
    if resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = set(manifest.get("completed_stages", []))
        log(f"前回の実行から再開します (完了済み: {', '.join(sorted(completed)) or 'なし'})")
        for st in completed:
            set_stage(st, "done")
    else:
        completed = set()
        manifest = {
            "mode": "build",
            "request": request_text,
            "model": args.model,
            "routing_mode": getattr(args, "routing_mode", "single"),
            "endpoint": args.endpoint,
            "status": "running",
            "completed_stages": [],
        }
        _save_manifest(run_dir, manifest)

    stage_models = _resolve_stage_models(args, log)
    manifest["stage_models"] = stage_models
    _save_manifest(run_dir, manifest)

    req_file = run_dir / "01_requirements.md"
    design_file = run_dir / "02_design.md"
    html_file = run_dir / "index.html"
    draft_file = run_dir / "index_draft_unverified.html"
    checks_file = run_dir / "03_static_checks.json"
    qa_file = run_dir / "04_qa_report.md"

    base_kwargs = {
        "endpoint": args.endpoint,
        "timeout": float(args.timeout),
        "stream": not getattr(args, "no_stream", False),
        "keep_alive": getattr(args, "keep_alive", "5m"),
        "disable_thinking": not getattr(args, "think", False),
        "token_callback": token_callback,
        "cancel_event": cancel_event,
    }

    try:
        # Stage 1: Requirements
        if "requirements" in completed and req_file.exists():
            req_text = req_file.read_text(encoding="utf-8")
            log("[1/4] 要件定義は完了済みのためスキップします。")
            set_stage("requirements", "done")
        else:
            set_stage("requirements", "running")
            log(f"[1/4] 要件定義を生成中... (model: {stage_models['requirements']})")
            req_text = call_ollama(
                model=stage_models["requirements"],
                prompt=f"次のアプリ要望について要件定義書を作成してください:\n\n{request_text}",
                system=REQ_SYSTEM,
                options=generation_options(args, "requirements"),
                stage="requirements",
                **base_kwargs,
            )
            req_file.write_text(req_text, encoding="utf-8")
            completed.add("requirements")
            manifest["completed_stages"] = sorted(completed)
            _save_manifest(run_dir, manifest)
            set_stage("requirements", "done")

        # Stage 2: Design
        if "design" in completed and design_file.exists():
            design_text = design_file.read_text(encoding="utf-8")
            log("[2/4] デザイン設計は完了済みのためスキップします。")
            set_stage("design", "done")
        else:
            set_stage("design", "running")
            log(f"[2/4] デザイン設計を生成中... (model: {stage_models['design']})")
            design_text = call_ollama(
                model=stage_models["design"],
                prompt=f"要件定義:\n{req_text}\n\nこの要件に対するUI/機能設計書を作成してください。",
                system=DESIGN_SYSTEM,
                options=generation_options(args, "design"),
                stage="design",
                **base_kwargs,
            )
            design_file.write_text(design_text, encoding="utf-8")
            completed.add("design")
            manifest["completed_stages"] = sorted(completed)
            _save_manifest(run_dir, manifest)
            set_stage("design", "done")

        # Stage 3: Implementation + Static Check Repair Loop
        if "implementation" in completed and html_file.exists():
            html_code = html_file.read_text(encoding="utf-8")
            log("[3/4] 実装は完了済みのためスキップします。")
            set_stage("implementation", "done")
        else:
            set_stage("implementation", "running")
            log(f"[3/4] HTMLアプリを実装中... (model: {stage_models['implementation']})")
            raw_impl = call_ollama(
                model=stage_models["implementation"],
                prompt=f"要件定義:\n{req_text}\n\n設計書:\n{design_text}\n\n完全な単一HTMLファイルを出力してください。",
                system=IMPL_SYSTEM,
                options=generation_options(args, "implementation", attempt=0),
                stage="implementation",
                **base_kwargs,
            )
            html_code = extract_html(raw_impl) or ""
            check_res = validate_html(html_code) if html_code else {"passed": False, "errors": ["HTMLタグを抽出できませんでした。"]}

            max_repairs = max(1, int(getattr(args, "max_impl_repairs", 3)))
            for repair_idx in range(1, max_repairs + 1):
                if check_res["passed"]:
                    break
                err_summary = "; ".join(check_res["errors"])
                log(f"[3/4] 静的チェック修復 ({repair_idx}/{max_repairs}): {err_summary}")
                repair_prompt = (
                    f"以下のエラーを修正し、<!DOCTYPE html> から </html> まで完全に完結する"
                    f"コンパクトな単一HTMLを出力してください。\nエラー: {err_summary}\n\n"
                    f"現在のHTML:\n{html_code or raw_impl}"
                )
                raw_impl = call_ollama(
                    model=stage_models["implementation"],
                    prompt=repair_prompt,
                    system=REPAIR_SYSTEM,
                    options=generation_options(args, "implementation", kind="repair", attempt=repair_idx),
                    stage="implementation",
                    **base_kwargs,
                )
                html_code = extract_html(raw_impl) or ""
                check_res = validate_html(html_code) if html_code else {"passed": False, "errors": ["HTMLタグを抽出できませんでした。"]}

            checks_file.write_text(json.dumps(check_res, ensure_ascii=False, indent=2), encoding="utf-8")
            if not check_res["passed"]:
                if html_code or raw_impl:
                    draft_file.write_text(html_code or raw_impl, encoding="utf-8")
                set_stage("implementation", "error")
                raise StageError("implementation", f"HTML検証に失敗しました: {'; '.join(check_res['errors'])}")

            html_file.write_text(html_code, encoding="utf-8")
            completed.add("implementation")
            manifest["completed_stages"] = sorted(completed)
            _save_manifest(run_dir, manifest)
            set_stage("implementation", "done")

        # Stage 4: QA Review Loop
        set_stage("qa", "running")
        log(f"[4/4] QAレビューを実施中... (model: {stage_models['qa']})")
        max_qa = max(1, int(getattr(args, "max_qa_revisions", 2)))
        approved = False
        for qa_round in range(1, max_qa + 1):
            render_info = check_render(html_code, run_dir)
            qa_prompt = (
                f"要件定義:\n{req_text}\n\nレンダー結果: {json.dumps(render_info, ensure_ascii=False)}\n\n"
                f"HTML:\n{html_code}"
            )
            qa_report = call_ollama(
                model=stage_models["qa"],
                prompt=qa_prompt,
                system=QA_SYSTEM,
                options=generation_options(args, "qa"),
                stage="qa",
                **base_kwargs,
            )
            qa_file.write_text(qa_report, encoding="utf-8")
            first_line = qa_report.strip().splitlines()[0].upper() if qa_report.strip() else ""
            if "APPROVED" in first_line and render_info.get("ok") is not False:
                approved = True
                break

            if qa_round < max_qa:
                log(f"[4/4] QA指摘に基づき修正中 ({qa_round}/{max_qa})...")
                fixed_raw = call_ollama(
                    model=stage_models["implementation"],
                    prompt=f"QA指摘:\n{qa_report}\n\n上記を修正した完全なHTMLを出力してください。\n\n現在のHTML:\n{html_code}",
                    system=REPAIR_SYSTEM,
                    options=generation_options(args, "implementation", kind="repair", attempt=qa_round),
                    stage="qa",
                    **base_kwargs,
                )
                fixed_html = extract_html(fixed_raw)
                if fixed_html and validate_html(fixed_html)["passed"]:
                    html_code = fixed_html
                    html_file.write_text(html_code, encoding="utf-8")

        completed.add("qa")
        manifest["completed_stages"] = sorted(completed)
        manifest["status"] = "completed" if approved else "completed_unapproved"
        _save_manifest(run_dir, manifest)
        set_stage("qa", "done")
        log(f"完了しました: {html_file}")
        return 0 if approved else 2

    except StageError as exc:
        set_stage(exc.stage, "error")
        manifest["status"] = "cancelled" if exc.cancelled else "failed"
        manifest["last_error"] = {"stage": exc.stage, "message": exc.message}
        _save_manifest(run_dir, manifest)
        raise


def run_revision(
    source_path: Path,
    notes: str,
    args: argparse.Namespace,
    output_root: Path | None = None,
    progress_callback: Callable[[str], None] | None = None,
    token_callback: Callable[[int, str], None] | None = None,
    stage_callback: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[int, Path]:
    """既存の run ディレクトリまたは単一 HTML ファイルを非破壊で改修する。"""
    def log(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def set_stage(st: str, status: str) -> None:
        if stage_callback:
            stage_callback(st, status)

    if source_path.is_dir():
        cand = source_path / "index.html"
        if not cand.exists():
            cand = source_path / "index_draft_unverified.html"
        if not cand.exists():
            raise StageError("revision", f"修正対象の index.html が見つかりません: {source_path}")
        orig_html = cand.read_text(encoding="utf-8")
        req_src = source_path / "01_requirements.md"
        orig_req = req_src.read_text(encoding="utf-8") if req_src.exists() else ""
    elif source_path.is_file():
        orig_html = source_path.read_text(encoding="utf-8")
        orig_req = ""
    else:
        raise StageError("revision", f"修正対象が見つかりません: {source_path}")

    new_dir = create_run_directory(output_root)
    (new_dir / "00_original_backup.html").write_text(orig_html, encoding="utf-8")
    (new_dir / "00_revision_request.md").write_text(notes, encoding="utf-8")
    if orig_req:
        (new_dir / "01_requirements.md").write_text(orig_req, encoding="utf-8")

    stage_models = _resolve_stage_models(args, log)
    manifest = {
        "mode": "revision",
        "source": str(source_path),
        "request": notes,
        "stage_models": stage_models,
        "status": "running",
        "completed_stages": [],
    }
    _save_manifest(run_dir=new_dir, manifest=manifest)

    base_kwargs = {
        "endpoint": args.endpoint,
        "timeout": float(args.timeout),
        "stream": not getattr(args, "no_stream", False),
        "keep_alive": getattr(args, "keep_alive", "5m"),
        "disable_thinking": not getattr(args, "think", False),
        "token_callback": token_callback,
        "cancel_event": cancel_event,
    }

    try:
        set_stage("requirements", "done")
        log("[1/4] 元のHTMLと修正依頼を確認しました。")

        set_stage("design", "running")
        log("[2/4] 修正方針を策定中...")
        try:
            plan_text = call_ollama(
                model=stage_models["design"],
                prompt=f"修正依頼:\n{notes}\n\n現在のHTML(冒頭):\n{orig_html[:6000]}",
                system=REVISION_PLAN_SYSTEM,
                options=generation_options(args, "design"),
                stage="design",
                **base_kwargs,
            )
        except StageError as exc:
            if exc.cancelled:
                raise
            plan_text = notes
        (new_dir / "02_revision_plan.md").write_text(plan_text, encoding="utf-8")
        set_stage("design", "done")

        set_stage("implementation", "running")
        log("[3/4] 修正を実装中...")
        raw_fixed = call_ollama(
            model=stage_models["implementation"],
            prompt=(
                f"ユーザーの修正依頼:\n{notes}\n\n修正方針:\n{plan_text}\n\n"
                f"既存機能を維持したまま上記を反映し、完全な単一HTMLを出力してください。\n\n"
                f"元のHTML:\n{orig_html}"
            ),
            system=REPAIR_SYSTEM,
            options=generation_options(args, "implementation", kind="repair", attempt=0),
            stage="implementation",
            **base_kwargs,
        )
        html_code = extract_html(raw_fixed) or ""
        check_res = validate_html(html_code) if html_code else {"passed": False, "errors": ["HTML抽出失敗"]}
        if not check_res["passed"]:
            raw_retry = call_ollama(
                model=stage_models["implementation"],
                prompt=f"エラー修正: {'; '.join(check_res['errors'])}\n\nHTML:\n{html_code or raw_fixed}",
                system=REPAIR_SYSTEM,
                options=generation_options(args, "implementation", kind="repair", attempt=1),
                stage="implementation",
                **base_kwargs,
            )
            html_code = extract_html(raw_retry) or ""
            check_res = validate_html(html_code) if html_code else {"passed": False, "errors": ["HTML抽出失敗"]}

        (new_dir / "03_static_checks.json").write_text(json.dumps(check_res, ensure_ascii=False, indent=2), encoding="utf-8")
        if not check_res["passed"]:
            set_stage("implementation", "error")
            raise StageError("implementation", f"修正版HTMLの検証に失敗しました: {'; '.join(check_res['errors'])}")
        (new_dir / "index.html").write_text(html_code, encoding="utf-8")
        set_stage("implementation", "done")

        set_stage("qa", "running")
        log("[4/4] 修正版のQA再検査中...")
        render_info = check_render(html_code, new_dir)
        qa_report = call_ollama(
            model=stage_models["qa"],
            prompt=f"修正依頼:\n{notes}\n\nレンダー結果: {json.dumps(render_info, ensure_ascii=False)}\n\nHTML:\n{html_code}",
            system=QA_SYSTEM,
            options=generation_options(args, "qa"),
            stage="qa",
            **base_kwargs,
        )
        (new_dir / "04_qa_report.md").write_text(qa_report, encoding="utf-8")
        approved = "APPROVED" in (qa_report.strip().splitlines()[0].upper() if qa_report.strip() else "")
        manifest["status"] = "completed" if approved else "completed_unapproved"
        manifest["completed_stages"] = list(STAGES)
        _save_manifest(new_dir, manifest)
        set_stage("qa", "done")
        log(f"改修完了: {new_dir / 'index.html'}")
        return (0 if approved else 2), new_dir
    except StageError as exc:
        set_stage(exc.stage, "error")
        manifest["status"] = "cancelled" if exc.cancelled else "failed"
        manifest["last_error"] = {"stage": exc.stage, "message": exc.message}
        _save_manifest(new_dir, manifest)
        raise


def ensure_ollama_running(endpoint: str = DEFAULT_ENDPOINT, wait_seconds: float = 10.0) -> bool:
    """Ollamaが未起動なら自動起動を試み、応答可能になるまで待機する。"""
    try:
        fetch_ollama_models(endpoint, timeout=2.0)
        return True
    except StageError:
        pass

    launched = False
    if sys.platform == "win32":
        local_app = os.environ.get("LOCALAPPDATA", "")
        ollama_app = Path(local_app) / "Programs" / "Ollama" / "ollama app.exe" if local_app else None
        if ollama_app and ollama_app.exists():
            subprocess.Popen([str(ollama_app)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            launched = True
    if not launched:
        ollama_bin = shutil.which("ollama")
        if ollama_bin:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(
                [ollama_bin, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=creationflags,
            )
            launched = True

    if not launched:
        return False

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        time.sleep(0.8)
        try:
            fetch_ollama_models(endpoint, timeout=2.0)
            return True
        except StageError:
            pass
    return False

def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    # Modern Dark Slate Palette
    BG = "#0f172a"
    SURFACE = "#1e293b"
    SURFACE_ALT = "#263349"
    BORDER = "#334155"
    TEXT = "#f8fafc"
    MUTED = "#94a3b8"
    ACCENT = "#3b82f6"
    ACCENT_HOVER = "#2563eb"
    SUCCESS = "#10b981"
    WARNING = "#f59e0b"
    DANGER = "#ef4444"
    FONT_FAMILY = "Yu Gothic UI" if sys.platform == "win32" else "TkDefaultFont"

    class AgentGUI:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("Local AI App Builder — ローカルAIアプリ開発エージェント")
            self.root.geometry("980x820")
            self.root.minsize(880, 720)
            self.root.configure(bg=BG)

            style = ttk.Style(root)
            try:
                style.theme_use("clam")
            except Exception:
                pass
            style.configure("TCombobox", fieldbackground=SURFACE_ALT, background=SURFACE_ALT, foreground=TEXT, bordercolor=BORDER)
            style.map("TCombobox", fieldbackground=[("readonly", SURFACE_ALT)], selectbackground=[("readonly", ACCENT)])
            style.configure("Modern.Horizontal.TProgressbar", troughcolor=SURFACE_ALT, background=ACCENT, bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
            style.configure("TNotebook", background=SURFACE, borderwidth=0)
            style.configure("TNotebook.Tab", background=SURFACE_ALT, foreground=MUTED, padding=(14, 6), font=(FONT_FAMILY, 9, "bold"))
            style.map("TNotebook.Tab", background=[("selected", BG)], foreground=[("selected", TEXT)])

            self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
            self.cancel_event: threading.Event | None = None
            self.busy = False
            self.last_run_dir: Path | None = None
            self.stage_labels_map: dict[str, tk.Label] = {}
            self.stage_frames_map: dict[str, tk.Frame] = {}

            # --- Header ---
            header = tk.Frame(root, bg=BG, padx=18, pady=12)
            header.pack(fill="x")
            tk.Label(header, text="Local AI App Builder", font=(FONT_FAMILY, 16, "bold"), bg=BG, fg=TEXT).pack(side="left")
            tk.Label(header, text="  ｜  3ステップで単一HTMLアプリを自動生成", font=(FONT_FAMILY, 10), bg=BG, fg=MUTED).pack(side="left", pady=(4, 0))
            self.conn_status_var = tk.StringVar(value="⚪ 未確認")
            tk.Label(header, textvariable=self.conn_status_var, font=(FONT_FAMILY, 9, "bold"), bg=SURFACE, fg=TEXT, padx=10, pady=4).pack(side="right")

            # --- STEP 1: Connection & Model ---
            step1 = self._make_step_card(root, "STEP 1", "接続とモデルを準備する（初回のみ確認）")
            row1 = tk.Frame(step1, bg=SURFACE)
            row1.pack(fill="x", pady=(6, 0))

            tk.Label(row1, text="モデル:", bg=SURFACE, fg=MUTED, font=(FONT_FAMILY, 9)).pack(side="left")
            self.model_var = tk.StringVar(value=DEFAULT_MODEL)
            self.model_combo = ttk.Combobox(row1, textvariable=self.model_var, values=[DEFAULT_MODEL], width=22)
            self.model_combo.pack(side="left", padx=(6, 14))

            tk.Label(row1, text="割当:", bg=SURFACE, fg=MUTED, font=(FONT_FAMILY, 9)).pack(side="left")
            self.routing_var = tk.StringVar(value="single")
            ttk.Combobox(row1, textvariable=self.routing_var, values=["single", "smart"], state="readonly", width=8).pack(side="left", padx=(6, 14))

            tk.Label(row1, text="タイムアウト(秒):", bg=SURFACE, fg=MUTED, font=(FONT_FAMILY, 9)).pack(side="left")
            self.timeout_var = tk.StringVar(value="600")
            tk.Entry(row1, textvariable=self.timeout_var, width=6, bg=SURFACE_ALT, fg=TEXT, insertbackground=TEXT, relief="flat", highlightthickness=1, highlightbackground=BORDER).pack(side="left", padx=(6, 14))

            self.think_var = tk.BooleanVar(value=False)
            tk.Checkbutton(row1, text="思考モード", variable=self.think_var, bg=SURFACE, fg=TEXT, selectcolor=SURFACE_ALT, activebackground=SURFACE, activeforeground=TEXT, font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 10))

            self._make_btn(row1, "🔄 接続確認", self.check_connection, bg=SURFACE_ALT, fg=TEXT).pack(side="right")

            # --- STEP 2: Prompt & Launch ---
            step2 = self._make_step_card(root, "STEP 2", "作りたいアプリを入力して実行する")
            preset_bar = tk.Frame(step2, bg=SURFACE)
            preset_bar.pack(fill="x", pady=(6, 6))
            tk.Label(preset_bar, text="テンプレート:", bg=SURFACE, fg=MUTED, font=(FONT_FAMILY, 9)).pack(side="left", padx=(0, 6))
            for title, body in PROMPT_PRESETS.items():
                self._make_btn(preset_bar, title, lambda b=body: self._apply_preset(b), bg=SURFACE_ALT, fg=TEXT, padx=8, pady=2).pack(side="left", padx=3)

            self.prompt_text = scrolledtext.ScrolledText(
                step2, height=4, wrap="word", bg=SURFACE_ALT, fg=TEXT, insertbackground=TEXT,
                relief="flat", highlightthickness=1, highlightbackground=BORDER, font=(FONT_FAMILY, 10), padx=8, pady=6
            )
            self.prompt_text.pack(fill="x")

            btns = tk.Frame(step2, bg=SURFACE)
            btns.pack(fill="x", pady=(8, 0))
            self.start_btn = self._make_btn(btns, "▶ 新規作成を開始", self.start_new, bg=ACCENT, fg="#ffffff", bold=True, padx=14, pady=5)
            self.start_btn.pack(side="left")
            self.resume_btn = self._make_btn(btns, "↻ つづきから再開", self.resume_last, bg=SURFACE_ALT, fg=TEXT, padx=10, pady=5)
            self.resume_btn.pack(side="left", padx=8)
            self.revise_btn = self._make_btn(btns, "✏ このアプリを修正", self.revise_dialog, bg=SURFACE_ALT, fg=TEXT, padx=10, pady=5)
            self.revise_btn.pack(side="left")
            self.stop_btn = self._make_btn(btns, "⏹ 中止", self.stop_run, bg=DANGER, fg="#ffffff", padx=10, pady=5)
            self.stop_btn.configure(state="disabled")
            self.stop_btn.pack(side="left", padx=8)

            self.open_browser_btn = self._make_btn(btns, "🌐 ブラウザで開く", self.open_in_browser, bg=SUCCESS, fg="#ffffff", bold=True, padx=12, pady=5)
            self.open_browser_btn.configure(state="disabled")
            self.open_browser_btn.pack(side="right")
            self.open_dir_btn = self._make_btn(btns, "📁 フォルダを開く", self.open_output_dir, bg=SURFACE_ALT, fg=TEXT, padx=10, pady=5)
            self.open_dir_btn.configure(state="disabled")
            self.open_dir_btn.pack(side="right", padx=8)

            # --- STEP 3: Pipeline Progress & Output ---
            step3 = self._make_step_card(root, "STEP 3", "自動開発の進捗と成果物を確認する", expand=True)

            cards_frame = tk.Frame(step3, bg=SURFACE)
            cards_frame.pack(fill="x", pady=(6, 8))
            for idx, st in enumerate(STAGES):
                card = tk.Frame(cards_frame, bg=SURFACE_ALT, highlightthickness=1, highlightbackground=BORDER, padx=10, pady=8)
                card.pack(side="left", fill="x", expand=True, padx=(0 if idx == 0 else 6, 0))
                tk.Label(card, text=STAGE_LABELS[st], font=(FONT_FAMILY, 9, "bold"), bg=SURFACE_ALT, fg=TEXT).pack(anchor="w")
                lbl = tk.Label(card, text="待機中", font=(FONT_FAMILY, 9), fg=MUTED, bg=SURFACE_ALT)
                lbl.pack(anchor="w", pady=(2, 0))
                self.stage_labels_map[st] = lbl
                self.stage_frames_map[st] = card

            self.progress_var = tk.DoubleVar(value=0.0)
            self.progress_bar = ttk.Progressbar(step3, variable=self.progress_var, maximum=100.0, style="Modern.Horizontal.TProgressbar")
            self.progress_bar.pack(fill="x", pady=(0, 6))

            self.status_var = tk.StringVar(value="待機中 — STEP 1 で接続確認後、STEP 2 からアプリ作成を開始できます。")
            tk.Label(step3, textvariable=self.status_var, anchor="w", bg=SURFACE, fg=TEXT, font=(FONT_FAMILY, 9, "bold")).pack(fill="x", pady=(0, 6))

            notebook = ttk.Notebook(step3)
            notebook.pack(fill="both", expand=True)
            log_opts = dict(state="disabled", wrap="word", bg=BG, fg="#e2e8f0", insertbackground=TEXT, relief="flat", font=("Consolas", 9), padx=8, pady=6)
            self.log_box = scrolledtext.ScrolledText(notebook, **log_opts)
            self.req_box = scrolledtext.ScrolledText(notebook, **log_opts)
            self.design_box = scrolledtext.ScrolledText(notebook, **log_opts)
            self.qa_box = scrolledtext.ScrolledText(notebook, **log_opts)

            notebook.add(self.log_box, text=" 実行ログ ")
            notebook.add(self.req_box, text=" 要件定義 (01) ")
            notebook.add(self.design_box, text=" 設計書 (02) ")
            notebook.add(self.qa_box, text=" QAレポート (04) ")

            self.root.after(150, self.poll_events)
            self.root.after(300, self.check_connection)

        def _make_step_card(self, parent: tk.Misc, badge: str, title: str, expand: bool = False) -> tk.Frame:
            outer = tk.Frame(parent, bg=SURFACE, highlightthickness=1, highlightbackground=BORDER, padx=14, pady=10)
            outer.pack(fill="both" if expand else "x", expand=expand, padx=18, pady=(0, 10))
            hdr = tk.Frame(outer, bg=SURFACE)
            hdr.pack(fill="x")
            tk.Label(hdr, text=badge, font=(FONT_FAMILY, 8, "bold"), bg=ACCENT, fg="#ffffff", padx=7, pady=1).pack(side="left")
            tk.Label(hdr, text=f"  {title}", font=(FONT_FAMILY, 10, "bold"), bg=SURFACE, fg=TEXT).pack(side="left")
            return outer

        def _make_btn(self, parent: tk.Misc, text: str, cmd: Callable[[], None], bg: str, fg: str, bold: bool = False, padx: int = 10, pady: int = 4) -> tk.Button:
            weight = "bold" if bold else "normal"
            return tk.Button(
                parent, text=text, command=cmd, bg=bg, fg=fg,
                activebackground=ACCENT_HOVER, activeforeground="#ffffff",
                relief="flat", bd=0, cursor="hand2", padx=padx, pady=pady,
                font=(FONT_FAMILY, 9, weight),
            )

        def _apply_preset(self, body: str) -> None:
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", body)

        def _set_text_readonly(self, widget: Any, content: str) -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", content)
            widget.configure(state="disabled")

        def append_log(self, msg: str) -> None:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")

        def check_connection(self) -> None:
            self.conn_status_var.set("🔄 確認中...")

            def worker() -> None:
                try:
                    models = fetch_ollama_models(DEFAULT_ENDPOINT)
                    names = [str(m.get("name", "")) for m in models if m.get("name")]
                    self.events.put(("models_ok", names))
                except Exception as exc:
                    self.events.put(("models_err", str(exc)))

            threading.Thread(target=worker, daemon=True).start()

        def start_new(self) -> None:
            req = self.prompt_text.get("1.0", "end").strip()
            if not req:
                messagebox.showwarning("入力不足", "STEP 2 に作りたいアプリの内容を入力してください。")
                return
            self._launch_build(req, resume_dir=None)

        def resume_last(self) -> None:
            target = self.last_run_dir
            if not target or not (target / "run.json").exists():
                chosen = filedialog.askdirectory(title="再開する runs フォルダを選択")
                if not chosen:
                    return
                target = Path(chosen)
            manifest = json.loads((target / "run.json").read_text(encoding="utf-8"))
            req = manifest.get("request", "").strip()
            if req:
                self._apply_preset(req)
            self._launch_build(req, resume_dir=target)

        def revise_dialog(self) -> None:
            notes = self.prompt_text.get("1.0", "end").strip()
            if not notes:
                messagebox.showwarning("入力不足", "STEP 2 の入力欄に「修正したい内容」を入力してから押してください。")
                return
            target: Path | None = self.last_run_dir
            if not target or not ((target / "index.html").exists() or (target / "index_draft_unverified.html").exists()):
                chosen = filedialog.askopenfilename(title="修正する HTML ファイルを選択", filetypes=[("HTML", "*.html")])
                if not chosen:
                    return
                target = Path(chosen)
            self._launch_revision(target, notes)

        def _prepare_run(self) -> argparse.Namespace | None:
            try:
                timeout = max(10.0, float(self.timeout_var.get().strip()))
            except ValueError:
                messagebox.showerror("入力エラー", "タイムアウトは数値で指定してください。")
                return None

            for st in STAGES:
                self.events.put(("stage", (st, "wait")))
            self.progress_var.set(5.0)

            self.busy = True
            self.cancel_event = threading.Event()
            for btn in (self.start_btn, self.resume_btn, self.revise_btn):
                btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.status_var.set("実行中...")

            return build_parser().parse_args(
                [
                    "gui_request",
                    "--model",
                    self.model_var.get().strip() or DEFAULT_MODEL,
                    "--routing-mode",
                    self.routing_var.get(),
                    "--timeout",
                    str(timeout),
                    *(["--think"] if self.think_var.get() else []),
                ]
            )

        def _launch_build(self, req: str, resume_dir: Path | None) -> None:
            if self.busy:
                return
            args = self._prepare_run()
            if args is None:
                return
            run_dir = resume_dir or create_run_directory()
            self.last_run_dir = run_dir
            self.append_log(f"--- 出力先: {run_dir} ---")

            def worker() -> None:
                try:
                    code = run_pipeline(
                        req,
                        args,
                        run_dir,
                        resume=(resume_dir is not None),
                        progress_callback=lambda m: self.events.put(("log", m)),
                        token_callback=lambda c, s: self.events.put(("tokens", (c, s))),
                        stage_callback=lambda s, st: self.events.put(("stage", (s, st))),
                        cancel_event=self.cancel_event,
                    )
                    self.events.put(("done", (code, run_dir)))
                except Exception as exc:
                    self.events.put(("error", (str(exc), run_dir)))

            threading.Thread(target=worker, daemon=True).start()

        def _launch_revision(self, source: Path, notes: str) -> None:
            if self.busy:
                return
            args = self._prepare_run()
            if args is None:
                return
            self.append_log(f"--- 改修元: {source} ---")

            def worker() -> None:
                try:
                    code, new_dir = run_revision(
                        source,
                        notes,
                        args,
                        progress_callback=lambda m: self.events.put(("log", m)),
                        token_callback=lambda c, s: self.events.put(("tokens", (c, s))),
                        stage_callback=lambda s, st: self.events.put(("stage", (s, st))),
                        cancel_event=self.cancel_event,
                    )
                    self.events.put(("done", (code, new_dir)))
                except Exception as exc:
                    self.events.put(("error", (str(exc), source)))

            threading.Thread(target=worker, daemon=True).start()

        def stop_run(self) -> None:
            if self.cancel_event:
                self.cancel_event.set()
                self.status_var.set("中止処理中（現在のチャンクで停止します）...")

        def open_in_browser(self) -> None:
            if self.last_run_dir and (self.last_run_dir / "index.html").exists():
                webbrowser.open((self.last_run_dir / "index.html").as_uri())

        def open_output_dir(self) -> None:
            if self.last_run_dir and self.last_run_dir.exists():
                if sys.platform == "win32":
                    os.startfile(self.last_run_dir)  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(self.last_run_dir)])
                else:
                    subprocess.Popen(["xdg-open", str(self.last_run_dir)])

        def _refresh_artifacts_view(self, run_dir: Path) -> None:
            for fname, box in (
                ("01_requirements.md", self.req_box),
                ("02_design.md", self.design_box),
                ("04_qa_report.md", self.qa_box),
            ):
                p = run_dir / fname
                if p.exists():
                    self._set_text_readonly(box, p.read_text(encoding="utf-8"))

        def poll_events(self) -> None:
            status_styles = {
                "wait": ("待機中", MUTED, BORDER),
                "running": ("実行中…", WARNING, WARNING),
                "done": ("完了 ✓", SUCCESS, SUCCESS),
                "error": ("停止 ✕", DANGER, DANGER),
            }
            stage_progress = {"requirements": 25.0, "design": 50.0, "implementation": 75.0, "qa": 95.0}
            while True:
                try:
                    kind, data = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind == "models_ok":
                    names = data
                    self.model_combo["values"] = names or [DEFAULT_MODEL]
                    if names and self.model_var.get() not in names:
                        self.model_var.set(names[0])
                    self.conn_status_var.set(f"🟢 接続OK ({len(names)}モデル)")
                elif kind == "models_err":
                    self.conn_status_var.set("🔴 Ollama未接続")
                    self.append_log(f"[接続確認] {data}")
                elif kind == "log":
                    self.append_log(data)
                elif kind == "stage":
                    st, st_status = data
                    if st in self.stage_labels_map:
                        txt, color, border_col = status_styles.get(st_status, ("待機中", MUTED, BORDER))
                        self.stage_labels_map[st].configure(text=txt, fg=color)
                        self.stage_frames_map[st].configure(highlightbackground=border_col)
                        if st_status == "done":
                            self.progress_var.set(max(self.progress_var.get(), stage_progress.get(st, 0.0)))
                elif kind == "tokens":
                    chars, stage = data
                    self.status_var.set(f"実行中 [{STAGE_LABELS.get(stage, stage)}] 生成文字数: {chars:,}")
                elif kind == "done":
                    code, rdir = data
                    self.last_run_dir = rdir
                    self.progress_var.set(100.0)
                    self.busy = False
                    for btn in (self.start_btn, self.resume_btn, self.revise_btn):
                        btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.open_browser_btn.configure(state="normal" if (rdir / "index.html").exists() else "disabled")
                    self.open_dir_btn.configure(state="normal")
                    self._refresh_artifacts_view(rdir)
                    self.status_var.set("完了しました！「ブラウザで開く」で動作確認できます。" if code == 0 else "完了 (QA要確認あり)")
                elif kind == "error":
                    msg, rdir = data
                    if isinstance(rdir, Path) and rdir.is_dir():
                        self.last_run_dir = rdir
                        self._refresh_artifacts_view(rdir)
                        self.open_dir_btn.configure(state="normal")
                    self.busy = False
                    for btn in (self.start_btn, self.resume_btn, self.revise_btn):
                        btn.configure(state="normal")
                    self.stop_btn.configure(state="disabled")
                    self.status_var.set("エラー停止 — 「つづきから再開」で完了済み工程をスキップして再開できます。")
                    self.append_log(f"[ERROR] {msg}")
            self.root.after(150, self.poll_events)

    root = tk.Tk()
    AgentGUI(root)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ローカルAIアプリ開発エージェント (完成・安定版)")
    p.add_argument("request", nargs="?", help="作りたいアプリの内容")
    p.add_argument("--gui", action="store_true", help="GUIモードで起動")
    p.add_argument("--check-connection", action="store_true", help="Ollama接続と自動モデル割当を確認して終了")
    p.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Ollama generate API URL")
    p.add_argument("--model", default=DEFAULT_MODEL, help="単一モード時に使用するOllamaモデル")
    p.add_argument("--routing-mode", choices=["smart", "single"], default="single", help="モデル割当モード (single=高速単一 / smart=工程別自動)")
    p.add_argument("--timeout", type=float, default=600.0, help="タイムアウト秒数")
    p.add_argument("--num-ctx", type=int, default=4096, help="コンテキスト長")
    p.add_argument("--num-predict", type=int, default=4096, help="基本最大出力トークン")
    p.add_argument("--temperature", type=float, default=0.3, help="基本temperature")
    p.add_argument("--keep-alive", default="5m", help="Ollamaモデル保持時間 (例: 5m)")
    p.add_argument("--think", action="store_true", help="思考モードを有効にする (デフォルト無効)")
    p.add_argument("--no-stream", action="store_true", help="非ストリーミングで実行")
    p.add_argument("--max-impl-repairs", type=int, default=3, help="実装自動修復の最大回数")
    p.add_argument("--max-qa-revisions", type=int, default=2, help="QA修正の最大回数")
    p.add_argument("--resume-dir", type=Path, help="中断した run フォルダから再開")
    p.add_argument("--revise-target", type=Path, help="既存 run フォルダまたは HTML ファイルを修正")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.check_connection:
        models = fetch_ollama_models(args.endpoint, timeout=float(args.timeout))
        assignments, reasons = smart_route_models(models)
        print(f"接続OK: {len(models)} モデル検出")
        for st in STAGES:
            print(f"  {STAGE_LABELS[st]}: {assignments[st]} ({reasons[st]})")
        return 0

    if args.gui or (not args.request and not args.resume_dir and not args.revise_target):
        try:
            ensure_ollama_running(args.endpoint, wait_seconds=8.0)
            launch_gui()
            return 0
        except Exception:
            import traceback
            traceback.print_exc()
            return 1

    if args.revise_target:
        if not args.request:
            parser.error("--revise-target 使用時は修正内容の request を指定してください。")
        code, _ = run_revision(args.revise_target, args.request, args, progress_callback=print)
        return code

    if args.resume_dir:
        manifest = json.loads((args.resume_dir / "run.json").read_text(encoding="utf-8"))
        req = args.request or manifest["request"]
        return run_pipeline(req, args, args.resume_dir, resume=True, progress_callback=print)

    run_dir = create_run_directory()
    return run_pipeline(args.request, args, run_dir, resume=False, progress_callback=print)


if __name__ == "__main__":
    sys.exit(main())
