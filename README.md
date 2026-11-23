# ローカルAIアプリ開発エージェント（完成・安定版）

ローカルの Ollama を使って「要件定義 → デザイン設計 → 実装 → QAレビュー」の4段階で単一HTMLアプリを自動生成・改修するデスクトップアプリです。

## 動作要件
- Python 3.9 以上（標準ライブラリ + Tkinter のみで動作）
- [Ollama](https://ollama.com/) がインストールされ、起動していること
  - 例: `ollama pull qwen3.5:9b`（または `qwen2.5-coder:7b` 等）

## 起動方法
### Windows
- `launch_gui.bat` をダブルクリックします。
- またはコマンドプロンプトで `python local_ai_agent.py --gui` を実行します。

### Mac / Linux
```bash
python3 local_ai_agent.py --gui
```

### CLI で実行する場合
```bash
# 接続とモデル自動割当の確認
python3 local_ai_agent.py --check-connection

# アプリ生成
python3 local_ai_agent.py "ポモドーロタイマーを作って"

# 中断した実行フォルダから再開
python3 local_ai_agent.py --resume-dir runs/run_20261123_030000
```

## 同梱ファイル
- `local_ai_agent.py` — エージェント本体（GUI / CLI 両対応）
- `launch_gui.bat` — Windows 用ダブルクリック起動スクリプト
- `test_local_ai_agent.py` — 動作確認用ユニットテスト（`python3 test_local_ai_agent.py`）
- `README.md` — この説明書
