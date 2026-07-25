# Jina CLIP 画像類似検索

`jinaai/jina-clip-v2` をローカルで実行し、画像EmbeddingをSQLiteに保存してFAISSで検索する日本語WPFアプリです。

## 新規環境で必要なもの

Windows x64の新しいPCでは、次の環境を先に用意してください。

- Windows 10/11 x64
- Python 3.10以降（x64版、`python` をPowerShellから実行できる状態）
- .NET 10 SDK（WPFをビルドするため。ランタイムだけではビルドできません）
- モデルと依存パッケージを保存するための空き容量（少なくとも8GBを推奨）
- GPUを使う場合は、対応するNVIDIAドライバー。GPUが使えない環境では自動的にCPUへ切り替わります。

確認コマンド：

```powershell
python --version
dotnet --info
```

Pythonと.NET SDKは、それぞれ公式サイトからインストールしてください。Pythonのインストール時は、`Add Python to PATH` を有効にします。

## モデルのダウンロードと配置

アプリは `jinaai/jina-clip-v2` のONNX版を使用します。推奨配置は次のとおりです。

```text
C:\Users\<ユーザー名>\.codex\TaggerImage\
├─ image_embedding\
└─ jinaai_jina-clip-v2\
   ├─ config.json
   ├─ tokenizer.json
   ├─ tokenizer_config.json
   ├─ special_tokens_map.json
   ├─ preprocessor_config.json
   └─ onnx\
      └─ model_fp16.onnx
```

`model_fp16.onnx` は約1.7GBあります。Hugging Face公式CLIで必要ファイルだけをダウンロードする場合は、`image_embedding` フォルダで次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
& .\.venv\Scripts\hf.exe download jinaai/jina-clip-v2 `
  config.json tokenizer.json tokenizer_config.json special_tokens_map.json preprocessor_config.json `
  onnx/model_fp16.onnx `
  --local-dir ..\jinaai_jina-clip-v2
```

`setup.ps1` はモデルを自動ダウンロードしません。上記コマンド、または[モデルのONNXフォルダ](https://huggingface.co/jinaai/jina-clip-v2/tree/main/onnx)からダウンロードして、必ず上記の配置にしてください。

アプリが使用するのは `onnx\model_fp16.onnx` です。`onnx\model.onnx` を使用する場合は別途 `model.onnx_data` が必要になるため、通常はダウンロード不要です。

## セットアップ

PowerShellでこのフォルダへ移動し、次を実行します。

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

初回の `setup.ps1` 実行時に、プロジェクト専用の `.venv` が作成され、ONNX Runtime、Transformers、Pillow、NumPy、FAISSなどがインストールされます。インストールにはインターネット接続が必要です。

完了後は `run.ps1`、または `bin\Release\net10.0-windows\ImageEmbedding.exe` を起動します。初回起動時は約1.7GBのONNXモデルを読み込むため時間がかかります。

## 操作

- 「フォルダを登録」でサブフォルダを含む画像を一括登録します。
- フォルダまたは画像をウィンドウへドロップしても登録できます。
- 日本語の検索文を入力し、Enterまたは「テキスト検索」で検索します。
- 「画像から検索」では画像同士の類似検索を行います。
- 「検索結果を保存」を押すと、現在テーブルに表示している結果と画像をHTML形式で保存します。
- 結果をダブルクリックすると縦横比を保った大きなプレビューを表示します。プレビューがすでに開いている場合は、そのウィンドウをアクティブにせず画像だけを切り替えます。
- プレビューを開いたまま結果の選択行を変えると、プレビュー画像も連動して切り替わります。
- プレビューではマウスホイールで拡大・縮小できます。拡大時は画像をドラッグして表示位置を移動できます。画面下部には画像の解像度と現在の拡大率を表示します。
- 「選択画像の場所を開く」でExplorerを開きます。
- 選択行でDeleteキーを押すと、SQLiteとFAISSから登録だけを解除します。元画像は削除しません。

### 検索結果の保存先

「検索結果を保存」を実行すると、カレントディレクトリの次の場所に日時ごとのフォルダを作成します。

```text
search_results\yyyyMMdd_HHmmss\
├─ result_yyyyMMdd_HHmmss.html
└─ 0001_<登録ID>_<元の画像ファイル名> など
```

HTMLには画面と同様に、プレビュー、類似度、ファイル名、画像サイズ、フルパスを出力します。プレビュー画像をクリックすると、同じ保存フォルダへコピーした画像をブラウザーで表示します。元画像は変更しません。

## 保存ファイル

通常の開発ビルドではプロジェクトフォルダ直下に、発行済みアプリでは実行ファイルと同じフォルダに次を作成します。

- `image_embedding.db`: パス、画像情報、1024次元Embeddingを保存するSQLite DB
- `image_embedding.db-shm`: SQLite WALモードの共有メモリファイル。アプリ実行中に作成されます。
- `image_embedding.db-wal`: SQLiteのWrite-Ahead Log（一時更新ログ）。正常終了後も残る場合があります。
- `image_embedding.faiss`: SQLiteから再構築可能なFAISS内積インデックス。検索高速化用で、SQLiteが正本です。
- `search_results\`: 「検索結果を保存」で作成されるHTMLと画像の出力先。Git管理対象外です。

`image_embedding.db`、`image_embedding.db-shm`、`image_embedding.db-wal` は、バックアップや移行を行う場合にアプリを終了してからまとめて扱ってください。FAISSファイルが壊れたり無くなったりしても、次回起動時にSQLiteから再構築されます。これらの保存ファイルは `.gitignore` によりGit管理対象外です。

## モデルとGPU

`jinaai_jina-clip-v2\onnx\model_fp16.onnx` とトークナイザーを使用します。CUDA Execution Providerを初期化できればGPUを使い、利用できなければCPUへ自動で切り替えます。

モデルのライセンスはCC BY-NC 4.0です。商用利用条件はJina AIの公式案内を確認してください。


https://huggingface.co/jinaai/jina-clip-v2/tree/main/onnx
