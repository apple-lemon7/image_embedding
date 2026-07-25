using ImageEmbedding;
using System.IO;
using System.Text.RegularExpressions;

if (args.Length != 1 || !File.Exists(args[0]))
    throw new ArgumentException("テスト用画像を1件指定してください。");

var testRoot = Path.Combine(Path.GetTempPath(), "ImageEmbeddingSmoke", Guid.NewGuid().ToString("N"));
var item = new SearchResultItem
{
    Id = 42,
    FullPath = Path.GetFullPath(args[0]),
    Score = 0.8123,
    Width = 1157,
    Height = 406
};
var progressEvents = new List<(int Current, int Total, string Message)>();
var result = await SearchResultsExporter.ExportAsync(
    new[] { item }, testRoot,
    new Progress<(int Current, int Total, string Message)>(value => progressEvents.Add(value)));

if (!Regex.IsMatch(Path.GetFileName(result.FolderPath), "^[0-9]{8}_[0-9]{6}$"))
    throw new InvalidOperationException("出力フォルダ名が指定形式ではありません。");
if (!File.Exists(result.HtmlPath) || result.CopiedCount != 1 || result.FailedCount != 0)
    throw new InvalidOperationException("検索結果ファイルを正しく保存できませんでした。");
if (Directory.GetFiles(result.FolderPath).Length != 2)
    throw new InvalidOperationException("画像とHTML以外のファイルが生成されました。");

var html = await File.ReadAllTextAsync(result.HtmlPath);
foreach (var expected in new[] { "プレビュー", "類似度", "画像ファイル名", "画像サイズ", "画像の場所（フルパス）", "81.23%", "target=\"_blank\"" })
    if (!html.Contains(expected, StringComparison.Ordinal))
        throw new InvalidOperationException($"HTMLに必要な内容がありません: {expected}");

Console.WriteLine($"PASS folder={result.FolderPath}");
