using System.Net;
using System.Text;

namespace ImageEmbedding;

internal sealed record SearchResultsExportResult(
    string FolderPath,
    string HtmlPath,
    int CopiedCount,
    int FailedCount);

internal static class SearchResultsExporter
{
    public static Task<SearchResultsExportResult> ExportAsync(
        IReadOnlyList<SearchResultItem> items,
        string currentDirectory,
        IProgress<(int Current, int Total, string Message)>? progress = null)
    {
        ArgumentNullException.ThrowIfNull(items);
        ArgumentException.ThrowIfNullOrWhiteSpace(currentDirectory);
        return Task.Run(() => Export(items, currentDirectory, progress));
    }

    private static SearchResultsExportResult Export(
        IReadOnlyList<SearchResultItem> items,
        string currentDirectory,
        IProgress<(int Current, int Total, string Message)>? progress)
    {
        var (timestamp, outputDirectory) = CreateOutputDirectory(currentDirectory);
        var rows = new List<ExportedRow>(items.Count);
        var copiedCount = 0;
        var failedCount = 0;

        for (var index = 0; index < items.Count; index++)
        {
            var item = items[index];
            progress?.Report((index, items.Count, $"画像をコピーしています… {index + 1:N0}/{items.Count:N0}"));
            string? copiedFileName = null;
            string? error = null;
            try
            {
                if (!File.Exists(item.FullPath)) throw new FileNotFoundException("元画像が見つかりません。", item.FullPath);
                copiedFileName = BuildCopiedFileName(index + 1, item);
                File.Copy(item.FullPath, Path.Combine(outputDirectory, copiedFileName), overwrite: false);
                copiedCount++;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
            {
                error = ex.Message;
                failedCount++;
            }
            rows.Add(new ExportedRow(item, copiedFileName, error));
        }

        progress?.Report((items.Count, items.Count, "HTMLを作成しています…"));
        var htmlPath = Path.Combine(outputDirectory, $"result_{timestamp}.html");
        File.WriteAllText(htmlPath, BuildHtml(timestamp, rows), new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        return new SearchResultsExportResult(outputDirectory, htmlPath, copiedCount, failedCount);
    }

    private static (string Timestamp, string Directory) CreateOutputDirectory(string currentDirectory)
    {
        var root = Path.Combine(Path.GetFullPath(currentDirectory), "search_results");
        Directory.CreateDirectory(root);
        var candidateTime = DateTime.Now;
        for (var attempt = 0; attempt < 120; attempt++, candidateTime = candidateTime.AddSeconds(1))
        {
            var timestamp = candidateTime.ToString("yyyyMMdd_HHmmss");
            var directory = Path.Combine(root, timestamp);
            if (Directory.Exists(directory)) continue;
            Directory.CreateDirectory(directory);
            return (timestamp, directory);
        }
        throw new IOException("検索結果の保存フォルダを作成できませんでした。");
    }

    private static string BuildCopiedFileName(int displayIndex, SearchResultItem item)
    {
        var name = Path.GetFileNameWithoutExtension(item.FileName);
        foreach (var invalid in Path.GetInvalidFileNameChars()) name = name.Replace(invalid, '_');
        if (name.Length > 100) name = name[..100];
        var extension = Path.GetExtension(item.FileName);
        return $"{displayIndex:D4}_{item.Id}_{name}{extension}";
    }

    private static string BuildHtml(string timestamp, IReadOnlyList<ExportedRow> rows)
    {
        static string E(string? value) => WebUtility.HtmlEncode(value ?? "");
        static string U(string value) => Uri.EscapeDataString(value);

        var html = new StringBuilder(4096 + rows.Count * 512);
        html.AppendLine("<!doctype html>");
        html.AppendLine("<html lang=\"ja\"><head><meta charset=\"utf-8\">");
        html.Append("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>検索結果 ")
            .Append(E(timestamp)).AppendLine("</title>");
        html.AppendLine("<style>body{font-family:'Yu Gothic UI','Meiryo',sans-serif;margin:20px;background:#f4f6f8;color:#20252b}h1{font-size:24px}table{width:100%;border-collapse:collapse;background:#fff}th,td{border:1px solid #d6dce2;padding:8px;text-align:left;vertical-align:middle}th{background:#243447;color:#fff;position:sticky;top:0}tr:nth-child(even){background:#f7f9fb}.preview{width:136px;height:104px;object-fit:contain;background:#eceff2}.path{word-break:break-all}.missing{color:#a32020;font-size:12px}</style></head><body>");
        html.Append("<h1>Jina CLIP 検索結果</h1><p>出力日時: ").Append(E(timestamp))
            .Append(" / 件数: ").Append(rows.Count.ToString("N0")).AppendLine("</p>");
        html.AppendLine("<table><thead><tr><th>プレビュー</th><th>類似度</th><th>画像ファイル名</th><th>画像サイズ</th><th>画像の場所（フルパス）</th></tr></thead><tbody>");
        foreach (var row in rows)
        {
            html.AppendLine("<tr><td>");
            if (row.CopiedFileName is not null)
            {
                var url = U(row.CopiedFileName);
                html.Append("<a href=\"").Append(url).Append("\" target=\"_blank\" rel=\"noopener\"><img class=\"preview\" src=\"")
                    .Append(url).Append("\" alt=\"").Append(E(row.Item.FileName)).AppendLine("\"></a>");
            }
            else
            {
                html.Append("<span class=\"missing\">コピー失敗: ").Append(E(row.Error)).AppendLine("</span>");
            }
            html.Append("</td><td>").Append(E(row.Item.Similarity))
                .Append("</td><td>").Append(E(row.Item.FileName))
                .Append("</td><td>").Append(E(row.Item.Size))
                .Append("</td><td class=\"path\">").Append(E(row.Item.FullPath))
                .AppendLine("</td></tr>");
        }
        html.AppendLine("</tbody></table></body></html>");
        return html.ToString();
    }

    private sealed record ExportedRow(SearchResultItem Item, string? CopiedFileName, string? Error);
}
