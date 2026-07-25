using Microsoft.Win32;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Text.Json;
using System.Windows;
using System.Windows.Input;

namespace ImageEmbedding;

public partial class MainWindow : Window
{
    private readonly ObservableCollection<SearchResultItem> _items = new();
    private WorkerClient? _worker;
    private ImagePreviewWindow? _previewWindow;
    private bool _busy;

    public MainWindow()
    {
        InitializeComponent();
        ResultsGrid.ItemsSource = _items;
        Loaded += async (_, _) => await InitializeWorkerAsync();
    }

    private async Task InitializeWorkerAsync()
    {
        try
        {
            SetBusy(true, "jina-clip-v2 とFAISSを読み込んでいます。初回は時間がかかります…");
            _worker = await WorkerClient.StartAsync(AppContext.BaseDirectory);
            _worker.Progress += (current, total, message) => Dispatcher.Invoke(() =>
            {
                ProgressBar.Value = total == 0 ? 0 : current * 100.0 / total;
                StatusText.Text = message;
            });
            _worker.Log += message => Dispatcher.Invoke(() => StatusText.Text = message);
            var status = await _worker.RequestAsync("status", new { });
            DeviceText.Text = $"実行デバイス: {status.GetProperty("device").GetString()} / FAISS";
            CountText.Text = $"登録数: {status.GetProperty("count").GetInt32():N0}";
            await ShowAllAsync();
            SetBusy(false, "準備完了。日本語テキスト、画像、またはドラッグ＆ドロップで操作できます。");
        }
        catch (Exception ex)
        {
            SetBusy(false, "初期化に失敗しました。");
            MessageBox.Show($"埋め込みエンジンを初期化できませんでした。\n\n{ex.Message}\n\nプロジェクト内の setup.ps1 を実行してください。", "初期化エラー", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    private void SetBusy(bool busy, string message)
    {
        _busy = busy;
        FolderButton.IsEnabled = ImageButton.IsEnabled = ImageSearchButton.IsEnabled = SearchButton.IsEnabled = !busy;
        ExportButton.IsEnabled = !busy && _items.Count > 0;
        ProgressBar.IsIndeterminate = busy && ProgressBar.Value == 0;
        if (!busy) { ProgressBar.IsIndeterminate = false; ProgressBar.Value = 0; }
        StatusText.Text = message;
    }

    private async Task RegisterAsync(IEnumerable<string> paths)
    {
        if (_worker is null || _busy) return;
        try
        {
            SetBusy(true, "対象画像を確認しています…");
            var result = await _worker.RequestAsync("index", new { paths = paths.ToArray() });
            CountText.Text = $"登録数: {result.GetProperty("count").GetInt32():N0}";
            var added = result.GetProperty("added").GetInt32();
            var updated = result.GetProperty("updated").GetInt32();
            var skipped = result.GetProperty("skipped").GetInt32();
            await ShowAllAsync();
            SetBusy(false, $"登録完了: 新規 {added:N0}、更新 {updated:N0}、スキップ {skipped:N0}");
        }
        catch (Exception ex) { SetBusy(false, "登録に失敗しました。"); ShowError(ex); }
    }

    private async Task SearchTextAsync()
    {
        if (_worker is null || _busy || string.IsNullOrWhiteSpace(QueryBox.Text)) return;
        try
        {
            SetBusy(true, "日本語テキストのEmbeddingとFAISS検索を実行中…");
            var result = await _worker.RequestAsync("search_text", new { query = QueryBox.Text.Trim(), limit = 200 });
            LoadResults(result);
            SetBusy(false, $"検索結果: {_items.Count:N0} 件");
        }
        catch (Exception ex) { SetBusy(false, "検索に失敗しました。"); ShowError(ex); }
    }

    private async Task SearchImageAsync(string path)
    {
        if (_worker is null || _busy) return;
        try
        {
            SetBusy(true, "検索画像のEmbeddingとFAISS検索を実行中…");
            var result = await _worker.RequestAsync("search_image", new { path, limit = 200 });
            LoadResults(result);
            SetBusy(false, $"画像検索結果: {_items.Count:N0} 件");
        }
        catch (Exception ex) { SetBusy(false, "画像検索に失敗しました。"); ShowError(ex); }
    }

    private async Task ShowAllAsync()
    {
        if (_worker is null) return;
        var result = await _worker.RequestAsync("list", new { limit = 1000 });
        LoadResults(result);
    }

    private void LoadResults(JsonElement result)
    {
        _items.Clear();
        foreach (var row in result.GetProperty("items").EnumerateArray())
            _items.Add(new SearchResultItem
            {
                Id = row.GetProperty("id").GetInt64(),
                FullPath = row.GetProperty("path").GetString() ?? "",
                Score = row.GetProperty("score").GetDouble(),
                Width = row.GetProperty("width").GetInt32(),
                Height = row.GetProperty("height").GetInt32()
            });
        ExportButton.IsEnabled = !_busy && _items.Count > 0;
    }

    private async void FolderButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "登録する画像フォルダを選択してください" };
        if (dialog.ShowDialog(this) == true) await RegisterAsync(new[] { dialog.FolderName });
    }

    private async void ImageButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = ImageDialog(true, "登録する画像を選択してください");
        if (dialog.ShowDialog(this) == true) await RegisterAsync(dialog.FileNames);
    }

    private async void ImageSearchButton_Click(object sender, RoutedEventArgs e)
    {
        var dialog = ImageDialog(false, "検索に使う画像を選択してください");
        if (dialog.ShowDialog(this) == true) await SearchImageAsync(dialog.FileName);
    }

    private static OpenFileDialog ImageDialog(bool multi, string title) => new()
    {
        Title = title, Multiselect = multi,
        Filter = "画像ファイル|*.jpg;*.jpeg;*.png;*.webp;*.bmp;*.gif;*.tif;*.tiff|すべてのファイル|*.*"
    };

    private async void SearchButton_Click(object sender, RoutedEventArgs e) => await SearchTextAsync();
    private async void QueryBox_KeyDown(object sender, KeyEventArgs e) { if (e.Key == Key.Enter) await SearchTextAsync(); }
    private async void ShowAllButton_Click(object sender, RoutedEventArgs e) { if (!_busy) await ShowAllAsync(); }

    private async void ExportButton_Click(object sender, RoutedEventArgs e)
    {
        if (_busy || _items.Count == 0) return;
        try
        {
            var snapshot = _items.ToArray();
            SetBusy(true, $"検索結果を保存しています… 0/{snapshot.Length:N0}");
            var progress = new Progress<(int Current, int Total, string Message)>(value =>
            {
                ProgressBar.IsIndeterminate = false;
                ProgressBar.Value = value.Total == 0 ? 0 : value.Current * 100.0 / value.Total;
                StatusText.Text = value.Message;
            });
            var result = await SearchResultsExporter.ExportAsync(snapshot, Environment.CurrentDirectory, progress);
            SetBusy(false, $"検索結果を保存しました: {result.CopiedCount:N0}件 / {result.FolderPath}");
            var detail = result.FailedCount == 0
                ? $"{result.CopiedCount:N0}件の画像とHTMLを保存しました。"
                : $"画像 {result.CopiedCount:N0}件を保存し、{result.FailedCount:N0}件は元画像が見つからないなどの理由でコピーできませんでした。";
            MessageBox.Show($"{detail}\n\n保存先:\n{result.FolderPath}\n\nHTML:\n{result.HtmlPath}",
                "検索結果の保存", MessageBoxButton.OK,
                result.FailedCount == 0 ? MessageBoxImage.Information : MessageBoxImage.Warning);
        }
        catch (Exception ex) { SetBusy(false, "検索結果の保存に失敗しました。"); ShowError(ex); }
    }

    private void OpenLocationButton_Click(object sender, RoutedEventArgs e)
    {
        if (ResultsGrid.SelectedItem is not SearchResultItem item) return;
        if (!File.Exists(item.FullPath)) { MessageBox.Show("元画像が見つかりません。", "確認", MessageBoxButton.OK, MessageBoxImage.Warning); return; }
        Process.Start(new ProcessStartInfo("explorer.exe", $"/select,\"{item.FullPath}\"") { UseShellExecute = true });
    }

    private void ResultsGrid_MouseDoubleClick(object sender, MouseButtonEventArgs e)
    {
        if (ResultsGrid.SelectedItem is not SearchResultItem item) return;

        if (_previewWindow is { IsLoaded: true })
        {
            _previewWindow.UpdateImage(item.FullPath);
            return;
        }

        _previewWindow = new ImagePreviewWindow(item.FullPath) { Owner = this };
        _previewWindow.Closed += (_, _) => _previewWindow = null;
        _previewWindow.Show();
    }

    private void ResultsGrid_SelectionChanged(object sender, System.Windows.Controls.SelectionChangedEventArgs e)
    {
        if (_previewWindow is { IsLoaded: true } && ResultsGrid.SelectedItem is SearchResultItem item)
            _previewWindow.UpdateImage(item.FullPath);
    }

    private async void Window_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key != Key.Delete || _worker is null || _busy || ResultsGrid.SelectedItems.Count == 0) return;
        var selected = ResultsGrid.SelectedItems.Cast<SearchResultItem>().ToArray();
        var answer = MessageBox.Show($"選択した {selected.Length:N0} 件の登録をDBから解除します。\n元の画像ファイルは削除しません。", "登録解除の確認", MessageBoxButton.YesNo, MessageBoxImage.Question);
        if (answer != MessageBoxResult.Yes) return;
        try
        {
            SetBusy(true, "DBとFAISSから登録を解除しています…");
            var result = await _worker.RequestAsync("delete", new { ids = selected.Select(x => x.Id).ToArray() });
            foreach (var item in selected) _items.Remove(item);
            ExportButton.IsEnabled = !_busy && _items.Count > 0;
            CountText.Text = $"登録数: {result.GetProperty("count").GetInt32():N0}";
            SetBusy(false, $"{selected.Length:N0} 件の登録を解除しました。元画像は変更していません。");
        }
        catch (Exception ex) { SetBusy(false, "登録解除に失敗しました。"); ShowError(ex); }
    }

    private async void Window_Drop(object sender, DragEventArgs e)
    {
        if (e.Data.GetData(DataFormats.FileDrop) is string[] paths) await RegisterAsync(paths);
    }
    private void Window_DragOver(object sender, DragEventArgs e)
    {
        e.Effects = e.Data.GetDataPresent(DataFormats.FileDrop) && !_busy ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }
    private static void ShowError(Exception ex) => MessageBox.Show(ex.Message, "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
    private async void Window_Closing(object? sender, System.ComponentModel.CancelEventArgs e) { if (_worker is not null) await _worker.DisposeAsync(); }
}
