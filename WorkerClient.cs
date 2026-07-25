using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace ImageEmbedding;

internal sealed class WorkerClient : IAsyncDisposable
{
    private const string ProtocolPrefix = "@@JSON@@";
    private readonly Process _process;
    private readonly ConcurrentDictionary<string, TaskCompletionSource<JsonElement>> _pending = new();
    private readonly Task _readerTask;
    public event Action<int, int, string>? Progress;
    public event Action<string>? Log;

    private WorkerClient(Process process)
    {
        _process = process;
        _readerTask = ReadLoopAsync();
        _ = Task.Run(async () =>
        {
            while (await _process.StandardError.ReadLineAsync() is { } line) Log?.Invoke(line);
        });
    }

    public static async Task<WorkerClient> StartAsync(string appDirectory)
    {
        var worker = Path.Combine(appDirectory, "worker", "embedding_worker.py");
        var python = FindPython(appDirectory);
        if (python is null)
            throw new InvalidOperationException("Pythonが見つかりません。setup.ps1 を実行してください。");

        var info = new ProcessStartInfo
        {
            FileName = python,
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardOutputEncoding = new UTF8Encoding(false),
            StandardErrorEncoding = new UTF8Encoding(false),
            CreateNoWindow = true,
            WorkingDirectory = appDirectory
        };
        info.ArgumentList.Add("-u");
        info.ArgumentList.Add(worker);
        info.ArgumentList.Add("--app-dir");
        info.ArgumentList.Add(appDirectory);
        var process = Process.Start(info) ?? throw new InvalidOperationException("埋め込みワーカーを開始できませんでした。");
        var client = new WorkerClient(process);
        await client.RequestAsync("init", new { });
        return client;
    }

    private static string? FindPython(string appDirectory)
    {
        var local = Path.Combine(appDirectory, ".venv", "Scripts", "python.exe");
        if (File.Exists(local)) return local;
        var sourceLocal = FindUp(appDirectory, Path.Combine("image_embedding", ".venv", "Scripts", "python.exe"));
        if (sourceLocal is not null) return sourceLocal;
        return "python";
    }

    private static string? FindUp(string start, string relative)
    {
        for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, relative);
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    public async Task<JsonElement> RequestAsync(string command, object payload)
    {
        var id = Guid.NewGuid().ToString("N");
        var completion = new TaskCompletionSource<JsonElement>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pending[id] = completion;
        var body = JsonSerializer.Serialize(new { id, command, payload });
        await _process.StandardInput.WriteLineAsync(body);
        await _process.StandardInput.FlushAsync();
        return await completion.Task;
    }

    private async Task ReadLoopAsync()
    {
        try
        {
            while (await _process.StandardOutput.ReadLineAsync() is { } line)
            {
                if (!line.StartsWith(ProtocolPrefix, StringComparison.Ordinal))
                {
                    if (!string.IsNullOrWhiteSpace(line)) Log?.Invoke(line);
                    continue;
                }

                JsonDocument doc;
                try
                {
                    var jsonBytes = Convert.FromBase64String(line[ProtocolPrefix.Length..]);
                    doc = JsonDocument.Parse(jsonBytes);
                }
                catch (Exception ex) when (ex is FormatException or JsonException)
                {
                    Log?.Invoke($"ワーカー応答を読み取れませんでした。処理を継続します: {ex.Message}");
                    continue;
                }
                using (doc)
                {
                var root = doc.RootElement;
                if (root.TryGetProperty("event", out var evt))
                {
                    var kind = evt.GetString();
                    if (kind == "progress")
                        Progress?.Invoke(root.GetProperty("current").GetInt32(), root.GetProperty("total").GetInt32(), root.GetProperty("message").GetString() ?? "");
                    else if (kind == "log") Log?.Invoke(root.GetProperty("message").GetString() ?? "");
                    continue;
                }
                var id = root.GetProperty("id").GetString();
                if (id is null || !_pending.TryRemove(id, out var tcs)) continue;
                if (root.GetProperty("ok").GetBoolean()) tcs.SetResult(root.GetProperty("result").Clone());
                else tcs.SetException(new InvalidOperationException(root.GetProperty("error").GetString()));
                }
            }
            throw new InvalidOperationException("埋め込みワーカーが終了しました。");
        }
        catch (Exception ex)
        {
            foreach (var pair in _pending) pair.Value.TrySetException(ex);
            _pending.Clear();
        }
    }

    public async ValueTask DisposeAsync()
    {
        try { await RequestAsync("shutdown", new { }); } catch { }
        if (!_process.HasExited) _process.Kill(true);
        _process.Dispose();
        try { await _readerTask; } catch { }
    }
}
