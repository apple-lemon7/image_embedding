using System.ComponentModel;
using System.Runtime.CompilerServices;
using System.Windows.Media.Imaging;

namespace ImageEmbedding;

public sealed class SearchResultItem : INotifyPropertyChanged
{
    private BitmapImage? _thumbnail;
    public long Id { get; init; }
    public string FullPath { get; init; } = "";
    public string FileName => Path.GetFileName(FullPath);
    public double Score { get; init; }
    public string Similarity => Score < 0 ? "-" : $"{Score:P2}";
    public int Width { get; init; }
    public int Height { get; init; }
    public string Size => Width > 0 ? $"{Width} × {Height}" : "-";
    public BitmapImage? Thumbnail => _thumbnail ??= LoadImage(FullPath, 128);

    public event PropertyChangedEventHandler? PropertyChanged;
    private void OnPropertyChanged([CallerMemberName] string? name = null) => PropertyChanged?.Invoke(this, new(name));

    public static BitmapImage? LoadImage(string path, int decodeSize = 0)
    {
        try
        {
            if (!File.Exists(path)) return null;
            using var stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            var image = new BitmapImage();
            image.BeginInit();
            image.CacheOption = BitmapCacheOption.OnLoad;
            if (decodeSize > 0) image.DecodePixelWidth = decodeSize;
            image.StreamSource = stream;
            image.EndInit();
            image.Freeze();
            return image;
        }
        catch { return null; }
    }
}
