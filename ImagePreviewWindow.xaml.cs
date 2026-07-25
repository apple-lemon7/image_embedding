using System.Windows;
using System.Windows.Input;
using System.Windows.Media;

namespace ImageEmbedding;

public partial class ImagePreviewWindow : Window
{
    private const double MinimumZoom = 0.1;
    private const double MaximumZoom = 20.0;
    private double _zoom = 1.0;
    private double _offsetX;
    private double _offsetY;
    private Point _dragStart;
    private bool _dragging;

    public ImagePreviewWindow(string path)
    {
        InitializeComponent();
        Height = Math.Max(MinHeight, SystemParameters.WorkArea.Height * 0.5);
        Width = Math.Max(MinWidth, Math.Min(SystemParameters.WorkArea.Width * 0.65, Height * 1.35));
        UpdateImage(path);
    }

    public void UpdateImage(string path)
    {
        Title = $"画像プレビュー - {Path.GetFileName(path)}";
        var image = SearchResultItem.LoadImage(path);
        PreviewImage.Source = image;
        ResolutionText.Text = image is null
            ? "画像を読み込めません"
            : $"解像度: {image.PixelWidth:N0} × {image.PixelHeight:N0}";
        ResetView();
    }

    private void ResetView()
    {
        _zoom = 1.0;
        _offsetX = 0;
        _offsetY = 0;
        _dragging = false;
        ApplyTransform();
    }

    private void ApplyTransform()
    {
        ImageTransform.Matrix = new Matrix(_zoom, 0, 0, _zoom, _offsetX, _offsetY);
        ZoomText.Text = $"拡大率: {_zoom:P0}";
        ImageViewport.Cursor = _zoom > 1.001 ? Cursors.Hand : Cursors.Arrow;
    }

    private void ImageViewport_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
    {
        if (PreviewImage.Source is null) return;
        var oldZoom = _zoom;
        var factor = e.Delta > 0 ? 1.2 : 1.0 / 1.2;
        var newZoom = Math.Clamp(oldZoom * factor, MinimumZoom, MaximumZoom);
        if (Math.Abs(newZoom - oldZoom) < 0.0001) return;

        var cursor = e.GetPosition(ImageViewport);
        var ratio = newZoom / oldZoom;
        _offsetX = cursor.X - (cursor.X - _offsetX) * ratio;
        _offsetY = cursor.Y - (cursor.Y - _offsetY) * ratio;
        _zoom = newZoom;
        ApplyTransform();
        e.Handled = true;
    }

    private void ImageViewport_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (_zoom <= 1.001 || PreviewImage.Source is null) return;
        _dragging = true;
        _dragStart = e.GetPosition(ImageViewport);
        ImageViewport.CaptureMouse();
        ImageViewport.Cursor = Cursors.Hand;
        e.Handled = true;
    }

    private void ImageViewport_MouseMove(object sender, MouseEventArgs e)
    {
        if (!_dragging) return;
        var current = e.GetPosition(ImageViewport);
        _offsetX += current.X - _dragStart.X;
        _offsetY += current.Y - _dragStart.Y;
        _dragStart = current;
        ApplyTransform();
    }

    private void ImageViewport_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        EndDragging();
    }

    private void ImageViewport_MouseLeave(object sender, MouseEventArgs e)
    {
        if (_dragging && e.LeftButton != MouseButtonState.Pressed) EndDragging();
    }

    private void EndDragging()
    {
        if (!_dragging) return;
        _dragging = false;
        ImageViewport.ReleaseMouseCapture();
        ApplyTransform();
    }
}
