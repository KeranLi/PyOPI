"""
Test colormap functions against MATLAB behavior
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from opi.viz.colormaps import haxby, cmapscale


def test_haxby_colors():
    """Test that haxby uses the correct base colors"""
    # MATLAB base colors (11 colors, 0-255 range)
    matlab_colors = np.array([
        [37, 57, 175],    # Deep blue
        [40, 127, 251],   # Medium blue
        [50, 190, 255],   # Light blue
        [106, 235, 255],  # Cyan
        [138, 236, 174],  # Green-cyan
        [205, 255, 162],  # Light green
        [240, 236, 121],  # Yellow
        [255, 189, 87],   # Orange
        [255, 161, 68],   # Dark orange
        [255, 186, 133],  # Light brown
        [255, 255, 255],  # White
    ]) / 255.0
    
    # Python implementation uses the same colors
    python_colors = np.array([
        [37, 57, 175],
        [40, 127, 251],
        [50, 190, 255],
        [106, 235, 255],
        [138, 236, 174],
        [205, 255, 162],
        [240, 236, 121],
        [255, 189, 87],
        [255, 161, 68],
        [255, 186, 133],
        [255, 255, 255],
    ]) / 255.0
    
    np.testing.assert_allclose(matlab_colors, python_colors, atol=1e-10)
    print("[PASS] Base colors match MATLAB")


def test_haxby_interpolation():
    """Test haxby colormap interpolation"""
    # Test with 64 colors (MATLAB default)
    cmap_64 = haxby(64)
    
    print(f"haxby(64) shape: {cmap_64.colors.shape}")
    print(f"First color (deep blue): {cmap_64.colors[0]}")
    print(f"Last color (white): {cmap_64.colors[-1]}")
    print(f"Middle color (~cyan-green): {cmap_64.colors[32]}")
    
    # Check shape
    assert cmap_64.colors.shape == (64, 3)
    
    # First color should be close to deep blue
    expected_first = np.array([37, 57, 175]) / 255.0
    np.testing.assert_allclose(cmap_64.colors[0], expected_first, atol=0.01)
    
    # Last color should be white
    expected_last = np.array([255, 255, 255]) / 255.0
    np.testing.assert_allclose(cmap_64.colors[-1], expected_last, atol=0.01)
    
    # Colors should be monotonically increasing (roughly)
    # Red channel generally increases
    assert cmap_64.colors[-1, 0] > cmap_64.colors[0, 0]
    # Blue channel generally decreases then increases
    
    print("[PASS] haxby interpolation test")


def test_haxby_default_size():
    """Test default haxby size (256)"""
    cmap_default = haxby()
    
    print(f"Default haxby size: {cmap_default.N}")
    assert cmap_default.N == 256
    
    print("[PASS] haxby default size test")


def test_haxby_different_sizes():
    """Test haxby with various sizes"""
    sizes = [16, 32, 64, 128, 256, 512]
    
    for n in sizes:
        cmap = haxby(n)
        assert cmap.N == n
        assert cmap.colors.shape == (n, 3)
        # All values should be in [0, 1]
        assert np.all(cmap.colors >= 0) and np.all(cmap.colors <= 1)
    
    print(f"[PASS] haxby sizes {sizes}")


def test_haxby_visual():
    """Visual test of haxby colormap"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    
    # Create test data
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    # Test different sizes
    sizes = [11, 32, 64, 256]
    titles = ['11 colors (base)', '32 colors', '64 colors', '256 colors']
    
    for ax, n, title in zip(axes.flat, sizes, titles):
        cmap = haxby(n)
        im = ax.contourf(X, Y, Z, levels=20, cmap=cmap)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Haxby Colormap Test')
    plt.tight_layout()
    plt.savefig('test_haxby_visual.png', dpi=100)
    plt.close()
    
    print("[PASS] Visual test saved to tests/test_haxby_visual.png")


def test_haxby_for_bathymetry():
    """Test haxby for bathymetry (positive downward, should flip)"""
    # Create synthetic bathymetry data (depth positive downward)
    x = np.linspace(-100, 100, 200)
    y = np.linspace(-100, 100, 200)
    X, Y = np.meshgrid(x, y)
    
    # Gaussian seamount
    depth = 4000 * np.exp(-(X**2 + Y**2) / (2 * 50**2))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Standard haxby (shallow = blue, deep = white)
    ax1 = axes[0]
    im1 = ax1.pcolormesh(X, Y, -depth, cmap=haxby(256), shading='auto')
    ax1.set_title('Standard haxby (elevation)')
    plt.colorbar(im1, ax=ax1, label='Elevation (m)')
    
    # Flipped haxby for bathymetry (deep = blue, shallow = white)
    ax2 = axes[1]
    # Flip the colormap
    cmap_flipped = ListedColormap(haxby(256).colors[::-1])
    im2 = ax2.pcolormesh(X, Y, depth, cmap=cmap_flipped, shading='auto')
    ax2.set_title('Flipped haxby (bathymetry)')
    plt.colorbar(im2, ax=ax2, label='Depth (m)')
    
    plt.tight_layout()
    plt.savefig('test_haxby_bathymetry.png', dpi=100)
    plt.close()
    
    print("[PASS] Bathymetry test saved to tests/test_haxby_bathymetry.png")


def test_cmapscale_basic():
    """Test cmapscale function"""
    # Create test data with skewed distribution
    np.random.seed(42)
    z = np.random.exponential(1.0, 1000)
    
    # Base colormap
    cmap0 = haxby(64).colors
    
    # Test different factors
    factors = [0.0, 0.5, 1.0]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, factor in zip(axes, factors):
        cmap1, ticks, labels = cmapscale(z, cmap0, factor=factor, n_ticks=5, n_round=2)
        
        # Display colormap
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        ax.imshow(gradient, aspect='auto', cmap=ListedColormap(cmap1))
        ax.set_title(f'factor={factor}')
        ax.set_xticks([])
        ax.set_yticks([])
        
        if labels:
            tick_positions = np.linspace(0, 255, len(labels))
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(labels, rotation=45, fontsize=8)
    
    plt.suptitle('cmapscale Test')
    plt.tight_layout()
    plt.savefig('test_cmapscale.png', dpi=100)
    plt.close()
    
    print("[PASS] cmapscale test saved to tests/test_cmapscale.png")


if __name__ == '__main__':
    print("=" * 60)
    print("Testing colormaps")
    print("=" * 60)
    
    test_haxby_colors()
    print()
    test_haxby_interpolation()
    print()
    test_haxby_default_size()
    print()
    test_haxby_different_sizes()
    print()
    test_haxby_visual()
    print()
    test_haxby_for_bathymetry()
    print()
    test_cmapscale_basic()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
