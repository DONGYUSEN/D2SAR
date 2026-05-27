"""Convert InSAR products to PNG — chunked processing for large arrays."""
import sys, gc
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, '/work')
try:
    from osgeo import gdal
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'gdal', '-q'])
    from osgeo import gdal
gdal.UseExceptions()

OUT = Path('/temp/tops_output/IW1/merged')
BURST_OUT = Path('/temp/tops_output/IW1/burst_ifg')

def read_tiff_complex(path):
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    n = ds.RasterCount
    r = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    if n >= 2:
        i = ds.GetRasterBand(2).ReadAsArray().astype(np.float32)
        return r + 1j * i
    return r

def phase_to_rgb_chunked(phase, valid_mask, chunk_rows=500):
    """Process wrapped phase in chunks to avoid OOM."""
    h, w = phase.shape
    result = np.zeros((h, w, 3), dtype=np.uint8)
    for r0 in range(0, h, chunk_rows):
        r1 = min(r0 + chunk_rows, h)
        ph = np.angle(phase[r0:r1])
        h_ch = ((ph + np.pi) / (2 * np.pi)) % 1.0
        vm = valid_mask[r0:r1] if valid_mask is not None else None

        # HSV→RGB (vectorized per chunk)
        s = np.where(vm, 0.9, 0.4) if vm is not None else 0.9
        v = np.where(vm, 1.0, 0.3) if vm is not None else 1.0
        c = s * v; x = c * (1 - np.abs(((h_ch * 6) % 2) - 1)); m = v - c

        h_f = h_ch.ravel(); c_f = c.ravel(); x_f = x.ravel(); m_f = m.ravel()
        n = len(h_f); sec = (h_f * 6).astype(int) % 6
        r = np.zeros(n, np.float32); g = np.zeros(n, np.float32); b = np.zeros(n, np.float32)

        m0 = sec == 0; m1 = sec == 1; m2 = sec == 2; m3 = sec == 3; m4 = sec == 4; m5 = sec == 5
        r[m0]=c_f[m0]; g[m0]=x_f[m0]; r[m1]=x_f[m1]; g[m1]=c_f[m1]
        g[m2]=c_f[m2]; b[m2]=x_f[m2]; g[m3]=x_f[m3]; b[m3]=c_f[m3]
        r[m4]=x_f[m4]; b[m4]=c_f[m4]; r[m5]=c_f[m5]; b[m5]=x_f[m5]

        rgb = np.stack([r+m_f, g+m_f, b+m_f], axis=1).reshape(3, r1-r0, w)
        chunk_rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
        result[r0:r1] = chunk_rgb
        if vm is not None:
            result[r0:r1][~vm] = [80, 80, 80]
        del ph, h_ch, rgb; gc.collect()

    return result

def save_png_pil(arr, path):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(str(path), optimize=False)
    print(f"  -> {path.name} ({arr.shape[1]}x{arr.shape[0]})")

for name, tiff in [
    ('wrapped', OUT/'merged_interferogram.tif'),
    ('coherence', OUT/'merged_coherence.tif'),
    ('unwrapped', OUT/'unwrapped.tif'),
    ('filtered', OUT/'filtered_ifg.tif'),
]:
    if not tiff.exists():
        print(f"SKIP {name}: not found"); continue
    data = read_tiff_complex(tiff)
    is_complex = np.iscomplexobj(data)
    print(f"{name}: {data.shape}, complex={is_complex}", end=" ", flush=True)
    if is_complex:
        valid = np.abs(data) > 1e-10
        print(f"valid={valid.sum()}/{valid.size}", flush=True)
        rgb = phase_to_rgb_chunked(data, valid)
        save_png_pil(rgb, OUT/f'{name}.png')
        del rgb
    elif name == 'coherence':
        arr = (np.clip(data, 0, 1) * 255).astype(np.uint8)
        save_png_pil(arr, OUT/f'{name}.png')
    else:
        v = np.isfinite(data)
        if v.any():
            lo, hi = np.percentile(data[v].ravel(), [2, 98])
            norm = np.zeros_like(data, dtype=np.uint8)
            mask = v & (data >= lo) & (data <= hi)
            norm[mask] = ((data[mask] - lo) / (hi - lo) * 255).astype(np.uint8)
            print(f"[{lo:.1f}, {hi:.1f}]", end=" ", flush=True)
        else:
            norm = np.zeros_like(data, dtype=np.uint8)
        save_png_pil(norm, OUT/f'{name}.png')
    del data; gc.collect()

print("Done.")

# ─────────────────────────────────────────────────────────────────────────────
# Burst IFG conversion (from npz files)
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Processing Burst IFGs ===")

if not BURST_OUT.exists():
    print(f"BURST_OUT not found: {BURST_OUT}")
else:
    BURST_OUT.mkdir(parents=True, exist_ok=True)

    # Handle both formats: burst_ifg_XXX.npz or burst_XXX/*.npz
    npz_files = sorted(BURST_OUT.glob("burst_ifg_*.npz"))
    if not npz_files:
        burst_dirs = sorted(BURST_OUT.glob("burst_???"))
        npz_files = []
        for bd in burst_dirs:
            npz_files.extend(sorted(bd.glob("*.npz")))
    
    if not npz_files:
        print("No burst IFG npz files found")
    else:
        for npz in npz_files:
            pair_idx = npz.name.split("_")[-1].replace(".npz", "")
            print(f"\n{npz.name}:", end=" ", flush=True)
            try:
                data = np.load(npz)
                ifg = None
                if "ifg" in data:
                    ifg = data["ifg"]
                elif "phase" in data:
                    ifg = data["phase"]
                elif "slc" in data:
                    ifg = data["slc"]
                else:
                    for key in data.files:
                        arr = data[key]
                        if np.iscomplexobj(arr):
                            ifg = arr
                            break
                
                if ifg is None:
                    print(f"skip (no ifg/phase key)"); continue

                if np.iscomplexobj(ifg):
                    valid = np.abs(ifg) > 1e-10
                    print(f"shape={ifg.shape}, valid={valid.sum()}/{valid.size}", flush=True)
                    rgb = phase_to_rgb_chunked(ifg, valid)
                    png_path = BURST_OUT / f"burst_ifg_{pair_idx}.png"
                else:
                    valid = np.isfinite(ifg)
                    print(f"shape={ifg.shape}, real", flush=True)
                    arr = (np.clip(ifg, 0, 1) * 255).astype(np.uint8)
                    rgb = np.stack([arr]*3, axis=-1) if arr.ndim == 2 else arr
                    png_path = BURST_OUT / f"burst_coh_{pair_idx}.png"

                save_png_pil(rgb, png_path)
                del rgb, ifg
            except Exception as e:
                print(f"error: {e}")

print("\nDone.")
