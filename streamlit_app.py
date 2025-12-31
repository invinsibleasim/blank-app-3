
import os
import io
import cv2
import time
import json
import zipfile
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any

# ---------------------------
# Safe OpenCV import guard
# ---------------------------
try:
    import cv2  # keep explicit import for clarity
except Exception as e:
    st.error(
        "Failed to import OpenCV (cv2). On Streamlit Cloud:\n"
        "• Add 'libgl1', 'libglib2.0-0', 'libstdc++6' in packages.txt\n"
        "• Use 'opencv-python-headless' in requirements.txt\n\n"
        f"Error:\n{e}"
    )
    st.stop()

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR uint8"""
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR -> PIL RGB"""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

def save_image(path: Path, image_bgr: np.ndarray, quality: int = 95):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

def zip_directory(src_dir: Path, zip_path: Path):
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src_dir):
            for f in files:
                fp = Path(root) / f
                zf.write(fp, fp.relative_to(src_dir))

def make_cells_zip_in_memory(cells: List[Dict[str, Any]], limit: int = 144, sequential: bool = True) -> io.BytesIO:
    """
    Create a ZIP in memory with up to `limit` cell images.
    Names:
      - sequential=True  -> cells/cell_001.jpg ... cells/cell_144.jpg
      - sequential=False -> cells/cell_rRR_cCC.jpg
    """
    buf_zip = io.BytesIO()
    with zipfile.ZipFile(buf_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        take = min(limit, len(cells))
        for idx in range(take):
            cell = cells[idx]
            img_bgr = cell["image"]
            # Write JPEG into memory
            is_ok, encoded = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not is_ok:
                continue
            if sequential:
                fname = f"cells/cell_{idx+1:03d}.jpg"
            else:
                fname = f"cells/cell_r{cell['row']:02d}_c{cell['col']:02d}.jpg"
            zf.writestr(fname, encoded.tobytes())
    buf_zip.seek(0)
    return buf_zip

# ---------------------------
# Optional alignment helpers (OpenCV)
# ---------------------------
def perspective_warp(img_bgr: np.ndarray) -> np.ndarray:
    """
    Find largest quadrilateral and warp to rectangular view.
    Helpful if module frame is visible and skewed.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return img_bgr

    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    rect = np.array([tl, tr, br, bl], dtype=np.float32)

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    if maxW < 100 or maxH < 100:
        return img_bgr

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (maxW, maxH), flags=cv2.INTER_LINEAR)
    return warped

def auto_deskew(img_bgr: np.ndarray, gray_u8: np.ndarray, hough_thresh: int = 120) -> np.ndarray:
    """
    Estimate dominant line angle via Hough transform and rotate to align grid.
    """
    edges = cv2.Canny(gray_u8, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, hough_thresh)
    if lines is None:
        return img_bgr
    angles = []
    for l in lines[:200]:
        theta = l[0][1]
        deg = np.rad2deg(theta)
        deg = ((deg + 90) % 180) - 90  # map to [-90,90)
        angles.append(deg)
    if len(angles) == 0:
        return img_bgr
    mean_angle = float(np.mean(angles))
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -mean_angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ---------------------------
# EL Preprocessing (OpenCV only)
# ---------------------------
def normalize_el(img_bgr: np.ndarray, clip_limit_frac: float = 0.02, tile_size: int = 8, sigma: float = 0.6) -> np.ndarray:
    """
    CLAHE + optional Gaussian blur.
    clip_limit_frac: skimage-like fraction mapped to OpenCV CLAHE clipLimit ~ [1..10]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cv_clip = max(1.0, min(10.0, clip_limit_frac * 100.0))
    clahe = cv2.createCLAHE(clipLimit=cv_clip, tileGridSize=(tile_size, tile_size))
    gray_norm = clahe.apply(gray)
    if sigma and sigma > 0:
        k = max(3, int(2 * sigma) * 2 + 1)  # odd kernel
        gray_norm = cv2.GaussianBlur(gray_norm, (k, k), 0)
    return gray_norm

# ---------------------------
# Grid line detection + cell building (OpenCV only)
# ---------------------------
def binarize(gray_u8: np.ndarray, mode: str = "otsu") -> np.ndarray:
    """Return binary image uint8 {0,255} using Otsu or adaptive threshold."""
    if mode == "adaptive":
        bw = cv2.adaptiveThreshold(gray_u8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
    else:
        _, bw = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return bw

def detect_grid_lines(gray_u8: np.ndarray,
                      polarity: str = "auto",
                      binarize_mode: str = "otsu",
                      ksize_v: int = 25,
                      ksize_h: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect vertical and horizontal line maps using OpenCV morphology.
    """
    bw = binarize(gray_u8, mode=binarize_mode)
    if polarity == "auto":
        use = 255 - bw if np.mean(gray_u8) > 127 else bw
    elif polarity == "dark":
        use = 255 - bw
    else:
        use = bw

    use_u8 = (use > 0).astype(np.uint8) * 255

    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ksize_v))
    vert = cv2.erode(use_u8, kernel_v, iterations=1)
    vert = cv2.dilate(vert, kernel_v, iterations=1)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize_h, 1))
    horiz = cv2.erode(use_u8, kernel_h, iterations=1)
    horiz = cv2.dilate(horiz, kernel_h, iterations=1)

    return vert, horiz

def project_peaks(line_map_u8: np.ndarray, axis: int = 0, min_dist: int = 30, min_strength: float = 0.3) -> List[int]:
    """
    Sum along axis and find local maxima separated by min_dist.
    axis=0 → columns (vertical lines), axis=1 → rows (horizontal lines)
    """
    proj = line_map_u8.sum(axis=axis).astype(np.float64)
    rng = proj.max() - proj.min()
    proj_norm = (proj - proj.min()) / (rng + 1e-6)
    peaks = []
    last_idx = -min_dist
    for i in range(1, len(proj_norm) - 1):
        if proj_norm[i] > min_strength and proj_norm[i] > proj_norm[i - 1] and proj_norm[i] > proj_norm[i + 1]:
            if i - last_idx >= min_dist:
                peaks.append(i)
                last_idx = i
    return peaks

def cuts_from_peaks(peaks: List[int], maxlen: int) -> List[int]:
    """Convert line positions (peaks) into cell boundary cut positions."""
    if len(peaks) < 2:
        return [0, maxlen - 1]
    cuts = [0]
    for i in range(len(peaks) - 1):
        cuts.append((peaks[i] + peaks[i + 1]) // 2)
    cuts.append(maxlen - 1)
    return sorted(list(set(cuts)))

def build_cells_from_grid(img_bgr: np.ndarray,
                          vert_map_u8: np.ndarray,
                          horiz_map_u8: np.ndarray,
                          min_cell_w: int = 40,
                          min_cell_h: int = 40) -> List[Dict[str, Any]]:
    H, W = vert_map_u8.shape
    xs = project_peaks(vert_map_u8, axis=0, min_dist=max(20, W // 30))
    ys = project_peaks(horiz_map_u8, axis=1, min_dist=max(20, H // 30))

    xcuts = cuts_from_peaks(xs, W)
    ycuts = cuts_from_peaks(ys, H)

    cells = []
    for r in range(len(ycuts) - 1):
        y0, y1 = ycuts[r], ycuts[r + 1]
        for c in range(len(xcuts) - 1):
            x0, x1 = xcuts[c], xcuts[c + 1]
            w, h = x1 - x0, y1 - y0
            if w >= min_cell_w and h >= min_cell_h:
                crop = img_bgr[y0:y1, x0:x1].copy()
                cells.append({"row": r, "col": c, "bbox": (x0, y0, x1, y1), "image": crop})
    return cells

def overlay_grid(img_bgr: np.ndarray, cells: List[Dict[str, Any]], color=(0, 255, 0), thickness=2) -> np.ndarray:
    vis = img_bgr.copy()
    for cell in cells:
        x0, y0, x1, y1 = cell["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, thickness)
    return vis

def manual_split(img_bgr: np.ndarray, n_rows: int, n_cols: int, margin: int = 0) -> List[Dict[str, Any]]:
    """
    Evenly split the image into n_rows × n_cols blocks as a fallback/preset (e.g., 6×24 = 144).
    """
    h, w = img_bgr.shape[:2]
    x0, y0 = margin, margin
    x1, y1 = w - margin, h - margin
    cell_w = (x1 - x0) // n_cols
    cell_h = (y1 - y0) // n_rows
    cells = []
    for r in range(n_rows):
        for c in range(n_cols):
            cx0 = x0 + c * cell_w
            cy0 = y0 + r * cell_h
            cx1 = cx0 + cell_w
            cy1 = cy0 + cell_h
            crop = img_bgr[cy0:cy1, cx0:cx1].copy()
            cells.append({"row": r, "col": c, "bbox": (cx0, cy0, cx1, cy1), "image": crop})
    return cells

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="EL PV Module → Cell Segregation (OpenCV-only)", layout="wide")
st.title("🔬 EL PV Module → Cell Segregation (OpenCV‑only)")

st.markdown("""
Upload **EL PV module** images to automatically detect the **cell grid** (OpenCV-only) and export **per‑cell crops**.
Use the **144‑cell presets** for half‑cut/full layouts, or the **manual fallback** if auto detection struggles.
""")

# Sidebar controls
st.sidebar.header("⚙️ Preprocessing & Alignment")
clip_limit = st.sidebar.slider("CLAHE clipLimit (mapped from fraction)", 0.005, 0.05, 0.02, 0.005)
tile_size  = st.sidebar.slider("CLAHE tile size", 4, 32, 8, 2)
sigma      = st.sidebar.slider("Gaussian sigma", 0.0, 2.0, 0.6, 0.1)
do_warp    = st.sidebar.checkbox("Perspective warp (rectify module)", True)
do_deskew  = st.sidebar.checkbox("Auto deskew (align grid)", True)

st.sidebar.header("🎛️ Detection")
polarity   = st.sidebar.selectbox("Line polarity", ["auto", "dark", "bright"], index=0)
binarize_mode = st.sidebar.selectbox("Binarization", ["otsu", "adaptive"], index=0)
ksize_v    = st.sidebar.slider("Vertical kernel size", 5, 75, 25, 1)
ksize_h    = st.sidebar.slider("Horizontal kernel size", 5, 75, 25, 1)
min_cell_w = st.sidebar.slider("Min cell width (px)", 20, 400, 40, 10)
min_cell_h = st.sidebar.slider("Min cell height (px)", 20, 400, 40, 10)

st.sidebar.header("📐 Layout preset (144)")
layout_preset = st.sidebar.selectbox("Preset", ["None", "6×24 (144)", "12×12 (144)"], index=0)

st.sidebar.header("🧩 Manual fallback")
use_manual    = st.sidebar.checkbox("Use manual rows × cols fallback", False)
n_rows        = st.sidebar.number_input("Rows", min_value=1, max_value=30, value=6)
n_cols        = st.sidebar.number_input("Cols", min_value=1, max_value=48, value=10)
manual_margin = st.sidebar.number_input("Manual margin (px)", min_value=0, max_value=200, value=0)

# Force manual split for 144 presets
if layout_preset == "6×24 (144)":
    use_manual = True
    n_rows, n_cols = 6, 24
elif layout_preset == "12×12 (144)":
    use_manual = True
    n_rows, n_cols = 12, 12

st.sidebar.header("💾 Output")
out_dir_str = st.sidebar.text_input("Output directory", "output")
save_144_to_disk = st.sidebar.checkbox("Also save 144‑cells ZIP to disk", True)
seq_names_144    = st.sidebar.checkbox("Use sequential names (001–144)", True)

start_btn = st.sidebar.button("🚀 Run")

uploads = st.file_uploader("Upload EL module image(s)", type=["jpg","jpeg","png","bmp","tif","tiff"], accept_multiple_files=True)

# ---------------------------
# Processing
# ---------------------------
def process_single_image(img_pil: Image.Image,
                         settings: Dict[str, Any],
                         save_root: Path) -> Dict[str, Any]:
    t0 = time.time()
    img_bgr = pil_to_cv(img_pil)

    # Optional perspective warp first
    if settings["do_warp"]:
        img_bgr = perspective_warp(img_bgr)

    # Preprocess EL for line detection
    gray_u8 = normalize_el(img_bgr,
                           clip_limit_frac=settings["clip_limit"],
                           tile_size=settings["tile_size"],
                           sigma=settings["sigma"])

    # Optional deskew
    if settings["do_deskew"]:
        img_bgr = auto_deskew(img_bgr, gray_u8)
        gray_u8 = normalize_el(img_bgr,
                               clip_limit_frac=settings["clip_limit"],
                               tile_size=settings["tile_size"],
                               sigma=settings["sigma"])

    # Auto detection or manual split
    if not settings["use_manual"]:
        vert_map, horiz_map = detect_grid_lines(gray_u8,
                                                polarity=settings["polarity"],
                                                binarize_mode=settings["binarize_mode"],
                                                ksize_v=settings["ksize_v"],
                                                ksize_h=settings["ksize_h"])
        cells = build_cells_from_grid(img_bgr, vert_map, horiz_map,
                                      min_cell_w=settings["min_cell_w"],
                                      min_cell_h=settings["min_cell_h"])
    else:
        cells = manual_split(img_bgr,
                             n_rows=settings["n_rows"],
                             n_cols=settings["n_cols"],
                             margin=settings["manual_margin"])

    grid_overlay = overlay_grid(img_bgr, cells)

    # Save outputs
    ensure_dir(save_root)
    save_image(save_root / "module_grid.jpg", grid_overlay)

    cells_dir = save_root / "cells"
    ensure_dir(cells_dir)
    meta = []
    for cell in cells:
        r, c = cell["row"], cell["col"]
        save_image(cells_dir / f"cell_{r:02d}_{c:02d}.jpg", cell["image"])
        meta.append({"row": r, "col": c, "bbox": cell["bbox"]})

    with open(save_root / "summary.json", "w") as f:
        json.dump({"n_cells": len(cells), "cells": meta}, f, indent=2)

    return {
        "n_cells": len(cells),
        "grid_overlay": grid_overlay,
        "cells": cells,
        "elapsed": time.time() - t0
    }

# Run pipeline
if start_btn:
    out_base = Path(out_dir_str)
    ensure_dir(out_base)

    if not uploads:
        st.warning("Please upload at least one image.")
    else:
        for upl in uploads:
            img_pil = Image.open(io.BytesIO(upl.read())).convert("RGB")
            save_dir = out_base / Path(upl.name).stem

            res = process_single_image(
                img_pil,
                settings={
                    "clip_limit": clip_limit,
                    "tile_size": tile_size,
                    "sigma": sigma,
                    "do_warp": do_warp,
                    "do_deskew": do_deskew,
                    "polarity": polarity,
                    "binarize_mode": binarize_mode,
                    "ksize_v": ksize_v,
                    "ksize_h": ksize_h,
                    "min_cell_w": min_cell_w,
                    "min_cell_h": min_cell_h,
                    "use_manual": use_manual,
                    "n_rows": int(n_rows),
                    "n_cols": int(n_cols),
                    "manual_margin": int(manual_margin)
                },
                save_root=save_dir
            )

            st.success(f"Processed {upl.name}: {res['n_cells']} cells in {res['elapsed']:.2f}s")
            st.image(cv_to_pil(res["grid_overlay"]), caption=f"Detected cell grid: {upl.name}", use_column_width=True)

            # Preview: show first 12 cells
            cols_show = st.columns(min(6, max(1, int(n_cols))))
            preview_count = min(len(res["cells"]), 12)
            for i in range(preview_count):
                r, c = res["cells"][i]["row"], res["cells"][i]["col"]
                cols_show[i % len(cols_show)].image(
                    cv_to_pil(res["cells"][i]["image"]),
                    caption=f"Cell r{r} c{c}",
                    use_column_width=True
                )

            # Full results ZIP (everything saved to disk)
            zip_path = save_dir / f"{Path(upl.name).stem}_results.zip"
            zip_directory(save_dir, zip_path)
            with open(zip_path, "rb") as f:
                st.download_button(f"📦 Download all outputs for {upl.name}", data=f, file_name=zip_path.name)

            # === 144-cells ZIP (in-memory + optional save to disk) ===
            n_detected = len(res["cells"])
            if n_detected < 144 and layout_preset == "None":
                st.warning(
                    f"{upl.name}: Only {n_detected} cells detected. "
                    "Use the 'Module layout preset' (6×24 or 12×12) to guarantee 144 crops."
                )

            zip144_mem = make_cells_zip_in_memory(res["cells"], limit=144, sequential=seq_names_144)
            st.download_button(
                "📦 Download 144 cells (001–144)" if seq_names_144 else "📦 Download 144 cells (row/col names)",
                data=zip144_mem,
                file_name=f"{Path(upl.name).stem}_cells_144.zip",
                mime="application/zip"
            )

            if save_144_to_disk:
                save_dir_144 = save_dir / f"{Path(upl.name).stem}_cells_144.zip"
                with open(save_dir_144, "wb") as f:
                    f.write(zip144_mem.getvalue())
                st.info(f"Saved 144-cells ZIP to disk: {save_dir_144}")

st.markdown("---")
st.caption("Tips: If busbars/gridlines are faint, increase kernel sizes (vertical/horizontal). Use 'adaptive' threshold for non‑uniform EL. For half‑cut modules, choose the 6×24 preset to guarantee 144 crops.")
