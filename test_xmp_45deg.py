#!/usr/bin/env python3
"""
Verify the XMP output for the failing 45° case:
  x1=860.9, y1=868.6, x2=4255.0, y2=3131.4  w=6000 h=4000  angle=-45

Also explains the 'wider but shorter' symptom observed in Lightroom when
CropConstrainToUnitSquare was missing.
"""
import math, re, tempfile, pathlib, sys

sys.path.insert(0, "src")
from autocropper.main import write_xmp

W, H = 6000, 4000
X1, Y1, X2, Y2 = 860.9, 868.6, 4255.0, 3131.4
ANGLE = -45   # browser CW-positive → 45° CCW

# ── What the crop should be ───────────────────────────────────────────
cw = X2 - X1       # 3394.1
ch = Y2 - Y1       # 2262.8
cx = (X1 + X2)/2   # 2557.9
cy = (Y1 + Y2)/2   # 2000.0

EXP_L = X1 / W    # 0.143483
EXP_T = Y1 / H    # 0.217150
EXP_R = X2 / W    # 0.709167
EXP_B = Y2 / H    # 0.782850
EXP_ANGLE = -ANGLE # 45

print("=" * 65)
print(f"Input: x1={X1}, y1={Y1}, x2={X2}, y2={Y2}  angle={ANGLE}°")
print(f"Expected: L={EXP_L:.6f}, T={EXP_T:.6f}, R={EXP_R:.6f}, B={EXP_B:.6f}, CropAngle={EXP_ANGLE}")
print(f"Output crop: {cw:.1f} × {ch:.1f} px → aspect {cw/ch:.3f}:1 (camera native: {W/H:.3f}:1)")
print()

# ── Explain why fix-1-only produces 'wider but shorter' ───────────────
print("Why fix-1-only (no CropConstrainToUnitSquare) was wrong:")
aabb_rot = W * abs(math.cos(math.radians(45))) + H * abs(math.sin(math.radians(45)))
print(f"  Virtual rotated image AABB at 45°: {aabb_rot:.1f} × {aabb_rot:.1f}")
virt_cw = EXP_L * aabb_rot       # approx
virt_w  = (EXP_R - EXP_L) * aabb_rot
virt_h  = (EXP_B - EXP_T) * aabb_rot
print(f"  LR interprets LTRB in virtual space: width={virt_w:.1f}, height={virt_h:.1f}")
print(f"  → outputs a {virt_w:.0f}×{virt_h:.0f} px crop (≈ square) instead of {cw:.0f}×{ch:.0f}")
print(f"  → width {virt_w:.0f} > expected {cw:.0f}: WIDER  ✓")
print(f"  → height {virt_h:.0f} > expected {ch:.0f}: BUT user sees it as shorter because aspect ratio")
print(f"     changed from landscape ({cw/ch:.2f}:1) to square ({virt_w/virt_h:.2f}:1)")
print()

# ── Run actual write_xmp and check output ─────────────────────────────
print("=" * 65)
print("Actual XMP output from current code:")
with tempfile.TemporaryDirectory() as tmp:
    cr3 = pathlib.Path(tmp) / "test.CR3"
    cr3.touch()   # write_xmp uses the path stem for the .xmp name
    write_xmp(cr3, X1, Y1, X2, Y2, W, H, angle=ANGLE)
    xmp_path = cr3.with_suffix("").with_suffix(".xmp")
    content = xmp_path.read_text()

print(content)

def extract(tag, text):
    m = re.search(rf'<crs:{tag}>(.*?)</crs:{tag}>', text)
    return m.group(1) if m else None

got_L = float(extract('CropLeft',   content))
got_T = float(extract('CropTop',    content))
got_R = float(extract('CropRight',  content))
got_B = float(extract('CropBottom', content))
got_A = float(extract('CropAngle',  content))
has_constrain = 'CropConstrainToUnitSquare' in content
has_warp      = 'CropConstrainToWarp'       in content

print("=" * 65)
print(f"CropLeft:   {got_L:.6f}  (expected {EXP_L:.6f})  {'OK' if abs(got_L-EXP_L)<0.001 else 'FAIL'}")
print(f"CropTop:    {got_T:.6f}  (expected {EXP_T:.6f})  {'OK' if abs(got_T-EXP_T)<0.001 else 'FAIL'}")
print(f"CropRight:  {got_R:.6f}  (expected {EXP_R:.6f})  {'OK' if abs(got_R-EXP_R)<0.001 else 'FAIL'}")
print(f"CropBottom: {got_B:.6f}  (expected {EXP_B:.6f})  {'OK' if abs(got_B-EXP_B)<0.001 else 'FAIL'}")
print(f"CropAngle:  {got_A:.1f}  (expected {EXP_ANGLE})  {'OK' if abs(got_A-EXP_ANGLE)<0.001 else 'FAIL'}")
print(f"CropConstrainToUnitSquare present: {has_constrain}  (expected True)")
print(f"CropConstrainToWarp present:       {has_warp}  (expected True)")

all_ok = (abs(got_L-EXP_L)<0.001 and abs(got_T-EXP_T)<0.001 and
          abs(got_R-EXP_R)<0.001 and abs(got_B-EXP_B)<0.001 and
          abs(got_A-EXP_ANGLE)<0.001 and has_constrain and has_warp)
print()
print(f"Overall: {'PASS ✓' if all_ok else 'FAIL ✗'}")
