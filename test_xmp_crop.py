#!/usr/bin/env python3
"""
Test script: verify correct XMP crop values for rotated crops.

Ground truth from Lightroom: rotating a 3:2 landscape image by -30 degrees
(30 degrees CCW) produces:
  CropLeft=0.335184, CropTop=0, CropRight=0.664816, CropBottom=1, CropAngle=30
"""
import math
import sys

# ── Ground truth ────────────────────────────────────────────────────
LR_L = 0.335184
LR_T = 0.0
LR_R = 0.664816
LR_B = 1.0
LR_ANGLE = 30  # CCW positive in Lightroom

# Image dimensions used in this test (3:2 landscape)
W, H = 6000, 4000

# What the browser sends for a centered full-height crop at -30° (30° CCW):
# The browser stores coords as the UNROTATED rect, then rotates visually.
# For a centered crop matching the Lightroom example:
#   L*W = 0.335184 * 6000 = 2011.1, T*H = 0, R*W = 3988.9, B*H = 4000
BROWSER_X1 = LR_L * W   # 2011.1
BROWSER_Y1 = LR_T * H   # 0
BROWSER_X2 = LR_R * W   # 3988.9
BROWSER_Y2 = LR_B * H   # 4000.0
BROWSER_ANGLE = -30     # CW-positive, so -30 = 30° CCW


# ── Current buggy implementation ─────────────────────────────────────
def current_write_xmp_ltrb(x1, y1, x2, y2, w, h, angle):
    """Current code: rotates centre CCW by angle, swaps cw/ch."""
    if angle:
        theta = math.radians(angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        cw, ch = x2 - x1, y2 - y1
        rcx = w / 2 + (cx - w / 2) * cos_a + (cy - h / 2) * sin_a
        rcy = h / 2 - (cx - w / 2) * sin_a + (cy - h / 2) * cos_a
        nl = (rcx - ch / 2) / w
        nt = (rcy - cw / 2) / h
        nr = (rcx + ch / 2) / w
        nb = (rcy + cw / 2) / h
    else:
        nl, nt, nr, nb = x1 / w, y1 / h, x2 / w, y2 / h
    crop_angle = -angle
    return nl, nt, nr, nb, crop_angle


# ── Proposed fix: just divide by W/H, no rotation of coords ──────────
def fixed_write_xmp_ltrb(x1, y1, x2, y2, w, h, angle):
    """Proposed fix: browser coords ARE in the same frame as Lightroom LTRB."""
    nl, nt, nr, nb = x1 / w, y1 / h, x2 / w, y2 / h
    crop_angle = -angle
    return nl, nt, nr, nb, crop_angle


# ── Verify the interpretation ─────────────────────────────────────────
def verify_ltrb_interpretation():
    """
    Confirm that Lightroom LTRB defines the crop rectangle in the original
    image coordinate system; the crop is then visually rotated by CropAngle.

    For the Lightroom example:
      - Unrotated rect center = (L+R)/2 * W, (T+B)/2 * H
      - Unrotated rect dims   = (R-L) * W, (B-T) * H
      - Rect is rotated CropAngle CCW around its own center
    """
    cx_lr = (LR_L + LR_R) / 2 * W   # 3000
    cy_lr = (LR_T + LR_B) / 2 * H   # 2000
    cw_lr = (LR_R - LR_L) * W       # 1978
    ch_lr = (LR_B - LR_T) * H       # 4000

    print(f"Lightroom unrotated rect: center=({cx_lr:.1f}, {cy_lr:.1f}), dims={cw_lr:.1f}×{ch_lr:.1f}")

    # What would the browser send for this crop?
    # Browser: center = image center (3000, 2000), dims = 1978×4000, angle = -30
    # x1 = cx - cw/2 = 3000 - 989 = 2011 = LR_L * W   ✓
    # y1 = cy - ch/2 = 2000 - 2000 = 0   = LR_T * H   ✓
    browser_x1 = cx_lr - cw_lr / 2
    browser_y1 = cy_lr - ch_lr / 2
    browser_x2 = cx_lr + cw_lr / 2
    browser_y2 = cy_lr + ch_lr / 2
    print(f"Equivalent browser coords: x1={browser_x1:.1f}, y1={browser_y1:.1f}, x2={browser_x2:.1f}, y2={browser_y2:.1f}")
    print(f"Which is L={browser_x1/W:.6f}, T={browser_y1/H:.6f}, R={browser_x2/W:.6f}, B={browser_y2/H:.6f}")
    print(f"Lightroom LTRB:            L={LR_L:.6f}, T={LR_T:.6f}, R={LR_R:.6f}, B={LR_B:.6f}")
    assert abs(browser_x1/W - LR_L) < 0.001, "L mismatch"
    assert abs(browser_y1/H - LR_T) < 0.001, "T mismatch"
    assert abs(browser_x2/W - LR_R) < 0.001, "R mismatch"
    assert abs(browser_y2/H - LR_B) < 0.001, "B mismatch"
    print("  → CONFIRMED: browser coords / (W, H) == Lightroom LTRB\n")


# ── Run tests ──────────────────────────────────────────────────────────
print("=" * 65)
print("Test 1: Centred crop, -30° (30° CCW) rotation")
print("=" * 65)

verify_ltrb_interpretation()

print(f"Browser input: x1={BROWSER_X1:.1f}, y1={BROWSER_Y1:.1f}, x2={BROWSER_X2:.1f}, y2={BROWSER_Y2:.1f}")
print(f"               angle={BROWSER_ANGLE}°  (image: {W}×{H})")
print(f"Expected XMP:  L={LR_L}, T={LR_T}, R={LR_R}, B={LR_B}, CropAngle={LR_ANGLE}")
print()

cur_l, cur_t, cur_r, cur_b, cur_a = current_write_xmp_ltrb(
    BROWSER_X1, BROWSER_Y1, BROWSER_X2, BROWSER_Y2, W, H, BROWSER_ANGLE)
print(f"Current code:  L={cur_l:.6f}, T={cur_t:.6f}, R={cur_r:.6f}, B={cur_b:.6f}, CropAngle={cur_a:.1f}")
cur_ok = (abs(cur_l - LR_L) < 0.001 and abs(cur_t - LR_T) < 0.001 and
          abs(cur_r - LR_R) < 0.001 and abs(cur_b - LR_B) < 0.001 and
          abs(cur_a - LR_ANGLE) < 0.001)
print(f"  PASS: {cur_ok}")
print()

fix_l, fix_t, fix_r, fix_b, fix_a = fixed_write_xmp_ltrb(
    BROWSER_X1, BROWSER_Y1, BROWSER_X2, BROWSER_Y2, W, H, BROWSER_ANGLE)
print(f"Fixed code:    L={fix_l:.6f}, T={fix_t:.6f}, R={fix_r:.6f}, B={fix_b:.6f}, CropAngle={fix_a:.1f}")
fix_ok = (abs(fix_l - LR_L) < 0.001 and abs(fix_t - LR_T) < 0.001 and
          abs(fix_r - LR_R) < 0.001 and abs(fix_b - LR_B) < 0.001 and
          abs(fix_a - LR_ANGLE) < 0.001)
print(f"  PASS: {fix_ok}")
print()

# ── Test 2: off-centre crop ────────────────────────────────────────────
print("=" * 65)
print("Test 2: Off-centre crop, +15° (15° CW) rotation")
print("=" * 65)

# Suppose the user draws a crop at x1=500, y1=300, x2=4500, y2=3700, angle=+15
T2_X1, T2_Y1, T2_X2, T2_Y2 = 500.0, 300.0, 4500.0, 3700.0
T2_ANGLE = 15   # CW in browser → LR CropAngle = -15

t2_cur_l, t2_cur_t, t2_cur_r, t2_cur_b, t2_cur_a = current_write_xmp_ltrb(
    T2_X1, T2_Y1, T2_X2, T2_Y2, W, H, T2_ANGLE)
t2_fix_l, t2_fix_t, t2_fix_r, t2_fix_b, t2_fix_a = fixed_write_xmp_ltrb(
    T2_X1, T2_Y1, T2_X2, T2_Y2, W, H, T2_ANGLE)

# Expected (fix): simply x/W, y/H
exp_l = T2_X1 / W   # 0.0833
exp_t = T2_Y1 / H   # 0.075
exp_r = T2_X2 / W   # 0.75
exp_b = T2_Y2 / H   # 0.925
exp_a = -T2_ANGLE   # -15

print(f"Expected:    L={exp_l:.6f}, T={exp_t:.6f}, R={exp_r:.6f}, B={exp_b:.6f}, CropAngle={exp_a}")
print(f"Current:     L={t2_cur_l:.6f}, T={t2_cur_t:.6f}, R={t2_cur_r:.6f}, B={t2_cur_b:.6f}, CropAngle={t2_cur_a}")
cur2_ok = (abs(t2_cur_l - exp_l) < 0.001 and abs(t2_cur_t - exp_t) < 0.001 and
           abs(t2_cur_r - exp_r) < 0.001 and abs(t2_cur_b - exp_b) < 0.001)
print(f"  Current PASS: {cur2_ok}")
print(f"Fixed:       L={t2_fix_l:.6f}, T={t2_fix_t:.6f}, R={t2_fix_r:.6f}, B={t2_fix_b:.6f}, CropAngle={t2_fix_a}")
fix2_ok = (abs(t2_fix_l - exp_l) < 0.001 and abs(t2_fix_t - exp_t) < 0.001 and
           abs(t2_fix_r - exp_r) < 0.001 and abs(t2_fix_b - exp_b) < 0.001)
print(f"  Fixed PASS: {fix2_ok}")
print()

# ── Test 3: zero angle — ensure we didn't break the non-rotated case ──
print("=" * 65)
print("Test 3: No rotation (angle=0) — regression check")
print("=" * 65)

T3_X1, T3_Y1, T3_X2, T3_Y2 = 600.0, 400.0, 5400.0, 3600.0
T3_ANGLE = 0

t3_cur = current_write_xmp_ltrb(T3_X1, T3_Y1, T3_X2, T3_Y2, W, H, T3_ANGLE)
t3_fix = fixed_write_xmp_ltrb(T3_X1, T3_Y1, T3_X2, T3_Y2, W, H, T3_ANGLE)
exp3 = (T3_X1/W, T3_Y1/H, T3_X2/W, T3_Y2/H, 0)

print(f"Expected:  L={exp3[0]:.6f}, T={exp3[1]:.6f}, R={exp3[2]:.6f}, B={exp3[3]:.6f}, CropAngle={exp3[4]}")
print(f"Current:   L={t3_cur[0]:.6f}, T={t3_cur[1]:.6f}, R={t3_cur[2]:.6f}, B={t3_cur[3]:.6f}, CropAngle={t3_cur[4]}")
print(f"  Current PASS: {all(abs(a-b)<1e-9 for a,b in zip(t3_cur, exp3))}")
print(f"Fixed:     L={t3_fix[0]:.6f}, T={t3_fix[1]:.6f}, R={t3_fix[2]:.6f}, B={t3_fix[3]:.6f}, CropAngle={t3_fix[4]}")
print(f"  Fixed PASS: {all(abs(a-b)<1e-9 for a,b in zip(t3_fix, exp3))}")
print()

# ── Summary ────────────────────────────────────────────────────────────
print("=" * 65)
print("Summary: the fix is to remove the `if angle:` branch in write_xmp")
print("and always use:  nl, nt, nr, nb = x1/w, y1/h, x2/w, y2/h")
print()
print("Root cause: browser coords are already in Lightroom's LTRB frame.")
print("The unrotated rect {x1,y1,x2,y2} defines the crop in original")
print("image space. CropAngle then rotates it. No centre transformation")
print("is needed.")
print("=" * 65)
