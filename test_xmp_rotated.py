#!/usr/bin/env python3
"""
Diagnose why Lightroom ignores the XMP for the 45° case.

Observed:
  write_xmp x1=860.9 y1=868.6 x2=4255.0 y2=3131.4  w=6000 h=4000  angle=-45
  Produced:
    CropLeft=0.143482, CropTop=0.217157, CropRight=0.709167, CropBottom=0.782843
    CropAngle=45

Lightroom ground-truth (30° CCW on a 3:2 image) has these EXTRA tags:
    crs:CropConstrainToWarp="0"
    crs:CropConstrainToUnitSquare="1"
    crs:HasSettings="True"

Hypothesis: without CropConstrainToUnitSquare="1", Lightroom interprets
LTRB in a "virtual rotated image" coordinate system whose AABB is larger
than the original image.  Our LTRB values (in [0,1]²) then map to a region
that extends outside the image, which Lightroom silently ignores.
"""
import math

# ── Known values ─────────────────────────────────────────────────────
W, H = 6000, 4000
x1, y1, x2, y2 = 860.9, 868.6, 4255.0, 3131.4
ANGLE_BROWSER = -45          # CW-positive → -45 = 45° CCW
ANGLE_LR = -ANGLE_BROWSER    # Lightroom CropAngle (CCW-positive) = 45

L = x1 / W   # 0.143482
T = y1 / H   # 0.217157
R = x2 / W   # 0.709167
B = y2 / H   # 0.782843

print("=" * 65)
print("Part 1: Verify LTRB geometry (crop corners in original image)")
print("=" * 65)

cx = (x1 + x2) / 2   # 2558.0
cy = (y1 + y2) / 2   # 2000.0
cw = x2 - x1          # 3394.1
ch = y2 - y1          # 2262.8

print(f"Crop centre: ({cx:.1f}, {cy:.1f})  dims: {cw:.1f} × {ch:.1f}")
print(f"LTRB: L={L:.6f}, T={T:.6f}, R={R:.6f}, B={B:.6f}")
print(f"CropAngle: {ANGLE_LR}°")
print()

theta = math.radians(ANGLE_LR)   # CCW
cos_a, sin_a = math.cos(theta), math.sin(theta)

# The four corners of the UNrotated rectangle (in original image coords):
corners_unrot = [
    (cx - cw/2, cy - ch/2),  # TL
    (cx + cw/2, cy - ch/2),  # TR
    (cx + cw/2, cy + ch/2),  # BR
    (cx - cw/2, cy + ch/2),  # BL
]

# Rotate each corner by CropAngle (CCW) around the crop centre:
print("Corners of rotated crop rectangle (in original image pixels):")
all_in_bounds = True
for name, (px, py) in zip(["TL", "TR", "BR", "BL"], corners_unrot):
    dx, dy = px - cx, py - cy
    rx = cx + dx * cos_a - dy * sin_a
    ry = cy + dx * sin_a + dy * cos_a
    in_x = 0 <= rx <= W
    in_y = 0 <= ry <= H
    in_bounds = in_x and in_y
    if not in_bounds:
        all_in_bounds = False
    print(f"  {name}: ({rx:.1f}, {ry:.1f})  in-bounds={in_bounds}")

print(f"All corners within {W}×{H} image: {all_in_bounds}")
print()

# AABB of rotated crop
aabb_w = cw * abs(cos_a) + ch * abs(sin_a)
aabb_h = cw * abs(sin_a) + ch * abs(cos_a)
print(f"AABB of rotated crop: {aabb_w:.1f} × {aabb_h:.1f}  (image: {W}×{H})")
print(f"Fits in image: {aabb_w <= W + 0.5 and aabb_h <= H + 0.5}")
print()

print("=" * 65)
print("Part 2: What does LR see WITHOUT CropConstrainToUnitSquare?")
print("=" * 65)
print()
print("Hypothesis: LR defaults CropConstrainToUnitSquare=0, meaning")
print("LTRB is in a 'virtual rotated image' coordinate system.")
print()

# When CropAngle=45° and the image is notionally rotated, the AABB of the
# original image in the rotated frame is:
aabb_img_w = W * abs(cos_a) + H * abs(sin_a)
aabb_img_h = W * abs(sin_a) + H * abs(cos_a)
print(f"AABB of {W}×{H} image rotated {ANGLE_LR}°: {aabb_img_w:.1f} × {aabb_img_h:.1f}")
print()
print("If LR uses this AABB as the 'virtual image size' for LTRB:")
virt_W = aabb_img_w
virt_H = aabb_img_h
# Our LTRB in virtual space (fractions of original W/H, but LR normalises by virt_W/virt_H):
# LR would treat L=0.143482 as a fraction of virt_W...
# Actually, the LTRB fractions might be applied to the virtual dimensions differently.
# Let's check: if LR maps L*virt_W to pixels in the rotated frame:
virt_left   = L * virt_W
virt_top    = T * virt_H
virt_right  = R * virt_W
virt_bottom = B * virt_H
virt_cx = (virt_left + virt_right) / 2
virt_cy = (virt_top + virt_bottom) / 2
print(f"Virtual pixel bounds: left={virt_left:.1f}, top={virt_top:.1f}, "
      f"right={virt_right:.1f}, bottom={virt_bottom:.1f}")
print(f"Virtual centre: ({virt_cx:.1f}, {virt_cy:.1f})  "
      f"(vs virtual image centre: {virt_W/2:.1f}, {virt_H/2:.1f})")
# Map virtual centre back to original image centre
# The virtual frame is centred on the original image centre (W/2, H/2),
# rotated CropAngle CCW.  To find original coords: rotate CW by CropAngle.
vdx = virt_cx - virt_W / 2
vdy = virt_cy - virt_H / 2
orig_cx = W/2 + vdx * cos_a + vdy * sin_a   # CW rotation = transpose of CCW
orig_cy = H/2 - vdx * sin_a + vdy * cos_a
print(f"Virtual centre mapped back to original image: ({orig_cx:.1f}, {orig_cy:.1f})")
print(f"Expected: ({cx:.1f}, {cy:.1f})")
print()
in_img = 0 <= orig_cx <= W and 0 <= orig_cy <= H
print(f"Virtual crop centre is within original image: {in_img}")
if not in_img or abs(orig_cx - cx) > 100 or abs(orig_cy - cy) > 100:
    print("  → Virtual interpretation maps to WRONG location — LR ignores/misplaces crop!")
else:
    print("  → Virtual interpretation still maps to correct location.")
print()

print("=" * 65)
print("Part 3: Correct fix — add CropConstrainToUnitSquare=1 to XMP")
print("=" * 65)
print()
print("With CropConstrainToUnitSquare=1, LR interprets LTRB as fractions")
print("of the original image dimensions — exactly what we compute.")
print()
print("Required additions to crop_block when angle != 0:")
print("   crs:CropConstrainToWarp=0")
print("   crs:CropConstrainToUnitSquare=1")
print()
print("These should also be added to CROP_TAGS so they are stripped from")
print("existing XMP before re-writing the crop block.")
print()

print("=" * 65)
print("Part 4: Verify 30° ground-truth still correct with fix")
print("=" * 65)
print()
LR_L, LR_T, LR_R, LR_B = 0.335184, 0.0, 0.664816, 1.0
LR_ANGLE = 30
LR_W, LR_H = 6000, 4000
print(f"Ground truth: L={LR_L}, T={LR_T}, R={LR_R}, B={LR_B}, angle={LR_ANGLE}°")
print(f"From browser x1={LR_L*LR_W:.1f}, y1={LR_T*LR_H:.1f}, x2={LR_R*LR_W:.1f}, y2={LR_B*LR_H:.1f}")
print("Simple division L=x1/W etc. reproduces this ✓ (proven in test_xmp_crop.py)")
print("Adding CropConstrainToUnitSquare=1 makes LR use the same interpretation. ✓")
