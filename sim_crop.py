#!/usr/bin/env python3
"""
Find the formula: LR -30 deg gives L=0.335184, T=0, R=0.664816, B=1.
LTRB is in the OUTPUT (rotated) frame, normalised by W and H.
All corners (TR, BL) of LR's crop are OUTSIDE the original image — so LR
doesn't require all corners to be in-bounds.

Key insight to check: maybe LR's LTRB, combined with CropAngle, means:
  - (cx, cy) = centre of the TILTED crop rectangle, in original image normalised space
  - (cw, ch) = dimensions of the crop rectangle IN THE TILTED FRAME, normalised by W,H

So L = (cx - cw/2) / W, T = (cy - ch/2) / H  — BUT cw/ch is the tilted-frame aspect ratio.

From -30 deg data: T=0, B=1 → ch/H = 1.0.  L=0.335184 → cw/W = 0.329632.

Question: what formula for cw gives 0.329632 for a 3:2 image at θ=30?
"""
import math

THETA_DEG = 30.0
THETA = math.radians(THETA_DEG)
cos_t, sin_t = math.cos(THETA), math.sin(THETA)

LR_L = 0.335184
LR_cw_norm = 1 - 2*LR_L   # = 0.329632

print(f"Target cw/W = {LR_cw_norm:.6f}  for θ={THETA_DEG}°")
print()

# For various image aspect ratios r=W/H, compute cw/W under several hypotheses:
print(f"{'r=W/H':>8}  {'tan(θ)·H/W':>12}  {'hyp_A':>12}  {'hyp_B':>12}  {'hyp_C':>12}  {'hyp_D':>12}  {'hyp_E':>12}")
for r_10 in range(8, 31):
    r = r_10 / 10.0
    # H=1 normalised, W=r
    W, H = r, 1.0

    # Hyp A: cw = W * cos(2θ)
    hyp_A = math.cos(2*THETA)

    # Hyp B: cw/W = (1 - H/W*tan(θ))
    hyp_B = 1 - (H/W)*math.tan(THETA)

    # Hyp C: cw = H * cos(θ) - W * sin(θ)  (fits if positive)
    hyp_C = (H*cos_t - W*sin_t) / W

    # Hyp D: cw = W*cos(θ)² - H*sin(θ)*cos(θ)  (max inscribed via trig formula)
    hyp_D = (W*cos_t**2 - H*sin_t*cos_t) / W

    # Hyp E: cw = (W*cos(θ) - H*sin(θ)) * ...
    # For the "max same-aspect tilted crop inscribed in image" where we want
    # TL corner at x=0 and BR corner at x=W: a*cos(θ) - b*sin(θ) = W/2
    # with a=r*b → b*(r*cos - sin) = W/2 → b = W/(2*(r*cos-sin))
    # a = r*W/(2*(r*cos-sin)) → cw_norm = a/W = r/(2*(r*cos-sin))
    denom = r*cos_t - sin_t
    hyp_E = r / (2*denom) if abs(denom) > 1e-9 else float('nan')

    err_A = abs(hyp_A - LR_cw_norm) if r == 1.5 else '-'
    err_B = abs(hyp_B - LR_cw_norm) if r == 1.5 else '-'
    err_C = abs(hyp_C - LR_cw_norm) if r == 1.5 else '-'
    err_D = abs(hyp_D - LR_cw_norm) if r == 1.5 else '-'
    err_E = abs(hyp_E - LR_cw_norm) if r == 1.5 else '-'

    print(f"  r={r:.1f}    {(H/W)*math.tan(THETA):>12.6f}  {hyp_A:>12.6f}  {hyp_B:>12.6f}  {hyp_C:>12.6f}  {hyp_D:>12.6f}  {hyp_E:>12.6f}")

print()
print(f"At r=1.5: target cw/W = {LR_cw_norm:.6f}")
W, H, r = 1.5, 1.0, 1.5
for name, val in [
    ("cos(2θ)", math.cos(2*THETA)),
    ("1 - H/W*tan(θ)", 1 - (H/W)*math.tan(THETA)),
    ("(H*cos-W*sin)/W", (H*cos_t - W*sin_t)/W),
    ("(W*cos²-H*sin*cos)/W", (W*cos_t**2 - H*sin_t*cos_t)/W),
    ("r/(2*(r*cos-sin))", r/(2*(r*cos_t-sin_t))),
    ("sin(θ)*H/W", sin_t*H/W),
    ("cos(θ)-H/W*sin(θ)", cos_t - (H/W)*sin_t),
    ("H/W*(cos(θ)-H/W*sin(θ))", (H/W)*(cos_t-(H/W)*sin_t)),
    ("(cos(θ)-sin(θ))²+(cos(θ)-sin(θ))*H/W", (cos_t-sin_t)**2 + (cos_t-sin_t)*H/W),
    ("H/W * 1/(1+tan(θ))", (H/W)/(1+math.tan(THETA))),
    ("H/W * cos(θ)/(sin(θ)+cos(θ))", (H/W)*cos_t/(sin_t+cos_t)),
    ("2*H/W*cos(θ)*sin(θ)", 2*(H/W)*cos_t*sin_t),
    ("(H/W)*sin(2θ)", (H/W)*math.sin(2*THETA)),
]:
    err = abs(val - LR_cw_norm)
    mark = " ✓✓✓" if err < 0.001 else (" ~" if err < 0.01 else "")
    print(f"  {name:40s} = {val:.6f}  err={err:.6f}{mark}")

print()
print("="*70)
print("Exact value analysis:")
print(f"  LR_cw_norm = {LR_cw_norm}")
print(f"  2*H/W*sin(θ)*cos(θ) = {2*(H/W)*sin_t*cos_t:.6f}")
print(f"  H/W * sin(2θ) = {(H/W)*math.sin(2*THETA):.6f}")
print(f"  Compare: 2*(1/1.5)*sin(30)*cos(30) = 2*(2/3)*(0.5)*(sqrt(3)/2) = 2/3 * sqrt(3)/2 = sqrt(3)/3 = {math.sqrt(3)/3:.6f}")
print(f"  But target = {LR_cw_norm:.6f}, sqrt(3)/3 = {math.sqrt(3)/3:.6f}")
