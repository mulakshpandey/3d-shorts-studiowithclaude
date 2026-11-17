#!/usr/bin/env python3
"""Generate "Hex", the 2D character for Pilot Scene #01 ("Why AI Hallucinates").

Writes 450 transparent 1024x1024 PNG frames to character_2d/frames/char_0001.png .. char_0450.png.
Deterministic: the same script always produces the same frames, so Kaggle can regenerate them
instead of storing them in git.

Usage (from the repo root):
    python3 character_2d/generate_character.py            # all 450 frames
    python3 character_2d/generate_character.py --only 150  # one frame, for a quick look

Scale and placement (shared with build_scene.py):
    canvas 1024 px = 2.0 m in the 3D scene; the sprite plane is centred at (0.35, -0.2, 1.0)
    feet land at canvas y ~ 990 (world z ~ 0); Hex is about 1.5 m tall.
    LEVER_HAND, LEVER_PULL, FACE_GUARD and POINT_AT are canvas pixels. They map to world
    space as x = 0.35 + (px - 512) / 512, z = 1.0 - (py - 512) / 512.
    The lever sits at world (0.8, 0, 1.1), which is canvas (742, 461) = LEVER_HAND.
"""
import math
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

FRAMES = 450
SIZE = 1024          # output size
SS = 2               # supersampling factor for smooth edges
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frames")

HIP_X, HIP_Y = 512, 600

# Hand targets in canvas pixels
LEVER_HAND = (742, 461)   # lever grip, matches lever pivot at world (0.8, 0, 1.1)
LEVER_PULL = (742, 560)   # lever pulled down
FACE_GUARD = (615, 385)   # forearm shielding the face from sparks
POINT_AT = (880, 880)     # pointing at the red cube on the floor (frame 360)

JACKET = (18, 18, 22, 255)
JACKET_B = (34, 34, 42, 255)   # back arm, a shade darker
PANTS = (44, 44, 52, 255)
SHOE = (12, 12, 14, 255)
SKIN = (238, 200, 170, 255)
HAIR = (22, 24, 44, 255)
CYAN = (0, 229, 255, 255)
LENS = (0, 229, 255, 110)
SPARK = (255, 220, 120, 255)


def smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3 - 2 * u)


def key(frame, keys):
    """Smoothstep interpolation through [(frame, value), ...]."""
    if frame <= keys[0][0]:
        return keys[0][1]
    for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
        if f0 <= frame <= f1:
            u = (frame - f0) / (f1 - f0) if f1 > f0 else 1.0
            return v0 + (v1 - v0) * smooth(u)
    return keys[-1][1]


def key2(frame, keys):
    """Interpolate an (x, y) pair through [(frame, (x, y)), ...]."""
    return (key(frame, [(f, v[0]) for f, v in keys]),
            key(frame, [(f, v[1]) for f, v in keys]))


def ik_arm(shoulder, target, l1=150.0, l2=140.0):
    """Two-bone IK. Returns the elbow point. Angles: 0 = straight down, + = forward (+x)."""
    dx, dy = target[0] - shoulder[0], target[1] - shoulder[1]
    d = math.hypot(dx, dy)
    d = max(abs(l1 - l2) + 1, min(l1 + l2 - 1, d))
    phi = math.atan2(dx, dy)
    cos_a = (l1 * l1 + d * d - l2 * l2) / (2 * l1 * d)
    alpha = math.acos(max(-1.0, min(1.0, cos_a)))
    a = phi - alpha                     # elbow hangs below the reach line
    elbow = (shoulder[0] + l1 * math.sin(a), shoulder[1] + l1 * math.cos(a))
    return elbow


def leg_points(hip, angle, bend):
    """Thigh and shin points. Angle 0 = straight down, + = forward."""
    a = math.radians(angle)
    knee = (hip[0] + 200 * math.sin(a), hip[1] + 200 * math.cos(a))
    b = math.radians(angle - bend)
    foot = (knee[0] + 190 * math.sin(b), knee[1] + 190 * math.cos(b))
    return knee, foot


def pose(f):
    """All joint positions for frame f (canvas pixels)."""
    walk_w = 1.0 - smooth((f - 40) / 20.0)       # 1 while walking in, 0 once settled
    ph = f * 0.45

    rx = key(f, [(1, -430), (60, 0), (271, 0), (300, -45), (330, -45), (345, 0), (450, 0)])
    lean = key(f, [(1, 0), (60, 4), (90, 20), (120, 18), (135, 25), (150, 20),
                   (270, 22), (285, -8), (330, -8), (360, 12), (450, 12)])
    hip = (HIP_X + rx, HIP_Y)

    t = math.radians(180 - lean)                 # torso direction (up = 180)
    up = (math.sin(t), math.cos(t))
    side = (math.cos(t), -math.sin(t))           # toward the front (+x)
    neck = (hip[0] + 230 * up[0], hip[1] + 230 * up[1])
    head = (neck[0] + 80 * up[0], neck[1] + 80 * up[1])
    sh_f = (neck[0] - 10 * up[0] + 40 * side[0], neck[1] - 10 * up[1] + 40 * side[1])
    sh_b = (neck[0] - 10 * up[0] - 40 * side[0], neck[1] - 10 * up[1] - 40 * side[1])

    # hand targets, with a walking swing that fades out by frame 60
    swing = 25.0 * (1.0 - smooth((f - 40) / 20.0)) * math.sin(ph)
    fh = key2(f, [(1, (560, 700)), (60, (560, 700)), (90, (560, 520)), (120, (560, 520)),
                  (135, LEVER_HAND), (150, LEVER_PULL), (270, LEVER_PULL),
                  (285, FACE_GUARD), (330, FACE_GUARD), (355, POINT_AT), (450, POINT_AT)])
    bh = key2(f, [(1, (450, 700)), (60, (450, 700)), (120, (455, 560)),
                  (135, (LEVER_HAND[0] - 22, LEVER_HAND[1] + 10)),
                  (150, (LEVER_PULL[0] - 22, LEVER_PULL[1] + 10)),
                  (270, (LEVER_PULL[0] - 22, LEVER_PULL[1] + 10)),
                  (285, (470, 700)), (450, (470, 700))])
    fh = (fh[0] + swing, fh[1])
    bh = (bh[0] - swing, bh[1])

    leg_f = 22 * math.sin(ph) * walk_w + 4 * (1 - walk_w)
    leg_b = -22 * math.sin(ph) * walk_w - 4 * (1 - walk_w)
    bend_f = 6 + 10 * walk_w * max(0.0, math.cos(ph))
    bend_b = 6 + 10 * walk_w * max(0.0, -math.cos(ph))

    head_shake = 2.5 * math.sin(f * 1.9) if 150 <= f < 270 else 0.0
    strobe = 0.35 if (150 <= f < 270 and (f // 5) % 2 == 0) else 0.0
    return dict(hip=hip, neck=neck, head=(head[0] + head_shake, head[1]), side=side, up=up,
                sh_f=sh_f, sh_b=sh_b, fh=fh, bh=bh,
                leg_f=leg_f, leg_b=leg_b, bend_f=bend_f, bend_b=bend_b, strobe=strobe)


def render(f):
    W = SIZE * SS
    img = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    cy = Image.new("RGBA", (W, W), (0, 0, 0, 0))     # cyan trim and glow, drawn sharp on top
    d = ImageDraw.Draw(img)
    dc = ImageDraw.Draw(cy)
    p = pose(f)

    def P(pt):
        return (pt[0] * SS, pt[1] * SS)

    def limb(dr, pts, width, color):
        q = [P(x) for x in pts]
        w = width * SS
        dr.line(q, fill=color, width=int(w), joint="curve")
        r = w / 2
        for x, y in (q[0], q[-1]):
            dr.ellipse([x - r, y - r, x + r, y + r], fill=color)

    def dot(dr, pt, r, color):
        x, y = P(pt)
        dr.ellipse([x - r * SS, y - r * SS, x + r * SS, y + r * SS], fill=color)

    hip, neck, side, up = p["hip"], p["neck"], p["side"], p["up"]

    # back leg
    knee, foot = leg_points(hip, p["leg_b"], p["bend_b"])
    limb(d, [hip, knee, foot], 60, PANTS)
    dot(d, (foot[0] + 18, foot[1] + 4), 22, SHOE)

    # back arm
    elbow = ik_arm(p["sh_b"], p["bh"])
    limb(d, [p["sh_b"], elbow, p["bh"]], 40, JACKET_B)
    dot(d, p["bh"], 20, SKIN)

    # torso and belt
    limb(d, [hip, neck], 128, JACKET)
    limb(d, [hip, (hip[0] + 40 * up[0], hip[1] + 40 * up[1])], 134, (26, 26, 32, 255))
    # cyan trim down the front of the jacket
    dc.line([P((neck[0] + 56 * side[0], neck[1] + 56 * side[1])),
             P((hip[0] + 56 * side[0] + 10 * up[0], hip[1] + 56 * side[1] + 10 * up[1]))],
            fill=CYAN, width=6 * SS)

    # head, hair and AR glasses
    hx, hy = p["head"]
    r = 75
    d.ellipse([(hx - r) * SS, (hy - r) * SS, (hx + r) * SS, (hy + r) * SS], fill=SKIN)
    d.chord([(hx - r - 4) * SS, (hy - r - 6) * SS, (hx + r + 4) * SS, (hy + r * 0.2) * SS],
            180, 360, fill=HAIR)
    for k in (-1, 0, 1):                                   # spiky back tufts
        bx = hx - 40 * side[0] + k * 22
        d.polygon([P((bx, hy - 40)), P((bx - 26, hy - 80)), P((bx + 18, hy - 52))], fill=HAIR)
    gx, gy = hx + 36 * side[0], hy + 4
    dc.rounded_rectangle([(gx - 42) * SS, (gy - 20) * SS, (gx + 42) * SS, (gy + 20) * SS],
                         radius=12 * SS, outline=CYAN, width=7 * SS)
    d.rounded_rectangle([(gx - 36) * SS, (gy - 14) * SS, (gx + 36) * SS, (gy + 14) * SS],
                        radius=10 * SS, fill=LENS)

    # front leg
    knee, foot = leg_points(hip, p["leg_f"], p["bend_f"])
    limb(d, [hip, knee, foot], 60, PANTS)
    dot(d, (foot[0] + 18, foot[1] + 4), 22, SHOE)

    # front arm, reaching for the lever, guarding, or pointing
    elbow = ik_arm(p["sh_f"], p["fh"])
    limb(d, [p["sh_f"], elbow, p["fh"]], 46, JACKET)
    cuff = (elbow[0] + 0.7 * (p["fh"][0] - elbow[0]), elbow[1] + 0.7 * (p["fh"][1] - elbow[1]))
    dc.line([P(cuff), P((cuff[0] + 12, cuff[1] + 4))], fill=CYAN, width=6 * SS)  # wrist console
    dot(d, p["fh"], 22, SKIN)

    # composite: blurred glow under sharp cyan
    glow = cy.filter(ImageFilter.GaussianBlur(10 * SS))
    out = Image.alpha_composite(img, glow)
    out = Image.alpha_composite(out, cy)

    # sparks while shielding the face (frames 271-330)
    if 271 <= f <= 330:
        sp = Image.new("RGBA", (W, W), (0, 0, 0, 0))
        ds = ImageDraw.Draw(sp)
        rng = random.Random(f * 7919)
        for _ in range(14):
            sx = FACE_GUARD[0] + rng.uniform(-90, 110)
            sy = FACE_GUARD[1] + rng.uniform(-70, 90)
            dot(ds, (sx, sy), rng.uniform(2, 6), SPARK)
        out = Image.alpha_composite(out, sp)

    out = out.resize((SIZE, SIZE), Image.LANCZOS)

    # red strobe wash during the lever sequence, following the character's alpha
    if p["strobe"]:
        tint = Image.new("RGBA", out.size, (255, 40, 40, 0))
        tint.putalpha(out.split()[3].point(lambda v: int(v * p["strobe"])))
        out = Image.alpha_composite(out, tint)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--only" in sys.argv:
        f = int(sys.argv[sys.argv.index("--only") + 1])
        render(f).save(os.path.join(OUT, f"char_{f:04d}.png"), optimize=True)
        print("wrote frame", f)
        return
    for f in range(1, FRAMES + 1):
        render(f).save(os.path.join(OUT, f"char_{f:04d}.png"), optimize=True)
        if f % 50 == 0:
            print("frame", f, "of", FRAMES)
    print("done:", FRAMES, "frames in", OUT)


if __name__ == "__main__":
    main()
