"""Pilot Scene #01 "Why AI Hallucinates": a 15 s, 1080x1920, 30 FPS Blender 4.2 scene.

Run on Kaggle (GPU T4 x2), from the repo root:
    python3 character_2d/generate_character.py          # once: makes character_2d/frames/
    blender -b -P build_scene.py -- --cycles-device OPTIX

Options after "--":
    --cycles-device OPTIX|CUDA|CPU   default OPTIX (falls back to CUDA, then CPU)
    --dry-run                        build the scene only: no physics bake, no render
    --sim-check                      step rigid-body physics to frame 300 and print positions

Output: ./renders/frame_0001.png .. frame_0450.png
"""
import math
import os
import random
import sys

import bpy
from mathutils import Matrix, Vector

# ---------------------------------------------------------------- options
ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default=None):
    if name not in ARGV:
        return default
    i = ARGV.index(name)
    if i + 1 < len(ARGV) and not ARGV[i + 1].startswith("--"):
        return ARGV[i + 1]
    return True


DEVICE = opt("--cycles-device", "OPTIX")
DRY_RUN = opt("--dry-run", False) is True
SIM_CHECK = opt("--sim-check", False) is True

ROOT = os.getcwd()
RENDER_DIR = os.path.join(ROOT, "renders")
CHAR_FIRST = os.path.join(ROOT, "character_2d", "frames", "char_0001.png")

W, H, FPS, FRAME_END = 1080, 1920, 30, 450
SAMPLES = 64
random.seed(7)


# ---------------------------------------------------------------- helpers
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def select_only(obj):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def add_mesh(kind, name, location=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1), **kw):
    getattr(bpy.ops.mesh, f"primitive_{kind}_add")(location=location, **kw)
    o = bpy.context.object
    o.name = name
    o.rotation_euler = rotation
    o.scale = scale
    return o


def mat(name, color=(0.5, 0.5, 0.5, 1), rough=0.5, metal=0.0, emit=None, emit_str=0.0,
        transmission=0.0, ior=1.45, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    b.inputs["IOR"].default_value = ior
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["Alpha"].default_value = alpha
    if emit is not None:
        b.inputs["Emission Color"].default_value = emit
        b.inputs["Emission Strength"].default_value = emit_str
    return m


def wet_concrete():
    """Procedural wet brushed concrete: noise colour, noise roughness and a bump map."""
    m = bpy.data.materials.new("WetConcrete")
    m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    bsdf = n["Principled BSDF"]
    tc = n.new("ShaderNodeTexCoord")
    noise = n.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 38
    noise.inputs["Detail"].default_value = 8
    ramp = n.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.035, 0.035, 0.04, 1)
    ramp.color_ramp.elements[1].color = (0.20, 0.20, 0.21, 1)
    rough = n.new("ShaderNodeValToRGB")
    rough.color_ramp.elements[0].color = (0.12, 0.12, 0.12, 1)   # wet patches
    rough.color_ramp.elements[1].color = (0.45, 0.45, 0.45, 1)   # dry, brushed
    bump = n.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    l.new(tc.outputs["Object"], noise.inputs["Vector"])
    l.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    l.new(noise.outputs["Fac"], rough.inputs["Fac"])
    l.new(noise.outputs["Fac"], bump.inputs["Height"])
    l.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    l.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    l.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def keyframe_all(obj, frame, **attrs):
    """Keyframe the given attributes. Use loc=, rot=, scale= for transform vectors."""
    for name, value in attrs.items():
        if name == "loc":
            obj.location = value
            obj.keyframe_insert("location", frame=frame)
        elif name == "rot":
            obj.rotation_euler = value
            obj.keyframe_insert("rotation_euler", frame=frame)
        elif name == "scale":
            obj.scale = value
            obj.keyframe_insert("scale", frame=frame)


def rigid(obj, kind="ACTIVE", shape="CONVEX_HULL", mass=0.2, friction=0.5, bounce=0.35):
    select_only(obj)
    bpy.ops.rigidbody.object_add(type=kind)
    r = obj.rigid_body
    r.collision_shape = shape
    r.friction = friction
    r.restitution = bounce
    if kind == "ACTIVE":
        r.mass = mass
    return r


# ---------------------------------------------------------------- GPU + render settings
def setup_gpu(preferred):
    if preferred == "CPU":
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for ctype in ([preferred] + (["CUDA"] if preferred == "OPTIX" else [])):
        try:
            prefs.compute_device_type = ctype
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type != "CPU"]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type != "CPU"
                print(f"[gpu] {ctype}: {[g.name for g in gpus]}")
                return ctype
        except Exception as e:           # try the next backend
            print(f"[gpu] {ctype} unavailable: {e}")
    print("[gpu] no GPU found, using CPU")
    return "CPU"


def setup_render(device):
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.render.resolution_x = W
    s.render.resolution_y = H
    s.render.resolution_percentage = 100
    s.render.fps = FPS
    s.render.fps_base = 1.0
    s.frame_start = 1
    s.frame_end = FRAME_END
    s.render.image_settings.file_format = "PNG"
    s.render.image_settings.color_mode = "RGB"
    s.render.filepath = os.path.join(RENDER_DIR, "frame_####")
    c = s.cycles
    c.device = "GPU" if device != "CPU" else "CPU"
    c.samples = SAMPLES
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    c.use_adaptive_sampling = True
    c.use_fast_gi = False
    s.view_settings.view_transform = "AgX"
    s.view_settings.look = "AgX - Medium High Contrast"
    return s


# ---------------------------------------------------------------- world
def build_world():
    s = bpy.context.scene
    w = bpy.data.worlds.new("NeuralFoundryWorld")
    s.world = w
    w.use_nodes = True
    n, l = w.node_tree.nodes, w.node_tree.links
    bg = n["Background"]
    bg.inputs["Color"].default_value = (0.012, 0.016, 0.022, 1)
    bg.inputs["Strength"].default_value = 0.6
    vol = n.new("ShaderNodeVolumeScatter")                 # faint fog scatter
    vol.inputs["Density"].default_value = 0.012
    vol.inputs["Color"].default_value = (0.55, 0.75, 0.95, 1)
    l.new(vol.outputs["Volume"], n["World Output"].inputs["Volume"])


def build_room():
    floor = add_mesh("plane", "Floor", scale=(30, 30, 1))
    floor.data.materials.append(wet_concrete())
    rigid(floor, kind="PASSIVE", shape="MESH", friction=0.8, bounce=0.1)

    for x in (-4.2, 4.2):                                   # steel pillars
        p = add_mesh("cylinder", f"Pillar_{x}", location=(x, -2.5, 4), radius=0.18, depth=8, vertices=24)
        p.data.materials.append(mat(f"Steel_{x}", (0.42, 0.44, 0.47, 1), rough=0.35, metal=1.0))


def build_hopper():
    """Glass-and-titanium Probability Hopper. The glass is visual only; the walls are 24 flat passive planes
    tangent to the cone, which hold the spheres reliably (a mesh collision leaked them out)."""
    body = add_mesh("cone", "HopperGlass", location=(0, 0, 2.6), vertices=48,
                    radius1=0.35, radius2=0.9, depth=1.4, end_fill_type="NOTHING")   # open outlet
    body.data.materials.append(mat("Glass", (0.9, 0.97, 1.0, 1), rough=0.03, transmission=0.95, ior=1.5))

    N = 24
    r_m, z_m = 0.35 + (2.6 - 1.9) / 1.4 * 0.55, 2.6              # mid-wall radius and height
    slant = math.hypot(0.55, 1.4)
    u_r, u_z = 0.55 / slant, 1.4 / slant                          # generator direction (radial, vertical)
    for i in range(N):
        a = 2 * math.pi * i / N
        radial = Vector((math.cos(a), math.sin(a), 0.0))
        tang = Vector((-math.sin(a), math.cos(a), 0.0))
        zdir = (radial * u_r + Vector((0, 0, u_z))).normalized()
        ydir = zdir.cross(tang).normalized()
        rot = Vector((0, 0, 0))
        wall = add_mesh("cube", f"HopperWall_{i:02d}", location=(0, 0, 0), size=1)
        wall.location = Vector((r_m * math.cos(a), r_m * math.sin(a), z_m))
        wall.rotation_mode = "QUATERNION"
        wall.rotation_quaternion = Matrix((tang, ydir, zdir)).transposed().to_quaternion()
        circ = 2 * math.pi * r_m / N
        wall.scale = (circ * 1.25, 0.02, slant * 1.0)   # exact length: no lip poking into the outlet
        wall.hide_render = True                                   # invisible collider
        wall.hide_viewport = False
        rigid(wall, kind="PASSIVE", shape="BOX", friction=0.4, bounce=0.2)
    rigid_walls_done = True
    body.hide_render = False

    rim = add_mesh("torus", "HopperRim", location=(0, 0, 3.3), major_radius=0.9, minor_radius=0.03)
    rim.data.materials.append(mat("Titanium", (0.75, 0.77, 0.8, 1), rough=0.25, metal=1.0))

    gate = add_mesh("cube", "Gate", location=(0, 0, 1.9), scale=(0.42, 0.42, 0.025))
    gate.data.materials.append(mat("GateSteel", (0.5, 0.52, 0.55, 1), rough=0.3, metal=1.0))
    # kinematic active body: Blender syncs its keyframed slide into the physics world
    # (a passive body kept its original footprint in the solver and held the facts up)
    rigid(gate, kind="ACTIVE", shape="BOX", friction=0.6, mass=1.0)
    gate.rigid_body.kinematic = True
    gate.rigid_body.keyframe_insert("kinematic", frame=1)
    keyframe_all(gate, 1, loc=(0, 0, 1.9))
    keyframe_all(gate, 119, loc=(0, 0, 1.9))
    keyframe_all(gate, 130, loc=(1.0, 0, 1.9))              # slides clear of the outlet (edge at x=0.58 > 0.41)
    return gate


def build_lever():
    """Physical lever: a pivot at (0.8, 0, 1.1) with the handle rotating down between frames 145 and 165."""
    pivot = bpy.data.objects.new("LeverPivot", None)
    bpy.context.collection.objects.link(pivot)
    pivot.location = (0.8, 0, 1.1)
    pivot.empty_display_type = "PLAIN_AXES"
    pivot.empty_display_size = 0.1

    handle = add_mesh("cylinder", "LeverHandle", location=(0.8, 0, 1.35), radius=0.03, depth=0.5, vertices=20)
    handle.data.materials.append(mat("Lever", (0.7, 0.72, 0.75, 1), rough=0.3, metal=1.0))
    handle.parent = pivot

    keyframe_all(pivot, 1, rot=(0, 0, 0))
    keyframe_all(pivot, 144, rot=(0, 0, 0))
    keyframe_all(pivot, 165, rot=(1.75, 0, 0))              # pulled down
    keyframe_all(pivot, 270, rot=(1.75, 0, 0))
    return pivot


def hopper_points(count, min_d, existing=()):
    """Non-overlapping points inside the hopper cone (z 2.0-3.1), so no bodies start interpenetrating."""
    pts = list(existing)
    out = []
    tries = 0
    while len(out) < count and tries < 20000:
        tries += 1
        z = random.uniform(2.0, 3.1)
        r_in = 0.35 + (z - 1.9) / 1.4 * 0.55 - 0.12           # inner wall radius minus a margin
        a = random.uniform(0, 2 * math.pi)
        rr = r_in * math.sqrt(random.uniform(0, 1))
        p = (rr * math.cos(a), rr * math.sin(a), z)
        if all(math.dist(p, q) >= min_d for q in pts):
            pts.append(p)
            out.append(p)
    if len(out) < count:
        raise RuntimeError(f"could only place {len(out)} of {count} bodies in the hopper")
    return out


def set_kinematic_until(obj, release):
    """Kinematic (held in place) on frame 1 until `release`, then free from `release + 1` on."""
    r = obj.rigid_body
    r.kinematic = True
    r.keyframe_insert("kinematic", frame=1)
    r.keyframe_insert("kinematic", frame=release)
    r.kinematic = False
    r.keyframe_insert("kinematic", frame=release + 1)


def build_foreground_physics(gate):
    """40 cyan 'Fact' spheres and 5 crimson 'Guess' cubes inside the hopper, plus the jam and the launch."""
    facts = []
    for i, p in enumerate(hopper_points(40, 0.16)):
        sph = add_mesh("uv_sphere", f"Fact_{i:02d}", location=p, radius=0.06, segments=16, ring_count=8)
        sph.data.materials.append(mat("FactCyan", (0.1, 0.8, 1.0, 1), rough=0.1, emit=(0.1, 0.8, 1.0, 1), emit_str=4.0))
        rigid(sph, shape="SPHERE", mass=0.05, friction=0.4, bounce=0.5)
        sph.rigid_body.use_deactivation = False      # settled spheres must wake when the gate opens
        facts.append(sph)

    cubes = []
    cube_pts = hopper_points(5, 0.3, existing=[tuple(o.location) for o in facts])
    for i, p in enumerate(cube_pts):
        cb = add_mesh("cube", f"Guess_{i}", location=p, scale=(0.05, 0.05, 0.05))
        cb.data.materials.append(mat("GuessCrimson", (0.9, 0.05, 0.05, 1), rough=0.2, emit=(1.0, 0.05, 0.05, 1), emit_str=3.0))
        rigid(cb, shape="BOX", mass=0.15, friction=0.5, bounce=0.2)
        cb.rigid_body.use_deactivation = False
        cubes.append(cb)

    # everything in the hopper is held still (kinematic) until the gate opens at frame 120, then released.
    # Keeping the contents still during the jam avoids the spheres being ejected from the start.
    for body in facts + cubes[1:]:
        set_kinematic_until(body, release=120)

    # the jammed crimson cube stays held until frame 279, then the launch frees it
    jammed = cubes[0]
    set_kinematic_until(jammed, release=279)

    # launch impulse at frame 280: a short, strong force field at the hopper outlet
    bpy.ops.object.effector_add(type="FORCE", location=(0, 0, 2.2))
    fe = bpy.context.object
    fe.name = "LaunchForce"
    fe.field.strength = 0.0
    fe.field.shape = "POINT"
    fe.field.falloff_type = "SPHERE"
    fe.field.falloff_power = 0.0
    fe.field.keyframe_insert("strength", frame=279)
    fe.field.strength = 500.0
    fe.field.keyframe_insert("strength", frame=280)
    fe.field.strength = 0.0
    fe.field.keyframe_insert("strength", frame=284)
    return facts, jammed


def build_background_physics():
    """Distant chutes (Y 12-25 m) with spheres dropping onto angled ramps, plus rotating turbines."""
    for j, y in enumerate((14, 18, 22)):
        ramp = add_mesh("cube", f"Ramp_{j}", location=(0, y, 3.0), rotation=(math.radians(-22), 0, 0),
                        scale=(5.0, 1.2, 0.05))
        ramp.data.materials.append(mat("SteelRamp", (0.35, 0.37, 0.4, 1), rough=0.35, metal=1.0))
        rigid(ramp, kind="PASSIVE", shape="BOX", friction=0.4, bounce=0.3)

    for i in range(120):
        x = random.uniform(-6, 6)
        y = random.uniform(12, 24)
        z = random.uniform(8, 16)
        s = add_mesh("uv_sphere", f"Chute_{i:03d}", location=(x, y, z), radius=0.08, segments=10, ring_count=6)
        s.data.materials.append(mat("ChuteBall", (0.6, 0.62, 0.66, 1), rough=0.3, metal=0.9))
        rigid(s, shape="SPHERE", mass=0.05, friction=0.4, bounce=0.45)

    for k, x in enumerate((-8.0, 8.0)):
        rotor = bpy.data.objects.new(f"TurbineRotor_{k}", None)
        bpy.context.collection.objects.link(rotor)
        rotor.location = (x, 20, 2.4)
        keyframe_all(rotor, 1, rot=(0, 0, 0))
        keyframe_all(rotor, FRAME_END, rot=(0, 0, 6 * math.tau))
        for b, ang in enumerate((0.0, 2.094, 4.189)):
            blade = add_mesh("cube", f"Blade_{k}_{b}", location=(math.sin(ang) * 0.6, 0, math.cos(ang) * 0.6),
                             rotation=(0, ang, 0), scale=(0.12, 0.05, 0.6))
            blade.data.materials.append(mat("Turbine", (0.55, 0.57, 0.6, 1), rough=0.4, metal=1.0))
            blade.parent = rotor


def build_lights():
    k = bpy.data.objects.new("KeyCyan", bpy.data.lights.new("KeyCyanData", "AREA"))
    bpy.context.collection.objects.link(k)
    k.location = (-2.6, -3.5, 4.2)
    k.data.energy = 700
    k.data.color = (0.45, 0.9, 1.0)
    k.data.size = 2.5

    r = bpy.data.objects.new("RimAmber", bpy.data.lights.new("RimAmberData", "AREA"))
    bpy.context.collection.objects.link(r)
    r.location = (3.2, 3.5, 3.2)
    r.data.energy = 500
    r.data.color = (1.0, 0.6, 0.25)
    r.data.size = 2.0

    fill = bpy.data.objects.new("Bounce", bpy.data.lights.new("BounceData", "AREA"))
    bpy.context.collection.objects.link(fill)
    fill.location = (0, -6, 0.6)
    fill.data.energy = 120
    fill.data.color = (0.8, 0.9, 1.0)
    fill.data.size = 3.0
    return k, r, fill


def build_hex(aim_target):
    """Hex as a sprite sequence on a vertical 2 m plane, with real shadows and occlusion."""
    bpy.ops.mesh.primitive_plane_add(size=1, location=(0.35, -0.2, 1.0))
    plane = bpy.context.object
    plane.name = "HexSprite"
    plane.scale = (2.0, 2.0, 1.0)
    plane.rotation_euler = (math.radians(90), 0, 0)

    img = bpy.data.images.load(CHAR_FIRST)
    img.source = "SEQUENCE"
    m = bpy.data.materials.new("HexMat")
    m.use_nodes = True
    n, l = m.node_tree.nodes, m.node_tree.links
    bsdf = n["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.6
    tex = n.new("ShaderNodeTexImage")
    tex.image = img
    tex.interpolation = "Closest"
    tex.image_user.frame_start = 1
    tex.image_user.frame_duration = FRAME_END
    tex.image_user.use_auto_refresh = True
    tex.image_user.use_cyclic = False
    l.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    l.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
    try:
        m.surface_render_method = "BLENDED"
    except Exception:
        pass
    plane.data.materials.append(m)
    return plane


def build_camera(target, cube):
    cam_data = bpy.data.cameras.new("Cam")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 2.8
    cam_data.dof.focus_object = target
    cam.constraints.new("TRACK_TO")
    c_target = cam.constraints[-1]
    c_target.target = target
    c_target.track_axis = "TRACK_NEGATIVE_Z"
    c_target.up_axis = "UP_Y"
    c_cube = cam.constraints.new("TRACK_TO")                # tracks the crimson cube from frame 271
    c_cube.target = cube
    c_cube.track_axis = "TRACK_NEGATIVE_Z"
    c_cube.up_axis = "UP_Y"
    c_cube.influence = 0.0
    c_cube.keyframe_insert("influence", frame=265)
    c_cube.influence = 1.0
    c_cube.keyframe_insert("influence", frame=271)

    T = Vector((0.4, 0.0, 1.2))

    def orbit(a_deg, radius):
        a = math.radians(a_deg)
        return (T.x + radius * math.sin(a), T.y - radius * math.cos(a), T.z + 0.4)

    keyframe_all(cam, 1, loc=(-1.2, -5.5, 0.9))
    keyframe_all(cam, 120, loc=(0.0, -3.4, 1.0))
    keyframe_all(cam, 121, loc=orbit(-8, 4.0))
    keyframe_all(cam, 195, loc=orbit(-30, 4.0))
    keyframe_all(cam, 270, loc=orbit(-45, 4.0))
    keyframe_all(cam, 360, loc=orbit(-38, 3.2))
    keyframe_all(cam, 450, loc=orbit(-22, 2.6))

    cam_data.lens = 28
    cam_data.keyframe_insert("lens", frame=1)
    cam_data.lens = 40
    cam_data.keyframe_insert("lens", frame=120)
    cam_data.lens = 50
    cam_data.keyframe_insert("lens", frame=270)
    cam_data.lens = 85
    cam_data.keyframe_insert("lens", frame=450)
    return cam


# ---------------------------------------------------------------- physics bake + checks
def bake_physics():
    s = bpy.context.scene
    s.frame_set(1)
    try:
        bpy.ops.ptcache.bake_all(bake=True)
        print("[bake] bake_all done")
    except Exception as e:
        print(f"[bake] bake_all failed ({e}); stepping frames to fill the cache")
        for f in range(1, FRAME_END + 1):
            s.frame_set(f)


def sim_check(facts, jammed):
    s = bpy.context.scene
    # rigid-body physics only advances one frame at a time, so step every frame in order
    report = {1, 60, 120, 130, 150, 200, 280, 300}
    for f in range(1, 301):
        s.frame_set(f)
        if f in report:
            c = jammed.matrix_world.translation
            low = sum(1 for o in facts if o.matrix_world.translation.z < 0.3)
            zs = [o.matrix_world.translation.z for o in facts]
            print(f"[sim] frame {f:3d}  jammed cube z={c.z:.2f}  facts below 0.3 m: {low}/40  facts z min/mean={min(zs):.2f}/{sum(zs)/len(zs):.2f}  gate x={bpy.data.objects['Gate'].matrix_world.translation.x:.2f}")
        for o in []:
            if o.matrix_world.translation.z < 1.8 and not getattr(o, "_flag", False):
                o["_flag"] = True
                t = o.matrix_world.translation
                print(f"[sim] first drop below 1.8 m: {o.name} at frame {f} pos=({t.x:.2f},{t.y:.2f},{t.z:.2f})")


# ---------------------------------------------------------------- main
def main():
    clear_scene()
    os.makedirs(RENDER_DIR, exist_ok=True)
    if not os.path.exists(CHAR_FIRST):
        raise SystemExit("Missing character frames. Run: python3 character_2d/generate_character.py")

    device = setup_gpu(DEVICE) if not DRY_RUN else "CPU"
    setup_render(device)
    s = bpy.context.scene
    if s.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()
    rw = s.rigidbody_world
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = FRAME_END

    build_world()
    build_room()
    gate = build_hopper()
    build_lever()
    facts, jammed = build_foreground_physics(gate)
    build_background_physics()
    build_lights()
    target = bpy.data.objects.new("CamTarget", None)
    bpy.context.collection.objects.link(target)
    target.location = (0.4, 0.0, 1.2)
    build_hex(target)
    build_camera(target, jammed)

    print(f"[scene] objects={len(bpy.data.objects)} rigid_bodies={len(rw.collection.objects)} device={device}")

    if SIM_CHECK:
        sim_check(facts, jammed)
        return
    if DRY_RUN:
        print("[dry-run] scene built, no bake or render")
        return

    bake_physics()
    print("[render] starting", FRAME_END, "frames at", W, "x", H)
    bpy.ops.render.render(animation=True)
    print("[render] done ->", RENDER_DIR)


if __name__ == "__main__":
    main()
