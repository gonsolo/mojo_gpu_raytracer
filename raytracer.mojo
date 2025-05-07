from collections import Optional
from gpu import thread_idx, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor
from math import sqrt
from time.time import monotonic

@value
struct Color:
    var r: Float32
    var g: Float32
    var b: Float32

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Color: ", self.r, ", ", self.g, ", ", self.b)

@value
struct Vec3(Copyable, Writable):
    var x: Float32
    var y: Float32
    var z: Float32

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Vec3: ", self.x, ", ", self.y, ", ", self.z)

struct Sphere:
    var center: Vec3
    var radius: Float32
    var color: Color

    fn __init__(out self, center: Vec3, radius: Float32, color: Color):
        self.center = center
        self.radius = radius
        self.color = color

    fn __copyinit__(out self, other: Self):
        self.center = other.center
        self.radius = other.radius
        self.color = other.color

    fn intersect(self, ray_origin: Vec3, ray_dir: Vec3) -> Optional[Float32]:
        oc = sub(ray_origin, self.center)
        a = dot(ray_dir, ray_dir)
        b = 2 * dot(oc, ray_dir)
        c = dot(oc, oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        t = (-b - sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t
        else:
            return None

fn dot(a: Vec3, b: Vec3) -> Float32:
    return a.x*b.x + a.y*b.y + a.z*b.z
fn sub(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x-b.x, a.y-b.y, a.z-b.z)
fn add(a: Vec3, b: Vec3) -> Vec3:
    return Vec3(a.x+b.x, a.y+b.y, a.z+b.z)
fn mul(a: Vec3, s: Float32) -> Vec3:
    return Vec3(a.x*s, a.y*s, a.z*s)
fn norm(v: Vec3) -> Vec3:
    length = sqrt(dot(v, v))
    if length > 0.01:
        return mul(v, 1/length)
    else:
        return v

alias width = 800
alias height = 600
alias dtype = DType.float32
alias blocks = width
alias threads = height
alias elements_in = blocks * threads

fn compute_direction(x: Int, y: Int) -> Vec3:
    px = Float32(x - width / 2) / width
    py = Float32(-(y - height / 2) / height)
    return norm(Vec3(px, py, 1))

fn trace(
    direction: Vec3,
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3
) -> Color:
    var min_t = Float32(999999999)
    var hit_color = Color(0, 0, 0)
    t = sphere.intersect(camera, direction)
    if t and t.value() < min_t:
        min_t = t.value()
        hit_point = add(camera, mul(direction, t.value()))
        normal = norm(sub(hit_point, sphere.center))
        light_dir = norm(sub(light_pos, hit_point))
        brightness = max(dot(normal, light_dir), 0)
        hit_color = Color(
            min(255, (brightness * sphere.color.r)),
            min(255, (brightness * sphere.color.g)),
            min(255, (brightness * sphere.color.b)))
    return hit_color

alias layout = Layout.row_major(blocks, threads)
alias xyzTensor = LayoutTensor[dtype, layout, MutableAnyOrigin]

fn trace_gpu(
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3,
    dir_x_tensor: xyzTensor,
    dir_y_tensor: xyzTensor,
    dir_z_tensor: xyzTensor,
    hit_r_tensor: xyzTensor,
    hit_g_tensor: xyzTensor,
    hit_b_tensor: xyzTensor
):
    var bix = block_idx.x
    var tix = thread_idx.x
    var direction = Vec3(
        dir_x_tensor[bix, tix][0],
        dir_y_tensor[bix, tix][0],
        dir_z_tensor[bix, tix][0])

    hit_color = trace(direction, sphere, camera, light_pos)

    hit_r_tensor[bix, tix][0] = hit_color.r
    hit_g_tensor[bix, tix][0] = hit_color.g
    hit_b_tensor[bix, tix][0] = hit_color.b


fn trace_pixel(x: Int, y: Int, sphere: Sphere, camera: Vec3, light_pos: Vec3) -> Color:
    var direction = compute_direction(x, y)
    var hit_color = trace(direction, sphere, camera, light_pos)
    return hit_color

def render_cpu(sphere: Sphere, camera: Vec3, light_pos: Vec3):
    with open("render.ppm", "w") as f:
            f.write("P3\n")
            f.write(String(width))
            f.write(" ")
            f.write(String(height))
            f.write("\n255\n")
            for y in range(height):
                for x in range(width):
                    hit_color = trace_pixel(x, y, sphere, camera, light_pos)
                    rgb = String(hit_color.r) + " " + String(hit_color.g) + " " + String(hit_color.b) + " "
                    f.write(rgb)
            f.write("\n")

def render_gpu(sphere: Sphere, camera: Vec3, light_pos: Vec3):

    var start_time = monotonic()

    var ctx = DeviceContext()

    var dir_x_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var dir_y_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var dir_z_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var hit_r_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var hit_g_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var hit_b_buffer = ctx.enqueue_create_buffer[dtype](elements_in)

    with dir_x_buffer.map_to_host() as host_x_buffer:
        with dir_y_buffer.map_to_host() as host_y_buffer:
            with dir_z_buffer.map_to_host() as host_z_buffer:
                for y in range(height):
                    for x in range(width):
                        index = y*width + x
                        direction = compute_direction(x, y)
                        host_x_buffer[index] = direction.x
                        host_y_buffer[index] = direction.y
                        host_z_buffer[index] = direction.z

    var dir_x_tensor = LayoutTensor[dtype, layout](dir_x_buffer)
    var dir_y_tensor = LayoutTensor[dtype, layout](dir_y_buffer)
    var dir_z_tensor = LayoutTensor[dtype, layout](dir_z_buffer)
    var hit_r_tensor = LayoutTensor[dtype, layout](hit_r_buffer)
    var hit_g_tensor = LayoutTensor[dtype, layout](hit_g_buffer)
    var hit_b_tensor = LayoutTensor[dtype, layout](hit_b_buffer)

    var enqueue_time = monotonic()
    print("Time before enqueue: ", (enqueue_time - start_time)/1000000, "ms")

    ctx.enqueue_function[trace_gpu](
        sphere,
        camera,
        light_pos,
        dir_x_tensor,
        dir_y_tensor,
        dir_z_tensor,
        hit_r_tensor,
        hit_g_tensor,
        hit_b_tensor,
        grid_dim=blocks,
        block_dim=threads,
    )
    var ppm_time = monotonic()
    print("Time before ppm: ", (ppm_time - enqueue_time)/1000000, "ms")

    with hit_r_buffer.map_to_host() as host_r_buffer:
        with hit_g_buffer.map_to_host() as host_g_buffer:
            with hit_b_buffer.map_to_host() as host_b_buffer:

                fn get_hitcolor(x: Int, y: Int) -> Color:
                    var index = y*width + x
                    r = Int(255 * host_r_buffer[index])
                    g = Int(255 * host_g_buffer[index])
                    b = Int(255 * host_b_buffer[index])
                    return Color(r, g, b)

                with open("render.ppm", "w") as f:
                    f.write("P3\n")
                    f.write(String(width))
                    f.write(" ")
                    f.write(String(height))
                    f.write("\n255\n")
                    for y in range(height):
                        for x in range(width):
                            var hit_color = get_hitcolor(x, y)
                            rgb = String(hit_color.r) + " " + String(hit_color.g) + " " + String(hit_color.b) + " "
                            f.write(rgb)
                    f.write("\n")
    var end_time = monotonic()
    print("Time before end: ", (end_time - ppm_time)/1000000, "ms")

def main():

    var sphere = Sphere(Vec3(0, -0.25, 3), 0.5, Color(255, 0, 0))
    var camera = Vec3(0, 0, -2)
    var light_pos = Vec3(5, 5, -10)

    var cpu = False #True
    if cpu:
        render_cpu(sphere, camera, light_pos)
    else:
        render_gpu(sphere, camera, light_pos)

