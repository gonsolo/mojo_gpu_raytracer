from collections import Optional
from gpu import thread_idx, block_idx
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from math import sqrt
from time.time import monotonic

alias width = 512
alias height = 512
alias dtype = DType.float32
alias blocks = width
alias threads = height
alias channels = 3
alias elements_in = blocks * threads * channels
alias layout = Layout.row_major(blocks, threads, channels)
alias xyzTensor = LayoutTensor[dtype, layout, MutableAnyOrigin]

@fieldwise_init
struct Color(Copyable, Movable):
    var r: Float32
    var g: Float32
    var b: Float32

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Color: ", self.r, ", ", self.g, ", ", self.b)

@fieldwise_init
struct Vec3(Copyable, Movable, Writable):
    var x: Float32
    var y: Float32
    var z: Float32

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Vec3: ", self.x, ", ", self.y, ", ", self.z)

    fn __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y, self.z - other.z)

    fn __add__(self, other: Self) -> Self:
        return Self(self.x+other.x, self.y+other.y, self.z+other.z)

    fn __mul__(self: Self, s: Float32) -> Self:
        return Self(self.x*s, self.y*s, self.z*s)

    fn dot(self, other: Self) -> Float32:
        return self.x*other.x + self.y*other.y + self.z*other.z

    fn normalize(self) -> Self:
        length = sqrt(self.dot(self))
        if length > 0.01:
            return self * (1/length)
        else:
            return self

@fieldwise_init
struct Sphere(Copyable, Movable):
    var center: Vec3
    var radius: Float32
    var color: Color

    fn intersect(self, ray_origin: Vec3, ray_dir: Vec3) -> Optional[Float32]:
        oc = ray_origin - self.center
        a = ray_dir.dot(ray_dir)
        b = 2 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius ** 2
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return None
        t = (-b - sqrt(discriminant)) / (2 * a)
        if t > 0:
            return t
        else:
            return None

fn nano_to_milliseconds(nanoseconds: UInt) -> UInt:
    return nanoseconds // 1000000

fn compute_direction(x: Int, y: Int) -> Vec3:
    px = Float32(x - width / 2) / width
    py = Float32(-(y - height / 2) / height)
    return (Vec3(px, py, 1)).normalize()

fn trace(
    direction: Vec3,
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3
) -> Color:
    var hit_color = Color(0, 0, 0)
    t = sphere.intersect(camera, direction)
    if t:
        hit_point = camera + direction * t.value()
        normal = (hit_point - sphere.center).normalize()
        light_dir = (light_pos - hit_point).normalize()
        brightness = normal.dot(light_dir)
        hit_color = Color(
            brightness * sphere.color.r,
            brightness * sphere.color.g,
            brightness * sphere.color.b)
    return hit_color

fn trace_gpu(
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3,
    hit_tensor: xyzTensor,
):
    var y = block_idx.x
    var x = thread_idx.x
    var direction = compute_direction(x, y)
    var hit_color = trace(direction, sphere, camera, light_pos)

    hit_tensor[y, x, 0][0] = hit_color.r
    hit_tensor[y, x, 1][0] = hit_color.g
    hit_tensor[y, x, 2][0] = hit_color.b

fn trace_pixel(x: Int, y: Int, sphere: Sphere, camera: Vec3, light_pos: Vec3) -> Color:

    var direction = compute_direction(x, y)
    var hit_color = trace(direction, sphere, camera, light_pos)
    return hit_color

def get_hitcolor_cpu(
    x: Int,
    y: Int,
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3
) -> Color:

    return trace_pixel(x, y, sphere, camera, light_pos)

fn get_hitcolor_gpu(
    x: Int,
    y: Int,
    buffer: HostBuffer[dtype],
) -> Color:

    var index = (y*width + x) * channels
    r = buffer[index+0]
    g = buffer[index+1]
    b = buffer[index+2]
    return Color(r, g, b)

def render_cpu(sphere: Sphere, camera: Vec3, light_pos: Vec3) -> List[Color]:

    buffer = List[Color]()
    for y in range(height):
        for x in range(width):
            hit_color = get_hitcolor_cpu(x, y, sphere, camera, light_pos)
            buffer.append(hit_color)
    return buffer

def render_gpu(sphere: Sphere, camera: Vec3, light_pos: Vec3) -> List[Color]:

    var start_time = monotonic()
    var ctx = DeviceContext()
    var create_time = monotonic()
    print("Time before create: ", nano_to_milliseconds(create_time - start_time), "ms")
    var hit_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var hit_tensor = LayoutTensor[dtype, layout](hit_buffer)
    var enqueue_time = monotonic()
    print("Time before enqueue: ", nano_to_milliseconds(enqueue_time - create_time), "ms")

    ctx.enqueue_function[trace_gpu](
        sphere,
        camera,
        light_pos,
        hit_tensor,
        grid_dim=blocks,
        block_dim=threads,
    )
    var ppm_time = monotonic()
    print("Time before ppm: ", nano_to_milliseconds(ppm_time - enqueue_time), "ms")

    buffer = List[Color]()
    with hit_buffer.map_to_host() as host_buffer:
        for y in range(height):
            for x in range(width):
                var hit_color = get_hitcolor_gpu(x, y, host_buffer)
                buffer.append(hit_color)
    var end_time = monotonic()
    print("Time before end: ", nano_to_milliseconds(end_time - ppm_time), "ms")
    return buffer

def write_ppm(name: String, buffer: List[Color]):
    with open(name, "w") as f:
        f.write("P3\n")
        f.write(String(width))
        f.write(" ")
        f.write(String(height))
        f.write("\n255\n")
        for y in range(height):
            for x in range(width):
                var index = y*width + x
                var hit_color = buffer[index]
                var ri = Int(255 * hit_color.r)
                var gi = Int(255 * hit_color.g)
                var bi = Int(255 * hit_color.b)
                var rgb = String(ri) + " " + String(gi) + " " + String(bi) + " "
                f.write(rgb)
        f.write("\n")

def main():

    var sphere = Sphere(Vec3(0, -0.25, 3), 1.5, Color(1, 0, 0))
    var camera = Vec3(0, 0, -2)
    var light_pos = Vec3(5, 5, -10)

    var cpu_buffer = render_cpu(sphere, camera, light_pos)
    var gpu_buffer = render_gpu(sphere, camera, light_pos)

    var pre_ppm_time = monotonic()
    write_ppm("cpu.ppm", cpu_buffer)
    write_ppm("gpu.ppm", gpu_buffer)
    var post_ppm_time = monotonic()
    print("Time to write ppm: ", nano_to_milliseconds(post_ppm_time - pre_ppm_time), "ms")
