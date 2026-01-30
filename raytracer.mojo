from algorithm import parallelize
from builtin.device_passable import DevicePassable
from collections import Optional
from gpu import block_idx, thread_idx
from gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from layout import Layout, LayoutTensor
from math import sqrt
from reflection import get_type_name
from time.time import monotonic
from sys.info import num_logical_cores

comptime width = 512
comptime height = 512
comptime dtype = DType.float32
comptime blocks = width
comptime threads = height
comptime channels = 3
comptime elements_in = blocks * threads * channels
comptime layout = Layout.row_major(blocks, threads, channels)
comptime xyzTensor = LayoutTensor[dtype, layout, MutAnyOrigin]
comptime readOnlyTensor = LayoutTensor[dtype, layout]


@fieldwise_init
struct Color(ImplicitlyCopyable, ImplicitlyDestructible, Movable, Writable):
    var r: Float32
    var g: Float32
    var b: Float32


@fieldwise_init
struct Vec3(
    DevicePassable,
    ImplicitlyCopyable,
    ImplicitlyDestructible,
    Movable,
    Writable,
):
    var x: Float32
    var y: Float32
    var z: Float32

    comptime device_type: AnyType = Self

    fn _to_device_type[
        origin: MutOrigin
    ](self, target: UnsafePointer[NoneType, origin]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return get_type_name[Self]()

    @staticmethod
    fn get_device_type_name() -> String:
        return get_type_name[Self]()

    fn __sub__(self, other: Self) -> Self:
        return Self(self.x - other.x, self.y - other.y, self.z - other.z)

    fn __add__(self, other: Self) -> Self:
        return Self(self.x + other.x, self.y + other.y, self.z + other.z)

    fn __mul__(self: Self, s: Float32) -> Self:
        return Self(self.x * s, self.y * s, self.z * s)

    fn dot(self, other: Self) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    fn normalize(self) -> Self:
        length = sqrt(self.dot(self))
        if length > 0.01:
            return self * (1 / length)
        else:
            return self.copy()


@fieldwise_init
struct Sphere(
    DevicePassable, ImplicitlyCopyable, ImplicitlyDestructible, Movable
):
    var center: Vec3
    var radius: Float32
    var color: Color

    comptime device_type: AnyType = Self

    fn _to_device_type[
        origin: MutOrigin
    ](self, target: UnsafePointer[NoneType, origin]):
        target.bitcast[Self.device_type]()[] = self

    @staticmethod
    fn get_type_name() -> String:
        return get_type_name[Self]()

    @staticmethod
    fn get_device_type_name() -> String:
        return get_type_name[Self]()

    fn intersect(self, ray_origin: Vec3, ray_dir: Vec3) -> Optional[Float32]:
        oc = ray_origin - self.center
        a = ray_dir.dot(ray_dir)
        b = 2 * oc.dot(ray_dir)
        c = oc.dot(oc) - self.radius**2
        discriminant = b**2 - 4 * a * c
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
    direction: Vec3, sphere: Sphere, camera: Vec3, light_pos: Vec3
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
            brightness * sphere.color.b,
        )
    return hit_color^


fn trace_gpu(
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3,
    hit_tensor: xyzTensor,
):
    var y = Int(block_idx.x)
    var x = Int(thread_idx.x)
    var direction = compute_direction(x, y)
    var hit_color = trace(direction, sphere, camera, light_pos)

    hit_tensor[y, x, 0] = hit_color.r
    hit_tensor[y, x, 1] = hit_color.g
    hit_tensor[y, x, 2] = hit_color.b


fn trace_pixel(
    x: Int, y: Int, sphere: Sphere, camera: Vec3, light_pos: Vec3
) -> Color:
    var direction = compute_direction(x, y)
    var hit_color = trace(direction, sphere, camera, light_pos)
    return hit_color^


fn get_hitcolor_cpu(
    x: Int, y: Int, sphere: Sphere, camera: Vec3, light_pos: Vec3
) -> Color:
    return trace_pixel(x, y, sphere, camera, light_pos)


fn get_hitcolor_gpu(
    x: Int,
    y: Int,
    buffer: HostBuffer[dtype],
) -> Color:
    var index = (y * width + x) * channels
    r = buffer[index + 0]
    g = buffer[index + 1]
    b = buffer[index + 2]
    return Color(r, g, b)


def write_ppm_tensor(name: String, buffer_tensor: readOnlyTensor):
    """
    Writes the content of a 3D LayoutTensor (width x height x channels)
    to a PPM image file (P3 format).
    """
    with open(name, "w") as f:
        f.write("P3\n")
        f.write(String(width))
        f.write(" ")
        f.write(String(height))
        f.write("\n255\n")

        for y in range(height):
            for x in range(width):
                var r = buffer_tensor[y, x, 0]
                var g = buffer_tensor[y, x, 1]
                var b = buffer_tensor[y, x, 2]
                var ri = Int(255 * r)
                var gi = Int(255 * g)
                var bi = Int(255 * b)
                var rgb = String(ri) + " " + String(gi) + " " + String(bi) + " "
                f.write(rgb)
        f.write("\n")


def write_ppm_tensor_gpu(name: String, hit_buffer: DeviceBuffer[dtype]):
    """
    Maps the GPU buffer to the host and writes it to a PPM file using a LayoutTensor wrapper.
    """
    with hit_buffer.map_to_host() as host_buffer:
        var hit_tensor = LayoutTensor[dtype, layout](host_buffer)
        write_ppm_tensor(name, hit_tensor)


fn render_cpu(
    ctx: DeviceContext,
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3,
) raises -> DeviceBuffer[dtype]:
    var hit_buffer = ctx.enqueue_create_buffer[dtype](elements_in)

    with hit_buffer.map_to_host() as host_buffer:
        var output_tensor = xyzTensor(host_buffer)

        @parameter
        fn worker(idx: Int):
            var y = idx // width
            var x = idx % width
            var hit_color = get_hitcolor_cpu(x, y, sphere, camera, light_pos)
            output_tensor[y, x, 0] = hit_color.r
            output_tensor[y, x, 1] = hit_color.g
            output_tensor[y, x, 2] = hit_color.b

        parallelize[worker](width * height, num_logical_cores())

    return hit_buffer^


fn render_gpu(
    ctx: DeviceContext,
    sphere: Sphere,
    camera: Vec3,
    light_pos: Vec3,
) raises -> DeviceBuffer[dtype]:
    var hit_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var hit_tensor = LayoutTensor[dtype, layout](hit_buffer)

    ctx.enqueue_function[trace_gpu, trace_gpu](
        sphere,
        camera,
        light_pos,
        hit_tensor,
        grid_dim=blocks,
        block_dim=threads,
    )

    return hit_buffer^


def main():
    print(num_logical_cores(), "cores")

    try:
        var ctx = DeviceContext()

        var sphere = Sphere(Vec3(0, -0.25, 3), 1.5, Color(1, 0, 0))
        var camera = Vec3(0, 0, -2)
        var light_pos = Vec3(5, 5, -10)

        var cpu_buffer = render_cpu(ctx, sphere, camera, light_pos)
        write_ppm_tensor_gpu("cpu.ppm", cpu_buffer)

        var gpu_buffer = render_gpu(ctx, sphere, camera, light_pos)
        write_ppm_tensor_gpu("gpu.ppm", gpu_buffer)

    except e:
        print("An error occurred during rendering:", e)
