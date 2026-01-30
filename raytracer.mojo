from algorithm import parallelize, vectorize
from builtin.device_passable import DevicePassable
from gpu import block_idx, thread_idx
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from math import sqrt, iota  # Imported iota here
from memory import UnsafePointer
from reflection import get_type_name
from time.time import monotonic
from sys.info import num_logical_cores
import sys

comptime width = 512
comptime height = 512
comptime dtype = DType.float32
comptime channels = 3
comptime simd_width = sys.info.simd_width_of[dtype]()

comptime layout = Layout.row_major(width, height, channels)
comptime xyzTensor = LayoutTensor[dtype, layout, MutAnyOrigin]
comptime elements_in = width * height * channels


@fieldwise_init
struct Color(ImplicitlyCopyable, ImplicitlyDestructible, Movable):
    var r: Float32
    var g: Float32
    var b: Float32


@fieldwise_init
struct Vec3(
    DevicePassable, ImplicitlyCopyable, ImplicitlyDestructible, Movable
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
        return "Vec3"

    @staticmethod
    fn get_device_type_name() -> String:
        return "Vec3"

    fn __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    fn __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    fn __mul__(self, s: Float32) -> Vec3:
        return Vec3(self.x * s, self.y * s, self.z * s)

    fn dot(self, other: Vec3) -> Float32:
        return self.x * other.x + self.y * other.y + self.z * other.z

    fn normalize(self) -> Vec3:
        var length = sqrt(self.dot(self))
        if length > 0.01:
            return self * (1.0 / length)
        return self


struct Vec3SIMD[v_width: Int]:
    var x: SIMD[dtype, Self.v_width]
    var y: SIMD[dtype, Self.v_width]
    var z: SIMD[dtype, Self.v_width]

    fn __init__(
        out self,
        x: SIMD[dtype, Self.v_width],
        y: SIMD[dtype, Self.v_width],
        z: SIMD[dtype, Self.v_width],
    ):
        self.x = x
        self.y = y
        self.z = z

    fn __sub__(self, other: Vec3SIMD[Self.v_width]) -> Vec3SIMD[Self.v_width]:
        return Vec3SIMD(self.x - other.x, self.y - other.y, self.z - other.z)

    fn __add__(self, other: Vec3SIMD[Self.v_width]) -> Vec3SIMD[Self.v_width]:
        return Vec3SIMD(self.x + other.x, self.y + other.y, self.z + other.z)

    fn __mul__(self, s: SIMD[dtype, Self.v_width]) -> Vec3SIMD[Self.v_width]:
        return Vec3SIMD(self.x * s, self.y * s, self.z * s)

    fn dot(self, other: Vec3SIMD[Self.v_width]) -> SIMD[dtype, Self.v_width]:
        return self.x * other.x + self.y * other.y + self.z * other.z

    fn normalize(self) -> Vec3SIMD[Self.v_width]:
        var length = sqrt(self.dot(self))
        return Vec3SIMD(self.x / length, self.y / length, self.z / length)


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
        return "Sphere"

    @staticmethod
    fn get_device_type_name() -> String:
        return "Sphere"

    fn intersect_simd[
        v_width: Int
    ](self, ray_origin: Vec3SIMD[v_width], ray_dir: Vec3SIMD[v_width]) -> SIMD[
        dtype, v_width
    ]:
        var sc = Vec3SIMD[v_width](self.center.x, self.center.y, self.center.z)
        var oc = ray_origin - sc
        var a = ray_dir.dot(ray_dir)
        var b = 2.0 * oc.dot(ray_dir)
        var c = oc.dot(oc) - (self.radius * self.radius)
        var discriminant = b * b - 4.0 * a * c
        var mask = discriminant.gt(0.0)
        var disc_clamped = mask.select(discriminant, SIMD[dtype, v_width](0.0))
        var t = (-b - sqrt(disc_clamped)) / (2.0 * a)

        return mask.select(t, SIMD[dtype, v_width](-1.0))


@always_inline
fn compute_direction_simd[
    v_width: Int
](x: SIMD[dtype, v_width], y: Int) -> Vec3SIMD[v_width]:
    var px = (x - Float32(width // 2)) / Float32(width)
    var py = Float32(-(y - height // 2)) / Float32(height)
    return Vec3SIMD[v_width](px, py, SIMD[dtype, v_width](1.0)).normalize()


fn trace_gpu(
    sphere: Sphere, camera: Vec3, light_pos: Vec3, hit_tensor: xyzTensor
):
    var y = Int(block_idx.x)
    var x = Int(thread_idx.x)
    var px = Float32(x - width // 2) / width
    var py = Float32(-(y - height // 2) / height)
    var direction = Vec3(px, py, 1.0).normalize()

    var oc = camera - sphere.center
    var a = direction.dot(direction)
    var b = 2.0 * oc.dot(direction)
    var c = oc.dot(oc) - (sphere.radius * sphere.radius)
    var disc = b * b - 4.0 * a * c

    if disc > 0:
        var t = (-b - sqrt(disc)) / (2.0 * a)
        var hit_point = camera + direction * t
        var normal = (hit_point - sphere.center).normalize()
        var light_dir = (light_pos - hit_point).normalize()
        var dot_p = normal.dot(light_dir)
        var brightness = dot_p if dot_p > 0 else 0.0
        hit_tensor[y, x, 0] = brightness * sphere.color.r
        hit_tensor[y, x, 1] = brightness * sphere.color.g
        hit_tensor[y, x, 2] = brightness * sphere.color.b
    else:
        hit_tensor[y, x, 0] = 0.0
        hit_tensor[y, x, 1] = 0.0
        hit_tensor[y, x, 2] = 0.0


fn write_ppm(name: String, hit_buffer: DeviceBuffer[dtype]) raises:
    with hit_buffer.map_to_host() as host_buffer:
        var hit_tensor = LayoutTensor[dtype, layout](host_buffer)
        with open(name, "w") as f:
            f.write("P3\n", width, " ", height, "\n255\n")
            for y in range(height):
                for x in range(width):
                    f.write(
                        Int(255 * hit_tensor[y, x, 0]),
                        " ",
                        Int(255 * hit_tensor[y, x, 1]),
                        " ",
                        Int(255 * hit_tensor[y, x, 2]),
                        " ",
                    )
                f.write("\n")


fn render_cpu(
    ctx: DeviceContext, sphere: Sphere, camera: Vec3, light_pos: Vec3
) raises -> DeviceBuffer[dtype]:
    var hit_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    with hit_buffer.map_to_host() as host_buffer:
        var output_tensor = xyzTensor(host_buffer)

        @parameter
        fn row_worker(y: Int):
            @always_inline
            fn v_call[
                v_width: Int
            ](x: Int) unified {
                var sphere, var camera, var light_pos, mut output_tensor, var y
            }:
                var x_vec = iota[dtype, v_width](x)
                var directions = compute_direction_simd[v_width](x_vec, y)
                var cam_simd = Vec3SIMD[v_width](camera.x, camera.y, camera.z)
                var light_simd = Vec3SIMD[v_width](
                    light_pos.x, light_pos.y, light_pos.z
                )

                var t = sphere.intersect_simd[v_width](cam_simd, directions)
                var hit_points = cam_simd + directions * t
                var sc = Vec3SIMD[v_width](
                    sphere.center.x, sphere.center.y, sphere.center.z
                )
                var normals = (hit_points - sc).normalize()
                var light_dirs = (light_simd - hit_points).normalize()

                var dot_p = normals.dot(light_dirs)

                var b_mask = dot_p.gt(0.0)
                var brightness = b_mask.select(dot_p, SIMD[dtype, v_width](0.0))

                var h_mask = t.gt(0.0)
                var final_b = h_mask.select(
                    brightness, SIMD[dtype, v_width](0.0)
                )

                var r_vec = final_b * sphere.color.r
                var g_vec = final_b * sphere.color.g
                var b_vec = final_b * sphere.color.b

                # FIX: Cast LegacyUnsafePointer to modern UnsafePointer
                var base_ptr = (
                    UnsafePointer(output_tensor.ptr) + (y * width + x) * 3
                )

                for i in range(v_width):
                    base_ptr.store(i * 3 + 0, r_vec[i])
                    base_ptr.store(i * 3 + 1, g_vec[i])
                    base_ptr.store(i * 3 + 2, b_vec[i])

            vectorize[simd_width](width, v_call)

        parallelize[row_worker](height, num_logical_cores())
    return hit_buffer^


fn render_gpu(
    ctx: DeviceContext, sphere: Sphere, camera: Vec3, light_pos: Vec3
) raises -> DeviceBuffer[dtype]:
    var hit_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var hit_tensor = LayoutTensor[dtype, layout](hit_buffer)
    ctx.enqueue_function[trace_gpu, trace_gpu](
        sphere, camera, light_pos, hit_tensor, grid_dim=width, block_dim=height
    )
    return hit_buffer^


def main():
    try:
        var ctx = DeviceContext()
        var sphere = Sphere(Vec3(0, -0.25, 3), 1.5, Color(1, 0, 0))
        var camera = Vec3(0, 0, -2)
        var light_pos = Vec3(5, 5, -10)

        print("Cores:", num_logical_cores(), "| SIMD Width:", simd_width)

        var start_cpu = monotonic()
        var cpu_buffer = render_cpu(ctx, sphere, camera, light_pos)
        var cpu_ms = Float64(monotonic() - start_cpu) / 1e6
        var cpu_str = String(cpu_ms)
        var cpu_dot = cpu_str.find(".")
        print("CPU Render Time:", cpu_str[:cpu_dot + 4] if cpu_dot != -1 else cpu_str, "ms")
        write_ppm("cpu.ppm", cpu_buffer)

        var start_gpu = monotonic()
        var gpu_buffer = render_gpu(ctx, sphere, camera, light_pos)
        ctx.synchronize()
        var gpu_ms = Float64(monotonic() - start_gpu) / 1e6
        var gpu_str = String(gpu_ms)
        var gpu_dot = gpu_str.find(".")
        print("GPU Render Time:", gpu_str[:gpu_dot + 4] if gpu_dot != -1 else gpu_str, "ms")
        write_ppm("gpu.ppm", gpu_buffer)

    except e:
        print("Error:", e)
