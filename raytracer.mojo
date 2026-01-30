from algorithm import parallelize, vectorize
from builtin.device_passable import DevicePassable
from gpu import block_idx, thread_idx
from gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor
from math import sqrt, iota
from memory import UnsafePointer
from reflection import get_type_name
from time.time import monotonic
from sys.info import num_logical_cores
import sys

# --- CONFIGURATION ---
comptime width = 1920
comptime height = 1080
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
struct Vec3(DevicePassable, ImplicitlyCopyable, ImplicitlyDestructible, Movable):
    var x: Float32
    var y: Float32
    var z: Float32
    comptime device_type: AnyType = Self
    fn _to_device_type[origin: MutOrigin](self, target: UnsafePointer[NoneType, origin]):
        target.bitcast[Self.device_type]()[] = self
    @staticmethod
    fn get_type_name() -> String: return "Vec3"
    @staticmethod
    fn get_device_type_name() -> String: return "Vec3"
    fn __sub__(self, other: Vec3) -> Vec3: return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    fn __add__(self, other: Vec3) -> Vec3: return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    fn __mul__(self, s: Float32) -> Vec3: return Vec3(self.x * s, self.y * s, self.z * s)
    fn dot(self, other: Vec3) -> Float32: return self.x * other.x + self.y * other.y + self.z * other.z
    fn normalize(self) -> Vec3:
        var length = sqrt(self.dot(self))
        if length > 0.001: return self * (1.0 / length)
        return self

struct Vec3SIMD[v_width: Int]:
    var x: SIMD[dtype, Self.v_width]
    var y: SIMD[dtype, Self.v_width]
    var z: SIMD[dtype, Self.v_width]
    fn __init__(out self, x: SIMD[dtype, Self.v_width], y: SIMD[dtype, Self.v_width], z: SIMD[dtype, Self.v_width]):
        self.x = x; self.y = y; self.z = z
    fn __sub__(self, other: Vec3SIMD[Self.v_width]) -> Vec3SIMD[Self.v_width]: return Vec3SIMD(self.x - other.x, self.y - other.y, self.z - other.z)
    fn __add__(self, other: Vec3SIMD[Self.v_width]) -> Vec3SIMD[Self.v_width]: return Vec3SIMD(self.x + other.x, self.y + other.y, self.z + other.z)
    fn __mul__(self, s: SIMD[dtype, Self.v_width]) -> Vec3SIMD[Self.v_width]: return Vec3SIMD(self.x * s, self.y * s, self.z * s)
    fn dot(self, other: Vec3SIMD[Self.v_width]) -> SIMD[dtype, Self.v_width]: return self.x * other.x + self.y * other.y + self.z * other.z
    fn normalize(self) -> Vec3SIMD[Self.v_width]:
        var length = sqrt(self.dot(self))
        return Vec3SIMD(self.x / length, self.y / length, self.z / length)

@fieldwise_init
struct Sphere(DevicePassable, ImplicitlyCopyable, ImplicitlyDestructible, Movable):
    var center: Vec3
    var radius: Float32
    var color: Color
    comptime device_type: AnyType = Self
    fn _to_device_type[origin: MutOrigin](self, target: UnsafePointer[NoneType, origin]):
        target.bitcast[Self.device_type]()[] = self
    @staticmethod
    fn get_type_name() -> String: return "Sphere"
    @staticmethod
    fn get_device_type_name() -> String: return "Sphere"
    fn intersect_simd[v_width: Int](self, ray_origin: Vec3SIMD[v_width], ray_dir: Vec3SIMD[v_width]) -> SIMD[dtype, v_width]:
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
fn compute_direction_simd[v_width: Int](x: SIMD[dtype, v_width], y: Int) -> Vec3SIMD[v_width]:
    var f_w = Float32(width)
    var f_h = Float32(height)
    var aspect_ratio = f_w / f_h
    # CRITICAL: Use explicit Float32 conversion to avoid Int division
    var px = (x / f_w - 0.5) * aspect_ratio
    var py = -(Float32(y) / f_h - 0.5)
    return Vec3SIMD[v_width](px, py, SIMD[dtype, v_width](1.0)).normalize()

fn trace_gpu(sphere: Sphere, camera: Vec3, light_pos: Vec3, hit_tensor: xyzTensor):
    var x = Int(block_idx.x * 16 + thread_idx.x)
    var y = Int(block_idx.y * 16 + thread_idx.y)
    if x >= width or y >= height: return

    var f_w = Float32(width)
    var f_h = Float32(height)
    var aspect_ratio = f_w / f_h
    var px = (Float32(x) / f_w - 0.5) * aspect_ratio
    var py = -(Float32(y) / f_h - 0.5)
    var direction = Vec3(px, py, 1.0).normalize()

    var oc = camera - sphere.center
    var a = direction.dot(direction)
    var b = 2.0 * oc.dot(direction)
    var c = oc.dot(oc) - (sphere.radius * sphere.radius)
    var disc = b * b - 4.0 * a * c

    var offset = (y * width + x) * 3
    if disc > 0:
        var t = (-b - sqrt(disc)) / (2.0 * a)
        var hit_point = camera + direction * t
        var normal = (hit_point - sphere.center).normalize()
        var light_dir = (light_pos - hit_point).normalize()
        var brightness = normal.dot(light_dir)
        if brightness < 0: brightness = 0.0
        
        hit_tensor.ptr.store(offset + 0, brightness * sphere.color.r)
        hit_tensor.ptr.store(offset + 1, brightness * sphere.color.g)
        hit_tensor.ptr.store(offset + 2, brightness * sphere.color.b)
    else:
        hit_tensor.ptr.store(offset + 0, 0.0)
        hit_tensor.ptr.store(offset + 1, 0.0)
        hit_tensor.ptr.store(offset + 2, 0.0)

fn write_ppm(name: String, hit_buffer: DeviceBuffer[dtype]) raises:
    with hit_buffer.map_to_host() as host_buffer:
        with open(name, "w") as f:
            f.write("P3\n", width, " ", height, "\n255\n")
            for y in range(height):
                for x in range(width):
                    var offset = (y * width + x) * 3
                    f.write(Int(255 * host_buffer[offset + 0]), " ", 
                            Int(255 * host_buffer[offset + 1]), " ", 
                            Int(255 * host_buffer[offset + 2]), " ")
                f.write("\n")









fn render_cpu(ctx: DeviceContext, sphere: Sphere, camera: Vec3, light_pos: Vec3) raises -> DeviceBuffer[dtype]:
    var hit_buffer = ctx.enqueue_create_buffer[dtype](elements_in)

    with hit_buffer.map_to_host() as host_buffer:
        var host_tensor = xyzTensor(host_buffer)
        var out_ptr = host_tensor.ptr

        @parameter
        fn row_worker(y: Int):
            # Pre-calculate float constants for this row
            var f_w = Float32(width)
            var f_h = Float32(height)
            var f_y = Float32(y)

            for x in range(0, width, simd_width):
                # 1. FORCE FLOAT SIMD: Ensure x_vec is float32 immediately
                # iota returns integers by default if dtype isn't float32
                var x_vec = iota[dtype, simd_width](x)

                # 2. BROADCAST SCALARS: Convert py to a full SIMD vector
                var px = (x_vec / f_w - 0.5) * (f_w / f_h)
                var py_scalar = -(f_y / f_h - 0.5)
                var py = SIMD[dtype, simd_width](py_scalar)
                var pz = SIMD[dtype, simd_width](1.0)

                var directions = Vec3SIMD[simd_width](px, py, pz).normalize()

                # 3. INTERSECTION & SHADING
                var cam_simd = Vec3SIMD[simd_width](camera.x, camera.y, camera.z)
                var t = sphere.intersect_simd[simd_width](cam_simd, directions)
                var h_mask = t.gt(0.0)

                var hit_points = cam_simd + directions * t
                var sc = Vec3SIMD[simd_width](sphere.center.x, sphere.center.y, sphere.center.z)
                var normals = (hit_points - sc).normalize()

                var light_simd = Vec3SIMD[simd_width](light_pos.x, light_pos.y, light_pos.z)
                var light_dirs = (light_simd - hit_points).normalize()
                var dot_p = normals.dot(light_dirs)

                # Zero out pixels that didn't hit or are in 'darkness'
                var final_b = h_mask.select(dot_p.gt(0.0).select(dot_p, 0.0), 0.0)

                # 4. THE STRIDE FIX: Calculate exact 1D offset for this chunk
                # (y * 1920 + x) * 3 channels
                var offset = (y * width + x) * 3
                var p = out_ptr + offset

                for i in range(simd_width):
                    if x + i < width:
                        p.store(i * 3 + 0, final_b[i] * sphere.color.r)
                        p.store(i * 3 + 1, final_b[i] * sphere.color.g)
                        p.store(i * 3 + 2, final_b[i] * sphere.color.b)

        parallelize[row_worker](height, num_logical_cores())

    return hit_buffer^


fn render_gpu(ctx: DeviceContext, sphere: Sphere, camera: Vec3, light_pos: Vec3) raises -> DeviceBuffer[dtype]:
    var hit_buffer = ctx.enqueue_create_buffer[dtype](elements_in)
    var hit_tensor = xyzTensor(hit_buffer)
    comptime b_size = 16
    ctx.enqueue_function[trace_gpu, trace_gpu](
        sphere, camera, light_pos, hit_tensor, 
        grid_dim=((width + b_size - 1) // b_size, (height + b_size - 1) // b_size), 
        block_dim=(b_size, b_size)
    )
    return hit_buffer^

def main():
    try:
        var ctx = DeviceContext()
        var sphere = Sphere(Vec3(0, -0.25, 3), 1.5, Color(1, 0, 0))
        var camera = Vec3(0, 0, -2)
        var light_pos = Vec3(5, 5, -10)
        
        var s_cpu = monotonic()
        var cpu_buf = render_cpu(ctx, sphere, camera, light_pos)
        print("CPU Time:", Float64(monotonic() - s_cpu) / 1e6, "ms")
        write_ppm("cpu.ppm", cpu_buf)

        var s_gpu = monotonic()
        var gpu_buf = render_gpu(ctx, sphere, camera, light_pos)
        ctx.synchronize()
        print("GPU Time:", Float64(monotonic() - s_gpu) / 1e6, "ms")
        write_ppm("gpu.ppm", gpu_buf)
    except e:
        print("Error:", e)
