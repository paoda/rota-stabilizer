//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const tracy = @import("tracy");
const gl = @import("gl");

const AvFrame = @import("lib/libav.zig").AvFrame;

pub var errors: Errors = undefined;

pub const c = @cImport({
    @cDefine("SDL_DISABLE_OLD_NAMES", {});
    @cDefine("SDL_MAIN_HANDLED", {});

    @cInclude("SDL3/SDL.h");
    @cInclude("SDL3/SDL_main.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libavfilter/avfilter.h");
    @cInclude("libavfilter/buffersink.h");
    @cInclude("libavfilter/buffersrc.h");
    @cInclude("libswscale/swscale.h");
    @cInclude("libswresample/swresample.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libavutil/opt.h");
    @cInclude("libavutil/display.h");
});

// bytes per pixel, i know... sorry
pub const RGB24_BPP = 3;
pub const Y_BPP = 1;
pub const UV_BPP = 2;

pub const magic_aspect_ratio = 1.7763157895;

pub const Resolution = struct {
    width: c_int,
    height: c_int,

    pub fn format(self: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{}x{}", .{ self.width, self.height });
    }
};

pub const BlurManager = struct {
    const Layer = struct { fbo: c_uint, tex: c_uint };

    front: Layer,
    back: Layer,

    resolution: Resolution,

    pub inline fn current(self: BlurManager, i: usize) Layer {
        return if (i & 1 == 0) self.front else self.back;
    }

    pub inline fn previous(self: BlurManager, i: usize) Layer {
        return if (i & 1 == 0) self.back else self.front;
    }
};

pub const GpuResourceManager = struct {
    vao: VertexArrayPool,
    vbo: VertexBufferPool,
    fbo: FramebufferPool,
    pbo: PixelBufferPool,
    tex: TexturePool,

    prog: ProgramPool,

    meta: Metadata,

    const log = std.log.scoped(.gpu);

    const Metadata = struct {
        circle_radius: f32,
        ring_radius: f32,
        ring_thickness: f32,

        blur_res: Resolution,
    };

    const VertexArrayPool = struct {
        const Index = enum(usize) { tex = 0, blur, empty };
        const len = @typeInfo(Index).@"enum".fields.len;

        id: [len]c_uint,

        fn init(self: *VertexArrayPool) void {
            var ids: [len]c_uint = undefined;
            gl.GenVertexArrays(len, ids[0..]);

            self.* = .{ .id = ids };
        }

        fn deinit(self: *VertexArrayPool) void {
            gl.DeleteVertexArrays(len, self.id[0..]);
        }

        pub fn get(self: VertexArrayPool, comptime idx: Index) c_uint {
            return self.id[@intFromEnum(idx)];
        }
    };

    const VertexBufferPool = struct {
        const Index = enum(usize) { tex };
        const len = @typeInfo(Index).@"enum".fields.len;

        id: [len]c_uint,

        fn init(self: *VertexBufferPool) void {
            var ids: [len]c_uint = undefined;
            gl.GenBuffers(len, ids[0..]);

            self.* = .{ .id = ids };
        }

        fn deinit(self: *VertexBufferPool) void {
            gl.DeleteBuffers(len, self.id[0..]);
        }

        pub fn get(self: VertexBufferPool, comptime idx: Index) c_uint {
            return self.id[@intFromEnum(idx)];
        }
    };

    const FramebufferPool = struct {
        const Index = enum(usize) { angle, blur_front, blur_back, y, uv, out };
        const len = @typeInfo(Index).@"enum".fields.len;

        id: [len]c_uint,

        fn init(self: *FramebufferPool) void {
            var ids: [len]c_uint = undefined;
            gl.GenFramebuffers(len, ids[0..]);

            self.* = .{ .id = ids };
        }

        fn deinit(self: *FramebufferPool) void {
            gl.DeleteFramebuffers(len, self.id[0..]);
        }

        pub fn get(self: FramebufferPool, comptime idx: Index) c_uint {
            return self.id[@intFromEnum(idx)];
        }
    };

    pub const PixelBufferPool = struct {
        pub const Index = enum(usize) { yuv_slot0, yuv_slot1, yuv_slot2, yuv_slot3 };
        const len = @typeInfo(Index).@"enum".fields.len;

        id: [len]c_uint,

        fn init(self: *PixelBufferPool) void {
            var ids: [len]c_uint = undefined;
            gl.GenBuffers(len, ids[0..]);

            self.* = .{ .id = ids };
        }

        fn deinit(self: *PixelBufferPool) void {
            gl.DeleteBuffers(len, self.id[0..]);
        }

        pub fn get(self: PixelBufferPool, idx: Index) c_uint {
            return self.id[@intFromEnum(idx)];
        }
    };

    const TexturePool = struct {
        const Index = enum(usize) {
            /// 1x1 texture that stores the calculated angle of the frame
            angle,
            // DoubleBuffer related
            y_front,
            y_back,
            uv_front,
            uv_back,
            // Blur related
            blur_front,
            blur_back,
            out,
            y_out,
            uv_out,
        };
        const len = @typeInfo(Index).@"enum".fields.len;

        id: [len]c_uint,

        fn init(self: *TexturePool) void {
            var ids: [len]c_uint = undefined;
            gl.GenTextures(len, ids[0..]);

            self.* = .{ .id = ids };
        }

        fn deinit(self: *TexturePool) void {
            gl.DeleteTextures(len, self.id[0..]);
        }

        pub fn get(self: TexturePool, idx: Index) c_uint {
            return self.id[@intFromEnum(idx)];
        }
    };

    const ProgramPool = struct {
        const Index = enum(usize) { tex = 0, bg, blur, ring, circle, angle, rgb_to_nv12 };
        const len = @typeInfo(Index).@"enum".fields.len;

        id: [len]c_uint,

        fn compile(self: *ProgramPool) !void {
            inline for (@typeInfo(Index).@"enum".fields, &self.id) |field, *ptr| {
                const tag: Index = @field(Index, field.name);

                ptr.* = switch (tag) {
                    .tex => try opengl_impl.program("shader/texture.vert", "shader/texture.frag"),
                    .bg => try opengl_impl.program("shader/texture.vert", "shader/bg.frag"),
                    .blur => try opengl_impl.program("shader/blur.vert", "shader/blur.frag"),
                    .ring => try opengl_impl.program("shader/texture.vert", "shader/ring.frag"),
                    .circle => try opengl_impl.program("shader/texture.vert", "shader/circle.frag"),
                    .angle => try opengl_impl.program("shader/blur.vert", "shader/rotation.frag"),
                    .rgb_to_nv12 => try opengl_impl.program("shader/blur.vert", "shader/rgb_to_nv12.frag"),
                };
            }
        }

        fn deinit(self: ProgramPool) void {
            for (self.id) |prog| {
                gl.DeleteProgram(prog);
            }
        }

        pub fn get(self: ProgramPool, comptime idx: Index) c_uint {
            return self.id[@intFromEnum(idx)];
        }
    };

    pub fn init(allocator: std.mem.Allocator, render_view: Viewport, dimensions: Resolution) !*GpuResourceManager {
        const manager = try allocator.create(GpuResourceManager);
        errdefer allocator.destroy(manager);

        const width: u32 = @intCast(dimensions.width);
        const height: u32 = @intCast(dimensions.height);

        {
            const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
            const gcd = std.math.gcd(width, height);

            log.debug("render resolution: {}x{}", .{ width, height });
            log.debug("render aspect ratio: {}:{} | {d:.3}", .{ width / gcd, height / gcd, aspect });
        }

        manager.vao.init();
        manager.vbo.init();
        manager.fbo.init();
        manager.pbo.init();
        manager.tex.init();

        try manager.prog.compile();
        try manager.setupAngleCalc();

        manager.setupVertexArrays(dimensions);
        manager.setupBlur(dimensions);
        manager.setupVideoTextures(dimensions);
        try manager.setupOffscreenTarget(render_view);

        return manager;
    }

    pub fn deinit(self: *GpuResourceManager, allocator: std.mem.Allocator) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "GpuResourceManager.deinit" });
        defer zone.end();

        self.vao.deinit();
        self.vbo.deinit();
        self.fbo.deinit();
        self.pbo.deinit();
        self.tex.deinit();

        allocator.destroy(self);
    }

    pub fn setupOffscreenTarget(self: GpuResourceManager, render_view: Viewport) error{setup_error}!void {
        const out_tex = self.tex.get(.out);
        const out_fbo = self.fbo.get(.out);
        const width, const height = render_view.get();

        gl.BindTexture(gl.TEXTURE_2D, out_tex);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGB8, width, height, 0, gl.RGB, gl.UNSIGNED_BYTE, null);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

        gl.BindFramebuffer(gl.FRAMEBUFFER, out_fbo);
        defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, out_tex, 0);
        if (gl.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) return error.setup_error;
    }

    pub fn blur(self: GpuResourceManager) BlurManager {
        const tex_front = self.tex.get(.blur_front);
        const tex_back = self.tex.get(.blur_back);
        const fbo_front = self.fbo.get(.blur_front);
        const fbo_back = self.fbo.get(.blur_back);

        return .{
            .front = .{ .fbo = fbo_front, .tex = tex_front },
            .back = .{ .fbo = fbo_back, .tex = tex_back },
            .resolution = self.meta.blur_res,
        };
    }

    pub fn setupAngleCalc(self: GpuResourceManager) error{setup_error}!void {
        const fbo_id = self.fbo.get(.angle);
        const tex_id = self.tex.get(.angle);
        const width, const height = .{ 1, 1 }; // 1x1

        gl.BindFramebuffer(gl.FRAMEBUFFER, fbo_id);
        defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        gl.BindTexture(gl.TEXTURE_2D, tex_id);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex_id, 0);

        const ret = gl.CheckFramebufferStatus(gl.FRAMEBUFFER);
        if (ret != gl.FRAMEBUFFER_COMPLETE) return error.setup_error;
    }

    pub fn setupEncodingTargets(self: *const GpuResourceManager, encode_view: Viewport, frame: AvFrame) error{setup_error}!void {
        const BufIndex = PixelBufferPool.Index;

        const width, const height = encode_view.get();

        const y_len = frame.ptr().linesize[0] * height;
        const uv_len = frame.ptr().linesize[1] * @divTrunc(height, 2);
        const len = y_len + uv_len;

        for ([_]BufIndex{ .yuv_slot0, .yuv_slot1, .yuv_slot2, .yuv_slot3 }) |idx| {
            gl.BindBuffer(gl.PIXEL_PACK_BUFFER, self.pbo.get(idx));
            gl.BufferData(gl.PIXEL_PACK_BUFFER, len, null, gl.STREAM_READ);
        }

        // setup y fbo + tex
        gl.BindFramebuffer(gl.FRAMEBUFFER, self.fbo.get(.y));
        gl.BindTexture(gl.TEXTURE_2D, self.tex.get(.y_out));
        gl.TexImage2D(gl.TEXTURE_2D, 0, gl.R8, @intCast(width), @intCast(height), 0, gl.RED, gl.UNSIGNED_BYTE, null);
        gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, self.tex.get(.y_out), 0);
        if (gl.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) return error.setup_error;

        // setup uv fbo + tex
        gl.BindFramebuffer(gl.FRAMEBUFFER, self.fbo.get(.uv));
        gl.BindTexture(gl.TEXTURE_2D, self.tex.get(.uv_out));
        gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RG8, @divTrunc(width, 2), @divTrunc(height, 2), 0, gl.RG, gl.UNSIGNED_BYTE, null);
        gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, self.tex.get(.uv_out), 0);
        if (gl.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE) return error.setup_error;

        gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);
        gl.BindFramebuffer(gl.FRAMEBUFFER, 0);
        gl.BindTexture(gl.TEXTURE_2D, 0);
    }

    pub fn setupVideoTextures(self: *const GpuResourceManager, size: Resolution) void {
        const TexIndex = TexturePool.Index;

        const width = size.width;
        const height = size.height;

        for ([_]TexIndex{ .y_front, .y_back, .uv_front, .uv_back }) |idx| {
            gl.BindTexture(gl.TEXTURE_2D, self.tex.get(idx));

            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            const internal: c_int, const format: c_uint, const res: Resolution = switch (idx) {
                .uv_front, .uv_back => .{ gl.RG8, gl.RG, .{ .width = @divTrunc(width, 2), .height = @divTrunc(height, 2) } },
                .y_front, .y_back => .{ gl.R8, gl.RED, .{ .width = width, .height = height } },
                else => unreachable,
            };

            gl.TexImage2D(
                gl.TEXTURE_2D,
                0,
                internal,
                res.width,
                res.height,
                0,
                format,
                gl.UNSIGNED_BYTE,
                null,
            );
        }

        gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);
    }

    pub fn setupBlur(self: *GpuResourceManager, full_size: Resolution) void {
        const full_width = full_size.width;
        const full_height = full_size.height;

        const width, const height = blk: {
            const limit: c_int = 128; // arbitrary
            var w = full_width;
            var h = full_height;

            while (true) {
                const next = .{ @divTrunc(w, 2), @divTrunc(h, 2) };
                if (@min(next[0], next[1]) < limit) break :blk .{ w, h };

                w, h = next;
            }
        };

        log.debug("blur resolution: {}x{}", .{ width, height });

        const tex_front = self.tex.get(.blur_front);
        const tex_back = self.tex.get(.blur_back);
        const fbo_front = self.fbo.get(.blur_front);
        const fbo_back = self.fbo.get(.blur_back);

        const layers: [2]BlurManager.Layer = .{
            .{ .fbo = fbo_front, .tex = tex_front },
            .{ .fbo = fbo_back, .tex = tex_back },
        };

        for (layers) |layer| {
            gl.BindFramebuffer(gl.FRAMEBUFFER, layer.fbo);
            defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

            gl.BindTexture(gl.TEXTURE_2D, layer.tex);
            defer gl.BindTexture(gl.TEXTURE_2D, 0);

            gl.TexImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGB16F,
                @intCast(width),
                @intCast(height),
                0,
                gl.RGB,
                gl.HALF_FLOAT,
                null,
            );

            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, layer.tex, 0);

            const ret = gl.CheckFramebufferStatus(gl.FRAMEBUFFER);
            if (ret != gl.FRAMEBUFFER_COMPLETE) @panic("FIXME: Framebuffer incomplete");
        }

        self.meta.blur_res = .{ .width = width, .height = height };
    }

    pub fn setupVertexArrays(self: *GpuResourceManager, size: Resolution) void {
        const width = size.width;
        const height = size.height;

        // zig fmt: off
        const tex_verts: [16]f32 = .{
            // pos      // uv
            -1.0, -1.0, 0.0, 1.0, // bottom left
             1.0, -1.0, 1.0, 1.0, // bottom right
            -1.0,  1.0, 0.0, 0.0, // top left
             1.0,  1.0, 1.0, 0.0, // top right
        };
        // zig fmt: on

        const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
        const ratio: [2]f32 = if (aspect > 1.0) .{ 1.0, 1.0 / aspect } else .{ aspect, 1.0 };

        // https://github.com/Lawrenceeeeeeee/python_rotaeno_stabilizer/blob/6e6504f5e3867404c66d94c5752daab5936eedc2/python_rotaeno_stabilizer.py#L253-L258
        const magic_radius_scale = 1.564; // -6
        const magic_thickness = 0.031; // + 6, trial and error
        const rota_height = if (aspect >= magic_aspect_ratio) ratio[1] else ratio[0] / magic_aspect_ratio;

        const radius = magic_radius_scale * rota_height;
        const circle_radius = radius * 1.05;
        const radius_thickness = rota_height * magic_thickness;

        { // Setup for FFMPEG Texture
            gl.BindVertexArray(self.vao.get(.tex));

            gl.BindBuffer(gl.ARRAY_BUFFER, self.vbo.get(.tex));
            gl.BufferData(gl.ARRAY_BUFFER, @sizeOf(@TypeOf(tex_verts)), tex_verts[0..].ptr, gl.STATIC_DRAW);

            gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 0);
            gl.EnableVertexAttribArray(0);

            gl.VertexAttribPointer(1, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 2 * @sizeOf(f32));
            gl.EnableVertexAttribArray(1);
        }

        self.meta.circle_radius = circle_radius;
        self.meta.ring_radius = radius;
        self.meta.ring_thickness = radius_thickness;
    }
};

pub fn sleep(duration: u64) void {
    const start = c.SDL_GetTicksNS();
    const target = start + duration;
    const threshold = 2 * std.time.ns_per_ms;

    while (true) {
        const now = c.SDL_GetTicksNS();
        if (now >= target) return;

        const remaining = target - now;
        if (remaining < threshold) break;

        c.SDL_Delay(1);
    }

    while (c.SDL_GetTicksNS() < target) std.atomic.spinLoopHint();
}

const opengl_impl = struct {
    const log = std.log.scoped(.shader);

    fn program(comptime vert_path: []const u8, comptime frag_path: []const u8) !c_uint {
        const vert_shader: [1][*]const u8 = .{@embedFile(vert_path)[0..].ptr};
        const frag_shader: [1][*]const u8 = .{@embedFile(frag_path)[0..].ptr};

        const vs = gl.CreateShader(gl.VERTEX_SHADER);
        defer gl.DeleteShader(vs);

        log.debug("compiling shader: {s}", .{vert_path});
        gl.ShaderSource(vs, 1, vert_shader[0..], null);
        gl.CompileShader(vs);

        if (!shader.didCompile(vs)) return error.VertexCompileError;

        const fs = gl.CreateShader(gl.FRAGMENT_SHADER);
        defer gl.DeleteShader(fs);

        log.debug("compiling shader: {s}", .{frag_path});
        gl.ShaderSource(fs, 1, frag_shader[0..], null);
        gl.CompileShader(fs);

        if (!shader.didCompile(fs)) return error.FragmentCompileError;

        const prog = gl.CreateProgram();
        gl.AttachShader(prog, vs);
        gl.AttachShader(prog, fs);
        gl.LinkProgram(prog);

        return prog;
    }

    const shader = struct {
        fn didCompile(id: c_uint) bool {
            var success: c_int = undefined;
            gl.GetShaderiv(id, gl.COMPILE_STATUS, @ptrCast(&success));

            if (success == 0) err(id);

            return success == 1;
        }

        fn err(id: c_uint) void {
            var error_msg: [512]u8 = undefined;

            gl.GetShaderInfoLog(id, error_msg.len, null, &error_msg);
            log.err("{s}", .{std.mem.sliceTo(&error_msg, 0)});
        }
    };
};

fn circle(allocator: std.mem.Allocator, radius: f32, len: usize) !std.ArrayList(f32) {
    var list: std.ArrayList(f32) = .empty;
    errdefer list.deinit(allocator);

    const _len: f32 = @floatFromInt(len);

    try list.appendSlice(allocator, &.{ 0.0, 0.0 });

    for (0..len) |i| {
        const angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / _len;
        const x = @cos(angle);
        const y = @sin(angle);

        try list.append(allocator, x * radius);
        try list.append(allocator, y * radius);
    }

    try list.appendSlice(allocator, list.items[2..][0..2]); // complete the loop
    return list;
}

fn ring(allocator: std.mem.Allocator, inner_radius: f32, outer_radius: f32, len: usize) !std.ArrayList(f32) {
    var list: std.ArrayList(f32) = .empty;
    errdefer list.deinit(allocator);

    const _len: f32 = @floatFromInt(len);

    for (0..len) |i| {
        const angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / _len;
        const x = @cos(angle);
        const y = @sin(angle);

        try list.append(allocator, x * outer_radius);
        try list.append(allocator, y * outer_radius);

        try list.append(allocator, x * inner_radius);
        try list.append(allocator, y * inner_radius);
    }

    try list.appendSlice(allocator, list.items[0..4]); // complete the loop

    return list;
}

pub fn Linesize(comptime fmt: c.AVPixelFormat) type {
    return switch (fmt) {
        c.AV_PIX_FMT_NV12 => struct {
            y: c_int,
            uv: c_int,

            pub fn init(frame: AvFrame) @This() {
                std.debug.assert(frame.ptr().format == fmt);

                return .{
                    .y = frame.ptr().linesize[0],
                    .uv = frame.ptr().linesize[1],
                };
            }
        },
        c.AV_PIX_FMT_RGB24 => struct {
            rgb: c_int,

            pub fn init(frame: AvFrame) @This() {
                std.debug.assert(frame.ptr().format == fmt);

                return .{
                    .rgb = frame.ptr().linesize[0],
                };
            }
        },
        else => unreachable,
    };
}

pub const UploadBuffer = struct {
    const PixelBufferPool = GpuResourceManager.PixelBufferPool;
    const Upload = struct { id: PixelBufferPool.Index, pts: i64 };
    const len = @typeInfo(PixelBufferPool.Index).@"enum".fields.len;

    pub const default: @This() = .{
        .id = std.meta.tags(PixelBufferPool.Index).*,
        .pts = @splat(null),
        .idx = 0,
    };

    id: [len]PixelBufferPool.Index,
    pts: [len]?i64,
    idx: usize = 0,

    pub fn current(self: @This()) PixelBufferPool.Index {
        return self.id[self.idx % len];
    }

    /// returns the oldest PBO in the Queue
    pub fn next(self: *@This()) ?Upload {
        const timestamp = self.pts[(self.idx + 1) % len] orelse return null;
        self.pts[(self.idx + 1) % len] = null;

        return .{
            .id = self.id[(self.idx + 1) % len],
            .pts = timestamp,
        };
    }

    /// Writes timestamp to current PBO, prepares next frame to overwrite oldest
    pub fn advance(self: *@This(), pts: i64) void {
        self.pts[self.idx % len] = pts;
        self.idx += 1;
    }

    pub fn skip(self: *@This()) void {
        std.debug.assert(self.pts[self.idx % len] == null);
        self.idx += 1;
    }

    pub fn flush(self: *@This()) ?Upload {
        const timestamp = self.pts[self.idx % len] orelse return null;
        self.pts[self.idx % len] = null;

        const ret: Upload = .{ .id = self.id[self.idx % len], .pts = timestamp };
        self.idx += 1;

        return ret;
    }
};

pub const DoubleBuffer = struct {
    const TexturePool = GpuResourceManager.TexturePool;
    pub const default: DoubleBuffer = .{};

    y: [2]TexturePool.Index = .{ .y_front, .y_back },
    uv: [2]TexturePool.Index = .{ .uv_front, .uv_back },

    display_times: [2]?f64 = .{ null, null },
    idx: u1 = 0,

    generation: usize = 0,

    pub const Channel = enum { y, uv };

    pub const Buffer = struct {
        super: *DoubleBuffer,
        idx: u1,

        generation: usize,

        pub fn tex(self: @This(), comptime ch: Channel) TexturePool.Index {
            std.debug.assert(self.generation == self.super.generation);

            switch (ch) {
                .y => return self.super.y[self.idx],
                .uv => return self.super.uv[self.idx],
            }
        }

        pub fn displayTime(self: @This()) f64 {
            std.debug.assert(self.generation == self.super.generation);

            return self.super.display_times[self.idx].?;
        }

        pub fn setDisplayTime(self: @This(), timestamp: f64) void {
            std.debug.assert(self.generation == self.super.generation);

            self.super.display_times[self.idx] = timestamp;
        }

        pub fn flip(self: @This()) @This() {
            std.debug.assert(self.generation == self.super.generation);

            return .{
                .super = self.super,
                .idx = self.idx +% 1,
                .generation = self.generation,
            };
        }
    };

    pub fn front(self: *DoubleBuffer) Buffer {
        return .{ .super = self, .idx = self.idx, .generation = self.generation };
    }

    pub fn back(self: *DoubleBuffer) Buffer {
        return .{ .super = self, .idx = self.idx +% 1, .generation = self.generation };
    }

    pub fn swap(self: *DoubleBuffer) void {
        self.idx +%= 1;
        self.generation += 1;
    }
};

pub fn trace(comptime fmt: []const u8, args: anytype) void {
    var buf: [0x40]u8 = undefined;
    tracy.message(.{ .text = std.fmt.bufPrint(&buf, fmt, args) catch unreachable });
}

pub const FbStack = struct {
    const cap = 4;
    pub const default: @This() = .{ .id = undefined, .idx = 0 };

    id: [cap]c_uint,
    idx: usize,

    const log = std.log.scoped(.framebuffer_stack);

    pub fn push(self: *FbStack, id: c_uint) !void {
        if (self.idx >= cap) return error.stack_overflow;

        self.id[self.idx] = id;
        self.idx += 1;

        gl.BindFramebuffer(gl.FRAMEBUFFER, id);
    }

    pub fn get(self: *const FbStack) c_uint {
        if (self.idx == 0) return 0;
        return self.id[self.idx - 1];
    }

    pub fn pop(self: *FbStack) void {
        if (self.idx == 0) return;
        self.idx -= 1;

        if (self.idx > 0) {
            const id = self.id[self.idx - 1];
            gl.BindFramebuffer(gl.FRAMEBUFFER, id);
        } else {
            gl.BindFramebuffer(gl.FRAMEBUFFER, 0);
        }
    }
};

pub const Viewport = struct {
    const cap = 5;
    pub const default: @This() = .{ .width = undefined, .height = undefined, .idx = 0 };

    width: [cap]c_int,
    height: [cap]c_int,

    idx: usize,

    const log = std.log.scoped(.viewport);

    pub fn push(self: *Viewport, width: c_int, height: c_int) !void {
        if (self.idx >= cap) return error.stack_overflow;

        self.width[self.idx] = width;
        self.height[self.idx] = height;
        self.idx += 1;

        gl.Viewport(0, 0, width, height);
    }

    pub fn reset(self: *Viewport, width: c_int, height: c_int) void {
        self.* = .default;
        self.push(width, height) catch unreachable; // cap == 0
    }

    pub fn get(self: Viewport) struct { c_int, c_int } {
        return .{ self.width[self.idx - 1], self.height[self.idx - 1] };
    }

    pub fn aspect(self: Viewport) f32 {
        const res = self.resolution();
        const width: f32 = @floatFromInt(res.width);
        const height: f32 = @floatFromInt(res.height);

        return width / height;
    }

    pub fn resolution(self: Viewport) Resolution {
        const width, const height = self.get();
        return .{ .width = width, .height = height };
    }

    pub fn pop(self: *Viewport) void {
        if (self.idx == 0) return;
        self.idx -= 1;

        if (self.idx > 0) {
            const w = self.width[self.idx - 1];
            const h = self.height[self.idx - 1];

            gl.Viewport(0, 0, w, h);
        }
    }
};

pub fn getPixelFormatName(kind: c.AVPixelFormat) [:0]const u8 {
    if (kind == c.AV_PIX_FMT_NONE) return "none";
    return std.mem.span(c.av_get_pix_fmt_name(kind));
}

// Error Policy:
// ---
// Errors are broken into two categories:
// 1. Recoverable, these errors must be gracefully handled and displayed to the user
// 2. Irrecoverable. These errors are allowed to crash the program if unhandled
//
// The way this program should work more or less, is that as soon as we have DearImgui drawing, all errors must more or less be treated as recoverable.
//
// Errors is a method that reports errors and their context
pub const Errors = struct {
    count: usize,

    pub fn init(self: *Errors) void {
        self.* = .{ .count = 0 };
    }

    pub fn add_local_ip_err(self: *Errors, e: std.posix.ConnectError) void {
        self.print("failed to determine local ip: {}\n", .{e});
    }

    pub fn add_win_signal_handler_err(self: *Errors, e: std.posix.ConnectError) void {
        self.print("failed to setup windows CTRL-C signal handler: {}\n", .{e});
    }

    pub fn add_set_volume_err(self: *Errors, volume: f32) void {
        self.print("failed to set device gain to {d:.2}: {s}\n", .{ volume, c.SDL_GetError() });
    }

    // TODO(paoda): maybe add a scope thing here?
    fn print(self: *Errors, comptime fmt: []const u8, args: anytype) void {
        std.debug.assert(fmt[fmt.len - 1] == '\n');

        self.count += 1;

        if (tracy.enabled) {
            var buf: [0x80]u8 = undefined;
            tracy.message(.{ .text = std.fmt.bufPrint(&buf, "err: " ++ fmt, args) catch unreachable });
        } else {
            std.debug.print("err: " ++ fmt, args);
        }
    }
};
