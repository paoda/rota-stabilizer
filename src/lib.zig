//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const gl = @import("gl");

pub const c = @cImport({
    @cDefine("SDL_DISABLE_OLD_NAMES", {});
    @cDefine("SDL_MAIN_HANDLED", {});

    @cInclude("SDL3/SDL.h");
    @cInclude("SDL3/SDL_main.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libswscale/swscale.h");
    @cInclude("libswresample/swresample.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libavutil/opt.h");
});

// bytes per pixel, i know... sorry
pub const RGB24_BPP = 3;
pub const Y_BPP = 1;
pub const UV_BPP = 2;

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

const Resolution = struct { width: u32, height: u32 };

pub const GpuResourceManager = struct {
    vao: VertexArrayPool,
    vbo: VertexBufferPool,
    fbo: FramebufferPool,
    pbo: PixelBufferPool,
    tex: TexturePool,

    prog: ProgramPool,

    meta: Metadata,
    allocator: std.mem.Allocator,

    const Metadata = struct {
        circle_len: usize,
        circle_radius: f32,

        ring_len: usize,

        strong_blur_res: Resolution,
        weak_blur_res: Resolution,
    };

    const VertexArrayPool = struct {
        const Index = enum(usize) { tex = 0, blur, ring, circle, empty };
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
        const Index = enum(usize) { tex, ring, circle };
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
        const Index = enum(usize) {
            angle,

            blur0_front,
            blur0_back,

            blur1_front,
            blur1_back,
        };
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

    const PixelBufferPool = struct {
        const Index = enum(usize) { y_front, y_back, uv_front, uv_back };
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
            angle,
            y_front,
            y_back,
            uv_front,
            uv_back,

            blur0_front,
            blur0_back,

            blur1_front,
            blur1_back,
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
        const Index = enum(usize) { tex = 0, bg, blur, ring, circle, angle };
        const len = @typeInfo(Index).@"enum".fields.len;

        id: [len]c_uint,

        fn compile(self: *ProgramPool) !void {
            inline for (@typeInfo(Index).@"enum".fields, &self.id) |field, *ptr| {
                const tag: Index = @field(Index, field.name);

                ptr.* = switch (tag) {
                    .tex => try opengl_impl.program("shader/texture.vert", "shader/texture.frag"),
                    .bg => try opengl_impl.program("shader/texture.vert", "shader/bg.frag"),
                    .blur => try opengl_impl.program("shader/blur.vert", "shader/blur.frag"),
                    .ring => try opengl_impl.program("shader/ring.vert", "shader/ring.frag"),
                    .circle => try opengl_impl.program("shader/ring.vert", "shader/circle.frag"),
                    .angle => try opengl_impl.program("./shader/blur.vert", "./shader/rotation.frag"),
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

    pub fn init(allocator: std.mem.Allocator, width: u32, height: u32) !*GpuResourceManager {
        const manager = try allocator.create(GpuResourceManager);
        errdefer allocator.destroy(manager);
        manager.allocator = allocator;

        manager.vao.init();
        manager.vbo.init();
        manager.fbo.init();
        manager.pbo.init();
        manager.tex.init();

        try manager.prog.compile();
        try manager.setupVertexArrays(allocator, width, height);
        try manager.setupAngleCalc();

        manager.setupBlur(.strong, width / 8, height / 8);
        manager.setupBlur(.weak, width / 8, height / 8);

        manager.setupVideoTextures(width, height);
        return manager;
    }

    pub fn deinit(self: *GpuResourceManager) void {
        self.vao.deinit();
        self.vbo.deinit();
        self.fbo.deinit();
        self.pbo.deinit();
        self.tex.deinit();

        self.allocator.destroy(self);
    }

    pub fn blur(self: GpuResourceManager, comptime kind: BlurKind) BlurManager {
        const tex_front = if (kind == .strong) self.tex.get(.blur0_front) else self.tex.get(.blur1_front);
        const tex_back = if (kind == .strong) self.tex.get(.blur0_back) else self.tex.get(.blur1_back);
        const fbo_front = if (kind == .strong) self.fbo.get(.blur0_front) else self.fbo.get(.blur1_front);
        const fbo_back = if (kind == .strong) self.fbo.get(.blur0_back) else self.fbo.get(.blur1_back);

        return .{
            .front = .{ .fbo = fbo_front, .tex = tex_front },
            .back = .{ .fbo = fbo_back, .tex = tex_back },

            .resolution = switch (kind) {
                .strong => self.meta.strong_blur_res,
                .weak => self.meta.weak_blur_res,
            },
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

    pub fn setupVideoTextures(self: *const GpuResourceManager, width: u32, height: u32) void {
        const TexIndex = TexturePool.Index;
        const BufIndex = PixelBufferPool.Index;

        for ([_]TexIndex{ .y_front, .y_back, .uv_front, .uv_back }) |idx| {
            gl.BindTexture(gl.TEXTURE_2D, self.tex.get(idx));

            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            const internal: c_int, const format: c_uint, const res: Resolution = switch (idx) {
                .uv_front, .uv_back => .{ gl.RG8, gl.RG, .{ .width = width / 2, .height = height / 2 } },
                .y_front, .y_back => .{ gl.R8, gl.RED, .{ .width = width, .height = height } },
                else => unreachable,
            };

            gl.TexImage2D(
                gl.TEXTURE_2D,
                0,
                internal,
                @intCast(res.width),
                @intCast(res.height),
                0,
                format,
                gl.UNSIGNED_BYTE,
                null,
            );
        }

        for ([_]BufIndex{ .y_front, .y_back, .uv_front, .uv_back }) |idx| {
            gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, self.pbo.get(idx));

            const len: c_int = switch (idx) {
                .uv_front, .uv_back => @intCast((width / 2) * (height / 2) * UV_BPP),
                .y_front, .y_back => @intCast(width * height * Y_BPP),
            };

            gl.BufferData(gl.PIXEL_UNPACK_BUFFER, len, null, gl.STREAM_DRAW);
        }

        gl.BindTexture(gl.TEXTURE_2D, 0);
        gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);
    }

    const BlurKind = enum { strong, weak };
    pub fn setupBlur(self: *GpuResourceManager, comptime kind: BlurKind, width: u32, height: u32) void {
        const tex_front = if (kind == .strong) self.tex.get(.blur0_front) else self.tex.get(.blur1_front);
        const tex_back = if (kind == .strong) self.tex.get(.blur0_back) else self.tex.get(.blur1_back);
        const fbo_front = if (kind == .strong) self.fbo.get(.blur0_front) else self.fbo.get(.blur1_front);
        const fbo_back = if (kind == .strong) self.fbo.get(.blur0_back) else self.fbo.get(.blur1_back);

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

        switch (kind) {
            .strong => self.meta.strong_blur_res = .{ .width = width, .height = height },
            .weak => self.meta.weak_blur_res = .{ .width = width, .height = height },
        }
    }

    pub fn setupVertexArrays(self: *GpuResourceManager, allocator: std.mem.Allocator, width: u32, height: u32) !void {
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
        const magic_aspect_ratio = 1.7763157895;
        const magic_radius_scale = 1.570;
        const magic_thickness = 0.02;
        const rota_height = if (aspect >= magic_aspect_ratio) ratio[1] else ratio[0] / magic_aspect_ratio;

        const radius = magic_radius_scale * rota_height;
        const circle_radius = radius * 1.05;
        const radius_thickness = rota_height * magic_thickness;
        const inner_radius = @max(radius - radius_thickness, 0.0);

        const ring_verts = try ring(allocator, inner_radius, radius, 0x80);
        defer ring_verts.deinit();

        // TODO: by messing with stride I think there's a way to combine the two ArrayLists
        // TODO: make the radius of the puck a runtime thing (scaling matrix + uniform)
        const circle_verts = try circle(allocator, circle_radius, 0x80);
        defer circle_verts.deinit();

        { // Setup for FFMPEG Texture
            gl.BindVertexArray(self.vao.get(.tex));

            gl.BindBuffer(gl.ARRAY_BUFFER, self.vbo.get(.tex));
            gl.BufferData(gl.ARRAY_BUFFER, @sizeOf(@TypeOf(tex_verts)), tex_verts[0..].ptr, gl.STATIC_DRAW);

            gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 0);
            gl.EnableVertexAttribArray(0);

            gl.VertexAttribPointer(1, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 2 * @sizeOf(f32));
            gl.EnableVertexAttribArray(1);
        }

        { // Setup for Ring
            gl.BindVertexArray(self.vao.get(.ring));

            gl.BindBuffer(gl.ARRAY_BUFFER, self.vbo.get(.ring));

            gl.BufferData(gl.ARRAY_BUFFER, @intCast(ring_verts.items.len * @sizeOf(f32)), ring_verts.items[0..].ptr, gl.STATIC_DRAW);
            gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 2 * @sizeOf(f32), 0);
            gl.EnableVertexAttribArray(0);
        }

        { // Setup for Circle
            gl.BindVertexArray(self.vao.get(.circle));
            defer gl.BindVertexArray(0);

            gl.BindBuffer(gl.ARRAY_BUFFER, self.vbo.get(.circle));
            defer gl.BindBuffer(gl.ARRAY_BUFFER, 0);

            gl.BufferData(gl.ARRAY_BUFFER, @intCast(circle_verts.items.len * @sizeOf(f32)), circle_verts.items[0..].ptr, gl.STATIC_DRAW);
            gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 2 * @sizeOf(f32), 0);
            gl.EnableVertexAttribArray(0);
        }

        self.meta.circle_len = circle_verts.items.len / 2;
        self.meta.ring_len = ring_verts.items.len / 2;
        self.meta.circle_radius = circle_radius;
    }
};

pub fn sleep(ns: u64) void {
    if (ns == 0) {
        @branchHint(.cold);
        return;
    }

    const freq = c.SDL_GetPerformanceFrequency();
    const start_counter = c.SDL_GetPerformanceCounter();

    // TODO: confirm I did the conversion right
    const duration: u64 = @intCast(@divTrunc(@as(i128, ns) * freq, std.time.ns_per_s));
    const target_counter: u64 = start_counter + duration;

    const threshold_ns = 2 * std.time.ns_per_ms; // Example threshold for coarse sleep

    // Perform coarse sleep if the total duration is significantly larger than the threshold
    if (ns > threshold_ns) {
        const ideal_ns = ns - threshold_ns;
        std.Thread.sleep(ideal_ns);
    }

    while (c.SDL_GetPerformanceCounter() < target_counter) std.atomic.spinLoopHint();

    // Optional: Debug print the actual elapsed time for verification
    // const end_counter = c.SDL_GetPerformanceCounter();
    // const actual_elapsed_ns: u64 = @intCast(@divTrunc(@as(i128, end_counter - start_counter) * std.time.ns_per_s, freq));

    // const log = std.log.scoped(.sleep);
    // log.debug("attempted {}ms late by {}ns", .{ ns / std.time.ns_per_ms, @as(i64, @intCast(actual_elapsed_ns)) - @as(i64, @intCast(ns)) });
}

const opengl_impl = struct {
    fn program(comptime vert_path: []const u8, comptime frag_path: []const u8) !c_uint {
        const vert_shader: [1][*]const u8 = .{@embedFile(vert_path)[0..].ptr};
        const frag_shader: [1][*]const u8 = .{@embedFile(frag_path)[0..].ptr};

        const vs = gl.CreateShader(gl.VERTEX_SHADER);
        defer gl.DeleteShader(vs);

        gl.ShaderSource(vs, 1, vert_shader[0..], null);
        gl.CompileShader(vs);

        if (!shader.didCompile(vs)) return error.VertexCompileError;

        const fs = gl.CreateShader(gl.FRAGMENT_SHADER);
        defer gl.DeleteShader(fs);

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
        const log = std.log.scoped(.shader);

        fn didCompile(id: c_uint) bool {
            var success: c_int = undefined;
            gl.GetShaderiv(id, gl.COMPILE_STATUS, &success);

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
    var list: std.ArrayList(f32) = .init(allocator);
    errdefer list.deinit();

    const _len: f32 = @floatFromInt(len);

    try list.appendSlice(&.{ 0.0, 0.0 });

    for (0..len) |i| {
        const angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / _len;
        const x = @cos(angle);
        const y = @sin(angle);

        try list.append(x * radius);
        try list.append(y * radius);
    }

    try list.appendSlice(list.items[2..][0..2]); // complete the loop
    return list;
}

fn ring(allocator: std.mem.Allocator, inner_radius: f32, outer_radius: f32, len: usize) !std.ArrayList(f32) {
    var list = std.ArrayList(f32).init(allocator);
    errdefer list.deinit();

    const _len: f32 = @floatFromInt(len);

    for (0..len) |i| {
        const angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / _len;
        const x = @cos(angle);
        const y = @sin(angle);

        try list.append(x * outer_radius);
        try list.append(y * outer_radius);

        try list.append(x * inner_radius);
        try list.append(y * inner_radius);
    }

    try list.appendSlice(list.items[0..4]); // complete the loop

    return list;
}
