const std = @import("std");
const builtin = @import("builtin");
const gl = @import("gl");
const zstbi = @import("zstbi");

const c = @import("librota").c;

const sleep = @import("librota").sleep;
const libavError = @import("librota").libavError;
const FrameQueue = @import("librota").FrameQueue;

var gl_procs: gl.ProcTable = undefined;

// bytes per pixel, i know... sorry
const RGB24_BPP = 3;
const Y_BPP = 1;
const UV_BPP = 2;

const muted_by_default = true;

/// set to enable hardware decoding
const hw_device: ?c.AVHWDeviceType = switch (builtin.os.tag) {
    .linux => c.AV_HWDEVICE_TYPE_VAAPI,
    .windows => c.AV_HWDEVICE_TYPE_D3D11VA,
    // .macos => c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
    else => null, // TODO: maybe use c.AV_HWDEVICE_TYPE_VULKAN on everything?
};

pub fn main() !void {
    const log = std.log.scoped(.ui);
    errdefer |err| if (err == error.sdl_error) log.err("SDL Error: {s}", .{c.SDL_GetError()});

    var gpa: std.heap.DebugAllocator(.{}) = .{ .backing_allocator = std.heap.c_allocator };
    defer std.debug.assert(gpa.deinit() == .ok);

    const allocator = gpa.allocator();

    // tmp: testing frame capture
    zstbi.init(allocator);
    defer zstbi.deinit();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.skip(); // self
    const path = args.next() orelse return error.missing_file;

    var vid_fmt_ctx = try createFormatContext(path);
    defer vid_fmt_ctx.deinit();

    var aud_fmt_ctx = try createFormatContext(path);
    defer aud_fmt_ctx.deinit();

    var vid_bundle = if (hw_device) |device| blk: {
        var bundle = try createHwAccelCodecContext(.video, vid_fmt_ctx.ptr(), device);
        bundle.hw.?.configure();

        break :blk bundle;
    } else try createCodecContext(.video, vid_fmt_ctx.ptr());
    defer vid_bundle.deinit();

    var aud_bundle = try createCodecContext(.audio, aud_fmt_ctx.ptr());
    defer aud_bundle.deinit();

    const vid_codec_ctx = vid_bundle.codec_ctx orelse return error.out_of_memory;
    const aud_codec_ctx = aud_bundle.codec_ctx orelse return error.out_of_memory;

    const vid_width: u32 = @intCast(vid_codec_ctx.width);
    const vid_height: u32 = @intCast(vid_codec_ctx.height);
    const win_size = @max(vid_width, vid_height) / 2;

    const ui = try createSdlWindow(win_size, win_size);
    defer ui.deinit();

    _ = c.SDL_GL_SetSwapInterval(0);

    const sdl_stream = try createSdlAudioStream(aud_codec_ctx);
    defer sdl_stream.deinit();

    var audio_clock = AudioClock.init(
        sdl_stream,
        @intCast(aud_codec_ctx.sample_rate),
        @intCast(aud_codec_ctx.ch_layout.nb_channels),
        c.AV_SAMPLE_FMT_FLT,
    );

    // -- opengl --

    // zig fmt: off
    const vertices: [16]f32 = .{
        // pos      // uv
        -1.0, -1.0, 0.0, 1.0, // bottom left
         1.0, -1.0, 1.0, 1.0, // bottom right
        -1.0,  1.0, 0.0, 0.0, // top left
         1.0,  1.0, 1.0, 0.0, // top right
    };
    // zig fmt: on

    gl.Enable(gl.BLEND);
    gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.Disable(gl.FRAMEBUFFER_SRGB);

    const aspect = @as(f32, @floatFromInt(vid_width)) / @as(f32, @floatFromInt(vid_height));
    const ratio: [2]f32 = if (aspect > 1.0) .{ 1.0, 1.0 / aspect } else .{ aspect, 1.0 };
    const scale = 1.0 / std.math.sqrt(ratio[0] * ratio[0] + ratio[1] * ratio[1]); // factor allows for ration w/out clipping

    {
        const gcd = std.math.gcd(vid_width, vid_height);
        log.debug("Resolution: {}x{}", .{ vid_width, vid_height });
        log.debug("Aspect Ratio: {}:{} | {d:.5}", .{ vid_width / gcd, vid_height / gcd, aspect });
    }

    // https://github.com/Lawrenceeeeeeee/python_rotaeno_stabilizer/blob/6e6504f5e3867404c66d94c5752daab5936eedc2/python_rotaeno_stabilizer.py#L253-L258
    const magic_aspect_ratio = 1.7763157895;
    const magic_radius_scale = 1.570;
    const magic_thickness = 0.02;
    const rota_height = if (aspect >= magic_aspect_ratio) ratio[1] else ratio[0] / magic_aspect_ratio;

    const radius = magic_radius_scale * rota_height;
    const radius_thickness = rota_height * magic_thickness;
    const inner_radius = @max(radius - radius_thickness, 0.0);

    const u_scale = Mat2.scale(scale);
    const u_aspect = Mat2.scaleXy(ratio[0], ratio[1]);
    const u_inv_scale = Mat2.scale((1.0 / @min(ratio[0], ratio[1])) * std.math.sqrt2); // |cos(π/4)| + |sin(π/4)| == sqrt(2)

    const ring_vertices = try ring(allocator, inner_radius, radius, 0x80);
    defer ring_vertices.deinit();

    // TODO: by messing with stride I think there's a way to combine the two ArrayLists
    // TODO: make the radius of the puck a runtime thing (scaling matrix + uniform)
    const circle_vertices = try circle(allocator, radius * 1.05, 0x80);
    defer circle_vertices.deinit();

    var vao_id = opengl_impl.vao(4);
    defer gl.DeleteVertexArrays(3, vao_id[0..]);

    var vbo_id = opengl_impl.vbo(3);
    defer gl.DeleteBuffers(3, vbo_id[0..]);

    var stable_buffer = DoubleBuffer.init(vid_width, vid_height);
    defer stable_buffer.deinit();

    const tex_prog = try opengl_impl.program("shader/texture.vert", "shader/texture.frag");
    defer gl.DeleteProgram(tex_prog);

    const bg_prog = try opengl_impl.program("shader/texture.vert", "shader/bg.frag");
    defer gl.DeleteProgram(bg_prog);

    const ring_prog = try opengl_impl.program("shader/ring.vert", "shader/ring.frag");
    defer gl.DeleteProgram(ring_prog);

    const circle_prog = try opengl_impl.program("shader/ring.vert", "shader/circle.frag");
    defer gl.DeleteProgram(circle_prog);

    const blur_prog = try opengl_impl.program("shader/blur.vert", "shader/blur.frag");
    defer gl.DeleteProgram(blur_prog);

    const prog_id: [5]c_uint = [_]c_uint{ tex_prog, ring_prog, circle_prog, bg_prog, blur_prog };

    var angle_calc = try AngleCalc.init(vid_codec_ctx.width, vid_codec_ctx.height);
    defer angle_calc.deinit();

    { // Setup for FFMPEG Texture
        gl.BindVertexArray(vao_id[@intFromEnum(Id.texture)]);
        defer gl.BindVertexArray(0);

        gl.BindBuffer(gl.ARRAY_BUFFER, vbo_id[@intFromEnum(Id.texture)]);
        defer gl.BindBuffer(gl.ARRAY_BUFFER, 0);

        gl.BufferData(gl.ARRAY_BUFFER, @sizeOf(@TypeOf(vertices)), vertices[0..].ptr, gl.STATIC_DRAW);

        gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 0);
        gl.EnableVertexAttribArray(0);

        gl.VertexAttribPointer(1, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 2 * @sizeOf(f32));
        gl.EnableVertexAttribArray(1);
    }

    { // Setup for Ring
        gl.BindVertexArray(vao_id[@intFromEnum(Id.ring)]);
        defer gl.BindVertexArray(0);

        gl.BindBuffer(gl.ARRAY_BUFFER, vbo_id[@intFromEnum(Id.ring)]);
        defer gl.BindBuffer(gl.ARRAY_BUFFER, 0);

        gl.BufferData(gl.ARRAY_BUFFER, @intCast(ring_vertices.items.len * @sizeOf(f32)), ring_vertices.items[0..].ptr, gl.STATIC_DRAW);
        gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 2 * @sizeOf(f32), 0);
        gl.EnableVertexAttribArray(0);
    }

    { // Setup for Circle
        gl.BindVertexArray(vao_id[@intFromEnum(Id.circle)]);
        defer gl.BindVertexArray(0);

        gl.BindBuffer(gl.ARRAY_BUFFER, vbo_id[@intFromEnum(Id.circle)]);
        defer gl.BindBuffer(gl.ARRAY_BUFFER, 0);

        gl.BufferData(gl.ARRAY_BUFFER, @intCast(circle_vertices.items.len * @sizeOf(f32)), circle_vertices.items[0..].ptr, gl.STATIC_DRAW);
        gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 2 * @sizeOf(f32), 0);
        gl.EnableVertexAttribArray(0);
    }

    const outer_blur = opengl_impl.setupBlur(vid_codec_ctx.width >> 1, vid_codec_ctx.height >> 1);
    defer for (outer_blur) |b| b.deinit();

    const inner_blur = opengl_impl.setupBlur(vid_codec_ctx.width, vid_codec_ctx.height);
    defer for (inner_blur) |b| b.deinit();

    // -- opengl end --
    var queue = try FrameQueue.init(allocator, 0x80); // 1s at 120fps?
    defer queue.deinit(allocator);

    var should_quit: std.atomic.Value(bool) = .init(false);

    const vid_decode: DecodeContext = .{ .bundle = vid_bundle, .fmt_ctx = vid_fmt_ctx.ptr(), .should_quit = &should_quit };
    const aud_decode: DecodeContext = .{ .bundle = aud_bundle, .fmt_ctx = aud_fmt_ctx.ptr(), .should_quit = &should_quit };

    const video_decode = try std.Thread.spawn(.{}, decodeVideo, .{ &queue, vid_decode });
    defer video_decode.join();

    const audio_decode = try std.Thread.spawn(.{}, decodeAudio, .{ sdl_stream.inner, &audio_clock, aud_decode });
    defer audio_decode.join();

    var w: c_int, var h: c_int = .{ undefined, undefined };
    try errify(c.SDL_GetWindowSizeInPixels(ui.window, &w, &h));

    var view = Viewport.init(w, h);

    while (!should_quit.load(.monotonic)) {
        var event: c.SDL_Event = undefined;

        while (c.SDL_PollEvent(&event)) {
            switch (event.type) {
                c.SDL_EVENT_QUIT => should_quit.store(true, .monotonic),
                c.SDL_EVENT_KEY_DOWN => switch (event.key.scancode) {
                    c.SDL_SCANCODE_P => {
                        log.debug("saving screenshot", .{});
                        var img: zstbi.Image = try .createEmpty(@intCast(w), @intCast(h), RGB24_BPP, .{});
                        defer img.deinit();

                        gl.BindFramebuffer(gl.READ_FRAMEBUFFER, 0);
                        gl.ReadPixels(0, 0, w, h, gl.RGB, gl.UNSIGNED_BYTE, img.data.ptr);

                        zstbi.setFlipVerticallyOnWrite(true);
                        try img.writeToFile("screenshot.png", .png);
                    },
                    c.SDL_SCANCODE_M => {
                        audio_clock.is_muted = !audio_clock.is_muted;
                        try errify(c.SDL_SetAudioStreamGain(audio_clock.stream.inner, if (audio_clock.is_muted) 0.0 else 1.0));
                    },
                    else => {},
                },
                else => {},
            }
        }

        if (queue.pop()) |frame| {
            defer queue.recycle(frame);
            defer stable_buffer.swap();

            // FIXME: we currently assume colour space. We can't do that.

            // Determine Colorspace
            // log.debug("colour space: {s}", .{c.av_color_space_name(frame.colorspace)});
            // log.debug("colour range: {s}", .{c.av_color_range_name(frame.color_range)});
            // log.debug("colour primaries: {s}", .{c.av_color_primaries_name(frame.color_primaries)});
            // log.debug("colour transfer: {s}", .{c.av_color_transfer_name(frame.color_trc)});

            const threshold = 0.1;

            const time_base: f64 = c.av_q2d(vid_fmt_ctx.ptr().streams[vid_bundle.stream].*.time_base);
            const pt_in_seconds = @as(f64, @floatFromInt(frame.best_effort_timestamp)) * time_base;
            stable_buffer.set_display_time(pt_in_seconds); // This is the timestamp for the frame currently being sent to the GPU

            // Skip frames that are too old
            const audio_time = audio_clock.seconds_passed();
            const frame_time = stable_buffer.invert().display_time();
            const diff = frame_time - audio_time;

            if (diff < -threshold and diff > -1.0) {
                // log.debug("skip: v: {d:.3}s a: {d:.3}s | \x1B[36m{d:.3}s\x1B[39m", .{ pt_in_seconds, audio_time, diff });
                continue;
            }

            if (diff > threshold * 0.1) {
                // log.debug("wait: v: {d:.3}s a: {d:.3}s | \x1B[31m{d:.3}s\x1B[39m", .{ pt_in_seconds, audio_time, diff });

                const wait_ns = @min(diff * std.time.ns_per_s, 1 * std.time.ns_per_s); // TODO: 1 second is way too long?
                sleep(@intFromFloat(wait_ns));
            }

            {
                gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, stable_buffer.pbo(.y));
                defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

                gl.BindTexture(gl.TEXTURE_2D, stable_buffer.tex(.y));
                defer gl.BindTexture(gl.TEXTURE_2D, 0);

                const pbo: ?[*]u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_UNPACK_BUFFER, gl.WRITE_ONLY));

                if (pbo) |ptr| {
                    // Copy line by line instead of a single large memcpy
                    const bytes_per_line: usize = @intCast(frame.linesize[0]);
                    const height: usize = @intCast(frame.height);

                    // Destination stride in the PBO may be different from the source
                    const dst_stride = @as(usize, @intCast(frame.width * Y_BPP));

                    for (0..height) |y| {
                        const src_offset = y * bytes_per_line;
                        const dst_offset = y * dst_stride;

                        @memcpy(ptr[dst_offset..][0..dst_stride], frame.data[0][src_offset..][0..dst_stride]);
                    }

                    _ = gl.UnmapBuffer(gl.PIXEL_UNPACK_BUFFER);
                }

                gl.TexSubImage2D(
                    gl.TEXTURE_2D,
                    0,
                    0,
                    0,
                    frame.width,
                    frame.height,
                    gl.RED,
                    gl.UNSIGNED_BYTE,
                    null,
                );
            }

            {
                gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, stable_buffer.pbo(.uv));
                defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

                gl.BindTexture(gl.TEXTURE_2D, stable_buffer.tex(.uv));
                defer gl.BindTexture(gl.TEXTURE_2D, 0);

                const pbo: ?[*]u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_UNPACK_BUFFER, gl.WRITE_ONLY));

                if (pbo) |ptr| {
                    // Copy line by line instead of a single large memcpy
                    const bytes_per_line: usize = @intCast(frame.linesize[1]);
                    const height: usize = @intCast(@divTrunc(frame.height, 2));

                    // Destination stride in the PBO may be different from the source
                    const dst_stride = @as(usize, @intCast(@divTrunc(frame.width, 2) * UV_BPP));

                    for (0..height) |y| {
                        const src_offset = y * bytes_per_line;
                        const dst_offset = y * dst_stride;

                        @memcpy(ptr[dst_offset..][0..dst_stride], frame.data[1][src_offset..][0..dst_stride]);
                    }

                    _ = gl.UnmapBuffer(gl.PIXEL_UNPACK_BUFFER);
                }

                gl.TexSubImage2D(
                    gl.TEXTURE_2D,
                    0,
                    0,
                    0,
                    @divTrunc(frame.width, 2),
                    @divTrunc(frame.height, 2),
                    gl.RG,
                    gl.UNSIGNED_BYTE,
                    null,
                );
            }

            try render(
                frame,
                &view,
                &stable_buffer,
                angle_calc,
                outer_blur,
                inner_blur,
                vao_id[0..],
                prog_id[0..],
                u_inv_scale,
                u_aspect,
                u_scale,
                .{ .inner = .{ @floatFromInt(w), @floatFromInt(h) } },
                radius * 1.05,
                circle_vertices.items.len,
                ring_vertices.items.len,
            );
        } else {
            // log.err("{}: video decode bottleneck", .{c.SDL_GetPerformanceCounter()}); // TODO: add adaptive sleeping here
            continue;
        }

        try errify(c.SDL_GL_SwapWindow(ui.window));
    }
}

const DoubleBuffer = struct {
    y: Setup,
    uv: Setup,

    display_times: [2]f64,

    current: u1,

    const Setup = struct {
        tex_id: [2]c_uint,
        pbo_id: [2]c_uint,

        pub fn init(comptime ch: Channel, width: usize, height: usize) Setup {
            const bytes_per_pixel = if (ch == .y) Y_BPP else UV_BPP;
            const internal_format = if (ch == .y) gl.R8 else gl.RG8;
            const format = if (ch == .y) gl.RED else gl.RG;
            const len: c_int = @intCast(width * height * bytes_per_pixel);

            var tex_id: [2]c_uint = undefined;
            gl.GenTextures(2, tex_id[0..]);

            for (tex_id) |id| {
                gl.BindTexture(gl.TEXTURE_2D, id);

                // TODO: confirm this should be bilinear
                gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
                gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
                gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
                gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

                gl.TexImage2D(
                    gl.TEXTURE_2D,
                    0,
                    internal_format,
                    @intCast(width),
                    @intCast(height),
                    0,
                    format,
                    gl.UNSIGNED_BYTE,
                    null,
                );
            }

            gl.BindTexture(gl.TEXTURE_2D, 0);

            return .{
                .tex_id = opengl_impl.vidTex(2, @intCast(width), @intCast(height)),
                .pbo_id = opengl_impl.pbo(2, len),
            };
        }

        fn deinit(self: *@This()) void {
            gl.DeleteTextures(2, self.tex_id[0..]);
            gl.DeleteBuffers(2, self.pbo_id[0..]);
        }
    };

    const Channel = enum { y, uv };

    const Inverted = struct {
        inner: DoubleBuffer,

        fn from(super: DoubleBuffer) Inverted {
            return .{
                .inner = .{
                    .y = super.y,
                    .uv = super.uv,
                    .display_times = super.display_times,
                    .current = super.current +% 1,
                },
            };
        }

        fn tex(self: @This(), comptime ch: Channel) c_uint {
            return switch (ch) {
                .y => self.inner.y.tex_id[self.inner.current],
                .uv => self.inner.uv.tex_id[self.inner.current],
            };
        }

        fn display_time(self: @This()) f64 {
            return self.inner.display_times[self.inner.current];
        }
    };

    pub fn init(width: usize, height: usize) DoubleBuffer {
        return .{
            .y = Setup.init(.y, width, height),
            .uv = Setup.init(.uv, width / 2, height / 2),
            .display_times = .{ 0.0, 0.0 },
            .current = 0,
        };
    }

    fn deinit(self: *@This()) void {
        self.y.deinit();
        self.uv.deinit();
    }

    fn tex(self: @This(), comptime ch: Channel) c_uint {
        return switch (ch) {
            .y => self.y.tex_id[self.current],
            .uv => self.uv.tex_id[self.current],
        };
    }

    fn pbo(self: @This(), comptime ch: Channel) c_uint {
        return switch (ch) {
            .y => self.y.pbo_id[self.current],
            .uv => self.uv.pbo_id[self.current],
        };
    }

    fn set_display_time(self: *@This(), in_seconds: f64) void {
        self.display_times[self.current] = in_seconds;
    }

    fn invert(self: @This()) Inverted {
        return Inverted.from(self);
    }

    fn swap(self: *@This()) void {
        self.current = self.current +% 1;
    }
};

const Id = enum(usize) {
    texture = 0,
    ring,
    circle,
    background,
    blur,
};

fn render(
    frame: *const c.AVFrame,
    view: *Viewport,
    stable_buffer: *DoubleBuffer,
    angle_calc: AngleCalc,
    outer_blur: [2]Blur,
    inner_blur: [2]Blur,
    vao_id: *const [4]c_uint,
    prog_id: *const [5]c_uint,
    u_inv_scale: Mat2,
    u_aspect: Mat2,
    u_scale: Mat2,
    u_viewport: Vec2,
    puck_radius: f32,
    circle_vertices_len: usize,
    ring_vertices_len: usize,
) !void {
    gl.ClearColor(0, 0, 0, 0);
    gl.Clear(gl.COLOR_BUFFER_BIT);

    const y_tex = stable_buffer.invert().tex(.y);
    const uv_tex = stable_buffer.invert().tex(.uv);

    const rad = angle_calc.execute(view, .{ y_tex, uv_tex });
    const u_rotation = mat2(@cos(rad), -@sin(rad), @sin(rad), @cos(rad));

    {
        blur(
            outer_blur,
            view,
            prog_id[@intFromEnum(Id.blur)],
            vao_id[@intFromEnum(Id.blur) - 1], // there is no Background VAO (it reuses the texture VAO) so Blur VAO is offset by one
            .{ y_tex, uv_tex },
            frame.width >> 1,
            frame.height >> 1,
            8,
        );

        blur(
            inner_blur,
            view,
            prog_id[@intFromEnum(Id.blur)],
            vao_id[@intFromEnum(Id.blur) - 1], // there is no Background VAO (it reuses the texture VAO) so Blur VAO is offset by one
            .{ y_tex, uv_tex },
            frame.width,
            frame.height,
            4,
        );

        gl.UseProgram(prog_id[@intFromEnum(Id.background)]);
        gl.BindVertexArray(vao_id[@intFromEnum(Id.texture)]);

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, outer_blur[0].tex); // guaranteed to be the last modified texture

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, inner_blur[0].tex); // guaranteed to be the last modified texture

        const u_transform = u_rotation.mul(u_inv_scale.mul(u_aspect));

        gl.UniformMatrix2fv(gl.GetUniformLocation(prog_id[@intFromEnum(Id.background)], "u_transform"), 1, gl.FALSE, &u_transform.inner);
        gl.Uniform1i(gl.GetUniformLocation(prog_id[@intFromEnum(Id.background)], "u_outer"), 0);
        gl.Uniform1i(gl.GetUniformLocation(prog_id[@intFromEnum(Id.background)], "u_inner"), 1);

        gl.Uniform2fv(gl.GetUniformLocation(prog_id[@intFromEnum(Id.background)], "u_viewport"), 1, &u_viewport.inner);
        gl.Uniform1f(gl.GetUniformLocation(prog_id[@intFromEnum(Id.background)], "u_radius"), u_scale.inner[0] * puck_radius);

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    {
        // Draw Transparent Puck
        gl.UseProgram(prog_id[@intFromEnum(Id.circle)]);
        gl.BindVertexArray(vao_id[@intFromEnum(Id.circle)]);

        gl.Uniform1f(gl.GetUniformLocation(prog_id[@intFromEnum(Id.circle)], "u_scale"), u_scale.inner[0]);
        gl.DrawArrays(gl.TRIANGLE_FAN, 0, @intCast(circle_vertices_len));

        // Draw Ring (matches ring in gameplay)
        gl.UseProgram(prog_id[@intFromEnum(Id.ring)]);
        gl.BindVertexArray(vao_id[@intFromEnum(Id.ring)]);

        gl.Uniform1f(gl.GetUniformLocation(prog_id[@intFromEnum(Id.ring)], "u_scale"), u_scale.inner[0]);
        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, @intCast(ring_vertices_len));
    }

    {
        gl.UseProgram(prog_id[@intFromEnum(Id.texture)]);
        defer gl.UseProgram(0);

        gl.BindVertexArray(vao_id[@intFromEnum(Id.texture)]);
        defer gl.BindVertexArray(0);

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, y_tex);

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, uv_tex);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        const u_transform = u_rotation.mul(u_scale.mul(u_aspect));

        gl.UniformMatrix2fv(gl.GetUniformLocation(prog_id[@intFromEnum(Id.texture)], "u_transform"), 1, gl.FALSE, &u_transform.inner);
        gl.Uniform1i(gl.GetUniformLocation(prog_id[@intFromEnum(Id.texture)], "u_y_tex"), 0);
        gl.Uniform1i(gl.GetUniformLocation(prog_id[@intFromEnum(Id.texture)], "u_uv_tex"), 1);

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
}

const DecodeContext = struct {
    bundle: AvBundle,
    fmt_ctx: *c.AVFormatContext,

    should_quit: *std.atomic.Value(bool),
};

const AvBundle = struct {
    codec_ctx: ?*c.AVCodecContext,
    stream: usize,

    hw: ?HwBundle = null,

    fn deinit(self: *@This()) void {
        if (self.hw) |*hw| hw.deinit();
        c.avcodec_free_context(&self.codec_ctx);
    }
};

fn decodeAudio(sample_queue: *c.SDL_AudioStream, clock: *AudioClock, decode_ctx: DecodeContext) !void {
    const log = std.log.scoped(.decode);

    log.info("audio decode thread start", .{});
    defer log.info("audio decode thread end", .{});

    var maybe_pkt: ?*c.AVPacket = c.av_packet_alloc();
    defer c.av_packet_free(&maybe_pkt);

    var maybe_src_frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(&maybe_src_frame);

    var maybe_dst_frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(&maybe_dst_frame);

    const pkt = maybe_pkt orelse return error.out_of_memory;
    const src_frame = maybe_src_frame orelse return error.out_of_memory;
    const dst_frame = maybe_dst_frame orelse return error.out_of_memory;

    const audio_ctx = decode_ctx.bundle.codec_ctx.?;

    _ = try libavError(c.av_channel_layout_copy(&dst_frame.ch_layout, &audio_ctx.ch_layout));
    dst_frame.sample_rate = audio_ctx.sample_rate;
    dst_frame.format = c.AV_SAMPLE_FMT_FLT;

    var maybe_swr = c.swr_alloc();
    defer c.swr_free(&maybe_swr);

    const swr = maybe_swr orelse return error.out_of_memory;

    var end_of_file = false;
    while (!decode_ctx.should_quit.load(.monotonic)) {
        // Try to receive a frame first if we have packets already in the decoder
        recv_loop: while (true) {
            switch (c.avcodec_receive_frame(audio_ctx, src_frame)) {
                0 => { // got a frame
                    defer c.av_frame_unref(src_frame);

                    const bytes_per_sec = clock.sample_rate * clock.bytes_per_sample * clock.channels;
                    const max_len = 1 * bytes_per_sec;

                    _ = try libavError(c.swr_convert_frame(swr, dst_frame, src_frame));

                    while (true) {
                        const queued_len = c.SDL_GetAudioStreamQueued(sample_queue);
                        if (queued_len < max_len) break;

                        std.Thread.sleep(5 * std.time.ns_per_ms);
                    }

                    const len = c.av_samples_get_buffer_size(null, dst_frame.ch_layout.nb_channels, dst_frame.nb_samples, dst_frame.format, 0);

                    _ = c.SDL_PutAudioStreamData(sample_queue, dst_frame.data[0], len);
                    _ = clock.bytes_sent.fetchAdd(@intCast(len), .monotonic);
                },
                c.AVERROR(c.EAGAIN) => break :recv_loop,
                c.AVERROR_EOF => return,
                else => return error.ffmpeg_error,
            }
        }

        if (end_of_file) return; // can be set by code below

        // Read the next packet
        switch (c.av_read_frame(decode_ctx.fmt_ctx, pkt)) {
            0...std.math.maxInt(c_int) => {
                defer c.av_packet_unref(pkt);

                if (pkt.stream_index != decode_ctx.bundle.stream) continue;

                const send_ret = c.avcodec_send_packet(audio_ctx, pkt);
                if (send_ret == c.AVERROR_EOF) return;
                if (send_ret == c.AVERROR(c.EAGAIN)) continue;
                if (send_ret < 0) return error.ffmpeg_error;
            },
            c.AVERROR_EOF => {
                defer end_of_file = true;

                // TODO: make this work with libavError
                const flush_ret = c.avcodec_send_packet(audio_ctx, null);
                if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) return error.ffmpeg_error;
            },
            else => return error.ffmpeg_error,
        }
    }
}

fn convertFrame(bundle: AvBundle, src_frame: *c.AVFrame, dst_frame: *c.AVFrame, sws: *c.SwsContext) !void {
    @setRuntimeSafety(false);

    if (bundle.hw) |hw| {
        std.debug.assert(src_frame.format == hw.pix_fmt);
        _ = try libavError(c.av_hwframe_transfer_data(dst_frame, src_frame, 0));
    } else {
        _ = try libavError(c.sws_scale_frame(sws, dst_frame, src_frame));
    }

    // timing
    dst_frame.pts = src_frame.pts;
    dst_frame.pkt_dts = src_frame.pkt_dts;
    dst_frame.best_effort_timestamp = src_frame.best_effort_timestamp;

    dst_frame.colorspace = src_frame.colorspace;
    dst_frame.color_range = src_frame.color_range;
    dst_frame.color_primaries = src_frame.color_primaries;
    dst_frame.color_trc = src_frame.color_trc;
}

fn decodeVideo(frame_queue: *FrameQueue, decode_ctx: DecodeContext) !void {
    const log = std.log.scoped(.video_decode);

    log.info("video decode thread start", .{});
    defer log.info("video decode thread end", .{});

    var maybe_pkt: ?*c.AVPacket = c.av_packet_alloc();
    defer c.av_packet_free(&maybe_pkt);

    var src_frame: AvFrame = try .init();
    defer src_frame.deinit();

    var dst_frame: AvFrame = try .init();
    defer dst_frame.deinit();

    const pkt = maybe_pkt orelse return error.out_of_memory;
    const video_ctx = decode_ctx.bundle.codec_ctx.?;

    try dst_frame.setup(video_ctx.width, video_ctx.height, c.AV_PIX_FMT_NV12);

    const maybe_sws = c.sws_getContext(
        video_ctx.width,
        video_ctx.height,
        video_ctx.pix_fmt,
        video_ctx.width,
        video_ctx.height,
        c.AV_PIX_FMT_NV12,
        c.SWS_FAST_BILINEAR,
        null,
        null,
        null,
    );
    defer c.sws_freeContext(maybe_sws);

    const sws = maybe_sws orelse return error.out_of_memory;

    // _ = try libavError(c.av_opt_set_int(sws, "threads", @intCast(try std.Thread.getCpuCount()), 0));

    var end_of_file = false;

    // TODO: using a fn ptr I think we can deduplicate this massive loop
    while (!decode_ctx.should_quit.load(.monotonic)) {
        // Try to receive a frame first if we have packets already in the decoder
        recv_loop: while (true) {
            switch (c.avcodec_receive_frame(video_ctx, src_frame.ptr())) {
                0 => {
                    defer c.av_frame_unref(dst_frame.ptr());

                    try convertFrame(decode_ctx.bundle, src_frame.ptr(), dst_frame.ptr(), sws);
                    try frame_queue.push(dst_frame.ptr());
                },
                c.AVERROR(c.EAGAIN) => break :recv_loop,
                c.AVERROR_EOF => return,
                else => return error.ffmpeg_error,
            }
        }

        if (end_of_file) return; // can be set by code below

        // Read the next packet
        switch (c.av_read_frame(decode_ctx.fmt_ctx, pkt)) {
            0...std.math.maxInt(c_int) => {
                defer c.av_packet_unref(pkt);

                if (pkt.stream_index != decode_ctx.bundle.stream) continue;

                const send_ret = c.avcodec_send_packet(video_ctx, pkt);
                if (send_ret == c.AVERROR_EOF) return;
                if (send_ret == c.AVERROR(c.EAGAIN)) continue;
                if (send_ret < 0) return error.ffmpeg_error;
            },
            c.AVERROR_EOF => {
                defer end_of_file = true;

                // TODO: make this work with libavError
                const flush_ret = c.avcodec_send_packet(video_ctx, null);
                if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) return error.ffmpeg_error;
            },
            else => return error.ffmpeg_error,
        }
    }
}

const Blur = struct {
    fbo: c_uint,
    tex: c_uint,

    fn deinit(self: @This()) void {
        var fbo_id: [1]c_uint = .{self.fbo};
        gl.DeleteFramebuffers(1, fbo_id[0..]);

        var tex_id: [1]c_uint = .{self.tex};
        gl.DeleteTextures(1, tex_id[0..]);
    }
};

// FIXME: this is the bottleneck of the main thread
fn blur(b: [2]Blur, view: *Viewport, prog: c_uint, src_vao: c_uint, src_tex: [2]c_uint, width: c_int, height: c_int, passes: u32) void {
    std.debug.assert(passes % 2 == 0);

    const fbo_cache: c_uint = blk: {
        var buf: [1]c_int = undefined;
        gl.GetIntegerv(gl.FRAMEBUFFER_BINDING, &buf);

        break :blk @intCast(buf[0]);
    };

    view.set(width, height);
    defer view.restore();

    gl.BindVertexArray(src_vao);

    gl.UseProgram(prog);
    defer gl.UseProgram(0);

    const y_tex, const uv_tex = .{ src_tex[0], src_tex[1] };

    gl.ActiveTexture(gl.TEXTURE1);
    gl.BindTexture(gl.TEXTURE_2D, y_tex);

    gl.ActiveTexture(gl.TEXTURE2);
    gl.BindTexture(gl.TEXTURE_2D, uv_tex);

    gl.Uniform2f(gl.GetUniformLocation(prog, "u_resolution"), @floatFromInt(width), @floatFromInt(height));
    gl.Uniform1i(gl.GetUniformLocation(prog, "u_screen"), 0);
    gl.Uniform1i(gl.GetUniformLocation(prog, "u_y_tex"), 1);
    gl.Uniform1i(gl.GetUniformLocation(prog, "u_uv_tex"), 2);

    const horiz_loc = gl.GetUniformLocation(prog, "u_horizontal");
    const use_nv12_loc = gl.GetUniformLocation(prog, "u_use_nv12");

    for (0..passes) |i| {
        const current = b[i % 2];
        const other = b[1 - i % 2];

        gl.BindFramebuffer(gl.FRAMEBUFFER, current.fbo);
        gl.ActiveTexture(gl.TEXTURE0);

        gl.Uniform1i(horiz_loc, @intFromBool(i % 2 == 0));

        switch (i) {
            0 => gl.Uniform1i(use_nv12_loc, @intFromBool(true)),
            else => gl.BindTexture(gl.TEXTURE_2D, other.tex),
        }
        defer if (i == 0) gl.Uniform1i(use_nv12_loc, @intFromBool(false));

        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }

    defer gl.BindFramebuffer(gl.FRAMEBUFFER, fbo_cache);
    defer gl.BindTexture(gl.TEXTURE_2D, 0);
    defer gl.BindVertexArray(0);
}

const opengl_impl = struct {
    fn vao(n: comptime_int) [n]c_uint {
        var ids: [n]c_uint = undefined;
        gl.GenVertexArrays(n, ids[0..]);

        return ids;
    }

    fn vbo(n: comptime_int) [n]c_uint {
        var ids: [n]c_uint = undefined;
        gl.GenBuffers(n, ids[0..]);

        return ids;
    }

    fn pbo(n: comptime_int, len: c_int) [n]c_uint {
        var ids: [n]c_uint = undefined;
        gl.GenBuffers(2, ids[0..]);

        for (ids) |pbo_id| {
            gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, pbo_id);
            defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

            gl.BufferData(gl.PIXEL_UNPACK_BUFFER, len, null, gl.STREAM_DRAW);
        }

        return ids;
    }

    fn setupBlur(width: c_int, height: c_int) [2]Blur {
        var fbo_ids: [2]c_uint = undefined;
        gl.GenFramebuffers(2, &fbo_ids);

        var tex_ids: [2]c_uint = undefined;
        gl.GenTextures(2, &tex_ids);

        for (fbo_ids, tex_ids) |fbo, tex| {
            gl.BindFramebuffer(gl.FRAMEBUFFER, fbo);
            defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

            gl.BindTexture(gl.TEXTURE_2D, tex);
            defer gl.BindTexture(gl.TEXTURE_2D, 0);

            gl.TexImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGB,
                width,
                height,
                0,
                gl.RGB,
                gl.UNSIGNED_BYTE,
                null,
            );

            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);

            const ret = gl.CheckFramebufferStatus(gl.FRAMEBUFFER);
            if (ret != gl.FRAMEBUFFER_COMPLETE) @panic("FIXME: Framebuffer incomplete");
        }

        return [_]Blur{ .{ .fbo = fbo_ids[0], .tex = tex_ids[0] }, .{ .fbo = fbo_ids[1], .tex = tex_ids[1] } };
    }

    fn vidTex(n: comptime_int, width: c_int, height: c_int) [n]c_uint {
        var ids: [n]c_uint = undefined;
        gl.GenTextures(n, ids[0..]);

        for (ids) |tex_id| {
            gl.BindTexture(gl.TEXTURE_2D, tex_id);

            // FIXME: gl.LINEAR looks nicer but I think for the angle sampling we must do gl.NEAREST for correctness
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            gl.TexImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGB,
                width,
                height,
                0,
                gl.RGB,
                gl.UNSIGNED_BYTE,
                null,
            );
        }

        gl.BindTexture(gl.TEXTURE_2D, 0);

        return ids;
    }

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

// https://github.com/castholm/zig-examples/blob/77a829c85b5ddbad673026d504626015db4093ac/opengl-sdl/main.zig#L200-L219
inline fn errify(value: anytype) error{sdl_error}!switch (@typeInfo(@TypeOf(value))) {
    .bool => void,
    .pointer, .optional => @TypeOf(value.?),
    .int => |info| switch (info.signedness) {
        .signed => @TypeOf(@max(0, value)),
        .unsigned => @TypeOf(value),
    },
    else => @compileError("unerrifiable type: " ++ @typeName(@TypeOf(value))),
} {
    return switch (@typeInfo(@TypeOf(value))) {
        .bool => if (!value) error.sdl_error,
        .pointer, .optional => value orelse error.sdl_error,
        .int => |info| switch (info.signedness) {
            .signed => if (value >= 0) @max(0, value) else error.sdl_error,
            .unsigned => if (value != 0) value else error.sdl_error,
        },
        else => comptime unreachable,
    };
}

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

const ContextKind = enum { video, audio };

fn createCodecContext(comptime kind: ContextKind, fmt_ctx: *c.AVFormatContext) !AvBundle {
    const av_type = switch (kind) {
        .video => c.AVMEDIA_TYPE_VIDEO,
        .audio => c.AVMEDIA_TYPE_AUDIO,
    };

    var codec_ptr: ?*const c.AVCodec = null;
    const stream = try libavError(c.av_find_best_stream(fmt_ctx, av_type, -1, -1, &codec_ptr, 0));

    var ctx_ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec_ptr);
    errdefer c.avcodec_free_context(&ctx_ptr);

    _ = try libavError(c.avcodec_parameters_to_context(ctx_ptr, fmt_ctx.streams[@intCast(stream)].*.codecpar));
    _ = try libavError(c.avcodec_open2(ctx_ptr, codec_ptr, null));

    return .{ .codec_ctx = ctx_ptr, .stream = @intCast(stream) };
}

const HwBundle = struct {
    device_ctx: ?*c.AVBufferRef,
    pix_fmt: c.AVPixelFormat,

    /// Must be called immediately after creation, sets a self-referential ptr in self.codec_ctx.@"opaque"
    ///
    /// Also sets the get_format fn pointer in AVCodecContext
    fn configure(self: *@This()) void {
        const super: *AvBundle = @fieldParentPtr("hw", @as(*?HwBundle, @ptrCast(self)));

        super.codec_ctx.?.@"opaque" = @ptrCast(self);
        super.codec_ctx.?.get_format = @This().getFormat;
    }

    fn deinit(self: *@This()) void {
        c.av_buffer_unref(&self.device_ctx);
    }

    fn getFormat(ctx: ?*c.AVCodecContext, formats: [*c]const c.AVPixelFormat) callconv(.c) c.AVPixelFormat {
        const self: *const @This() = @alignCast(@ptrCast(ctx.?.@"opaque"));

        const fmts = std.mem.sliceTo(formats, c.AV_PIX_FMT_NONE);
        for (fmts) |fmt| {
            // if (hw_pix_fmt == null) break; TODO: maybe i still need this?
            if (fmt == self.pix_fmt) return self.pix_fmt;
        }

        return c.AV_PIX_FMT_NONE;
    }
};

fn createHwAccelCodecContext(comptime kind: ContextKind, fmt_ctx: *c.AVFormatContext, device_type: c.AVHWDeviceType) !AvBundle {
    comptime std.debug.assert(kind == .video);
    const log = std.log.scoped(.hwdevice);

    var codec_ptr: ?*const c.AVCodec = null;
    const stream = try libavError(c.av_find_best_stream(fmt_ctx, c.AVMEDIA_TYPE_VIDEO, -1, -1, &codec_ptr, 0));

    var ctx_ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec_ptr);
    errdefer c.avcodec_free_context(&ctx_ptr);

    _ = try libavError(c.avcodec_parameters_to_context(ctx_ptr, fmt_ctx.streams[@intCast(stream)].*.codecpar));

    var hw_device_ctx: ?*c.AVBufferRef = null;
    var hw_pix_fmt: c.AVPixelFormat = undefined;

    {
        var i: c_int = 0;
        while (true) {
            defer i += 1;

            const config: ?*const c.AVCodecHWConfig = c.avcodec_get_hw_config(codec_ptr.?, i);
            if (config == null) std.debug.panic("decoder {s} does not support {s}", .{ codec_ptr.?.name, c.av_hwdevice_get_type_name(device_type) });

            if (config.?.methods & c.AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX == 0) continue;
            if (config.?.device_type != device_type) continue;

            hw_pix_fmt = config.?.pix_fmt;
            log.info("found {s} hwdevice for {s} decoder", .{ c.av_hwdevice_get_type_name(device_type), codec_ptr.?.name });

            // FIXME: does this need to be deallocated????
            _ = try libavError(c.av_hwdevice_ctx_create(&hw_device_ctx, device_type, null, null, 0));
            ctx_ptr.?.hw_device_ctx = c.av_buffer_ref(hw_device_ctx);

            {
                var constraints: ?*c.AVHWFramesConstraints = c.av_hwdevice_get_hwframe_constraints(hw_device_ctx, null);
                defer c.av_hwframe_constraints_free(&constraints);

                log.debug("valid sw formats:", .{});
                const valid_sw = std.mem.sliceTo(constraints.?.valid_sw_formats, c.AV_SAMPLE_FMT_NONE);
                for (valid_sw) |fmt| log.debug("\t{s}", .{c.av_get_pix_fmt_name(fmt)});

                log.debug("valid hw formats:", .{});
                const valid_hw = std.mem.sliceTo(constraints.?.valid_hw_formats, c.AV_SAMPLE_FMT_NONE);
                for (valid_hw) |fmt| log.debug("\t{s}", .{c.av_get_pix_fmt_name(fmt)});
            }

            break;
        }
    }

    _ = try libavError(c.avcodec_open2(ctx_ptr, codec_ptr, null));

    return .{
        .codec_ctx = ctx_ptr,
        .stream = @intCast(stream),
        .hw = .{
            .pix_fmt = hw_pix_fmt,
            .device_ctx = hw_device_ctx,
        },
    };
}

const AvFormatContext = struct {
    inner: ?*c.AVFormatContext,

    fn ptr(self: @This()) *c.AVFormatContext {
        return self.inner.?;
    }

    fn deinit(self: *@This()) void {
        c.avformat_close_input(&self.inner);
    }
};

fn createFormatContext(path: []const u8) !AvFormatContext {
    var ptr: ?*c.AVFormatContext = c.avformat_alloc_context();
    errdefer c.avformat_close_input(&ptr);

    _ = try libavError(c.avformat_open_input(&ptr, path.ptr, null, null));
    const fmt_ctx = ptr orelse return error.out_of_memory;

    _ = try libavError(c.avformat_find_stream_info(fmt_ctx, null));

    return .{ .inner = ptr };
}

const Ui = struct {
    window: *c.SDL_Window,
    gl_ctx: c.SDL_GLContext,

    fn deinit(self: @This()) void {
        gl.makeProcTableCurrent(null);
        _ = c.SDL_GL_MakeCurrent(self.window, null);
        _ = c.SDL_GL_DestroyContext(self.gl_ctx);
        c.SDL_DestroyWindow(self.window);
    }
};

fn createSdlWindow(width: u32, height: u32) !Ui {
    c.SDL_SetMainReady();
    try errify(c.SDL_Init(c.SDL_INIT_AUDIO | c.SDL_INIT_VIDEO));

    try errify(c.SDL_SetAppMetadata("Rotaeno Stabilizer", "0.1.0", "moe.paoda.rota-stabilizer"));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_MULTISAMPLEBUFFERS, 1));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_MULTISAMPLESAMPLES, 4));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_ALPHA_SIZE, 8));

    const window: *c.SDL_Window = try errify(c.SDL_CreateWindow("Rotaeno Stabilizer", @intCast(width), @intCast(height), c.SDL_WINDOW_OPENGL));
    errdefer c.SDL_DestroyWindow(window);

    const gl_ctx = try errify(c.SDL_GL_CreateContext(window));
    errdefer errify(c.SDL_GL_DestroyContext(gl_ctx)) catch {};

    try errify(c.SDL_GL_MakeCurrent(window, gl_ctx));
    errdefer errify(c.SDL_GL_MakeCurrent(window, null)) catch {};

    if (!gl_procs.init(c.SDL_GL_GetProcAddress)) return error.gl_init_failed;

    gl.makeProcTableCurrent(&gl_procs);
    errdefer gl.makeProcTableCurrent(null);

    return .{ .window = window, .gl_ctx = gl_ctx };
}

const SdlAudioStream = struct {
    inner: *c.SDL_AudioStream,

    fn deinit(self: @This()) void {
        c.SDL_DestroyAudioStream(self.inner);
    }
};

fn createSdlAudioStream(ctx: *const c.AVCodecContext) !SdlAudioStream {
    var desired: c.SDL_AudioSpec = std.mem.zeroes(c.SDL_AudioSpec);
    desired.freq = ctx.sample_rate;
    desired.format = c.SDL_AUDIO_F32;
    desired.channels = ctx.ch_layout.nb_channels;

    const stream = try errify(c.SDL_OpenAudioDeviceStream(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired, null, null));
    errdefer c.SDL_DestroyAudioStream(stream);

    return .{ .inner = stream };
}

const Vec2 = struct { inner: [2]f32 };

const Mat2 = struct {
    inner: [4]f32,

    const identity: @This() = .{ .inner = .{ 1, 0, 0, 1 } };

    fn scale(s: f32) @This() {
        return .{ .inner = .{ s, 0, 0, s } };
    }

    fn scaleXy(sx: f32, sy: f32) @This() {
        return .{ .inner = .{ sx, 0, 0, sy } };
    }

    fn mul(left: @This(), right: @This()) @This() {
        const l = left.inner;
        const r = right.inner;

        return .{
            .inner = .{
                l[0] * r[0] + l[2] * r[1],
                l[1] * r[0] + l[3] * r[1],
                l[0] * r[2] + l[2] * r[3],
                l[1] * r[2] + l[3] * r[3],
            },
        };
    }
};

const AudioClock = struct {
    bytes_sent: std.atomic.Value(u64) = .init(0),
    start_time: u64,

    sample_rate: u16,
    channels: u8,
    bytes_per_sample: u32,

    is_muted: bool,

    stream: SdlAudioStream,

    const log = std.log.scoped(.audio_clock);

    pub fn init(stream: SdlAudioStream, sample_rate: u16, channels: u8, fmt: c.AVSampleFormat) @This() {
        defer _ = c.SDL_ResumeAudioStreamDevice(stream.inner);

        const ret = c.SDL_SetAudioStreamGain(stream.inner, if (muted_by_default) 0.0 else 1.0);
        if (!ret) log.err("failed to set audio gain on init", .{});

        return .{
            .start_time = c.SDL_GetPerformanceCounter(),
            .sample_rate = sample_rate,
            .channels = channels,
            .bytes_per_sample = @intCast(c.av_get_bytes_per_sample(fmt)),

            .is_muted = muted_by_default,
            .stream = stream,
        };
    }

    fn seconds_passed(self: @This()) f64 {
        const bytes_per_sec: f64 = @floatFromInt(self.bytes_per_sample * self.sample_rate * self.channels);

        const queued: f64 = @floatFromInt(c.SDL_GetAudioStreamQueued(self.stream.inner));
        const bytes_sent: f64 = @floatFromInt(self.bytes_sent.load(.monotonic));

        const pos = bytes_sent - queued;
        return pos / bytes_per_sec;
    }
};

fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) Mat2 {
    return .{ .inner = .{ m00, m01, m10, m11 } };
}

const AvFrame = struct {
    inner: ?*c.AVFrame,

    pub fn init() !AvFrame {
        const p: ?*c.AVFrame = c.av_frame_alloc();
        if (p == null) return error.out_of_memory;

        return .{ .inner = p };
    }

    pub fn setup(self: *@This(), width: c_int, height: c_int, fmt: c.AVPixelFormat) !void {
        self.inner.?.width = width;
        self.inner.?.height = height;
        self.inner.?.format = fmt;

        _ = try libavError(c.av_frame_get_buffer(self.inner, 32));
    }

    pub fn deinit(self: *@This()) void {
        c.av_frame_free(&self.inner);
    }

    pub fn ptr(self: @This()) *c.AVFrame {
        return self.inner.?;
    }
};

const Viewport = struct {
    width: c_int,
    height: c_int,

    cached: ?[2]c_int = null,

    fn init(width: c_int, height: c_int) Viewport {
        gl.Viewport(0, 0, width, height);

        return .{ .width = width, .height = height };
    }

    fn set(self: *@This(), width: c_int, height: c_int) void {
        gl.Viewport(0, 0, width, height);
        self.cached = .{ self.width, self.height };

        self.width = width;
        self.height = height;
    }

    fn restore(self: *@This()) void {
        if (self.cached == null) @panic("viewport manager state corrupted");

        const view = self.cached.?;
        gl.Viewport(0, 0, view[0], view[1]);

        self.width = view[0];
        self.height = view[1];
        self.cached = null;
    }
};

const AngleCalc = struct {
    vao: [1]c_uint,
    fbo: [1]c_uint,
    tex: [1]c_uint,
    prog: c_uint,

    u_dimension: [2]f32,

    const log = std.log.scoped(.angle_calc);

    pub fn init(tex_width: c_int, tex_height: c_int) !AngleCalc {
        var fbo_id: [1]c_uint = undefined;
        gl.GenFramebuffers(1, fbo_id[0..]);

        var vao_id: [1]c_uint = undefined;
        gl.GenVertexArrays(1, vao_id[0..]);

        var tex_id: [1]c_uint = undefined;
        gl.GenTextures(1, tex_id[0..]);

        gl.BindFramebuffer(gl.FRAMEBUFFER, fbo_id[0]);
        defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        gl.BindTexture(gl.TEXTURE_2D, tex_id[0]);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        const program = try opengl_impl.program("./shader/blur.vert", "./shader/rotation.frag");

        gl.TexImage2D(gl.TEXTURE_2D, 0, gl.R32F, 1, 1, 0, gl.RED, gl.FLOAT, null);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        gl.FramebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex_id[0], 0);

        const ret = gl.CheckFramebufferStatus(gl.FRAMEBUFFER);
        if (ret != gl.FRAMEBUFFER_COMPLETE) @panic("FIXME: Framebuffer incomplete");

        return .{
            .vao = vao_id,
            .fbo = fbo_id,
            .tex = tex_id,
            .prog = program,
            .u_dimension = .{ @floatFromInt(tex_width), @floatFromInt(tex_height) },
        };
    }

    pub fn execute(self: @This(), view: *Viewport, tex_id: [2]c_uint) f32 {
        view.set(1, 1);
        defer view.restore();

        gl.BindVertexArray(self.vao[0]);
        defer gl.BindVertexArray(0);

        gl.BindFramebuffer(gl.FRAMEBUFFER, self.fbo[0]);
        defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        const y_tex, const uv_tex = .{ tex_id[0], tex_id[1] };

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, y_tex);

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, uv_tex);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.UseProgram(self.prog);
        defer gl.UseProgram(0);

        gl.Uniform1i(gl.GetUniformLocation(self.prog, "u_y_tex"), 0);
        gl.Uniform1i(gl.GetUniformLocation(self.prog, "u_uv_tex"), 1);
        gl.Uniform2f(gl.GetUniformLocation(self.prog, "u_dimension"), self.u_dimension[0], self.u_dimension[1]);

        gl.DrawArrays(gl.TRIANGLES, 0, 3);

        // FIXME: massive bottleneck
        var angle: f32 = 0;
        gl.ReadPixels(0, 0, 1, 1, gl.RED, gl.FLOAT, &angle);

        return angle;
    }

    pub fn deinit(self: *@This()) void {
        gl.DeleteFramebuffers(1, self.fbo[0..]);
        gl.DeleteTextures(1, self.tex[0..]);
        gl.DeleteVertexArrays(1, self.vao[0..]);
        gl.DeleteProgram(self.prog);
    }
};
