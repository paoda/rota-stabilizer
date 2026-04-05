const std = @import("std");
const gl = @import("gl");
const clap = @import("clap");
const tracy = @import("tracy");

const c = @import("lib.zig").c;
const platform = @import("lib/platform.zig");
const signal = @import("lib/platform.zig").signal;

const Encoder = @import("lib/codec.zig").Encoder;
const Decoder = @import("lib/codec.zig").Decoder;

const GpuResourceManager = @import("lib.zig").GpuResourceManager;
const BlurManager = @import("lib.zig").BlurManager;

const Mat2 = @import("lib/math.zig").Mat2;
const Vec2 = @import("lib/math.zig").Vec2;
const mat2 = @import("lib/math.zig").mat2;
const vec2 = @import("lib/math.zig").vec2;

const sleep = @import("lib.zig").sleep;

const RGB24_BPP = @import("lib.zig").RGB24_BPP;

const Y_BPP = @import("lib.zig").Y_BPP;
const UV_BPP = @import("lib.zig").UV_BPP;

const magic_aspect_ratio = @import("lib.zig").magic_aspect_ratio;

pub const tracy_impl = @import("tracy_impl"); // configured from build.zig
pub const tracy_options: tracy.Options = .{ .default_callstack_depth = 5 };

pub fn main() !void {
    defer signal.should_quit.store(true, .monotonic);

    const log = std.log.scoped(.main);
    errdefer |err| if (err == error.sdl_error) log.err("SDL Error: {s}", .{c.SDL_GetError()});

    signal.setupHandler();

    var gpa: std.heap.DebugAllocator(.{}) = .{ .backing_allocator = std.heap.c_allocator };
    defer std.debug.assert(gpa.deinit() == .ok);

    var tracy_alloc: tracy.Allocator = .{ .parent = gpa.allocator() };
    const allocator = tracy_alloc.allocator();

    const params = comptime clap.parseParamsComptime(
        \\-h, --help            Display this help and exit.
        \\-i, --input <str>     Path to input file.
        \\<str>                 Path to output file.
        \\
    );

    var diag: clap.Diagnostic = .{};
    var cli = clap.parse(clap.Help, &params, clap.parsers.default, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| return diag.reportToFile(.stderr(), err);
    defer cli.deinit();

    if (cli.args.help != 0) return clap.helpToFile(.stdout(), clap.Help, &params, .{});

    const ui = if (cli.positionals[0]) |_| try platform.createHeadless(1920, 1080) else try platform.createWindow(1600, 900);
    defer ui.deinit();

    const src_path = cli.args.input orelse return error.missing_input_path;

    const hw_decode, const hw_encode = platform.guessHardware();
    log.debug("guessed {s} hw decode and {s} hw encode support", .{
        if (hw_decode) |t| std.mem.span(c.av_hwdevice_get_type_name(t)) else "no",
        if (hw_encode) |t| std.mem.span(c.av_hwdevice_get_type_name(t)) else "no",
    });

    var decoder = try Decoder.init(allocator, &signal.should_quit, hw_decode, src_path, cli.positionals[0] != null);
    defer decoder.deinit(allocator);

    // -- opengl --
    log.info("OpenGL device: {?s}", .{gl.GetString(gl.RENDERER)});
    log.info("OpenGL support (want 3.3): {?s}", .{gl.GetString(gl.VERSION)});

    gl.Enable(gl.BLEND);
    gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.Disable(gl.FRAMEBUFFER_SRGB);
    _ = c.SDL_GL_SetSwapInterval(0);

    var res = try GpuResourceManager.init(allocator, decoder.dimensions);
    defer res.deinit(allocator);

    var stable_buffer = DoubleBuffer.init(res);

    const w, const h = try ui.windowSize();
    var camera = Camera.init(decoder.dimensions, w, h, decoder.colour_space);
    var view = Viewport.init(w, h);

    const angle_calc = try AngleCalc.init(res, camera);
    // -- opengl end --

    const frame_rate = decoder.framerate();
    const frame_period = 1.0 / c.av_q2d(frame_rate);

    const should_write = try checkFile(cli.positionals[0]);
    if (!should_write) return error.wont_overwrite;

    const handles = try decoder.spawn(cli.positionals[0]);
    defer handles.deinit();

    if (cli.positionals[0]) |dst_path| {
        // Initialize encoder
        var encoder = try Encoder.init(.{
            .width = @intCast(view.width),
            .height = @intCast(view.height),
            .decoder = &decoder,
        }, hw_encode, dst_path);
        defer encoder.deinit();

        const render_start_time = c.SDL_GetPerformanceCounter();
        const perf_freq: f64 = @floatFromInt(c.SDL_GetPerformanceFrequency());

        const video_stream = decoder.stream(.video);
        const audio_stream = decoder.stream(.audio);

        const estimated_frames: usize = blk: {
            if (video_stream.nb_frames > 0) break :blk @intCast(video_stream.nb_frames);
            if (decoder.fmt_ctx.ptr().duration <= 0) break :blk 0;

            // estimate from duration
            const duration_secs = @as(f64, @floatFromInt(decoder.fmt_ctx.ptr().duration)) / @as(f64, c.AV_TIME_BASE);
            break :blk @intFromFloat(duration_secs * c.av_q2d(frame_rate));
        };

        // Initialize progress
        var progress_buffer: [256]u8 = undefined;
        const progress_node = std.Progress.start(.{
            .root_name = "Encoding",
            .estimated_total_items = estimated_frames,
            .draw_buffer = &progress_buffer,
        });
        defer progress_node.end();

        try res.setupEncodingTargets(@intCast(view.width), @intCast(view.height));

        var current_pbo: u1 = 0;
        var pending_frame_pts: ?i64 = null; // PTS of frame in the "previous" PBO waiting to be encoded
        var frame_count: u64 = 0;

        while (!signal.should_quit.load(.monotonic)) {
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "encode loop" });
            defer zone.end();

            // Process any pending audio packets (remux to output)
            while (decoder.queue.pkt.audio.tryPop()) |audio_pkt| {
                try encoder.writeAudioPacket(audio_stream, audio_pkt);

                var pkt: ?*c.AVPacket = audio_pkt;
                c.av_packet_free(&pkt);
            }

            if (decoder.queue.frame.pop()) |frame| {
                defer decoder.queue.frame.recycle(frame);
                defer stable_buffer.swap();
                defer frame_count += 1; // increment after frame is sent to the GPU

                const z = tracy.Zone.begin(.{ .src = @src(), .name = "frame received" });
                defer z.end();

                uploadPlane(stable_buffer, frame, .y);
                uploadPlane(stable_buffer, frame, .uv);

                try render(&view, &stable_buffer, angle_calc, res, camera);
                writeToNv12Tex(res, &view, camera);

                {
                    const pbo_z = tracy.Zone.begin(.{ .src = @src(), .name = "read from opengl" });
                    defer pbo_z.end();

                    const idx = current_pbo;

                    // read Y and UV into single contiguous PBO
                    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, res.pbo.get(if (idx == 1) .dl_front else .dl_back));
                    defer gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);

                    {
                        const read_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.ReadPixels y" });
                        defer read_z.end();

                        gl.BindFramebuffer(gl.READ_FRAMEBUFFER, res.fbo.get(.y));
                        gl.ReadPixels(0, 0, view.width, view.height, gl.RED, gl.UNSIGNED_BYTE, @ptrFromInt(0));
                    }

                    {
                        const read_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.ReadPixels uv" });
                        defer read_z.end();

                        const uv_offset: usize = @intCast(view.width * view.height);
                        gl.BindFramebuffer(gl.READ_FRAMEBUFFER, res.fbo.get(.uv));
                        gl.ReadPixels(0, 0, @divTrunc(view.width, 2), @divTrunc(view.height, 2), gl.RG, gl.UNSIGNED_BYTE, @ptrFromInt(uv_offset));
                    }

                    gl.BindFramebuffer(gl.READ_FRAMEBUFFER, 0);

                    {
                        const flush_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.Flush" });
                        defer flush_z.end();

                        gl.Flush(); // trigger readback immediately
                    }

                    // encode buffered frame
                    if (pending_frame_pts) |pts| blk: {
                        const y, const uv = mapNv12Frame(res, view, current_pbo) orelse break :blk;
                        defer unmapNv12Frame(res, current_pbo);

                        try encoder.encodeNv12Frame(y, uv, pts);
                    }
                }

                // Store current frame's PTS for next iteration, swap PBOs
                pending_frame_pts = frame.best_effort_timestamp;
                current_pbo +%= 1;

                // Update progress
                progress_node.setCompletedItems(frame_count);
            } else if (decoder.queue.frame.end_of_stream.load(.monotonic)) {
                defer signal.should_quit.store(true, .monotonic); // exit program once flush is done
                defer std.Progress.setStatus(.success);

                const z = tracy.Zone.begin(.{ .src = @src(), .name = "final frame received" });
                defer z.end();

                // encode buffered frame
                if (pending_frame_pts) |pts| blk: {
                    const y, const uv = mapNv12Frame(res, view, current_pbo) orelse break :blk;
                    defer unmapNv12Frame(res, current_pbo);

                    try encoder.encodeNv12Frame(y, uv, pts);
                }

                // Calculate and display final stats
                const elapsed: f64 = @as(f64, @floatFromInt(c.SDL_GetPerformanceCounter() - render_start_time)) / perf_freq;
                const video_duration = @as(f64, @floatFromInt(frame_count)) / c.av_q2d(frame_rate);
                const speed = video_duration / elapsed;

                break log.info("Finished rendering {} frames in {d:.1}s ({d:.2}x realtime)", .{ frame_count, elapsed, speed });
            }
        }
    } else {
        // ===== PLAYBACK MODE =====
        const audio_clock = &(decoder.audio_clock orelse return error.uninitialized_audio_clock);
        const time_base = c.av_q2d(decoder.stream(.video).time_base);

        const start_time = @as(f64, @floatFromInt(decoder.stream(.video).start_time)) * time_base;
        log.debug("video start time: {d}s", .{start_time});

        try audio_clock.start(start_time);

        const delay_threshold = 0.020;

        log.debug("frame period: {d:.2}ms ({d:.2}fps)", .{ frame_period * std.time.ms_per_s, c.av_q2d(frame_rate) });
        log.debug("delay_threshold: {d:.2}ms", .{delay_threshold * std.time.ms_per_s});

        tracy.plotConfig(.{ .name = "A/V Sync Drift (ms)" });

        var next_frame: ?*c.AVFrame = null;
        var should_render: bool = true;

        var buf: [0x20]u8 = undefined;

        while (!signal.should_quit.load(.monotonic)) {
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "ui loop" });
            defer zone.end();

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "query input" });
                defer z.end();

                var event: c.SDL_Event = undefined;
                while (c.SDL_PollEvent(&event)) {
                    switch (event.type) {
                        c.SDL_EVENT_QUIT => {
                            signal.should_quit.store(true, .monotonic);
                            should_render = true;
                        },
                        c.SDL_EVENT_MOUSE_WHEEL => {
                            const wheel_delta = event.wheel.y * 0.1;
                            camera.adjustZoom(wheel_delta);
                            log.debug("Zoom: {d:.2}x", .{camera.zoom});

                            should_render = true;
                        },
                        c.SDL_EVENT_WINDOW_RESIZED => {
                            view = Viewport.init(event.window.data1, event.window.data2);
                            camera.updateWindow(view.width, view.height);

                            should_render = true;
                        },
                        c.SDL_EVENT_KEY_UP => switch (event.key.scancode) {
                            c.SDL_SCANCODE_M => {
                                try if (audio_clock.is_muted) audio_clock.unmute() else audio_clock.mute();
                                should_render = true;
                            },
                            c.SDL_SCANCODE_F11 => {
                                try ui.toggleFullscreen();
                                should_render = true;
                            },
                            else => {},
                        },
                        else => {},
                    }
                }
            }

            if (next_frame == null) {
                tracy.frameMarkStart("video timing");
                next_frame = decoder.queue.frame.tryPop();
            }

            if (next_frame) |frame| blk: {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "have next frame" });
                defer z.end();

                std.debug.assert(frame.format == c.AV_PIX_FMT_NV12);

                const back_buffer = stable_buffer.invert();
                const next_frame_time = @as(f64, @floatFromInt(frame.best_effort_timestamp)) * time_base;
                const already_uploaded = @abs(next_frame_time - back_buffer.display_time()) < std.math.floatEps(f64);

                if (!already_uploaded) {
                    tracy.plot(.{ .name = "Buffered Frames", .value = .{ .i64 = @intCast(decoder.queue.frame.len()) } });

                    const upload_z = tracy.Zone.begin(.{ .src = @src(), .name = "upload next frame" });
                    defer upload_z.end();

                    // this is a new frame, upload it to the BACK buffer
                    uploadPlane(back_buffer, frame, .y);
                    uploadPlane(back_buffer, frame, .uv);

                    stable_buffer.display_times[back_buffer.current] = next_frame_time;
                }

                // after upload, to account for it maybe taking a while
                const audio_time = audio_clock.seconds_passed();

                if (next_frame_time - audio_time < -delay_threshold) {
                    tracy.message(.{ .text = "dropped frame" });
                    tracy.frameMarkEnd("video timing");

                    decoder.queue.frame.recycle(frame);
                    next_frame = null;
                    break :blk;
                }

                const lead_time = 0; // TODO: 1ms for upload and rende
                const diff_s = next_frame_time - audio_time - lead_time;

                if (diff_s <= 0) { // Time to display the newer frame!
                    tracy.plot(.{ .name = "Audio Time (s)", .value = .{ .f64 = audio_time } });
                    tracy.plot(.{ .name = "Next Frame Time (s)", .value = .{ .f64 = next_frame_time } });
                    tracy.plot(.{ .name = "A/V Sync Drift (ms)", .value = .{ .f64 = (diff_s + lead_time) * std.time.ms_per_s } });
                    tracy.frameMarkEnd("video timing");

                    stable_buffer.swap();

                    decoder.queue.frame.recycle(frame);
                    next_frame = null;
                    should_render = true;
                } else {
                    const wait_z = tracy.Zone.begin(.{ .src = @src(), .name = "wait for next frame", .color = .gray25 });
                    defer wait_z.end();

                    const sleep_s = @min(diff_s, 0.005); // wake up slightly earlier than max theoretical
                    wait_z.text(try std.fmt.bufPrint(&buf, "expected to wait {d:.2}ms", .{sleep_s * std.time.ms_per_s}));

                    sleep(@intFromFloat(sleep_s * std.time.ns_per_s));
                }
            } else {
                // no next frame
                tracy.message(.{ .text = "FrameQueue starved!" });
                sleep(1 * std.time.ns_per_ms);
            }

            if (should_render) {
                try render(&view, &stable_buffer, angle_calc, res, camera);
                try ui.swap();

                should_render = false;
            }
        }
    }
}

const DoubleBuffer = struct {
    res: *const GpuResourceManager,

    display_times: [2]f64,
    current: u1,

    const Channel = enum { y, uv };

    pub fn init(res: *const GpuResourceManager) DoubleBuffer {
        return .{
            .res = res,
            .display_times = .{ 0.0, 0.0 },
            .current = 0,
        };
    }

    fn tex(self: @This(), comptime ch: Channel) c_uint {
        return switch (ch) {
            .y => self.res.tex.get(if (self.current == 0) .y_front else .y_back),
            .uv => self.res.tex.get(if (self.current == 0) .uv_front else .uv_back),
        };
    }

    fn pbo(self: @This(), comptime ch: Channel) c_uint {
        return switch (ch) {
            .y => self.res.pbo.get(if (self.current == 0) .y_front else .y_back),
            .uv => self.res.pbo.get(if (self.current == 0) .uv_front else .uv_back),
        };
    }

    fn set_display_time(self: *@This(), in_seconds: f64) void {
        self.display_times[self.current] = in_seconds;
    }

    fn display_time(self: @This()) f64 {
        return self.display_times[self.current];
    }

    fn invert(self: @This()) DoubleBuffer {
        return .{
            .res = self.res,
            .display_times = self.display_times,
            .current = self.current +% 1,
        };
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
    view: *Viewport,
    stable_buffer: *DoubleBuffer,
    angle_calc: AngleCalc,
    res: *const GpuResourceManager,
    camera: Camera,
) !void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    gl.ClearColor(0, 0, 0, 0);
    gl.Clear(gl.COLOR_BUFFER_BIT);

    const tex: Nv12Tex = .{ .y = stable_buffer.tex(.y), .uv = stable_buffer.tex(.uv) };
    angle_calc.execute(view, tex);

    {
        const z = tracy.Zone.begin(.{ .src = @src(), .name = "background pass" });
        defer z.end();

        blur(res.blur(), res, view, tex, camera, 6);
        const prog = res.prog.get(.bg);

        gl.UseProgram(prog);
        gl.BindVertexArray(res.vao.get(.tex));

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, res.blur().front.tex); // guaranteed to be the last modified texture

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.angle));

        const u_world_transform = camera.getBackgroundWorldTransform();
        const u_view_transform = Mat2.identity; // don't zoom in on background
        const u_clip_transform = camera.getViewClipTransform();

        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_angle"), 1);

        gl.Uniform1i(gl.GetUniformLocation(prog, "u_blur"), 0);
        gl.Uniform1f(gl.GetUniformLocation(prog, "u_radius"), res.meta.circle_radius * camera.scale * camera.zoom);

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    {
        const z = tracy.Zone.begin(.{ .src = @src(), .name = "ui pass" });
        defer z.end();

        const circle_prog = res.prog.get(.circle);
        const ring_prog = res.prog.get(.ring);

        const u_world_transform = camera.getUiWorldTransform();
        const u_view_transform = camera.getWorldViewTransform();
        const u_clip_transform = camera.getViewClipTransform();

        // Draw Transparent Puck
        gl.UseProgram(circle_prog);
        gl.BindVertexArray(res.vao.get(.tex));

        gl.UniformMatrix2fv(gl.GetUniformLocation(circle_prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(circle_prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(circle_prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});

        gl.Uniform1f(gl.GetUniformLocation(circle_prog, "u_radius"), res.meta.circle_radius);
        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);

        // Draw Ring (matches ring in gameplay)
        gl.UseProgram(ring_prog);
        gl.BindVertexArray(res.vao.get(.tex));

        gl.UniformMatrix2fv(gl.GetUniformLocation(ring_prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(ring_prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(ring_prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});

        gl.Uniform1f(gl.GetUniformLocation(ring_prog, "u_radius"), res.meta.ring_radius);
        gl.Uniform1f(gl.GetUniformLocation(ring_prog, "u_thickness"), res.meta.ring_thickness);
        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    {
        const z = tracy.Zone.begin(.{ .src = @src(), .name = "video pass" });
        defer z.end();

        const prog = res.prog.get(.tex);

        gl.UseProgram(prog);
        defer gl.UseProgram(0);

        gl.BindVertexArray(res.vao.get(.tex));
        defer gl.BindVertexArray(0);

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, tex.y);

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, tex.uv);

        gl.ActiveTexture(gl.TEXTURE2);
        gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.angle));
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        const u_world_transform = camera.getVideoWorldTransform();
        const u_view_transform = camera.getWorldViewTransform();
        const u_clip_transform = camera.getViewClipTransform();

        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});

        gl.Uniform1i(gl.GetUniformLocation(prog, "u_y_tex"), 0);
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_uv_tex"), 1);
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_angle"), 2);

        gl.UniformMatrix3fv(gl.GetUniformLocation(prog, "u_colour_space"), 1, gl.FALSE, camera.colourSpaceMatrix());
        gl.Uniform1f(gl.GetUniformLocation(prog, "u_ratio"), magic_aspect_ratio);
        gl.Uniform2i(gl.GetUniformLocation(prog, "u_resolution"), camera.video_resolution[0], camera.video_resolution[1]);

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
}

fn blur(b: BlurManager, res: *const GpuResourceManager, view: *Viewport, src_tex: Nv12Tex, camera: Camera, comptime passes: u32) void {
    if (passes == 0) return;

    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    std.debug.assert(passes & 1 == 0);

    const width: c_int = @intCast(b.resolution.width);
    const height: c_int = @intCast(b.resolution.height);
    const program = res.prog.get(.blur);

    const cache: c_uint = blk: {
        var buf: [1]c_int = undefined;
        gl.GetIntegerv(gl.FRAMEBUFFER_BINDING, &buf);

        break :blk @intCast(buf[0]);
    };
    defer gl.BindFramebuffer(gl.FRAMEBUFFER, cache);

    view.set(width, height);
    defer view.restore();

    gl.BindVertexArray(res.vao.get(.blur));
    defer gl.BindVertexArray(0);

    gl.UseProgram(program);
    defer gl.UseProgram(0);

    gl.ActiveTexture(gl.TEXTURE1);
    gl.BindTexture(gl.TEXTURE_2D, src_tex.y);

    gl.ActiveTexture(gl.TEXTURE2);
    gl.BindTexture(gl.TEXTURE_2D, src_tex.uv);

    gl.Uniform1i(gl.GetUniformLocation(program, "u_screen"), 0);
    gl.Uniform1i(gl.GetUniformLocation(program, "u_y_tex"), 1);
    gl.Uniform1i(gl.GetUniformLocation(program, "u_uv_tex"), 2);

    const f_width: f32 = @floatFromInt(width);
    const f_height: f32 = @floatFromInt(height);
    gl.Uniform2f(gl.GetUniformLocation(program, "u_texel_size"), 1.0 / f_width, 1.0 / f_height);

    gl.UniformMatrix3fv(gl.GetUniformLocation(program, "u_colour_space"), 1, gl.FALSE, camera.colourSpaceMatrix());

    const dir_loc = gl.GetUniformLocation(program, "u_direction");
    const nv12_loc = gl.GetUniformLocation(program, "u_use_nv12");

    gl.ActiveTexture(gl.TEXTURE0);

    for (0..passes) |i| {
        const current = b.current(i);
        const other = b.previous(i);

        gl.BindFramebuffer(gl.FRAMEBUFFER, current.fbo);
        gl.BindTexture(gl.TEXTURE_2D, other.tex);

        gl.Uniform2f(dir_loc, @floatFromInt(i & 1), @floatFromInt((i + 1) & 1));
        gl.Uniform1i(nv12_loc, @intFromBool(i == 0));

        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }
}

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

const Nv12Tex = struct { y: c_uint, uv: c_uint };

const AngleCalc = struct {
    res: *const GpuResourceManager,
    colour_matrix: [*]const [9]f32,

    const log = std.log.scoped(.angle_calc);

    pub fn init(res: *const GpuResourceManager, camera: Camera) !AngleCalc {
        return .{
            .res = res,
            .colour_matrix = camera.colourSpaceMatrix(),
        };
    }

    pub fn execute(self: @This(), view: *Viewport, tex: Nv12Tex) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "angle calc pass" });
        defer zone.end();

        const program = self.res.prog.get(.angle);

        view.set(1, 1);
        defer view.restore();

        gl.BindVertexArray(self.res.vao.get(.empty));
        defer gl.BindVertexArray(0);

        gl.BindFramebuffer(gl.FRAMEBUFFER, self.res.fbo.get(.angle));
        defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, tex.y);

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, tex.uv);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.UseProgram(program);
        defer gl.UseProgram(0);

        gl.Uniform1i(gl.GetUniformLocation(program, "u_y_tex"), 0);
        gl.Uniform1i(gl.GetUniformLocation(program, "u_uv_tex"), 1);
        gl.UniformMatrix3fv(gl.GetUniformLocation(program, "u_colour_space"), 1, gl.FALSE, self.colour_matrix);

        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }
};

const Camera = struct {
    view_to_clip: Mat2,

    video_resolution: struct { c_int, c_int }, // TODO: consolidate this with Resolution struct in GpuResourceManager

    world_aspect: f32,
    video_aspect: f32,
    gameplay_aspect: f32,

    scale: f32,
    inv_scale: f32,

    colour_space: c.AVColorSpace,

    zoom: f32 = 1.0,

    // zig fmt: off
    const bt601 = [9]f32{
        1.0,      1.0,      1.0,
        0.0,     -0.39465,  2.03211,
        1.13983, -0.58060,  0.0
    };

    const bt709 = [9]f32{
        1.0,      1.0,      1.0,
        0.0,     -0.1873,   1.8556,
        1.5748,  -0.4681,   0.0
    };
    // zig fmt: on

    pub fn init(video_resolution: struct { u32, u32 }, window_width: c_int, window_height: c_int, colour_space: c.AVColorSpace) Camera {
        const video_width, const video_height = video_resolution;

        const video_aspect = @as(f32, @floatFromInt(video_width)) / @as(f32, @floatFromInt(video_height));
        const window_aspect = @as(f32, @floatFromInt(window_width)) / @as(f32, @floatFromInt(window_height));

        const world_aspect = 1.0; // assume square world

        const viewport_bounds = if (window_aspect > 1.0) vec2(window_aspect, 1.0) else vec2(1.0, 1.0 / window_aspect);
        const viewport_diagonal = std.math.sqrt(viewport_bounds.x() * viewport_bounds.x() + viewport_bounds.y() * viewport_bounds.y());

        const gameplay_aspect = @max(video_aspect, magic_aspect_ratio);
        const gameplay_bounds = if (gameplay_aspect > 1.0) vec2(1.0, 1.0 / gameplay_aspect) else vec2(gameplay_aspect, 1.0);

        const scale = 1.0 / std.math.sqrt(gameplay_bounds.x() * gameplay_bounds.x() + gameplay_bounds.y() * gameplay_bounds.y());
        const inv_scale = viewport_diagonal / @min(gameplay_bounds.x(), gameplay_bounds.y());

        // std.log.debug("gameplay width: {d}, height: {d}", .{ gameplay_bounds.x(), gameplay_bounds.y() });
        // std.log.debug("angle: {d}", .{90.0 + std.math.atan(gameplay_bounds.x() / gameplay_bounds.y()) * std.math.deg_per_rad});

        return .{
            .view_to_clip = calculateAspectCorrection(world_aspect, window_aspect),
            .video_resolution = .{ @intCast(video_width), @intCast(video_height) },

            .world_aspect = world_aspect,
            .video_aspect = video_aspect,
            .gameplay_aspect = gameplay_aspect,

            .colour_space = colour_space,

            .scale = scale,
            .inv_scale = inv_scale,

            .zoom = defaultZoom(window_aspect),
        };
    }

    pub fn colourSpaceMatrix(self: Camera) [*]const [9]f32 {
        return @ptrCast(switch (self.colour_space) {
            c.AVCOL_SPC_BT470BG, c.AVCOL_SPC_SMPTE170M => &bt601,
            c.AVCOL_SPC_BT709 => &bt709,
            else => &bt709,
        });
    }

    fn defaultZoom(window_aspect: f32) f32 {
        if (@abs(window_aspect - 1.0) <= std.math.floatEps(f32)) return 1.0;
        if (window_aspect > 1.0) return 1.45; // landscape

        return 1.0; // portrait
    }

    pub fn updateWindow(self: *@This(), width: c_int, height: c_int) void {
        const window_aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));

        const viewport_bounds = if (window_aspect > 1.0) vec2(window_aspect, 1.0) else vec2(1.0, 1.0 / window_aspect);
        const viewport_diagonal = std.math.sqrt(viewport_bounds.x() * viewport_bounds.x() + viewport_bounds.y() * viewport_bounds.y());

        const gameplay_bounds = if (self.gameplay_aspect > 1.0) vec2(1.0, 1.0 / self.gameplay_aspect) else vec2(self.gameplay_aspect, 1.0);

        self.inv_scale = viewport_diagonal / @min(gameplay_bounds.x(), gameplay_bounds.y());
        self.view_to_clip = calculateAspectCorrection(self.world_aspect, window_aspect);
    }

    fn calculateAspectCorrection(world_aspect: f32, window_aspect: f32) Mat2 {
        if (window_aspect > world_aspect) {
            // Window wider than world - letterbox horizontally
            return Mat2.scaleXy(world_aspect / window_aspect, 1.0);
        } else {
            // Window taller than world - letterbox vertically
            return Mat2.scaleXy(1.0, window_aspect / world_aspect);
        }
    }

    pub fn getUiWorldTransform(self: @This()) Mat2 {
        return Mat2.scale(self.scale);
    }

    pub fn getVideoWorldTransform(self: @This()) Mat2 {
        const wide_scale = Mat2.scaleXy(1.0, 1.0 / self.video_aspect);
        const tall_scale = Mat2.scaleXy(self.video_aspect, 1.0);
        const aspect_transform = if (self.video_aspect > 1.0) wide_scale else tall_scale;

        const scale_transform = Mat2.scale(self.scale);
        return aspect_transform.mul(scale_transform);
    }

    pub fn getBackgroundWorldTransform(self: @This()) Mat2 {
        const wide_scale = Mat2.scaleXy(1.0, 1.0 / self.video_aspect);
        const tall_scale = Mat2.scaleXy(self.video_aspect, 1.0);
        const aspect_transform = if (self.video_aspect > 1.0) wide_scale else tall_scale;

        const scale_transform = Mat2.scale(self.inv_scale);
        return aspect_transform.mul(scale_transform);
    }

    pub fn getWorldViewTransform(self: @This()) Mat2 {
        return Mat2.scale(self.zoom);
    }

    pub fn getViewClipTransform(self: @This()) Mat2 {
        return self.view_to_clip;
    }

    pub fn adjustZoom(self: *@This(), delta: f32) void {
        self.setZoom(self.zoom + delta);
    }

    fn setZoom(self: *@This(), new_zoom: f32) void {
        std.log.info("zoom: {d:.2}", .{new_zoom});
        self.zoom = @max(1.0, @min(10.0, new_zoom));
    }
};

pub fn uploadPlane(stable_buffer: DoubleBuffer, frame: *c.AVFrame, comptime ch: DoubleBuffer.Channel) void {
    const zone = tracy.Zone.begin(.{ .src = @src(), .name = "upload " ++ @tagName(ch) ++ " plane" });
    defer zone.end();

    gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, stable_buffer.pbo(ch));
    defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

    gl.BindTexture(gl.TEXTURE_2D, stable_buffer.tex(ch));
    defer gl.BindTexture(gl.TEXTURE_2D, 0);

    const is_y_plane = ch == .y;

    const width: usize = @intCast(if (is_y_plane) frame.width else @divTrunc(frame.width, 2));
    const height: usize = @intCast(if (is_y_plane) frame.height else @divTrunc(frame.height, 2));
    const idx: usize = if (is_y_plane) 0 else 1;

    const pbo: ?[*]u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_UNPACK_BUFFER, gl.WRITE_ONLY));

    if (pbo) |ptr| {
        defer _ = gl.UnmapBuffer(gl.PIXEL_UNPACK_BUFFER);

        const bytes_per_line: usize = @intCast(frame.linesize[idx]);
        const dst_stride = width * if (is_y_plane) Y_BPP else UV_BPP;

        for (0..height) |row| {
            const src_offset = row * bytes_per_line;
            const dst_offset = row * dst_stride;

            @memcpy(ptr[dst_offset..][0..dst_stride], frame.data[idx][src_offset..][0..dst_stride]);
        }
    }

    const fmt: c_uint = if (is_y_plane) gl.RED else gl.RG;

    gl.PixelStorei(gl.UNPACK_ALIGNMENT, if (is_y_plane) 1 else 2);
    gl.TexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, @intCast(width), @intCast(height), fmt, gl.UNSIGNED_BYTE, null);
}

pub fn writeToNv12Tex(res: *const GpuResourceManager, view: *Viewport, camera: Camera) void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    // TODO: When we implement DearImgui, we will need an Offscreen FBO

    gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.out));
    defer gl.BindTexture(gl.TEXTURE_2D, 0);

    // copy from headless screen to rgb texture
    gl.BindFramebuffer(gl.READ_FRAMEBUFFER, 0);
    gl.CopyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, view.width, view.height);

    gl.BindVertexArray(res.vao.get(.empty));
    defer gl.BindVertexArray(0);

    gl.ActiveTexture(gl.TEXTURE0);

    gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.out));
    defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

    const program = res.prog.get(.rgb_to_nv12);
    gl.UseProgram(program);
    defer gl.UseProgram(0);

    gl.Uniform1i(gl.GetUniformLocation(program, "u_rgb_tex"), 0);
    gl.UniformMatrix3fv(gl.GetUniformLocation(program, "u_colour_space"), 1, gl.FALSE, camera.colourSpaceMatrix());

    const is_y_loc = gl.GetUniformLocation(program, "u_is_y");

    {
        view.set(view.width, view.height);
        defer view.restore();

        // render Y plane
        gl.BindFramebuffer(gl.FRAMEBUFFER, res.fbo.get(.y));
        gl.Uniform1i(is_y_loc, 1);
        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }

    {
        view.set(@divTrunc(view.width, 2), @divTrunc(view.height, 2));
        defer view.restore();

        // render UV plane
        gl.BindFramebuffer(gl.FRAMEBUFFER, res.fbo.get(.uv));
        gl.Uniform1i(is_y_loc, 0);
        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }
}

pub fn mapNv12Frame(res: *const GpuResourceManager, view: Viewport, current_pbo: u1) ?struct { []const u8, []const u8 } {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    const idx = current_pbo +% 1;

    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, res.pbo.get(if (idx == 1) .dl_front else .dl_back));
    defer gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);

    const maybe_ptr: ?[*]const u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY));

    if (maybe_ptr) |ptr| {
        const y_len: usize = @intCast(view.width * view.height);
        const uv_len: usize = @intCast(@divTrunc(view.width, 2) * @divTrunc(view.height, 2) * UV_BPP);

        return .{ ptr[0..y_len], ptr[y_len..][0..uv_len] };
    }

    return null;
}

pub fn unmapNv12Frame(res: *const GpuResourceManager, current_pbo: u1) void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    const idx = current_pbo +% 1;

    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, res.pbo.get(if (idx == 1) .dl_front else .dl_back));
    _ = gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER);

    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);
}

fn checkFile(maybe_path: ?[]const u8) !bool {
    const path = maybe_path orelse return true;

    var buf: [0x40]u8 = undefined;

    var writer = std.fs.File.stdout().writer(buf[0..][0..0x20]);
    var reader = std.fs.File.stdin().reader(buf[0x20..][0..0x20]);

    var stdout = &writer.interface;
    var stdin = &reader.interface;

    std.fs.cwd().access(path, .{}) catch |e| switch (e) {
        error.FileNotFound => return true,
        else => return e,
    };

    try stdout.print("File '{s}' already exists. Overwrite? (y/n): ", .{path});
    try stdout.flush();

    const line = try stdin.takeDelimiter('\n') orelse return false;
    const answer = std.mem.trim(u8, line, "\r\t\n");

    return std.ascii.eqlIgnoreCase(answer, "y");
}
