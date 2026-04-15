const std = @import("std");
const gl = @import("gl");
const tracy = @import("tracy");
const zgui = @import("zgui");

const c = @import("lib.zig").c;
const platform = @import("lib/platform.zig");
const signal = @import("lib/platform.zig").signal;

const Encoder = @import("lib/codec.zig").Encoder;
const Decoder = @import("lib/codec.zig").Decoder;

const UploadBuffer = @import("lib.zig").UploadBuffer;
const DoubleBuffer = @import("lib.zig").DoubleBuffer;
const BlurManager = @import("lib.zig").BlurManager;
const Viewport = @import("lib.zig").Viewport;
const FbStack = @import("lib.zig").FbStack;
const GpuResourceManager = @import("lib.zig").GpuResourceManager;
const PixelBufferPool = GpuResourceManager.PixelBufferPool;
const Ui = @import("lib/platform.zig").Ui;
const App = @import("app.zig").App;

const Mat2 = @import("lib/math.zig").Mat2;
const Vec2 = @import("lib/math.zig").Vec2;
const Linesize = @import("lib.zig").Linesize;
const Resolution = @import("lib.zig").Resolution;
const mat2 = @import("lib/math.zig").mat2;
const vec2 = @import("lib/math.zig").vec2;

const sleep = @import("lib.zig").sleep;

const RGB24_BPP = @import("lib.zig").RGB24_BPP;

const Y_BPP = @import("lib.zig").Y_BPP;
const UV_BPP = @import("lib.zig").UV_BPP;

const magic_aspect_ratio = @import("lib.zig").magic_aspect_ratio;
var zoom: f32 = 1.0;

pub const startup = struct {
    pub const ui_window: Resolution = .{ .width = 1600, .height = 900 };
    pub const encode_target: Resolution = .{ .width = 1920, .height = 1080 };
    pub const render_target: Resolution = .{ .width = 1920, .height = 1080 };
};

pub const tracy_impl = @import("tracy_impl"); // configured from build.zig
pub const tracy_options: tracy.Options = .{ .default_callstack_depth = 5 };

pub fn main() !void {
    const log = std.log.scoped(.main);
    errdefer |err| if (err == error.sdl_error) log.err("SDL Error: {s}", .{c.SDL_GetError()});

    var gpa: std.heap.DebugAllocator(.{}) = .{ .backing_allocator = std.heap.c_allocator };
    defer std.debug.assert(gpa.deinit() == .ok);

    var tracy_alloc: tracy.Allocator = .{ .parent = gpa.allocator() };
    const allocator = tracy_alloc.allocator();

    const ui = try Ui.init(allocator, startup.ui_window);
    defer ui.deinit();

    var ui_view: Viewport = .default;
    const ui_size = try ui.windowSize();
    try ui_view.push(ui_size[0], ui_size[1]);

    // const should_write = try checkFile(cli.positionals[0]);
    // if (!should_write) return error.wont_overwrite;

    try signal.setupHandler();

    // if (cli.positionals[0]) |dst_path| {
    //     // Initialize encoder
    //     var encoder = try Encoder.init(.{
    //         .encode_view = encode_view,
    //         .decoder = &decoder,
    //     }, hw_encode, dst_path);
    //     defer encoder.deinit();

    //     const render_start_time = c.SDL_GetPerformanceCounter();
    //     const perf_freq: f64 = @floatFromInt(c.SDL_GetPerformanceFrequency());

    //     const video_stream = decoder.stream(.video);
    //     const audio_stream = decoder.stream(.audio);

    //     const estimated_frames: usize = blk: {
    //         if (video_stream.nb_frames > 0) break :blk @intCast(video_stream.nb_frames);
    //         if (decoder.fmt_ctx.ptr().duration <= 0) break :blk 0;

    //         // estimate from duration
    //         const duration_secs = @as(f64, @floatFromInt(decoder.fmt_ctx.ptr().duration)) / @as(f64, c.AV_TIME_BASE);
    //         break :blk @intFromFloat(duration_secs * c.av_q2d(frame_rate));
    //     };

    //     // Initialize progress
    //     var progress_buffer: [256]u8 = undefined;
    //     const progress_node = std.Progress.start(.{
    //         .root_name = "Encoding",
    //         .estimated_total_items = estimated_frames,
    //         .draw_buffer = &progress_buffer,
    //     });
    //     defer progress_node.end();

    //     try res.setupEncodingTargets(encode_view, encoder._frame);

    //     var upload_buffer: UploadBuffer = .default;
    //     var frame_count: u64 = 0;

    //     _ = try preload(res, &decoder, double_buffer);
    //     try render(&render_view, &fbs, double_buffer.front(), angle_calc, res, camera);
    //     try writeToNv12Tex(res, &encode_view, fbs, camera);

    //     const linesize: Linesize(c.AV_PIX_FMT_NV12) = .init(encoder._frame);

    //     while (!signal.should_quit.load(.monotonic)) {
    //         const zone = tracy.Zone.begin(.{ .src = @src(), .name = "encode loop" });
    //         defer zone.end();

    //         // Process any pending audio packets (remux to output)
    //         while (decoder.queue.pkt.audio.tryPop()) |audio_pkt| {
    //             try encoder.writeAudioPacket(audio_stream, audio_pkt);

    //             var pkt: ?*c.AVPacket = audio_pkt;
    //             c.av_packet_free(&pkt);
    //         }

    //         if (decoder.queue.frame.pop()) |frame| {
    //             defer decoder.queue.frame.recycle(frame);
    //             defer double_buffer.swap();
    //             defer frame_count += 1; // increment after frame is sent to the GPU

    //             const z = tracy.Zone.begin(.{ .src = @src(), .name = "frame received" });
    //             defer z.end();

    //             const back = double_buffer.back();
    //             uploadPlane(.y, res, back, frame);
    //             uploadPlane(.uv, res, back, frame);

    //             try render(&render_view, &fbs, back.flip(), angle_calc, res, camera);
    //             try writeToNv12Tex(res, &encode_view, fbs, camera);

    //             const width, const height = encode_view.get();

    //             {
    //                 const pbo_z = tracy.Zone.begin(.{ .src = @src(), .name = "read from opengl" });
    //                 defer pbo_z.end();

    //                 const idx = upload_buffer.current();

    //                 // read Y and UV into single contiguous PBO with linesize padding
    //                 gl.BindBuffer(gl.PIXEL_PACK_BUFFER, res.pbo.get(idx));
    //                 defer gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);

    //                 {
    //                     const read_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.ReadPixels y" });
    //                     defer read_z.end();

    //                     gl.PixelStorei(gl.PACK_ROW_LENGTH, @intCast(linesize.y));
    //                     gl.BindFramebuffer(gl.READ_FRAMEBUFFER, res.fbo.get(.y));
    //                     gl.ReadPixels(0, 0, width, height, gl.RED, gl.UNSIGNED_BYTE, @ptrFromInt(0));
    //                 }

    //                 {
    //                     const read_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.ReadPixels uv" });
    //                     defer read_z.end();

    //                     // after y plane in memory
    //                     const uv_offset: usize = @intCast(linesize.y * height);
    //                     gl.PixelStorei(gl.PACK_ROW_LENGTH, @divTrunc(linesize.uv, 2)); // FFMpeg linesize is bytes, RG is 2 bytes per pixel
    //                     gl.BindFramebuffer(gl.READ_FRAMEBUFFER, res.fbo.get(.uv));
    //                     gl.ReadPixels(0, 0, @divTrunc(width, 2), @divTrunc(height, 2), gl.RG, gl.UNSIGNED_BYTE, @ptrFromInt(uv_offset));
    //                 }

    //                 gl.PixelStorei(gl.PACK_ROW_LENGTH, 0);
    //                 gl.BindFramebuffer(gl.READ_FRAMEBUFFER, fbs.get());

    //                 {
    //                     const flush_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.Flush" });
    //                     defer flush_z.end();

    //                     gl.Flush(); // trigger readback immediately
    //                 }

    //                 if (upload_buffer.next()) |upload| blk: {
    //                     const y, const uv = mapNv12Frame(res, encode_view, upload.id, linesize) orelse break :blk;
    //                     defer unmapNv12Frame(res, upload.id);

    //                     try encoder.encodeNv12Frame(y, uv, upload.pts);
    //                 }
    //             }

    //             upload_buffer.advance(frame.pts);

    //             // Update progress
    //             progress_node.setCompletedItems(frame_count);
    //         } else if (decoder.queue.frame.end_of_stream.load(.monotonic)) {
    //             defer shutdown(&decoder.queue);
    //             defer std.Progress.setStatus(.success);

    //             const z = tracy.Zone.begin(.{ .src = @src(), .name = "final frame received" });
    //             defer z.end();

    //             upload_buffer.skip(); // to access the pending frames

    //             while (upload_buffer.flush()) |upload| {
    //                 const y, const uv = mapNv12Frame(res, encode_view, upload.id, linesize) orelse continue;
    //                 defer unmapNv12Frame(res, upload.id);

    //                 try encoder.encodeNv12Frame(y, uv, upload.pts);
    //             }

    //             // Calculate and display final stats
    //             const elapsed: f64 = @as(f64, @floatFromInt(c.SDL_GetPerformanceCounter() - render_start_time)) / perf_freq;
    //             const video_duration = @as(f64, @floatFromInt(frame_count)) / c.av_q2d(frame_rate);
    //             const speed = video_duration / elapsed;

    //             break log.info("Finished rendering {} frames in {d:.1}s ({d:.2}x realtime)", .{ frame_count, elapsed, speed });
    //         }
    //     }

    var app: App = .default;
    defer app.deinit(allocator);

    const state = try allocator.create(platform.gui.State);
    defer allocator.destroy(state);

    state.* = .default;

    while (!signal.should_quit.load(.monotonic)) {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "ui loop" });
        defer zone.end();

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "query input" });
            defer z.end();

            var event: c.SDL_Event = undefined;
            while (c.SDL_PollEvent(&event)) {
                _ = zgui.backend.processEvent(&event);

                switch (event.type) {
                    c.SDL_EVENT_QUIT => signal.should_quit.store(true, .monotonic),
                    c.SDL_EVENT_WINDOW_RESIZED => ui_view.reset(event.window.data1, event.window.data2),
                    else => {},
                }
            }
        }

        gl.ClearColor(0, 0, 0, 1.0);
        gl.Clear(gl.COLOR_BUFFER_BIT);

        try app.poll(allocator, ui, state);
        try app.run();

        try platform.gui.draw(state, ui_view, app.video());
        zgui.backend.draw();

        try ui.swap();
    }
}

const Id = enum(usize) {
    texture = 0,
    ring,
    circle,
    background,
    blur,
};

pub fn render(
    view: *Viewport,
    fbs: *FbStack,
    front: DoubleBuffer.Buffer,
    angle_calc: AngleCalc,
    res: *const GpuResourceManager,
    camera: Camera,
) !void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    try fbs.push(res.fbo.get(.out));
    defer fbs.pop();

    const tex: Nv12Tex = .{ .y = res.tex.get(front.tex(.y)), .uv = res.tex.get(front.tex(.uv)) };
    try angle_calc.execute(view, fbs, tex);

    {
        const z = tracy.Zone.begin(.{ .src = @src(), .name = "background pass" });
        defer z.end();

        try blur(res.blur(), res, view, fbs, tex, camera, 6);
        const prog = res.prog.get(.bg);

        gl.UseProgram(prog);
        gl.BindVertexArray(res.vao.get(.tex));

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, res.blur().front.tex); // guaranteed to be the last modified texture

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.angle));

        gl.ActiveTexture(gl.TEXTURE2);
        gl.BindTexture(gl.TEXTURE_2D, tex.y);

        gl.ActiveTexture(gl.TEXTURE3);
        gl.BindTexture(gl.TEXTURE_2D, tex.uv);

        const u_world_transform = camera.getBackgroundWorldTransform();
        const u_view_transform = Mat2.identity; // don't zoom in on background
        const u_clip_transform = camera.getViewClipTransform();

        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_angle"), 1);

        gl.Uniform1i(gl.GetUniformLocation(prog, "u_blur"), 0);
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_y_tex"), 2);
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_uv_tex"), 3);
        gl.Uniform1f(gl.GetUniformLocation(prog, "u_radius"), res.meta.circle_radius * camera.scale * camera.zoom);
        gl.Uniform1f(gl.GetUniformLocation(prog, "u_zoom"), zoom);

        gl.UniformMatrix3fv(gl.GetUniformLocation(prog, "u_colour_space"), 1, gl.FALSE, camera.colourSpaceMatrix());

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
        gl.Uniform2i(gl.GetUniformLocation(prog, "u_resolution"), camera.video_resolution.width, camera.video_resolution.height);

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
}

fn blur(
    b: BlurManager,
    res: *const GpuResourceManager,
    view: *Viewport,
    fbs: *FbStack,
    src_tex: Nv12Tex,
    camera: Camera,
    comptime passes: u32,
) !void {
    if (passes == 0) return;

    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    std.debug.assert(passes & 1 == 0);

    const width: c_int = @intCast(b.resolution.width);
    const height: c_int = @intCast(b.resolution.height);
    const program = res.prog.get(.blur);

    try view.push(width, height);
    defer view.pop();

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

    gl.BindFramebuffer(gl.FRAMEBUFFER, fbs.get());
}

const Nv12Tex = struct { y: c_uint, uv: c_uint };

pub const AngleCalc = struct {
    res: *const GpuResourceManager,
    colour_matrix: [*]const [9]f32,

    const log = std.log.scoped(.angle_calc);

    pub fn init(res: *const GpuResourceManager, camera: Camera) !AngleCalc {
        return .{
            .res = res,
            .colour_matrix = camera.colourSpaceMatrix(),
        };
    }

    pub fn execute(self: @This(), view: *Viewport, fbs: *FbStack, tex: Nv12Tex) !void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "angle calc pass" });
        defer zone.end();

        const program = self.res.prog.get(.angle);

        try view.push(1, 1);
        defer view.pop();

        gl.BindVertexArray(self.res.vao.get(.empty));
        defer gl.BindVertexArray(0);

        try fbs.push(self.res.fbo.get(.angle));
        defer fbs.pop();

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

pub const Camera = struct {
    view_to_clip: Mat2,

    video_resolution: Resolution,

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

    pub fn init(render_view: Viewport, input_video_resolution: Resolution, colour_space: c.AVColorSpace) Camera {
        const video_resolution = input_video_resolution;
        const video_width = video_resolution.width;
        const video_height = video_resolution.height;

        const window_width, const window_height = render_view.get();

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
            .video_resolution = video_resolution,

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

pub fn uploadPlane(comptime ch: DoubleBuffer.Channel, res: *const GpuResourceManager, buffer: DoubleBuffer.Buffer, frame: *c.AVFrame) void {
    const zone = tracy.Zone.begin(.{ .src = @src(), .name = "upload " ++ @tagName(ch) ++ " plane" });
    defer zone.end();

    gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0); // Disable PBO to read directly from FFMpeg strided memory

    gl.BindTexture(gl.TEXTURE_2D, res.tex.get(buffer.tex(ch)));
    defer gl.BindTexture(gl.TEXTURE_2D, 0);

    const is_y_plane = ch == .y;

    const width: usize = @intCast(if (is_y_plane) frame.width else @divTrunc(frame.width, 2));
    const height: usize = @intCast(if (is_y_plane) frame.height else @divTrunc(frame.height, 2));
    const idx: usize = if (is_y_plane) 0 else 1;

    // TODO: merge?
    const bpp: c_int = if (is_y_plane) Y_BPP else UV_BPP;
    const alignment: c_int = if (is_y_plane) 1 else 2;

    const fmt: c_uint = if (is_y_plane) gl.RED else gl.RG;

    gl.PixelStorei(gl.UNPACK_ALIGNMENT, alignment);
    gl.PixelStorei(gl.UNPACK_ROW_LENGTH, @intCast(@divTrunc(frame.linesize[idx], bpp)));
    defer gl.PixelStorei(gl.UNPACK_ROW_LENGTH, 0);

    gl.TexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, @intCast(width), @intCast(height), fmt, gl.UNSIGNED_BYTE, frame.data[idx]);
}

pub fn writeToNv12Tex(res: *const GpuResourceManager, view: *Viewport, fbs: FbStack, camera: Camera) !void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    const width, const height = view.get();

    gl.BindVertexArray(res.vao.get(.empty));
    defer gl.BindVertexArray(0);

    gl.ActiveTexture(gl.TEXTURE0);

    gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.out));
    defer gl.BindTexture(gl.TEXTURE_2D, 0);

    const program = res.prog.get(.rgb_to_nv12);
    gl.UseProgram(program);
    defer gl.UseProgram(0);

    gl.Uniform1i(gl.GetUniformLocation(program, "u_rgb_tex"), 0);
    gl.UniformMatrix3fv(gl.GetUniformLocation(program, "u_colour_space"), 1, gl.FALSE, camera.colourSpaceMatrix());

    const is_y_loc = gl.GetUniformLocation(program, "u_is_y");

    {
        try view.push(width, height);
        defer view.pop();

        // render Y plane
        gl.BindFramebuffer(gl.FRAMEBUFFER, res.fbo.get(.y));
        gl.Uniform1i(is_y_loc, 1);
        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }

    {
        try view.push(@divTrunc(width, 2), @divTrunc(height, 2));
        defer view.pop();

        // render UV plane
        gl.BindFramebuffer(gl.FRAMEBUFFER, res.fbo.get(.uv));
        gl.Uniform1i(is_y_loc, 0);
        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }

    gl.BindFramebuffer(gl.FRAMEBUFFER, fbs.get());
}

pub fn mapNv12Frame(res: *const GpuResourceManager, view: Viewport, idx: PixelBufferPool.Index, linesize: Linesize(c.AV_PIX_FMT_NV12)) ?struct { []const u8, []const u8 } {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, res.pbo.get(idx));
    defer gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);

    const maybe_ptr: ?[*]const u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY));

    if (maybe_ptr) |ptr| {
        _, const height = view.get();
        const y_len: usize = @intCast(linesize.y * height);
        const uv_len: usize = @intCast(linesize.uv * @divTrunc(height, 2));

        return .{ ptr[0..y_len], ptr[y_len..][0..uv_len] };
    }

    return null;
}

pub fn unmapNv12Frame(res: *const GpuResourceManager, idx: PixelBufferPool.Index) void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, res.pbo.get(idx));
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

pub fn shutdown(queues: *Decoder.Queues) void {
    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    while (!signal.should_quit.load(.monotonic)) {
        signal.should_quit.store(true, .monotonic);
        std.atomic.spinLoopHint();
    }

    // wake up all the Queues
    queues.pkt.video.interrupt();
    queues.pkt.audio.interrupt();
    queues.frame.interrupt();
}

pub fn preload(res: *const GpuResourceManager, decoder: *Decoder, double_buffer: *DoubleBuffer) !f64 {
    const FrameQueue = @import("lib/codec.zig").FrameQueue;

    const zone = tracy.Zone.begin(.{ .src = @src() });
    defer zone.end();

    const frame = decoder.queue.frame.pop() orelse {
        shutdown(&decoder.queue);
        return error.early_exit;
    };
    defer decoder.queue.frame.recycle(frame);

    const time_base = c.av_q2d(decoder.stream(.video).time_base);
    const timestamp = @as(f64, @floatFromInt(frame.pts)) * time_base;

    const front = double_buffer.front();
    uploadPlane(.y, res, front, frame);
    uploadPlane(.uv, res, front, frame);
    front.setDisplayTime(timestamp);

    const back = double_buffer.back();
    uploadPlane(.y, res, back, frame);
    uploadPlane(.uv, res, back, frame);
    back.setDisplayTime(timestamp);

    // might as well completely fill the queue before we start
    {
        const z = tracy.Zone.begin(.{ .src = @src(), .name = "wait for queue to fill", .color = .gray25 });
        defer z.end();

        while (decoder.queue.frame.len() + 1 != FrameQueue.capacity) {
            if (decoder.queue.frame.end_of_stream.load(.monotonic)) break;
            sleep(1 * std.time.ns_per_ms);
        }
    }

    return timestamp;
}
