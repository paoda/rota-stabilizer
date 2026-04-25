const std = @import("std");
const tracy = @import("tracy");
const gl = @import("gl");
const c = @import("lib.zig").c;

const Camera = @import("main.zig").Camera;
const AngleCalc = @import("main.zig").AngleCalc;

const Viewport = @import("lib.zig").Viewport;
const FbStack = @import("lib.zig").FbStack;
const GpuResourceManager = @import("lib.zig").GpuResourceManager;
const DoubleBuffer = @import("lib.zig").DoubleBuffer;
const UploadBuffer = @import("lib.zig").UploadBuffer;
const Linesize = @import("lib.zig").Linesize;

const Ui = @import("lib/platform.zig").Ui;
const GuiState = @import("lib/platform.zig").gui.State;
const VideoContext = @import("lib/platform.zig").gui.VideoContext;

const Encoder = @import("lib/codec.zig").Encoder;
const Decoder = @import("lib/codec.zig").Decoder;

const signal = @import("lib/platform.zig").signal;

const preload = @import("main.zig").preload;
const render = @import("main.zig").render;
const uploadPlane = @import("main.zig").uploadPlane;
const writeToNv12Tex = @import("main.zig").writeToNv12Tex;
const mapNv12Frame = @import("main.zig").mapNv12Frame;
const unmapNv12Frame = @import("main.zig").unmapNv12Frame;
const trace = @import("lib.zig").trace;

const State = enum { idle, playback, encode };

pub const Request = union(State) {
    idle: void,
    playback: []const u8,
    encode: struct { src_path: []const u8, dst_path: []const u8 },
};

pub const Action = union(enum) {
    SetVolume: f32,
    Seek: f32,
};

const Session = union(State) {
    idle: IdleSession,
    playback: PlaybackSession,
    encode: EncodeSession,

    pub fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "Session.deinit" });
        defer zone.end();

        switch (self.*) {
            .idle => |s| s.deinit(),
            .playback => |*s| s.deinit(allocator),
            .encode => |*s| s.deinit(allocator),
        }
    }
};

const IdleSession = struct {
    pub fn init() IdleSession {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "IdleSession.init" });
        defer zone.end();

        // NB: see PlaybackSession.deinit;
        while (signal.should_quit.load(.monotonic)) {
            signal.should_quit.store(false, .monotonic);
            std.atomic.spinLoopHint();
        }

        return .{};
    }

    pub fn deinit(_: IdleSession) void {}
};

const PlaybackSession = struct {
    decoder: *Decoder,
    manager: *GpuResourceManager,

    camera: Camera,
    angle_calc: AngleCalc,

    double_buffer: *DoubleBuffer,
    handles: Decoder.Handles,

    render_view: Viewport,

    fbs: FbStack,

    lookahead: f64,
    delay_threshold: f64,

    next_frame: ?*c.AVFrame = null,

    const log = std.log.scoped(.playback_session);

    fn stop(self: *const @This()) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "PlaybackSession.stop" });
        defer zone.end();

        // TODO: session unique atomic
        while (!signal.should_quit.load(.monotonic)) {
            signal.should_quit.store(true, .monotonic);
            std.atomic.spinLoopHint();
        }

        self.decoder.queue.pkt.video.interrupt();
        self.decoder.queue.pkt.audio.interrupt();
        self.decoder.queue.frame.interrupt();
    }

    pub fn init(allocator: std.mem.Allocator, state: *const GuiState, ui: Ui, path: []const u8) !PlaybackSession {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "PlaybackSession.init" });
        defer zone.end();

        const hw_device: c.AVHWDeviceType = @intFromEnum(state.hw_dec);
        log.debug("using {s} for hw decode", .{getHwDeviceName(hw_device)});

        const decoder = try allocator.create(Decoder);
        errdefer allocator.destroy(decoder);

        decoder.* = try Decoder.init(allocator, hw_device, state.volume.value, path, false);
        errdefer decoder.deinit(allocator);

        const double_buffer = try allocator.create(DoubleBuffer);
        errdefer allocator.destroy(double_buffer);

        double_buffer.* = .{};

        const handles = try decoder.spawn(false);
        errdefer handles.deinit();

        var render_view: Viewport = .default;
        try render_view.push(state.resolution[0], state.resolution[1]);

        const manager = try GpuResourceManager.init(allocator, render_view, decoder.resolution);
        errdefer manager.deinit(allocator);

        const camera = Camera.init(render_view, decoder.resolution, decoder.colour_space);
        const angle_calc = try AngleCalc.init(manager, camera);

        const start_time = try preload(manager, decoder, double_buffer);
        log.debug("video start time: {d}s", .{start_time});

        var fbs: FbStack = .default;

        try render(
            &render_view,
            &fbs,
            double_buffer.front(),
            angle_calc,
            manager,
            camera,
        );
        try ui.swap();

        try decoder.audio_clock.?.start(start_time);

        const refresh_rate = ui.refreshRate() catch 60.0;

        const frame_rate = decoder.framerate();
        const frame_period = 1.0 / c.av_q2d(frame_rate);

        const lookahead = (1.0 / refresh_rate) / 2.0;
        const delay_threshold = @max(0.010, frame_period * 1.25);

        log.debug("frame period: {d:.2}ms ({d:.2}fps)", .{ frame_period * std.time.ms_per_s, c.av_q2d(frame_rate) });
        log.debug("display refresh rate: {d:.2}Hz (lookahead: {d:.2}ms)", .{ refresh_rate, lookahead * std.time.ms_per_s });
        log.debug("delay_threshold: {d:.2}ms", .{delay_threshold * std.time.ms_per_s});

        tracy.plotConfig(.{ .name = "A/V Sync Drift (ms)" });

        return .{
            .decoder = decoder,
            .manager = manager,
            .double_buffer = double_buffer,
            .camera = camera,
            .angle_calc = angle_calc,
            .fbs = fbs,
            .lookahead = lookahead,
            .delay_threshold = delay_threshold,

            .handles = handles,
            .render_view = render_view,
        };
    }

    pub fn deinit(self: *PlaybackSession, allocator: std.mem.Allocator) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "PlaybackSession.deinit" });
        defer zone.end();

        self.stop();

        self.handles.deinit();
        self.manager.deinit(allocator);

        allocator.destroy(self.double_buffer);

        self.decoder.deinit(allocator);
        allocator.destroy(self.decoder);
    }

    pub fn run(self: *PlaybackSession) !void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "PlaybackSession.run" });
        defer zone.end();

        const time_base = c.av_q2d(self.decoder.stream(.video).time_base);
        const audio_clock = &(self.decoder.audio_clock orelse return error.uninitialized_audio_clock);

        var is_dirty: bool = false;

        while (true) {
            if (self.next_frame == null) self.next_frame = self.decoder.queue.frame.tryPop() orelse break;

            // FIXME: self.next_frame guaranteed to be true here

            if (self.next_frame) |frame| {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "frame obtained" });
                defer z.end();

                std.debug.assert(frame.format == c.AV_PIX_FMT_NV12);

                const back = self.double_buffer.back();
                const frame_time = @as(f64, @floatFromInt(frame.pts)) * time_base;

                const audio_time = audio_clock.seconds_passed();
                const diff_s = frame_time - audio_time;

                const already_uploaded = @abs(frame_time - back.displayTime()) < std.math.floatEps(f64);

                if (diff_s < -self.delay_threshold) {
                    trace("dropped frame | d: {d:.3} a: {d:.3} v: {d:.3}", .{ diff_s, audio_time, frame_time });

                    // invalidate frame and mark the memory as free as we aren't gonna display it
                    self.decoder.queue.frame.recycle(frame);
                    self.next_frame = null;

                    continue; // we want to try the next frame
                }

                // based on some tracy data
                const estimated_upload_s = 0.001; // 1ms

                // frame falls within +/- self.lookahead, but is greater than -self.delay_threshold
                if (!already_uploaded and diff_s - self.lookahead <= 0) blk: {
                    const next = self.decoder.queue.frame.peek() orelse break :blk;
                    const next_time = @as(f64, @floatFromInt(next.pts)) * time_base;

                    if ((next_time - (audio_time + estimated_upload_s)) <= 0) {
                        // the next next frame happens to also be ready
                        trace("skipped frame | d: {d:.3} a: {d:.3} v: {d:.3}", .{ diff_s, audio_time, frame_time });

                        self.decoder.queue.frame.recycle(frame);
                        self.next_frame = null;

                        continue; // we want to now grab that frame next iteration
                    }
                }

                // at this point we know that this is the frame we want to consider

                if (!already_uploaded) {
                    const upload_z = tracy.Zone.begin(.{ .src = @src(), .name = "upload next frame" });
                    defer upload_z.end();

                    tracy.plot(.{ .name = "Buffered Frames", .value = .{ .i64 = @intCast(self.decoder.queue.frame.len()) } });

                    uploadPlane(.y, self.manager, back, frame);
                    uploadPlane(.uv, self.manager, back, frame);
                    back.setDisplayTime(frame_time);
                }

                if (diff_s - self.lookahead <= 0) {
                    tracy.plot(.{ .name = "Audio Time (s)", .value = .{ .f64 = audio_time } });
                    tracy.plot(.{ .name = "Next Frame Time (s)", .value = .{ .f64 = frame_time } });
                    tracy.plot(.{ .name = "A/V Sync Drift (ms)", .value = .{ .f64 = diff_s * std.time.ms_per_s } });

                    self.double_buffer.swap(); // frame has been uploaded, and marked for display so swap the buffers

                    // frame has been copied to gpu so we are ready to cleanup
                    self.decoder.queue.frame.recycle(frame);
                    self.next_frame = null;

                    tracy.frameMark("video");
                    is_dirty = true;

                    break;
                }
            }

            break;
        }

        if (is_dirty) {
            try render(
                &self.render_view,
                &self.fbs,
                self.double_buffer.front(),
                self.angle_calc,
                self.manager,
                self.camera,
            );
        }
    }
};

const EncodeSession = struct {
    encoder: *Encoder,
    decoder: *Decoder,

    manager: *GpuResourceManager,

    camera: Camera,
    angle_calc: AngleCalc,

    double_buffer: *DoubleBuffer,
    handles: Decoder.Handles,

    render_view: Viewport,
    encode_view: Viewport,

    upload_buffer: UploadBuffer,

    fbs: FbStack,

    frame_count: usize = 0,
    frame_total: usize,

    refresh_rate: f32,
    is_finished: bool = false,

    const log = std.log.scoped(.encode_session);

    fn stop(self: *const @This()) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "EncodeSession.stop" });
        defer zone.end();

        // TODO: session unique atomic
        while (!signal.should_quit.load(.monotonic)) {
            signal.should_quit.store(true, .monotonic);
            std.atomic.spinLoopHint();
        }

        self.decoder.queue.pkt.video.interrupt();
        self.decoder.queue.pkt.audio.interrupt();
        self.decoder.queue.frame.interrupt();
    }

    pub fn init(allocator: std.mem.Allocator, state: *const GuiState, ui: Ui, src_path: []const u8, dst_path: []const u8) !EncodeSession {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "EncodeSession.init" });
        defer zone.end();

        const hw_dec: c.AVHWDeviceType = @intFromEnum(state.hw_dec);
        const hw_enc: c.AVHWDeviceType = @intFromEnum(state.hw_enc);
        log.debug("using {s} for hw decode", .{getHwDeviceName(hw_dec)});
        log.debug("using {s} for hw decode", .{getHwDeviceName(hw_enc)});

        var render_view: Viewport = .default;
        try render_view.push(state.resolution[0], state.resolution[1]);

        var encode_view: Viewport = .default;
        try encode_view.push(state.resolution[0], state.resolution[1]);

        const decoder = try allocator.create(Decoder);
        errdefer allocator.destroy(decoder);

        decoder.* = try Decoder.init(allocator, hw_dec, 0.0, src_path, true);
        errdefer decoder.deinit(allocator);

        const encoder = try allocator.create(Encoder);
        errdefer allocator.destroy(encoder);

        encoder.* = try Encoder.init(
            .{
                .encode_view = encode_view,
                .decoder = decoder,
                .bit_rate = state.bit_rate,
            },
            hw_enc,
            dst_path,
        );
        errdefer encoder.deinit();

        const double_buffer = try allocator.create(DoubleBuffer);
        errdefer allocator.destroy(double_buffer);

        double_buffer.* = .{};

        const manager = try GpuResourceManager.init(allocator, render_view, decoder.resolution);
        errdefer manager.deinit(allocator);

        try manager.setupEncodingTargets(encode_view, encoder._frame);

        const handles = try decoder.spawn(true);
        errdefer handles.deinit();

        const camera = Camera.init(render_view, decoder.resolution, decoder.colour_space);
        const angle_calc = try AngleCalc.init(manager, camera);

        // TODO: Track + Display Encode Progress

        var fbs: FbStack = .default;

        _ = try preload(manager, decoder, double_buffer);
        try render(&render_view, &fbs, double_buffer.front(), angle_calc, manager, camera);
        try writeToNv12Tex(manager, &encode_view, fbs, camera);

        const frame_estimate: usize = blk: {
            const stream = decoder.stream(.video);
            if (stream.nb_frames > 0) break :blk @intCast(stream.nb_frames);
            if (decoder.fmt_ctx.ptr().duration <= 0) break :blk 0;

            const duration_secs = @as(f64, @floatFromInt(decoder.fmt_ctx.ptr().duration)) / @as(f64, c.AV_TIME_BASE);
            break :blk @intFromFloat(duration_secs * c.av_q2d(decoder.framerate()));
        };

        return .{
            .encoder = encoder,
            .decoder = decoder,
            .manager = manager,
            .double_buffer = double_buffer,
            .upload_buffer = .default,
            .camera = camera,
            .angle_calc = angle_calc,
            .fbs = fbs,
            .refresh_rate = ui.refreshRate() catch 60.0,

            .handles = handles,

            .render_view = render_view,
            .encode_view = encode_view,

            .frame_total = frame_estimate,
        };
    }

    pub fn deinit(self: *EncodeSession, allocator: std.mem.Allocator) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "EncodeSession.deinit" });
        defer zone.end();

        self.stop();

        self.handles.deinit();
        self.manager.deinit(allocator);

        allocator.destroy(self.double_buffer);

        self.encoder.deinit();
        allocator.destroy(self.encoder);

        self.decoder.deinit(allocator);
        allocator.destroy(self.decoder);
    }

    pub fn run(self: *EncodeSession) !void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "EncodeSession.run" });
        defer zone.end();

        const audio_stream = self.decoder.stream(.audio);
        const linesize: Linesize(c.AV_PIX_FMT_NV12) = .init(self.encoder._frame);

        var timer = try std.time.Timer.start();

        const offset_s = 7 * 0.001; // P99.9 for input + draw is gonna be ~6ms

        // NB: displays above (1 / offset_s) will have an interval_s of 0.0
        const interval_s = @max(0.0, (1.0 / self.refresh_rate) - offset_s);

        const target_ns: u64 = @intFromFloat(interval_s * std.time.ns_per_s);

        // hack for when interval_s is 0.0;
        // FIXME: I feel like this really should just be a ran in another thread....
        var just_once = interval_s < std.math.floatEps(f32);

        while (timer.read() < target_ns or just_once) {
            just_once = false;

            // Process any pending audio packets (remux to output)
            while (self.decoder.queue.pkt.audio.tryPop()) |pkt| {
                try self.encoder.writeAudioPacket(audio_stream, pkt);

                var tmp: ?*c.AVPacket = pkt;
                c.av_packet_free(&tmp);
            }

            if (self.decoder.queue.frame.pop()) |frame| {
                defer self.decoder.queue.frame.recycle(frame);
                defer self.double_buffer.swap();

                const z = tracy.Zone.begin(.{ .src = @src(), .name = "frame received" });
                defer z.end();

                const back = self.double_buffer.back();
                uploadPlane(.y, self.manager, back, frame);
                uploadPlane(.uv, self.manager, back, frame);

                try render(&self.render_view, &self.fbs, back.flip(), self.angle_calc, self.manager, self.camera);
                try writeToNv12Tex(self.manager, &self.encode_view, self.fbs, self.camera);

                const width, const height = self.encode_view.get();

                {
                    const pbo_z = tracy.Zone.begin(.{ .src = @src(), .name = "read from opengl" });
                    defer pbo_z.end();

                    const idx = self.upload_buffer.current();

                    // read Y and UV into single contiguous PBO with linesize padding
                    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, self.manager.pbo.get(idx));
                    defer gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);

                    {
                        const read_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.ReadPixels y" });
                        defer read_z.end();

                        gl.PixelStorei(gl.PACK_ROW_LENGTH, @intCast(linesize.y));
                        gl.BindFramebuffer(gl.READ_FRAMEBUFFER, self.manager.fbo.get(.y));
                        gl.ReadPixels(0, 0, width, height, gl.RED, gl.UNSIGNED_BYTE, @ptrFromInt(0));
                    }

                    {
                        const read_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.ReadPixels uv" });
                        defer read_z.end();

                        // after y plane in memory
                        const uv_offset: usize = @intCast(linesize.y * height);
                        gl.PixelStorei(gl.PACK_ROW_LENGTH, @divTrunc(linesize.uv, 2)); // FFMpeg linesize is bytes, RG is 2 bytes per pixel
                        gl.BindFramebuffer(gl.READ_FRAMEBUFFER, self.manager.fbo.get(.uv));
                        gl.ReadPixels(0, 0, @divTrunc(width, 2), @divTrunc(height, 2), gl.RG, gl.UNSIGNED_BYTE, @ptrFromInt(uv_offset));
                    }

                    gl.PixelStorei(gl.PACK_ROW_LENGTH, 0);
                    gl.BindFramebuffer(gl.READ_FRAMEBUFFER, self.fbs.get());

                    {
                        const flush_z = tracy.Zone.begin(.{ .src = @src(), .name = "gl.Flush" });
                        defer flush_z.end();

                        gl.Flush(); // trigger readback immediately
                    }

                    if (self.upload_buffer.next()) |upload| blk: {
                        const y, const uv = mapNv12Frame(self.manager, self.encode_view, upload.id, linesize) orelse break :blk;
                        defer unmapNv12Frame(self.manager, upload.id);

                        try self.encoder.encodeNv12Frame(y, uv, upload.pts);
                        self.frame_count += 1;
                    }
                }

                self.upload_buffer.advance(frame.pts);
            } else if (self.decoder.queue.frame.end_of_stream.load(.monotonic)) {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "final frame received" });
                defer z.end();

                self.upload_buffer.skip(); // to access the pending frames

                while (self.upload_buffer.flush()) |upload| {
                    const y, const uv = mapNv12Frame(self.manager, self.encode_view, upload.id, linesize) orelse continue;
                    defer unmapNv12Frame(self.manager, upload.id);

                    try self.encoder.encodeNv12Frame(y, uv, upload.pts);
                    self.frame_count += 1;
                }

                self.is_finished = true;
                break;
            }
        }
    }
};

pub const App = struct {
    pub const default: @This() = .{ .session = .idle };

    session: Session,

    const log = std.log.scoped(.app);

    pub fn poll(self: *App, allocator: std.mem.Allocator, ui: Ui, state: *GuiState) !void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "App.poll" });
        defer zone.end();

        switch (self.session) {
            .encode => {
                const num: f32 = @floatFromInt(self.session.encode.frame_count);
                const den: f32 = @floatFromInt(self.session.encode.frame_total);
                state.encode_progress = num / den;

                if (self.session.encode.is_finished) {
                    state.request = .idle;

                    if (@abs(state.encode_progress - 1.0) < std.math.floatEps(f32)) {
                        log.warn("encode_progress finished at {d:.2}", .{state.encode_progress});
                    }
                }
            },
            .playback => |playback| {
                const audio = &(playback.decoder.audio_clock orelse @panic("invariant broken"));
                state.volume.value = audio.volume;

                state.progress = .{
                    .timestamp = @floatCast(playback.double_buffer.back().displayTime()),
                    .end_timestamp = @floatCast(self.session.playback.decoder.duration()),
                };

                if (state.action) |action| {
                    defer state.action = null;

                    switch (action) {
                        .SetVolume => |volume| try audio.setVolume(volume),
                        .Seek => |timestamp| log.warn("TODO: Seek to {d:.3}s", .{timestamp}),
                    }
                }
            },
            else => {},
        }

        const request = state.request orelse return;
        self.reset(allocator, state);

        switch (request) {
            .idle => {}, // already idle
            .encode => |paths| {
                const session = try EncodeSession.init(allocator, state, ui, paths.src_path, paths.dst_path);
                self.session = .{ .encode = session };
            },
            .playback => |path| {
                const session = try PlaybackSession.init(allocator, state, ui, path);
                self.session = .{ .playback = session };
            },
        }
    }

    pub fn reset(self: *App, allocator: std.mem.Allocator, state: *GuiState) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "App.reset" });
        defer zone.end();

        self.session.deinit(allocator);
        self.session = .{ .idle = IdleSession.init() };

        state.encode_progress = 0.0;
        state.progress = .default;

        state.request = null;
    }

    pub fn run(self: *App) !void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "App.run" });
        defer zone.end();

        switch (self.session) {
            .idle => {},
            inline .playback, .encode => |*s| try s.run(),
        }
    }

    pub fn video(self: *const App) ?VideoContext {
        return switch (self.session) {
            inline .playback, .encode => |s| .{ .tex_id = s.manager.tex.get(.out), .render_view = s.render_view },
            else => null,
        };
    }

    pub fn deinit(self: *App, allocator: std.mem.Allocator) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "App.deinit" });
        defer zone.end();

        self.session.deinit(allocator);
        self.session = .idle;
    }
};

fn getHwDeviceName(t: c.AVHWDeviceType) []const u8 {
    if (t == c.AV_HWDEVICE_TYPE_NONE) return "software";

    return std.mem.span(c.av_hwdevice_get_type_name(t));
}
