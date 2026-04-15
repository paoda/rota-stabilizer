const std = @import("std");
const tracy = @import("tracy");
const c = @import("lib.zig").c;

const Decoder = @import("lib/codec.zig").Decoder;
const Camera = @import("main.zig").Camera;

const DoubleBuffer = @import("lib.zig").DoubleBuffer;
const GpuResourceManager = @import("lib.zig").GpuResourceManager;
const Viewport = @import("lib.zig").Viewport;
const Ui = @import("lib/platform.zig").Ui;
const FbStack = @import("lib.zig").FbStack;
const VideoContext = @import("lib/platform.zig").gui.VideoContext;

const AngleCalc = @import("main.zig").AngleCalc;

const preload = @import("main.zig").preload;
const render = @import("main.zig").render;
const uploadPlane = @import("main.zig").uploadPlane;

const startup = @import("main.zig").startup;
const shutdown = @import("main.zig").shutdown;
const platform = @import("lib/platform.zig");
const signal = platform.signal;

const GuiState = @import("lib/platform.zig").gui.State;

const State = enum { idle, playback, encode };

const log = std.log.scoped(.app);

pub const Request = union(State) {
    idle: void,
    playback: []const u8,
    encode: struct { src_path: []const u8, dst_path: []const u8 },
};

const Session = union(State) {
    idle: void,
    playback: PlaybackSession,
    encode: EncodeSession,

    pub fn deinit(self: *Session, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .idle => {},
            .playback => |*s| s.deinit(allocator),
            .encode => |*s| s.deinit(allocator),
        }
    }
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

    // tracy output buffer
    buf: [0x100]u8 = undefined,

    pub fn init(allocator: std.mem.Allocator, ui: Ui, path: []const u8) !PlaybackSession {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "PlaybackSession.init" });
        defer zone.end();

        const hwdec, _ = platform.guessHardware();
        log.debug("guessed {s} hw decode", .{if (hwdec) |t| std.mem.span(c.av_hwdevice_get_type_name(t)) else "no"});

        const decoder = try allocator.create(Decoder);
        errdefer allocator.destroy(decoder);

        decoder.* = try Decoder.init(allocator, hwdec, path, false);
        errdefer decoder.deinit(allocator);

        const double_buffer = try allocator.create(DoubleBuffer);
        errdefer allocator.destroy(double_buffer);

        double_buffer.* = .{};

        const handles = try decoder.spawn(null);
        errdefer handles.deinit();

        var render_view: Viewport = .default;
        try render_view.push(startup.render_target.width, startup.render_target.height);

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

        shutdown(&self.decoder.queue);

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

        var should_render: bool = false;

        while (true) { // necessary for when Display Hz !== Video FPS

            if (self.next_frame == null) {
                if (self.decoder.queue.frame.tryPop()) |next| {
                    self.next_frame = next;
                    tracy.frameMarkStart("video timing");
                }
            }

            if (self.next_frame) |frame| {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "have next frame" });
                defer z.end();

                std.debug.assert(frame.format == c.AV_PIX_FMT_NV12);

                const back = self.double_buffer.back();
                const next_frame_time = @as(f64, @floatFromInt(frame.pts)) * time_base;
                const drop_diff_s = next_frame_time - audio_clock.seconds_passed();

                if (drop_diff_s < -self.delay_threshold) {
                    const msg = try std.fmt.bufPrint(
                        &self.buf,
                        "dropped frame | d: {d:.3} a: {d:.3} v: {d:.3}",
                        .{ drop_diff_s, -(drop_diff_s - next_frame_time), next_frame_time },
                    );

                    tracy.message(.{ .text = msg });
                    tracy.frameMarkEnd("video timing");

                    self.decoder.queue.frame.recycle(frame);
                    self.next_frame = null;
                    continue;
                }

                const already_uploaded = @abs(next_frame_time - back.displayTime()) < std.math.floatEps(f64);

                if (!already_uploaded) {
                    tracy.plot(.{ .name = "Buffered Frames", .value = .{ .i64 = @intCast(self.decoder.queue.frame.len()) } });

                    const upload_z = tracy.Zone.begin(.{ .src = @src(), .name = "upload next frame" });
                    defer upload_z.end();

                    // this is a new frame, upload it to the BACK buffer
                    uploadPlane(.y, self.manager, back, frame);
                    uploadPlane(.uv, self.manager, back, frame);
                    back.setDisplayTime(next_frame_time);
                }

                // after upload, recalculate to check if it's time to swap
                const audio_time = audio_clock.seconds_passed();
                const diff_s = next_frame_time - audio_time;

                if (diff_s - self.lookahead <= 0) { // Time to display the newer frame!
                    tracy.plot(.{ .name = "Audio Time (s)", .value = .{ .f64 = audio_time } });
                    tracy.plot(.{ .name = "Next Frame Time (s)", .value = .{ .f64 = next_frame_time } });
                    tracy.plot(.{ .name = "A/V Sync Drift (ms)", .value = .{ .f64 = diff_s * std.time.ms_per_s } });
                    tracy.frameMarkEnd("video timing");

                    self.double_buffer.swap();

                    self.decoder.queue.frame.recycle(frame);
                    self.next_frame = null;
                    should_render = true;
                    continue;
                }
            }

            break;
        }

        if (should_render) {
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
    pub fn init() EncodeSession {
        @panic("TODO: implement");
    }

    pub fn deinit(self: EncodeSession, allocator: std.mem.Allocator) void {
        _ = self;
        _ = allocator;
        @panic("TODO: deinit");
    }
};

pub const App = struct {
    pub const default: @This() = .{ .session = .idle };

    session: Session,

    pub fn poll(self: *App, allocator: std.mem.Allocator, ui: Ui, state: *GuiState) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const request = state.request orelse return;

        // reset necessary state
        self.session.deinit(allocator);
        self.session = .idle;

        state.request = null;

        // we don't actually want to exit the program, but we did set should_quit in self.session.deinit
        while (signal.should_quit.load(.monotonic)) signal.should_quit.store(false, .monotonic);

        switch (request) {
            .idle => self.session = .idle,
            .encode => |_| self.session = .{
                .encode = @panic("TODO: call EncodeSession.init"),
            },
            .playback => |path| self.session = .{
                .playback = try PlaybackSession.init(allocator, ui, path),
            },
        }
    }

    pub fn run(self: *App) !void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "Session.run" });
        defer zone.end();

        switch (self.session) {
            .idle, .encode => {},
            .playback => |*s| try s.run(),
        }
    }

    pub fn video(self: *const App) ?VideoContext {
        return switch (self.session) {
            .playback => |s| .{ .tex_id = s.manager.tex.get(.out), .render_view = s.render_view },
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
