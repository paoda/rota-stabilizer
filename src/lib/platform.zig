const std = @import("std");
const builtin = @import("builtin");
const gl = @import("gl");
const tracy = @import("tracy");
const zgui = @import("zgui");
const nfd = @import("nfd");

const version = @import("build.zig.zon").version;

const Request = @import("../app.zig").Request;
const Resolution = @import("../lib.zig").Resolution;
const Viewport = @import("../lib.zig").Viewport;
const GpuResourceManager = @import("../lib.zig").GpuResourceManager;

const AV_HWDEVICE_TYPE_AMF: c_int = if (@hasDecl(c, "AV_HWDEVICE_TYPE_AMF")) c.AV_HWDEVICE_TYPE_AMF else 13;

pub const HwDeviceType = enum(c.AVHWDeviceType) {
    VideoToolbox = c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
    CUDA = c.AV_HWDEVICE_TYPE_CUDA,
    QSV = c.AV_HWDEVICE_TYPE_QSV,
    VAAPI = c.AV_HWDEVICE_TYPE_VAAPI,
    D3D11VA = c.AV_HWDEVICE_TYPE_D3D11VA,
    AMF = AV_HWDEVICE_TYPE_AMF,
    Software = c.AV_HWDEVICE_TYPE_NONE,
};

const c = @import("../lib.zig").c;

pub var gl_procs: gl.ProcTable = undefined;

pub const Ui = struct {
    window: *c.SDL_Window,
    gl_ctx: c.SDL_GLContext,

    const log = std.log.scoped(.ui);

    pub fn init(allocator: std.mem.Allocator, resolution: Resolution) !Ui {
        const width = resolution.width;
        const height = resolution.height;

        c.SDL_SetMainReady();

        try errify(c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_AUDIO));
        errdefer c.SDL_Quit();

        try errify(c.SDL_SetAppMetadata("Rotaeno Stabilizer", version, "moe.paoda.rota-stabilizer"));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_ALPHA_SIZE, 8));

        const window_flags: c.SDL_WindowFlags = c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE;
        const window: *c.SDL_Window = try errify(c.SDL_CreateWindow("Rotaeno Stabilizer", width, height, window_flags));
        errdefer c.SDL_DestroyWindow(window);

        const gl_ctx = try errify(c.SDL_GL_CreateContext(window));
        errdefer errify(c.SDL_GL_DestroyContext(gl_ctx)) catch {};

        try errify(c.SDL_GL_MakeCurrent(window, gl_ctx));
        errdefer errify(c.SDL_GL_MakeCurrent(window, null)) catch {};

        if (!gl_procs.init(c.SDL_GL_GetProcAddress)) return error.gl_init_failed;

        gl.makeProcTableCurrent(&gl_procs);
        errdefer gl.makeProcTableCurrent(null);

        zgui.init(allocator);
        zgui.backend.init(window, gl_ctx);
        zgui.io.setIniFilename(null);

        log.info("OpenGL device: {?s}", .{gl.GetString(gl.RENDERER)});
        log.info("OpenGL support (want 3.3): {?s}", .{gl.GetString(gl.VERSION)});

        gl.Enable(gl.BLEND);
        gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.Disable(gl.FRAMEBUFFER_SRGB);
        _ = c.SDL_GL_SetSwapInterval(1);

        return .{ .window = window, .gl_ctx = gl_ctx };
    }

    pub fn deinit(self: @This()) void {
        zgui.backend.deinit();
        zgui.deinit();

        gl.makeProcTableCurrent(null);
        _ = c.SDL_GL_MakeCurrent(self.window, null);
        _ = c.SDL_GL_DestroyContext(self.gl_ctx);
        c.SDL_DestroyWindow(self.window);

        c.SDL_Quit();
    }

    pub fn windowSize(self: @This()) ![2]c_int {
        var w: c_int, var h: c_int = .{ undefined, undefined };
        try errify(c.SDL_GetWindowSizeInPixels(self.window, &w, &h));

        return .{ w, h };
    }

    pub fn refreshRate(self: @This()) !f32 {
        const id = try errify(c.SDL_GetDisplayForWindow(self.window));
        const mode = try errify(c.SDL_GetCurrentDisplayMode(id));

        return mode.*.refresh_rate;
    }

    pub fn toggleFullscreen(self: @This()) !void {
        if (c.SDL_GetWindowFlags(self.window) & c.SDL_WINDOW_FULLSCREEN != 0) {
            try errify(c.SDL_SetWindowFullscreen(self.window, false));
        } else {
            try errify(c.SDL_SetWindowFullscreen(self.window, true));
        }
    }

    pub fn swap(self: @This()) !void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .color = .gray25 });
        defer zone.end();

        try errify(c.SDL_GL_SwapWindow(self.window));
        tracy.frameMark(null);
    }
};

// https://github.com/castholm/zig-examples/blob/77a829c85b5ddbad673026d504626015db4093ac/opengl-sdl/main.zig#L200-L219
pub inline fn errify(value: anytype) error{sdl_error}!switch (@typeInfo(@TypeOf(value))) {
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

pub fn guessHardware() struct { HwDeviceType, HwDeviceType } {
    const vendor = std.mem.span(gl.GetString(gl.VENDOR)) orelse return .{ .Software, .Software };

    const is_apple = std.mem.indexOf(u8, vendor, "Apple") != null;
    if (is_apple) return .{ .VideoToolbox, .VideoToolbox };

    const is_nvidia = std.mem.indexOf(u8, vendor, "NVIDIA") != null;
    if (is_nvidia) return .{ .CUDA, .CUDA };

    const is_intel = std.mem.indexOf(u8, vendor, "Intel") != null;
    if (is_intel) return .{ .QSV, .QSV };

    const is_amd = std.mem.indexOf(u8, vendor, "AMD") != null;
    const is_ati = std.mem.indexOf(u8, vendor, "ATI") != null;

    if (is_amd or is_ati) return switch (builtin.os.tag) {
        .linux => .{ .VAAPI, .VAAPI },
        .windows => .{ .D3D11VA, .AMF },
        else => unreachable,
    };

    return .{ .Software, .Software };
}

pub const signal = struct {
    const AtomicBool = std.atomic.Value(bool);
    const log = std.log.scoped(.signal_handler);

    pub var should_quit: AtomicBool = .init(false); // should be global

    pub fn setupHandler() !void {
        switch (builtin.os.tag) {
            .windows => try std.os.windows.SetConsoleCtrlHandler(windowsHandler, true),
            else => std.posix.sigaction(std.posix.SIG.INT, &.{ .handler = .{ .handler = posixHandler }, .mask = std.posix.sigemptyset(), .flags = 0 }, null),
        }

        log.debug("setup {s} CTRL-C signal handler", .{@tagName(builtin.os.tag)});
    }

    fn windowsHandler(ctrl_type: std.os.windows.DWORD) callconv(.winapi) std.os.windows.BOOL {
        switch (ctrl_type) {
            std.os.windows.CTRL_C_EVENT, std.os.windows.CTRL_BREAK_EVENT => {
                signal.should_quit.store(true, .monotonic);
                return std.os.windows.TRUE;
            },
            else => return std.os.windows.FALSE,
        }
    }

    fn posixHandler(_: i32) callconv(.c) void {
        signal.should_quit.store(true, .monotonic);
    }
};

const startup = @import("../main.zig").startup;

pub const gui = struct {
    pub const VideoContext = struct { tex_id: c_uint, render_view: Viewport };

    pub const State = struct {
        pub const default: @This() = .{
            .input_path = [_:0]u8{0} ** std.fs.max_path_bytes,
            .output_path = [_:0]u8{0} ** std.fs.max_path_bytes,
            .request = null,
            .hw_dec = .Software,
            .hw_enc = .Software,
            .bit_rate = 30_000,
            .encode_progress = 0.0,
            .resolution = .{ startup.render_target.width, startup.render_target.height },
        };

        input_path: [std.fs.max_path_bytes:0]u8,
        output_path: [std.fs.max_path_bytes:0]u8,

        hw_dec: HwDeviceType,
        hw_enc: HwDeviceType,

        bit_rate: i32,
        resolution: [2]i32,

        encode_progress: f32,

        request: ?Request,

        pub fn defaultHardware(self: *State) void {
            self.hw_dec, self.hw_enc = guessHardware();
        }
    };

    pub fn draw(state: *State, ui_view: Viewport, maybe_video: ?VideoContext) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const width, const height = ui_view.get();

        zgui.backend.newFrame(@intCast(width), @intCast(height));
        defer zgui.backend.draw();

        if (builtin.mode == .Debug) zgui.showDemoWindow(null);

        try drawSettings(state);

        if (maybe_video) |vid| {
            drawVideoWindow(@floatFromInt(width), vid.render_view, vid.tex_id);
        }
    }

    fn drawSettings(state: *State) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        // zgui.setNextWindowSize(.{ .w = 500, .h = 0 });

        const showing = zgui.begin("Settings", .{ .flags = .{ .always_auto_resize = true } });
        defer zgui.end();

        if (!showing) return;

        {
            zgui.textDisabled("Hardware Acceleration", .{});
            _ = zgui.comboFromEnum("Decoder", &state.hw_dec);
            _ = zgui.comboFromEnum("Encoder", &state.hw_enc);

            zgui.spacing();
            zgui.textDisabled("Output Settings", .{});

            _ = zgui.dragInt2("Resolution", .{ .v = &state.resolution });

            _ = zgui.sliderInt("Target Bitrate (kbps)", .{
                .v = &state.bit_rate,
                .max = 60_000,
                .min = 10_000,
            });
        }

        zgui.spacing();
        zgui.textDisabled("Media Files", .{});

        {
            const filter = "mp4,mkv,mov,webm";

            if (zgui.button("Browse...##input", .{})) {
                const maybe_path = try nfd.openFileDialog(filter, null);
                if (maybe_path) |path| setPath(&state.input_path, path);
            }

            zgui.sameLine(.{});
            _ = zgui.inputText("Input", .{ .buf = &state.input_path });

            if (zgui.button("Browse...##output", .{})) {
                const maybe_path = try nfd.saveFileDialog(filter, null);
                if (maybe_path) |path| setPath(&state.output_path, path);
            }

            zgui.sameLine(.{});
            _ = zgui.inputTextWithHint("Output", .{ .hint = "output.mp4", .buf = &state.output_path });

            zgui.spacing();
            zgui.separator();
            zgui.spacing();

            const input_path: [:0]const u8 = std.mem.sliceTo(state.input_path[0..], 0);
            const is_possible = input_path.len != 0;

            {
                if (!is_possible) zgui.beginDisabled(.{});
                defer if (!is_possible) zgui.endDisabled();

                if (zgui.button("Play", .{})) state.request = .{ .playback = input_path };
            }

            zgui.sameLine(.{});

            {
                if (!is_possible) zgui.beginDisabled(.{});
                defer if (!is_possible) zgui.endDisabled();

                if (zgui.button("Start Encode", .{})) {
                    const output_path = blk: {
                        const str = std.mem.sliceTo(state.output_path[0..], 0);
                        if (str.len == 0) setPath(&state.output_path, "output.mp4");

                        break :blk std.mem.sliceTo(state.output_path[0..], 0);
                    };

                    state.request = .{ .encode = .{ .src_path = input_path, .dst_path = output_path } };
                    state.encode_progress = 0.0;
                }
            }

            zgui.sameLine(.{});

            if (zgui.button("Stop", .{})) {
                state.request = .idle;
                state.encode_progress = 0.0;
            }

            if (state.encode_progress > 0.0) {
                zgui.spacing();

                zgui.progressBar(.{
                    .fraction = state.encode_progress,
                    .w = -1.0,
                    .h = 0.0,
                    .overlay = "Encoding...",
                });
            }
        }
    }

    fn drawVideoWindow(window_width: f32, render_view: Viewport, tex_id: c_uint) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const vw, const vh = render_view.get();
        const video_aspect = @as(f32, @floatFromInt(vw)) / @as(f32, @floatFromInt(vh));

        const scale = 0.80;
        const width = window_width * scale;
        const height = @ceil(width / video_aspect);

        zgui.setNextWindowSize(.{
            .w = width,
            .h = height + zgui.getFrameHeight(),
            .cond = .first_use_ever,
        });

        zgui.pushStyleVar2f(.{ .idx = .window_padding, .v = .{ 0.0, 0.0 } });
        defer zgui.popStyleVar(.{ .count = 1 });

        const showing = zgui.begin("Video", .{});

        defer zgui.end();

        if (!showing) return;

        const dw, const dh = zgui.getContentRegionAvail();
        if (dw <= 0 or dh <= 0) return;

        const window_aspect = dw / dh;
        const w = if (window_aspect > video_aspect) dh * video_aspect else dw;
        const h = if (window_aspect > video_aspect) dh else dw / video_aspect;

        // horizontal + vertical center
        const pos = zgui.getCursorPos();
        zgui.setCursorPos(.{
            pos[0] + (dw - w) * 0.5,
            pos[1] + (dh - h) * 0.5,
        });

        zgui.image(.{ .tex_data = null, .tex_id = @enumFromInt(tex_id) }, .{
            .w = w,
            .h = h,
            .uv0 = .{ 0.0, 1.0 },
            .uv1 = .{ 1.0, 0.0 },
        });
    }

    // FIXME: does zig not handle this?
    fn setPath(dst: *[std.fs.max_path_bytes:0]u8, src: [:0]const u8) void {
        const payload = @min(src.len, std.fs.max_path_bytes - 1);

        @memset(dst[0..], 0);
        @memcpy(dst[0..payload], src[0..payload]);
    }
};
