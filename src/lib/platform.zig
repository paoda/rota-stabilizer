const std = @import("std");
const builtin = @import("builtin");
const gl = @import("gl");
const tracy = @import("tracy");
const zgui = @import("zgui");
const nfd = @import("nfd");

const version = @import("build.zig.zon").version;
const default_font = @embedFile("asset/Inter-Medium.ttf");

const Request = @import("../app.zig").Request;
const Action = @import("../app.zig").Action;
const Resolution = @import("../lib.zig").Resolution;
const Viewport = @import("../lib.zig").Viewport;
const GpuResourceManager = @import("../lib.zig").GpuResourceManager;

const AV_HWDEVICE_TYPE_AMF: c_int = if (@hasDecl(c, "AV_HWDEVICE_TYPE_AMF")) c.AV_HWDEVICE_TYPE_AMF else 13;

pub const HwDeviceType = switch (builtin.os.tag) {
    .macos => enum(c.AVHWDeviceType) {
        VideoToolbox = c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
        Software = c.AV_HWDEVICE_TYPE_NONE,
    },
    .windows => enum(c.AVHWDeviceType) {
        CUDA = c.AV_HWDEVICE_TYPE_CUDA,
        QSV = c.AV_HWDEVICE_TYPE_QSV,
        D3D11VA = c.AV_HWDEVICE_TYPE_D3D11VA,
        AMF = AV_HWDEVICE_TYPE_AMF,
        Vulkan = c.AV_HWDEVICE_TYPE_VULKAN,
        Software = c.AV_HWDEVICE_TYPE_NONE,
    },
    .linux => enum(c.AVHWDeviceType) {
        CUDA = c.AV_HWDEVICE_TYPE_CUDA,
        QSV = c.AV_HWDEVICE_TYPE_QSV,
        VAAPI = c.AV_HWDEVICE_TYPE_VAAPI,
        AMF = AV_HWDEVICE_TYPE_AMF,
        Vulkan = c.AV_HWDEVICE_TYPE_VULKAN,
        Software = c.AV_HWDEVICE_TYPE_NONE,
    },
    else => unreachable,
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
        zgui.io.setConfigFlags(.{ .dock_enable = true });
        _ = zgui.io.addFontFromMemory(default_font, 16.0);

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

    if (builtin.os.tag == .macos) {
        const is_apple = std.mem.indexOf(u8, vendor, "Apple") != null;
        if (is_apple) return .{ .VideoToolbox, .VideoToolbox } else return .{ .Software, .Software };
    }

    const is_nvidia = std.mem.indexOf(u8, vendor, "NVIDIA") != null;
    if (is_nvidia) return .{ .CUDA, .CUDA };

    // FIXME is it fine to use QSV like this?
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
const RenderOptions = @import("../main.zig").RenderOptions;

pub const gui = struct {
    pub const VideoContext = struct { tex_id: c_uint, render_view: Viewport };

    var built_layout: bool = false;

    pub const State = struct {
        const default_volume = 0.5;

        input_path: [std.fs.max_path_bytes:0]u8 = [_:0]u8{0} ** std.fs.max_path_bytes,
        output_path: [std.fs.max_path_bytes:0]u8 = [_:0]u8{0} ** std.fs.max_path_bytes,

        hw_dec: HwDeviceType,
        hw_enc: HwDeviceType,

        bit_rate: i32 = 30_000,
        resolution: [2]i32,

        encode_progress: f32 = 0.0,
        progress: VideoProgress = .default,

        local_addr: std.net.Address,

        request: ?Request = null, // session change
        action: ?Action = null,

        volume: VolumeState = .default,

        render: RenderOptions = .{},

        const VideoProgress = struct {
            pub const default: @This() = .{ .duration = null, .timestamp = 0.0 };

            duration: ?f32,
            timestamp: f32,
        };

        const VolumeState = struct {
            pub const default: @This() = .{ .value = 0.5, .cache = 0.5 };

            value: f32,
            cache: f32,
        };

        pub fn init(render_target: Resolution) !State {
            const hw_dec, const hw_enc = guessHardware();

            return .{
                .local_addr = try getLocalIpAddress(),
                .hw_dec = hw_dec,
                .hw_enc = hw_enc,
                .resolution = .{ render_target.width, render_target.height },
            };
        }

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

        const view_pos = zgui.getMainViewport().getWorkPos();

        zgui.setNextWindowPos(.{ .x = view_pos[0], .y = view_pos[1], .cond = .always });
        zgui.setNextWindowSize(.{ .w = @floatFromInt(width), .h = @floatFromInt(height), .cond = .always });

        const show_dockspace = zgui.begin("MainDockSpace", .{
            .flags = .{
                .no_title_bar = true,
                .no_move = true,
                .no_resize = true,
                .no_collapse = true,
                .no_bring_to_front_on_focus = true,
                .no_nav_focus = true,
                .no_docking = true,
                .no_background = true,
            },
        });
        defer zgui.end();

        if (show_dockspace) {
            if (!built_layout) {
                setupDockingLayout("MainDockSpace", ui_view);
                built_layout = true;
            }

            _ = zgui.dockSpace("MainDockSpace", .{ 0.0, 0.0 }, .{ .passthru_central_node = true });
        }

        zgui.pushStyleVar1f(.{ .idx = .frame_rounding, .v = 1.0 });
        defer zgui.popStyleVar(.{});

        try drawSettings(state);
        drawVideoWindow(maybe_video);
        try drawControls(state);
    }

    fn setupDockingLayout(str_id: [:0]const u8, ui_view: Viewport) void {
        const view_size = ui_view.get();
        const dock_id = zgui.getStrIdZ(str_id);
        const size: [2]f32 = .{ @floatFromInt(view_size[0]), @floatFromInt(view_size[1]) };

        _ = zgui.dockBuilderAddNode(dock_id, .{});
        zgui.dockBuilderSetNodeSize(dock_id, size);

        var left_id: zgui.Ident = undefined;
        var video_id: zgui.Ident = undefined;
        _ = zgui.dockBuilderSplitNode(dock_id, .left, 0.25, &left_id, &video_id);

        var settings_id: zgui.Ident = undefined;
        var controls_id: zgui.Ident = undefined;
        _ = zgui.dockBuilderSplitNode(left_id, .down, 0.33, &controls_id, &settings_id);

        zgui.dockBuilderDockWindow("Settings", settings_id);
        zgui.dockBuilderDockWindow("Video", video_id);
        zgui.dockBuilderDockWindow("Controls", controls_id);

        zgui.dockBuilderFinish(dock_id);
    }

    fn drawSettings(state: *State) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const showing = zgui.begin("Settings", .{});
        defer zgui.end();

        if (!showing) return;

        zgui.textDisabled("Hardware Acceleration", .{});

        {
            _ = zgui.comboFromEnum("Decoder", &state.hw_dec);
            _ = zgui.comboFromEnum("Encoder", &state.hw_enc);

            zgui.spacing();

            zgui.textDisabled("Output Settings", .{});
            _ = zgui.inputInt2("Resolution", .{ .v = &state.resolution });
            if (zgui.inputInt("Bitrate (kbps)", .{ .v = &state.bit_rate })) {
                state.bit_rate = @min(100_000, @max(1000, state.bit_rate));
            }
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
        }

        zgui.spacing();
        zgui.textDisabled("Configuration", .{});

        {
            _ = zgui.checkbox("Ring", .{ .v = &state.render.show_ring });

            zgui.sameLine(.{});
            _ = zgui.checkbox("Circle", .{ .v = &state.render.show_circle });

            zgui.sameLine(.{});
            _ = zgui.checkbox("Background", .{ .v = &state.render.show_background });

            zgui.sameLine(.{});
            _ = zgui.checkbox("Border", .{ .v = &state.render.show_border });

            _ = zgui.sliderFloat("Border Radius", .{ .v = &state.render.border_radius, .min = 0.0, .max = 200.0, .cfmt = "%.1f" });

            if (zgui.inputFloat("Zoom", .{ .v = &state.render.zoom, .step = 0.05, .cfmt = "%.2f" })) {
                state.render.zoom = @max(1.0, state.render.zoom);
                state.action = .{ .SetCameraZoom = state.render.zoom };
            }

            if (zgui.inputFloat("BG Zoom", .{ .v = &state.render.background_zoom, .step = 0.05, .cfmt = "%.2f" })) {
                state.render.background_zoom = @max(1.0, state.render.background_zoom);
            }

            zgui.sameLine(.{});
            zgui.textDisabled("(?)", .{});
            if (zgui.isItemHovered(.{}) and zgui.beginTooltip()) {
                defer zgui.endTooltip();

                zgui.text("Note: Zoom of the background texture that lives within the ring", .{});
            }

            _ = zgui.sliderFloat("Border Opacity", .{ .v = &state.render.border_opacity, .min = 0.0, .max = 1.0 });
            _ = zgui.sliderFloat("Ring Opacity", .{ .v = &state.render.ring_opacity, .min = 0.0, .max = 1.0 });
            _ = zgui.sliderFloat("Circle Opacity", .{ .v = &state.render.circle_opacity, .min = 0.0, .max = 1.0 });

            _ = zgui.sliderFloat("##TintIntensity", .{ .v = &state.render.tint_intensity, .min = 0.0, .max = 1.0 });

            zgui.sameLine(.{});
            _ = zgui.colorEdit3("Tint", .{ .col = &state.render.tint, .flags = .{ .no_inputs = true } });
        }

        zgui.spacing();
        zgui.textDisabled("Information", .{});

        {
            const addr = std.mem.toBytes(state.local_addr.in.sa.addr);
            zgui.text("Local IP: {}.{}.{}.{}", .{ addr[0], addr[1], addr[2], addr[3] });
            zgui.text("Tip: CTRL+Click to manually input a value!", .{});
            zgui.text("Tip: For Youtube, I recommend 3840x2160 @ 60_000kbps", .{});
        }

        zgui.spacing();
        zgui.separator();
        zgui.spacing();

        {
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
        }
    }

    fn drawVideoWindow(maybe_video: ?VideoContext) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        zgui.pushStyleVar2f(.{ .idx = .window_padding, .v = .{ 0.0, 0.0 } });
        defer zgui.popStyleVar(.{ .count = 1 });

        const showing = zgui.begin("Video", .{});

        defer zgui.end();

        if (!showing) return;

        const vid = maybe_video orelse {
            return zgui.textDisabled("Video preview will appear here once a frame is available.", .{});
        };
        const video_aspect = vid.render_view.aspect();

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

        zgui.image(.{ .tex_data = null, .tex_id = @enumFromInt(vid.tex_id) }, .{
            .w = w,
            .h = h,
            .uv0 = .{ 0.0, 1.0 },
            .uv1 = .{ 1.0, 0.0 },
        });
    }

    fn drawControls(state: *State) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const showing = zgui.begin("Controls", .{});
        defer zgui.end();

        if (!showing) return;

        const is_muted = state.volume.value < std.math.floatEps(f32);
        if (zgui.button(if (is_muted) "Unmute" else "Mute", .{})) {
            state.action = .{ .SetVolume = if (is_muted) state.volume.cache else 0.0 };
        }

        zgui.sameLine(.{});

        {
            zgui.pushItemWidth(-1.0);
            defer zgui.popItemWidth();

            if (zgui.sliderFloat("##Volume", .{ .min = 0.0, .max = 1.0, .v = &state.volume.value })) {
                state.volume.cache = state.volume.value;
                state.action = .{ .SetVolume = state.volume.value };
            }
        }

        zgui.spacing();

        if (state.encode_progress > 0.0) {
            zgui.progressBar(.{
                .fraction = state.encode_progress,
                .w = -1.0,
                .overlay = "Encoding...",
            });
        } else {
            zgui.pushItemWidth(-1.0);
            defer zgui.popItemWidth();

            const duration = state.progress.duration orelse 0.0;

            // FIXME: until seeking impl, always disabled
            if (duration == 0.0 or true) zgui.beginDisabled(.{});
            defer if (duration == 0.0 or true) zgui.endDisabled();

            if (zgui.sliderFloat("##Progress", .{ .min = 0.0, .max = duration, .v = &state.progress.timestamp })) {
                state.action = .{ .Seek = state.progress.timestamp };
            }
        }

        zgui.textDisabled("TODO: add more controls here", .{});
    }

    // FIXME: does zig not handle this?
    fn setPath(dst: *[std.fs.max_path_bytes:0]u8, src: [:0]const u8) void {
        const payload = @min(src.len, std.fs.max_path_bytes - 1);

        @memset(dst[0..], 0);
        @memcpy(dst[0..payload], src[0..payload]);
    }
};

/// dummy udp connection to figure out what the active local ip is
fn getLocalIpAddress() !std.net.Address {
    const target = try std.net.Address.parseIp4("8.8.8.8", 53);

    const sock = try std.posix.socket(std.posix.AF.INET, std.posix.SOCK.DGRAM, std.posix.IPPROTO.UDP);
    defer std.posix.close(sock);

    try std.posix.connect(sock, &target.any, target.getOsSockLen());

    var local_addr: std.net.Address = undefined;
    var addr_len: std.posix.socklen_t = @sizeOf(@TypeOf(local_addr.any));
    try std.posix.getsockname(sock, &local_addr.any, &addr_len);

    return local_addr;
}
