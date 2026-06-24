const std = @import("std");
const builtin = @import("builtin");
const gl = @import("gl");
const tracy = @import("tracy");
const zgui = @import("zgui");
const nfd = @import("znfde");
const known_folders = @import("known-folders");
const qrcodegen = @import("qrcodegen");

const version = @import("build.zig.zon").version;
const default_font = @embedFile("asset/Inter-Medium.ttf");

const Request = @import("../app.zig").Request;
const Action = @import("../app.zig").Action;
const Resolution = @import("../lib.zig").Resolution;
const Viewport = @import("../lib.zig").Viewport;
const Errors = @import("../lib.zig").Errors;
const GpuResourceManager = @import("../lib.zig").GpuResourceManager;

const AV_HWDEVICE_TYPE_AMF: c_int = if (@hasDecl(c, "AV_HWDEVICE_TYPE_AMF")) c.AV_HWDEVICE_TYPE_AMF else 13;

const errors = &@import("../lib.zig").errors;

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
    const title: [:0]const u8 = "Rotaeno Stabilizer";

    window: *c.SDL_Window,
    gl_ctx: c.SDL_GLContext,

    const log = std.log.scoped(.ui);

    pub fn init(allocator: std.mem.Allocator, resolution: Resolution) !Ui {
        const width = resolution.width;
        const height = resolution.height;

        c.SDL_SetMainReady();

        try errify(c.SDL_Init(c.SDL_INIT_VIDEO | c.SDL_INIT_AUDIO));
        errdefer c.SDL_Quit();

        try errify(c.SDL_SetAppMetadata(title, version, "moe.paoda.rota-stabilizer"));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));
        try errify(c.SDL_GL_SetAttribute(c.SDL_GL_ALPHA_SIZE, 8));

        const window_flags: c.SDL_WindowFlags = c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE;
        const window: *c.SDL_Window = try errify(c.SDL_CreateWindow(title, width, height, window_flags));
        errdefer c.SDL_DestroyWindow(window);

        const gl_ctx = try errify(c.SDL_GL_CreateContext(window));
        errdefer _ = c.SDL_GL_DestroyContext(gl_ctx);

        try errify(c.SDL_GL_MakeCurrent(window, gl_ctx));
        errdefer _ = c.SDL_GL_MakeCurrent(window, null);

        if (!gl_procs.init(c.SDL_GL_GetProcAddress)) return error.gl_init_failed;

        gl.makeProcTableCurrent(&gl_procs);
        errdefer gl.makeProcTableCurrent(null);

        zgui.init(allocator);
        errdefer zgui.deinit();

        zgui.backend.init(window, gl_ctx);
        errdefer zgui.backend.deinit();

        zgui.io.setIniFilename(null);
        zgui.io.setConfigFlags(.{ .dock_enable = true });

        var config = zgui.FontConfig.init();
        @memcpy(config.name[0..12], "Inter Medium");

        _ = zgui.io.addFontFromMemoryWithConfig(default_font, 16, config, null);

        log.info("OpenGL device: {?s}", .{gl.GetString(gl.RENDERER)});
        log.info("OpenGL support (want 3.3): {?s}", .{gl.GetString(gl.VERSION)});

        gl.Enable(gl.BLEND);
        gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.Disable(gl.FRAMEBUFFER_SRGB);
        _ = c.SDL_GL_SetSwapInterval(1);

        try nfd.init();
        errdefer nfd.deinit();

        return .{ .window = window, .gl_ctx = gl_ctx };
    }

    pub fn deinit(self: @This()) void {
        nfd.deinit();

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
    if (builtin.os.tag == .macos) return .{ .VideoToolbox, .VideoToolbox };

    const vendor = std.mem.span(gl.GetString(gl.VENDOR)) orelse return .{ .Software, .Software };

    const is_nvidia = std.mem.indexOf(u8, vendor, "NVIDIA") != null;
    if (is_nvidia) return .{ .CUDA, .CUDA };

    const is_intel = std.mem.indexOf(u8, vendor, "Intel") != null;
    if (is_intel) return switch (builtin.os.tag) {
        .linux => .{ .VAAPI, .VAAPI },
        // FIXME(paoda): qsv encoding?
        .windows => .{ .D3D11VA, .QSV },
        else => unreachable,
    };

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

    pub fn setupHandler() void {
        switch (builtin.os.tag) {
            .windows => {
                std.os.windows.SetConsoleCtrlHandler(windowsHandler, true) catch |e| {
                    errors.add_win_signal_handler_err(e);
                    // but then ignore the error
                };
            },
            else => std.posix.sigaction(
                std.posix.SIG.INT,
                &.{ .handler = .{ .handler = posixHandler }, .mask = std.posix.sigemptyset(), .flags = 0 },
                null,
            ),
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
        input_path: [std.fs.max_path_bytes:0]u8 = @splat(0),
        output_path: [std.fs.max_path_bytes:0]u8 = @splat(0),
        default_path: ?[:0]const u8,

        hw_dec: HwDeviceType,
        hw_enc: HwDeviceType,

        bit_rate: i32 = 30_000,
        resolution: [2]i32,

        encode_progress: f32 = 0.0,
        progress: VideoProgress = .default,

        net: Network,

        request: ?Request = null, // session change
        action: ?Action = null,

        volume: VolumeState = .default,

        render: RenderOptions = .{},

        const Network = struct {
            local_addr: ?std.net.Address,
            qr: QrCode,

            fn init(allocator: std.mem.Allocator) !Network {
                var qr = QrCode.init();

                const local_addr = getLocalIpAddress();
                if (local_addr) |addr| try qr.updateTexture(allocator, addr);

                return .{ .local_addr = local_addr, .qr = qr };
            }

            fn deinit(self: Network, allocator: std.mem.Allocator) void {
                self.qr.deinit(allocator);
            }
        };

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

        pub fn init(self: *State, allocator: std.mem.Allocator, render_target: Resolution) !void {
            const hw_dec, const hw_enc = guessHardware();

            self.* = .{
                .default_path = getVideoDirectory(allocator) catch null,
                .net = try Network.init(allocator),
                .hw_dec = hw_dec,
                .hw_enc = hw_enc,
                .resolution = .{ render_target.width, render_target.height },
            };
        }

        pub fn deinit(self: State, allocator: std.mem.Allocator) void {
            if (self.default_path) |path| allocator.free(path);
            self.net.deinit(allocator);
        }

        pub fn defaultHardware(self: *State) void {
            self.hw_dec, self.hw_enc = guessHardware();
        }
    };

    pub fn draw(allocator: std.mem.Allocator, state: *State, ui_view: Viewport, maybe_video: ?VideoContext) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const width, const height = ui_view.get();

        zgui.backend.newFrame(@intCast(width), @intCast(height));

        if (builtin.mode == .Debug) zgui.showDemoWindow(null);

        {
            const view_pos = zgui.getMainViewport().getWorkPos();
            zgui.setNextWindowPos(.{ .x = view_pos[0], .y = view_pos[1], .cond = .always });
            zgui.setNextWindowSize(.{ .w = @floatFromInt(width), .h = @floatFromInt(height), .cond = .always });

            _ = zgui.begin("MainDockSpace", .{
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

            if (!built_layout) {
                setupDockingLayout("MainDockSpace", ui_view);
                built_layout = true;
            }

            _ = zgui.dockSpace("MainDockSpace", .{ 0.0, 0.0 }, .{ .passthru_central_node = true });
        }

        {
            zgui.pushStyleVar1f(.{ .idx = .frame_rounding, .v = 1.0 });
            defer zgui.popStyleVar(.{});

            try drawSettings(allocator, state);
            drawVideoWindow(maybe_video);
            drawControls(state);

            if (errors.messages.items.len != 0) {
                const x, const y = zgui.getMainViewport().getCenter();
                zgui.setNextWindowPos(.{ .x = x, .y = y, .cond = .appearing, .pivot_x = 0.5, .pivot_y = 0.5 });
                zgui.openPopup("Error", .{});
            }

            if (zgui.beginPopupModal("Error", .{ .flags = .{ .always_auto_resize = true } })) {
                defer zgui.endPopup();

                const message = errors.messages.items[0];
                zgui.text("{s}", .{message});

                const remaining = errors.messages.items.len -| 1;
                if (remaining != 0) {
                    zgui.textDisabled("({} more error(s) pending)", .{remaining});
                }

                zgui.spacing();
                zgui.separator();

                if (zgui.button("OK", .{ .w = -1.0 })) {
                    _ = errors.messages.orderedRemove(0);
                    errors.allocator.free(message);
                    zgui.closeCurrentPopup();
                }
            }
        }

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "zgui.backend.draw" });
            defer z.end();

            zgui.backend.draw();
        }
    }

    fn setupDockingLayout(str_id: [:0]const u8, ui_view: Viewport) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const view_size = ui_view.get();
        const dock_id = zgui.getStrIdZ(str_id);
        const size: [2]f32 = .{ @floatFromInt(view_size[0]), @floatFromInt(view_size[1]) };

        // TODO(paoda): figure out how to do this with the other tabs
        _ = zgui.dockBuilderAddNode(dock_id, .{ .auto_hide_tab_bar = true });
        zgui.dockBuilderSetNodeSize(dock_id, size);

        var left_id: zgui.Ident = undefined;
        var video_id: zgui.Ident = undefined;
        _ = zgui.dockBuilderSplitNode(dock_id, .left, 0.25, &left_id, &video_id);

        var settings_id: zgui.Ident = undefined;
        var controls_id: zgui.Ident = undefined;
        _ = zgui.dockBuilderSplitNode(left_id, .down, 0.15, &controls_id, &settings_id);

        zgui.dockBuilderDockWindow("Settings", settings_id);
        zgui.dockBuilderDockWindow("Video", video_id);
        zgui.dockBuilderDockWindow("Controls", controls_id);

        zgui.dockBuilderFinish(dock_id);
    }

    fn drawSettings(allocator: std.mem.Allocator, state: *State) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const showing = zgui.begin("Settings", .{});
        defer zgui.end();

        if (!showing) return;

        drawActionButtons(state);

        zgui.spacing();

        try drawMediaSettings(allocator, state);

        zgui.spacing();
        zgui.separator();
        zgui.spacing();

        if (zgui.beginTabBar("ConfigTabs", .{})) {
            defer zgui.endTabBar();

            if (zgui.beginTabItem("Render", .{})) {
                defer zgui.endTabItem();

                zgui.spacing();
                drawRenderSettings(state);
            }

            if (zgui.beginTabItem("Hardware & Output", .{})) {
                defer zgui.endTabItem();

                zgui.spacing();
                drawHardwareSettings(state);
            }

            if (zgui.beginTabItem("Upload", .{})) {
                defer zgui.endTabItem();

                zgui.spacing();
                drawUploadPanel(state);
            }

            if (zgui.beginTabItem("Info", .{})) {
                defer zgui.endTabItem();

                zgui.spacing();
                drawInfoPanel(state);
            }
        }
    }

    fn drawMediaSettings(allocator: std.mem.Allocator, state: *State) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const name = "Screen Recordings";
        const spec = "mp4,mkv,mov,webm";

        zgui.pushItemWidth(-zgui.calcTextSize("Browse...", .{})[0] - 20.0);
        defer zgui.popItemWidth();

        _ = zgui.inputTextWithHint("##Input", .{ .hint = "Input Video Path...", .buf = &state.input_path });

        zgui.sameLine(.{});
        if (zgui.button("Browse...##input", .{})) {
            if (try nfd.openFileDialog(allocator, &.{.{ .name = name, .spec = spec }}, state.default_path)) |path| {
                defer allocator.free(path);
                setPath(&state.input_path, path);
            }
        }

        _ = zgui.inputTextWithHint("##Output", .{ .hint = "Output Video Path (Optional)...", .buf = &state.output_path });

        zgui.sameLine(.{});
        if (zgui.button("Browse...##output", .{})) {
            if (try nfd.saveFileDialog(allocator, &.{.{ .name = name, .spec = spec }}, state.default_path, "output.mp4")) |path| {
                defer allocator.free(path);
                setPath(&state.output_path, path);
            }
        }
    }

    fn drawActionButtons(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const input_path: [:0]const u8 = std.mem.sliceTo(state.input_path[0..], 0);
        const is_possible = input_path.len != 0;

        if (!is_possible) zgui.beginDisabled(.{});
        defer if (!is_possible) zgui.endDisabled();

        // center the cursor
        const w = 80;
        const spacing = zgui.getStyle().item_spacing[0];
        const width = (w * 3.0) + (spacing * 2.0);

        const avail_width = zgui.getContentRegionAvail()[0];
        zgui.setCursorPosX((avail_width - width) / 2.0);

        if (zgui.button("\u{25ba} Play", .{ .w = 80, .h = 30 })) {
            state.request = .{ .playback = input_path };
        }

        zgui.sameLine(.{});

        if (zgui.button("\u{25cf} Encode", .{ .w = 80, .h = 30 })) {
            const path = blk: {
                const str = std.mem.sliceTo(state.output_path[0..], 0);

                if (str.len == 0) {
                    if (state.default_path) |path| {
                        var buf: [std.fs.max_path_bytes]u8 = undefined;
                        const file_path = std.fmt.bufPrintZ(&buf, "{s}{s}{s}", .{ path, std.fs.path.sep_str, "output.mp4" }) catch unreachable;

                        setPath(&state.output_path, file_path);
                    }
                }

                break :blk std.mem.sliceTo(state.output_path[0..], 0);
            };

            state.request = .{ .encode = .{ .src_path = input_path, .dst_path = path } };
            state.encode_progress = 0.0; // FIXME: don't set this here
        }

        zgui.sameLine(.{});

        if (zgui.button("\u{25a0} Stop", .{ .w = 80, .h = 30 })) {
            state.request = .idle;
            state.encode_progress = 0.0; // FIXME: don't set this here
        }
    }

    fn drawHardwareSettings(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        if (zgui.beginTable("HardwareForm", .{ .column = 2 })) {
            defer zgui.endTable();

            zgui.tableSetupColumn("Label", .{ .flags = .{ .width_fixed = true } });
            zgui.tableSetupColumn("Input", .{ .flags = .{ .width_stretch = true } });

            _ = zgui.tableNextColumn();
            zgui.alignTextToFramePadding();
            zgui.text("Decoder", .{});

            _ = zgui.tableNextColumn();
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                _ = zgui.comboFromEnum("##Decoder", &state.hw_dec);
            }

            _ = zgui.tableNextColumn();
            zgui.alignTextToFramePadding();
            zgui.text("Encoder", .{});

            _ = zgui.tableNextColumn();
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                _ = zgui.comboFromEnum("##Encoder", &state.hw_enc);
            }

            _ = zgui.tableNextColumn();
            zgui.alignTextToFramePadding();
            zgui.text("Resolution", .{});

            _ = zgui.tableNextColumn();
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                if (zgui.inputInt2("##Resolution", .{ .v = &state.resolution })) {
                    state.resolution[0] = @max(1, state.resolution[0]);
                    state.resolution[1] = @max(1, state.resolution[1]);
                }
            }

            _ = zgui.tableNextColumn();
            zgui.alignTextToFramePadding();
            zgui.text("Bitrate (kbps)", .{});

            _ = zgui.tableNextColumn();
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                if (zgui.inputInt("##Bitrate", .{ .v = &state.bit_rate })) {
                    state.bit_rate = @min(100_000, @max(1000, state.bit_rate));
                }
            }
        }
    }

    fn drawRenderSettings(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        // Push a negative width so all sliders and inputs stretch consistently,
        // leaving a uniform margin on the right side.
        zgui.pushItemWidth(-100.0);
        defer zgui.popItemWidth();

        {
            zgui.textDisabled("Background", .{});
            _ = zgui.checkbox("##BackgroundEnabled", .{ .v = &state.render.show_background });

            if (!state.render.show_background) zgui.beginDisabled(.{});
            defer if (!state.render.show_background) zgui.endDisabled();

            zgui.sameLine(.{});
            zgui.text("Zoom", .{});

            zgui.sameLine(.{ .spacing = 2 });
            zgui.textDisabled("(?)", .{});
            if (zgui.isItemHovered(.{}) and zgui.beginTooltip()) {
                defer zgui.endTooltip();
                zgui.text("Zoom of the background texture that lives within the ring", .{});
            }

            zgui.sameLine(.{});

            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                if (zgui.inputFloat("##BackgroundZoom", .{ .v = &state.render.background_zoom, .step = 0.05, .cfmt = "%.2f" })) {
                    state.render.background_zoom = @max(1.0, state.render.background_zoom);
                }
            }
        }

        zgui.spacing();
        zgui.separator();
        zgui.spacing();

        {
            zgui.textDisabled("Border", .{});

            if (zgui.beginTable("BorderOptions", .{ .column = 5 })) {
                defer zgui.endTable();

                zgui.tableSetupColumn("Enabled", .{ .flags = .{ .width_fixed = true } });
                zgui.tableSetupColumn("Opacity", .{ .flags = .{ .width_fixed = true } });
                zgui.tableSetupColumn("Slider1", .{ .flags = .{ .width_stretch = true } });
                zgui.tableSetupColumn("Radius", .{ .flags = .{ .width_fixed = true } });
                zgui.tableSetupColumn("Slider2", .{ .flags = .{ .width_stretch = true } });

                _ = zgui.tableNextColumn();
                _ = zgui.checkbox("##BorderEnabled", .{ .v = &state.render.show_border });

                if (!state.render.show_border) zgui.beginDisabled(.{});
                defer if (!state.render.show_border) zgui.endDisabled();

                _ = zgui.tableNextColumn();
                zgui.text("Opacity", .{});

                _ = zgui.tableNextColumn();
                {
                    zgui.pushItemWidth(-1.0);
                    defer zgui.popItemWidth();

                    _ = zgui.sliderFloat("##Opacity", .{ .v = &state.render.border_opacity, .min = 0.0, .max = 1.0 });
                }

                _ = zgui.tableNextColumn();
                zgui.text("Radius", .{});

                _ = zgui.tableNextColumn();
                {
                    zgui.pushItemWidth(-1.0);
                    defer zgui.popItemWidth();

                    _ = zgui.sliderFloat("##Radius", .{ .v = &state.render.border_radius, .min = 0.0, .max = 200.0 });
                }
            }
        }

        zgui.spacing();
        zgui.separator();
        zgui.spacing();

        {
            zgui.textDisabled("Ring", .{});
            _ = zgui.checkbox("##RingEnabled", .{ .v = &state.render.show_ring });

            zgui.sameLine(.{});

            if (!state.render.show_ring) zgui.beginDisabled(.{});
            defer if (!state.render.show_ring) zgui.endDisabled();

            zgui.text("Opacity", .{});

            zgui.sameLine(.{});
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                _ = zgui.sliderFloat("##RingOpacity", .{ .v = &state.render.ring_opacity, .min = 0.0, .max = 1.0 });
            }
        }

        zgui.spacing();
        zgui.separator();
        zgui.spacing();

        {
            zgui.textDisabled("Circle", .{});
            _ = zgui.checkbox("##CircleEnabled", .{ .v = &state.render.show_circle });

            zgui.sameLine(.{});

            if (!state.render.show_circle) zgui.beginDisabled(.{});
            defer if (!state.render.show_circle) zgui.endDisabled();

            zgui.text("Opacity", .{});

            zgui.sameLine(.{});
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                _ = zgui.sliderFloat("##CircleOpacity", .{ .v = &state.render.circle_opacity, .min = 0.0, .max = 1.0 });
            }
        }

        zgui.spacing();
        zgui.separator();
        zgui.spacing();

        {
            zgui.textDisabled("Global View", .{});

            zgui.alignTextToFramePadding();
            zgui.text("Zoom", .{});

            zgui.sameLine(.{});
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                if (zgui.inputFloat("##Zoom", .{ .v = &state.render.zoom, .step = 0.05, .cfmt = "%.2f" })) {
                    state.render.zoom = @max(1.0, state.render.zoom);
                    state.action = .{ .SetCameraZoom = state.render.zoom };
                }
            }

            zgui.spacing();

            zgui.alignTextToFramePadding();
            zgui.text("Tint", .{});

            zgui.sameLine(.{});
            _ = zgui.colorEdit3("##Tint", .{ .col = &state.render.tint, .flags = .{ .no_inputs = true } });

            zgui.sameLine(.{});
            zgui.text("Intensity", .{});

            zgui.sameLine(.{});
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                _ = zgui.sliderFloat("##Intensity", .{ .v = &state.render.tint_intensity, .min = 0.0, .max = 1.0 });
            }
        }
    }

    fn drawUploadPanel(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const panel_width = zgui.getContentRegionAvail()[0];

        zgui.text("Scan to upload a video over Wi-Fi", .{});

        if (state.default_path) |path| {
            zgui.sameLine(.{ .spacing = 2 });
            zgui.textDisabled("(?)", .{});
            if (zgui.isItemHovered(.{}) and zgui.beginTooltip()) {
                defer zgui.endTooltip();
                zgui.text("files will be uploaded to '{s}'", .{path});
            }
        }

        zgui.spacing();
        zgui.separator();
        zgui.spacing();

        if (state.net.local_addr) |address| {
            const qr_size = @min(panel_width - 16.0, 180.0);

            zgui.setCursorPosX((panel_width - qr_size) / 2.0);
            zgui.image(
                .{ .tex_data = null, .tex_id = @enumFromInt(state.net.qr.tex_id[0]) },
                .{ .w = qr_size, .h = qr_size, .uv0 = .{ 0.0, 0.0 }, .uv1 = .{ 1.0, 1.0 } },
            );

            zgui.spacing();

            const ip_label = blk: {
                const bytes = std.mem.toBytes(address.in.sa.addr);

                var buf: [0xF]u8 = undefined;
                break :blk std.fmt.bufPrint(&buf, "{d}.{d}.{d}.{d}", .{ bytes[0], bytes[1], bytes[2], bytes[3] }) catch unreachable;
            };
            const ip_width = zgui.calcTextSize(ip_label, .{})[0];

            zgui.setCursorPosX((panel_width - ip_width) / 2.0);
            zgui.textDisabled("{s}", .{ip_label});
        } else {
            const label = "Waiting for network...";
            const label_width = zgui.calcTextSize(label, .{})[0];

            zgui.setCursorPosX((panel_width - label_width) / 2.0);
            zgui.textDisabled("{s}", .{label});
        }
    }

    fn drawInfoPanel(_: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        zgui.text("TIPS:", .{});
        zgui.bulletText("CTRL+Click to manually input a value!", .{});
        zgui.bulletText("For YouTube: 3840x2160 @ 60_000kbps", .{});

        const version_label = blk: {
            var buf: [64]u8 = undefined;
            break :blk std.fmt.bufPrintZ(&buf, "rota-stabilizer v{s}", .{version}) catch unreachable;
        };
        const version_width = zgui.calcTextSize(version_label, .{})[0];
        const line_height = zgui.getTextLineHeightWithSpacing();
        const panel_height = zgui.getContentRegionAvail()[1];

        zgui.setCursorPosY(zgui.getCursorPosY() + panel_height - line_height);
        zgui.setCursorPosX((zgui.getContentRegionAvail()[0] - version_width) / 2.0);
        zgui.textDisabled("{s}", .{version_label});
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
            const text = "VIDEO PREVIEW";

            const panel_width, const panel_height = zgui.getContentRegionAvail();
            const text_width, const text_height = zgui.calcTextSize(text, .{});

            zgui.setCursorPosX((panel_width - text_width) / 2.0);
            zgui.setCursorPosY((panel_height - text_height) / 2.0);

            return zgui.textDisabled(text, .{});
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

    fn drawControls(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const showing = zgui.begin("Controls", .{});
        defer zgui.end();

        if (!showing) return;

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

        zgui.spacing();

        if (zgui.beginTable("VolumeControl", .{ .column = 3 })) {
            defer zgui.endTable();

            zgui.tableSetupColumn("Label", .{ .flags = .{ .width_fixed = true } });
            zgui.tableSetupColumn("Slider", .{ .flags = .{ .width_stretch = true } });
            zgui.tableSetupColumn("Button", .{ .flags = .{ .width_fixed = true } });

            _ = zgui.tableNextColumn();
            zgui.alignTextToFramePadding();
            zgui.text("Volume", .{});

            _ = zgui.tableNextColumn();
            {
                zgui.pushItemWidth(-1.0);
                defer zgui.popItemWidth();

                if (zgui.sliderFloat("##Volume", .{ .min = 0.0, .max = 1.0, .v = &state.volume.value })) {
                    state.volume.cache = state.volume.value;
                    state.action = .{ .SetVolume = state.volume.value };
                }
            }

            _ = zgui.tableNextColumn();

            const w = zgui.calcTextSize("Unmute", .{})[0] + 2 * zgui.getStyle().frame_padding[0];
            const is_muted = state.volume.value < std.math.floatEps(f32);
            if (zgui.button(if (is_muted) "Unmute" else "Mute", .{ .w = w })) {
                state.action = .{ .SetVolume = if (is_muted) state.volume.cache else 0.0 };
            }
        }

        // TODO(paoda): add more controls here
    }

    // FIXME: does zig not handle this?
    fn setPath(dst: *[std.fs.max_path_bytes:0]u8, src: [:0]const u8) void {
        const payload = @min(src.len, std.fs.max_path_bytes - 1);

        @memset(dst[0..], 0);
        @memcpy(dst[0..payload], src[0..payload]);
    }
};

/// dummy udp connection to maybe figure out what the active local ip is
fn getLocalIpAddress() ?std.net.Address {
    const target = std.net.Address.parseIp4("8.8.8.8", 53) catch return null;

    const sock = std.posix.socket(
        std.posix.AF.INET,
        std.posix.SOCK.DGRAM,
        std.posix.IPPROTO.UDP,
    ) catch return null;
    defer std.posix.close(sock);

    std.posix.connect(sock, &target.any, target.getOsSockLen()) catch |e| {
        errors.add_local_ip_err(e);
        return null;
    };

    var local_addr: std.net.Address = undefined;
    var addr_len: std.posix.socklen_t = @sizeOf(@TypeOf(local_addr.any));
    std.posix.getsockname(sock, &local_addr.any, &addr_len) catch return null;

    return local_addr;
}

pub fn getVideoDirectory(allocator: std.mem.Allocator) !?[:0]const u8 {
    const base_path = try known_folders.getPath(allocator, .videos) orelse return null;
    defer allocator.free(base_path);

    const path = try std.fs.path.joinZ(allocator, &.{ base_path, "rota-stabilizer" });
    errdefer allocator.free(path);

    std.fs.makeDirAbsolute(path) catch |e| if (e != error.PathAlreadyExists) return e;

    return path;
}

const QrCode = struct {
    const RGB24_BPP = @import("../lib.zig").RGB24_BPP;

    buf: []u8 = &.{},
    tex_id: [1]c_uint,

    // FIXME(paoda): generating textures outside of GpuResourceManager
    pub fn init() QrCode {
        var ids: [1]c_uint = undefined;
        gl.GenTextures(1, ids[0..]);

        gl.BindTexture(gl.TEXTURE_2D, ids[0]);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

        return .{ .tex_id = ids };
    }

    fn deinit(self: QrCode, allocator: std.mem.Allocator) void {
        if (self.buf.len != 0) allocator.free(self.buf);
        gl.DeleteTextures(1, self.tex_id[0..]);
    }

    pub fn updateTexture(self: *QrCode, allocator: std.mem.Allocator, address: std.net.Address) !void {
        const MAX_LEN: usize = "http://255.255.255.255:65535".len;

        const url = blk: {
            var addr = address;
            addr.setPort(8080); // TODO(paoda): make configurable

            var buf: [MAX_LEN + 1]u8 = undefined;
            break :blk std.fmt.bufPrintZ(&buf, "http://{f}", .{addr}) catch unreachable;
        };

        const qrcode = try qrcodegen.encodeText(allocator, url, .Low, .Auto, .{});
        defer allocator.free(qrcode);

        const size: usize = @intCast(qrcodegen.getSize(qrcode));
        const padding = 1;
        const padded_size = size + (padding * 2);
        const required_len = padded_size * padded_size * RGB24_BPP;

        if (self.buf.len < required_len) {
            self.buf = try allocator.realloc(self.buf, required_len);
        }

        for (0..padded_size) |y| {
            for (0..padded_size) |x| {
                const i = (y * padded_size + x) * RGB24_BPP;

                const is_quiet_zone =
                    x < padding or
                    x >= size + padding or
                    y < padding or
                    y >= size + padding;

                if (is_quiet_zone) {
                    @memset(self.buf[i..][0..RGB24_BPP], 0xFF);
                } else {
                    const is_dark = qrcodegen.getModule(qrcode, @intCast(x - padding), @intCast(y - padding));
                    @memset(self.buf[i..][0..RGB24_BPP], if (is_dark) 0x00 else 0xFF);
                }
            }
        }

        gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);
        gl.BindTexture(gl.TEXTURE_2D, self.tex_id[0]);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.PixelStorei(gl.UNPACK_ALIGNMENT, 1);
        defer gl.PixelStorei(gl.UNPACK_ALIGNMENT, 4);

        // Allocate and upload in one step using the actual dimension
        gl.TexImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGB8,
            @intCast(padded_size),
            @intCast(padded_size),
            0,
            gl.RGB,
            gl.UNSIGNED_BYTE,
            self.buf.ptr,
        );
    }
};
