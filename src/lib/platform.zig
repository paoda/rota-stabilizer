const std = @import("std");
const builtin = @import("builtin");
const gl = @import("gl");
const tracy = @import("tracy");
const nfd = @import("znfde");
const known_folders = @import("known-folders");
const qrcodegen = @import("qrcodegen");
const imgui = @import("imgui.zig");

const version = @import("build.zig.zon").version;
const default_font = @embedFile("asset/Inter-Medium.ttf");

const Request = @import("../app.zig").Request;
const Action = @import("../app.zig").Action;
const Resolution = @import("../lib.zig").Resolution;
const ContentRect = @import("../lib.zig").ContentRect;
const Viewport = @import("../lib.zig").Viewport;
const Errors = @import("../lib.zig").Errors;
const GpuResourceManager = @import("../lib.zig").GpuResourceManager;

const AV_HWDEVICE_TYPE_AMF: c_int = if (@hasDecl(c, "AV_HWDEVICE_TYPE_AMF")) c.AV_HWDEVICE_TYPE_AMF else 13;

const errors = &@import("../lib.zig").errors;

const DecodeHwDeviceType = switch (builtin.os.tag) {
    .macos => enum(c.AVHWDeviceType) {
        VideoToolbox = c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
        Software = c.AV_HWDEVICE_TYPE_NONE,
    },
    .windows => enum(c.AVHWDeviceType) {
        CUDA = c.AV_HWDEVICE_TYPE_CUDA,
        D3D11VA = c.AV_HWDEVICE_TYPE_D3D11VA,
        Vulkan = c.AV_HWDEVICE_TYPE_VULKAN,
        Software = c.AV_HWDEVICE_TYPE_NONE,
    },
    .linux => enum(c.AVHWDeviceType) {
        CUDA = c.AV_HWDEVICE_TYPE_CUDA,
        VAAPI = c.AV_HWDEVICE_TYPE_VAAPI,
        Vulkan = c.AV_HWDEVICE_TYPE_VULKAN,
        Software = c.AV_HWDEVICE_TYPE_NONE,
    },
    else => unreachable,
};

const EncodeHwDeviceType = switch (builtin.os.tag) {
    .macos => enum(c.AVHWDeviceType) {
        VideoToolbox = c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
        Software = c.AV_HWDEVICE_TYPE_NONE,
    },
    .windows => enum(c.AVHWDeviceType) {
        CUDA = c.AV_HWDEVICE_TYPE_CUDA,
        QSV = c.AV_HWDEVICE_TYPE_QSV,
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

    view: Viewport,

    const log = std.log.scoped(.ui);

    pub fn init(resolution: Resolution) !Ui {
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

        const main_scale = c.SDL_GetDisplayContentScale(c.SDL_GetPrimaryDisplay());
        const w: c_int = @intFromFloat(@as(f32, @floatFromInt(width)) * main_scale);
        const h: c_int = @intFromFloat(@as(f32, @floatFromInt(height)) * main_scale);

        const window_flags: c.SDL_WindowFlags = c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE | c.SDL_WINDOW_HIGH_PIXEL_DENSITY;
        const window: *c.SDL_Window = try errify(c.SDL_CreateWindow(title, w, h, window_flags));
        errdefer c.SDL_DestroyWindow(window);

        const gl_ctx = try errify(c.SDL_GL_CreateContext(window));
        errdefer _ = c.SDL_GL_DestroyContext(gl_ctx);

        try errify(c.SDL_GL_MakeCurrent(window, gl_ctx));
        errdefer _ = c.SDL_GL_MakeCurrent(window, null);

        if (!gl_procs.init(c.SDL_GL_GetProcAddress)) return error.gl_init_failed;

        gl.makeProcTableCurrent(&gl_procs);
        errdefer gl.makeProcTableCurrent(null);

        var view: Viewport = .default;
        try view.push(width, height);

        imgui.init(null);
        errdefer imgui.deinit();

        imgui.styleColourDark(null);

        imgui.backend.init(window, gl_ctx);
        errdefer imgui.backend.deinit();

        const config_flags: imgui.ConfigFlags = .{ .dock_enable = true };

        const io = imgui.getIo();
        io.inner.IniFilename = null;
        io.inner.ConfigFlags |= @bitCast(config_flags);

        const style = imgui.getStyle();
        style.scaleAllSizes(main_scale);
        style.inner.FontScaleDpi = main_scale;

        var config: imgui.FontConfig = undefined;
        config.init("Inter Medium");

        _ = io.addFontFromMemoryTTF(default_font, 16, &config);

        log.info("OpenGL device: {?s}", .{gl.GetString(gl.RENDERER)});
        log.info("OpenGL support (want 3.3): {?s}", .{gl.GetString(gl.VERSION)});

        gl.Enable(gl.BLEND);
        gl.BlendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.Disable(gl.FRAMEBUFFER_SRGB);
        _ = c.SDL_GL_SetSwapInterval(1);

        try nfd.init();
        errdefer nfd.deinit();

        return .{
            .window = window,
            .gl_ctx = gl_ctx,
            .view = view,
        };
    }

    pub fn deinit(self: @This()) void {
        nfd.deinit();

        imgui.backend.deinit();
        imgui.deinit();

        gl.makeProcTableCurrent(null);
        _ = c.SDL_GL_MakeCurrent(self.window, null);
        _ = c.SDL_GL_DestroyContext(self.gl_ctx);
        c.SDL_DestroyWindow(self.window);

        c.SDL_Quit();
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

pub fn guessHardware() struct { DecodeHwDeviceType, EncodeHwDeviceType } {
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
            .windows => return, // TODO(paoda): unsupported
            else => std.posix.sigaction(
                std.posix.SIG.INT,
                &.{ .handler = .{ .handler = handler }, .mask = std.posix.sigemptyset(), .flags = 0 },
                null,
            ),
        }

        log.debug("setup {s} CTRL-C signal handler", .{@tagName(builtin.os.tag)});
    }

    fn handler(_: std.posix.SIG) callconv(.c) void {
        signal.should_quit.store(true, .monotonic);
    }
};

const startup = @import("../main.zig").startup;
const RenderOptions = @import("../main.zig").RenderOptions;

pub const gui = struct {
    pub const VideoContext = struct {
        tex_id: c_uint,
        render_view: Viewport,

        resolution: Resolution,
    };

    var built_layout: bool = false;

    pub const State = struct {
        input_path: [std.fs.max_path_bytes:0]u8 = @splat(0),
        output_path: [std.fs.max_path_bytes:0]u8 = @splat(0),
        default_path: ?[:0]const u8,

        hw_dec: DecodeHwDeviceType,
        hw_enc: EncodeHwDeviceType,

        bit_rate: i32 = 30_000,
        resolution: [2]i32,

        is_listening: bool = false,
        encode_progress: f32 = 0.0,
        progress: VideoProgress = .default,

        net: Network,

        request: ?Request = null, // session change
        action: ?Action = null,

        volume: VolumeState = .default,

        render: RenderOptions = .{},
        source: VideoSourceInfo = .{},

        fullscreen: bool = false,
        init_view: Viewport,

        const Network = struct {
            local_addr: ?std.Io.net.IpAddress,
            qr: QrCode,

            fn init(io: std.Io, allocator: std.mem.Allocator) !Network {
                var qr = QrCode.init();

                const local_addr = getLocalIpAddress(io);
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

        pub const VideoSourceInfo = struct {
            const DetectStatus = enum { idle, searching, validating, locked, failed };

            resolution: [2]i32 = .{ 0, 0 },
            detected: ?struct { rect: ContentRect, target: Resolution } = null,

            // TODO(paoda): merge with enum in DetectStatus?
            detect_status: DetectStatus = .idle,

            pub fn isSetManually(self: VideoSourceInfo) bool {
                const width, const height = self.resolution;
                return width > 0 and height > 0;
            }

            pub fn effectiveRect(self: VideoSourceInfo, res: Resolution) ContentRect {
                if (self.isSetManually()) {
                    const width, const height = self.resolution;

                    const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
                    return .fromAspect(res, aspect);
                }

                if (self.detected) |result| {
                    if (result.target.eql(res)) return result.rect;
                }

                return .full(res);
            }
        };

        const VolumeState = struct {
            pub const default: @This() = .{ .value = 0.5, .cache = 0.5 };

            value: f32,
            cache: f32,
        };

        pub fn init(self: *State, io: std.Io, environ_map: *std.process.Environ.Map, allocator: std.mem.Allocator, view: Viewport, render_target: Resolution) !void {
            const hw_dec, const hw_enc = guessHardware();

            self.* = .{
                .default_path = getVideoDirectory(io, environ_map, allocator) catch null,
                .net = try Network.init(io, allocator),
                .hw_dec = hw_dec,
                .hw_enc = hw_enc,
                .resolution = .{ render_target.width, render_target.height },
                .init_view = view,
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

    pub fn draw(allocator: std.mem.Allocator, ui: Ui, state: *State, maybe_video: ?VideoContext) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        imgui.backend.newFrame();

        if (builtin.mode == .Debug) imgui.showDemoWindow(null);

        if (!state.fullscreen) {
            const viewport = imgui.getMainViewport();

            const dock_id = imgui.getIDFromString("MainDockSpace");

            if (imgui.dockBuilderGetNode(dock_id) == null) {
                setupDockingLayout(dock_id, viewport);
            }

            _ = imgui.dockSpaceOverViewport(dock_id, viewport, .{ .passthru_central_node = true });
        }

        {
            imgui.pushStyleVar(.frame_rounding, 1.0);
            defer imgui.popStyleVar(.{});

            if (state.fullscreen) {
                drawVideoWindow(ui, state, maybe_video);
            } else {
                const maybe_frame = if (maybe_video) |vid| vid.resolution else null;

                try drawSettings(allocator, state, maybe_frame);
                drawVideoWindow(ui, state, maybe_video);
                drawControls(state);
            }

            drawErrorPopup();
        }

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "imgui.backend.render" });
            defer z.end();

            imgui.backend.render();
        }
    }

    fn drawErrorPopup() void {
        errors.mutex.lock();
        defer errors.mutex.unlock();

        if (errors.messages.len() != 0 and !imgui.isPopupOpen("Error")) {
            const center = imgui.getCenter(imgui.getMainViewport());
            // FIXME(paoda): what about the (0.5, 0.5) pivot?
            imgui.setNextWindowPosCentered(center, .appearing);
            imgui.openPopup("Error");
        }

        if (imgui.beginPopupModal("Error", .{ .always_auto_resize = true })) {
            defer imgui.endPopup();

            // this and the allocator.free line assert that the mutex is live for at least as long as this block
            imgui.text("{?s}", .{errors.messages.peek()});

            const remaining = errors.messages.len() -| 1;
            if (remaining != 0) {
                imgui.textDisabled("({} more error(s) pending)", .{remaining});
            }

            imgui.spacing();
            imgui.separator();

            if (imgui.button("OK", .{ .w = -1.0 })) {
                errors.allocator.free(errors.messages.pop() orelse unreachable);
                imgui.closeCurrentPopup();
            }
        }
    }

    fn setupDockingLayout(id: imgui.Id, viewport: *imgui.Viewport) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        _ = imgui.dockBuilderAddNode(id, .{ .dock_space = true });
        _ = imgui.dockBuilderSetNodeSize(id, .{ viewport.Size.x, viewport.Size.y });

        var left_id: imgui.Id = undefined;
        var video_id: imgui.Id = undefined;
        _ = imgui.dockBuilderSplitNode(
            id,
            .left,
            0.30,
            &left_id,
            &video_id,
        );

        var settings_id: imgui.Id = undefined;
        var controls_id: imgui.Id = undefined;
        _ = imgui.dockBuilderSplitNode(
            left_id,
            .down,
            0.10,
            &controls_id,
            &settings_id,
        );

        imgui.dockBuilderDockWindow("Settings", settings_id);
        imgui.dockBuilderDockWindow("Video", video_id);
        imgui.dockBuilderDockWindow("Controls", controls_id);

        imgui.dockBuilderFinish(id);
    }

    fn drawSettings(allocator: std.mem.Allocator, state: *State, maybe_frame: ?Resolution) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const overrides: imgui.DockNodeFlags = .{ .no_tab_bar = true };
        var class: imgui.WindowClass = .{ .DockNodeFlagsOverrideSet = @bitCast(overrides) };
        imgui.setNextWindowClass(&class);

        const showing = imgui.begin("Settings", null, .{});
        defer imgui.end();

        if (!showing) return;

        drawActionButtons(state);

        imgui.spacing();

        try drawMediaSettings(allocator, state);

        imgui.spacing();
        imgui.separator();
        imgui.spacing();

        if (imgui.beginTabBar("ConfigTabs")) {
            defer imgui.endTabBar();

            if (imgui.beginTabItem("Render")) {
                defer imgui.endTabItem();

                imgui.spacing();
                drawRenderSettings(state, maybe_frame);
            }

            if (imgui.beginTabItem("Hardware & Output")) {
                defer imgui.endTabItem();

                imgui.spacing();
                drawHardwareSettings(state);
            }

            if (imgui.beginTabItem("Upload")) {
                defer imgui.endTabItem();

                imgui.spacing();
                drawUploadPanel(state);
            }

            if (imgui.beginTabItem("Info")) {
                defer imgui.endTabItem();

                imgui.spacing();
                drawInfoPanel(state);
            }
        }
    }

    fn drawMediaSettings(allocator: std.mem.Allocator, state: *State) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const name = "Screen Recordings";
        const spec = "mp4,mkv,mov,webm";

        imgui.pushItemWidth(-imgui.calcTextSize("Browse...")[0] - 20.0);
        defer imgui.popItemWidth();

        _ = imgui.inputTextWithHint("##Input", "Input Video Path...", state.input_path[0..]);

        imgui.sameLine(.{});
        if (imgui.button("Browse...##input", .{})) {
            if (try nfd.openFileDialog(allocator, &.{.{ .name = name, .spec = spec }}, state.default_path)) |path| {
                defer allocator.free(path);
                setPath(&state.input_path, path);
            }
        }

        _ = imgui.inputTextWithHint("##Output", "Output Video Path (Optional)...", state.output_path[0..]);

        imgui.sameLine(.{});
        if (imgui.button("Browse...##output", .{})) {
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

        const w = 75;
        const spacing = imgui.getStyle().inner.ItemSpacing.x;
        const width = (w * 4.0) + (spacing * 3.0);

        imgui.beginDisabled(!is_possible);

        const off = (imgui.getContentRegionAvail()[0] - width) / 2.0;
        if (off > 0) imgui.setCursorPosX(imgui.getCursorPosX() + off);

        if (imgui.button("\u{25ba} Play", .{ .w = w, .h = 30 })) {
            state.request = .{ .playback = input_path };
        }

        imgui.sameLine(.{});

        if (imgui.button("\u{2913} Encode", .{ .w = w, .h = 30 })) {
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

        imgui.endDisabled();

        imgui.sameLine(.{});

        {
            imgui.beginDisabled(state.is_listening);
            defer imgui.endDisabled();

            if (imgui.button("\u{25cf} Stream", .{ .w = w, .h = 30 })) {
                state.request = .listen;
            }
        }

        imgui.sameLine(.{});

        if (imgui.button("\u{25a0} Stop", .{ .w = w, .h = 30 })) {
            state.request = .idle;
            state.encode_progress = 0.0; // FIXME: don't set this here
        }

        if (state.is_listening) {
            const msg = "Awaiting SRT connection on 0.0.0.0:8090\u{2026}";
            const text_size = imgui.calcTextSize(msg);

            const off_text = (imgui.getContentRegionAvail()[0] - text_size[0]) / 2.0;
            if (off_text > 0) imgui.setCursorPosX(imgui.getCursorPosX() + off_text);

            imgui.textDisabled(msg, .{});
        }
    }

    fn drawHardwareSettings(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        if (imgui.beginTable("HardwareForm", 2)) {
            defer imgui.endTable();

            imgui.tableSetupColumn("Label", .{ .width_fixed = true });
            imgui.tableSetupColumn("Input", .{ .width_stretch = true });

            _ = imgui.tableNextColumn();
            imgui.alignTextToFramePadding();
            imgui.text("Decoder", .{});

            _ = imgui.tableNextColumn();
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                _ = imgui.comboFromEnum("##Decoder", &state.hw_dec);
            }

            _ = imgui.tableNextColumn();
            imgui.alignTextToFramePadding();
            imgui.text("Encoder", .{});

            _ = imgui.tableNextColumn();
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                _ = imgui.comboFromEnum("##Encoder", &state.hw_enc);
            }

            _ = imgui.tableNextColumn();
            imgui.alignTextToFramePadding();
            imgui.text("Resolution", .{});

            _ = imgui.tableNextColumn();
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                if (imgui.inputInt2("##Resolution", &state.resolution)) {
                    state.resolution[0] = @max(1, state.resolution[0]);
                    state.resolution[1] = @max(1, state.resolution[1]);
                }
            }

            _ = imgui.tableNextColumn();
            imgui.alignTextToFramePadding();
            imgui.text("Bitrate (kbps)", .{});

            _ = imgui.tableNextColumn();
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                if (imgui.inputInt("##Bitrate", &state.bit_rate)) {
                    state.bit_rate = @min(100_000, @max(1000, state.bit_rate));
                }
            }
        }
    }

    fn drawRenderSettings(state: *State, maybe_frame: ?Resolution) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        // Push a negative width so all sliders and inputs stretch consistently,
        // leaving a uniform margin on the right side.
        imgui.pushItemWidth(-100.0);
        defer imgui.popItemWidth();

        {
            imgui.textDisabled("Background", .{});
            _ = imgui.checkbox("##BackgroundEnabled", &state.render.show_background);

            imgui.beginDisabled(!state.render.show_background);
            defer imgui.endDisabled();

            imgui.sameLine(.{});
            imgui.text("Zoom", .{});

            imgui.sameLine(.{ .spacing = 2 });
            imgui.textDisabled("(?)", .{});
            if (imgui.isItemHovered() and imgui.beginTooltip()) {
                defer imgui.endTooltip();
                imgui.text("Zoom of the background texture that lives within the ring", .{});
            }

            imgui.sameLine(.{});

            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                if (imgui.inputFloat("##BackgroundZoom", &state.render.background_zoom, .{ .step = 0.05, .cfmt = "%.2f" })) {
                    state.render.background_zoom = @max(1.0, state.render.background_zoom);
                }
            }
        }

        imgui.spacing();
        imgui.separator();
        imgui.spacing();

        {
            imgui.textDisabled("Border", .{});

            if (imgui.beginTable("BorderOptions", 5)) {
                defer imgui.endTable();

                imgui.tableSetupColumn("Enabled", .{ .width_fixed = true });
                imgui.tableSetupColumn("Opacity", .{ .width_fixed = true });
                imgui.tableSetupColumn("Slider1", .{ .width_stretch = true });
                imgui.tableSetupColumn("Radius", .{ .width_fixed = true });
                imgui.tableSetupColumn("Slider2", .{ .width_stretch = true });

                _ = imgui.tableNextColumn();
                _ = imgui.checkbox("##BorderEnabled", &state.render.show_border);

                imgui.beginDisabled(!state.render.show_border);
                defer imgui.endDisabled();

                _ = imgui.tableNextColumn();
                imgui.text("Opacity", .{});

                _ = imgui.tableNextColumn();
                {
                    imgui.pushItemWidth(-1.0);
                    defer imgui.popItemWidth();

                    _ = imgui.sliderFloat("##Opacity", &state.render.border_opacity, 0.0, 1.0);
                }

                _ = imgui.tableNextColumn();
                imgui.text("Radius", .{});

                _ = imgui.tableNextColumn();
                {
                    imgui.pushItemWidth(-1.0);
                    defer imgui.popItemWidth();

                    _ = imgui.sliderFloat("##Radius", &state.render.border_radius, 0.0, 200.0);
                }
            }
        }

        imgui.spacing();
        imgui.separator();
        imgui.spacing();

        {
            imgui.textDisabled("Ring", .{});
            _ = imgui.checkbox("##RingEnabled", &state.render.show_ring);

            imgui.sameLine(.{});

            imgui.beginDisabled(!state.render.show_ring);
            defer imgui.endDisabled();

            imgui.text("Opacity", .{});

            imgui.sameLine(.{});
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                _ = imgui.sliderFloat("##RingOpacity", &state.render.ring_opacity, 0.0, 1.0);
            }
        }

        imgui.spacing();
        imgui.separator();
        imgui.spacing();

        {
            imgui.textDisabled("Circle", .{});

            _ = imgui.checkbox("##CircleEnabled", &state.render.show_circle);

            imgui.sameLine(.{});

            imgui.beginDisabled(!state.render.show_circle);
            defer imgui.endDisabled();

            imgui.text("Opacity", .{});

            imgui.sameLine(.{});
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                _ = imgui.sliderFloat("##CircleOpacity", &state.render.circle_opacity, 0.0, 1.0);
            }
        }

        imgui.spacing();
        imgui.separator();
        imgui.spacing();

        {
            imgui.textDisabled("Global View", .{});

            imgui.alignTextToFramePadding();
            imgui.text("Zoom", .{});

            imgui.sameLine(.{});
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                if (imgui.inputFloat("##Zoom", &state.render.zoom, .{ .step = 0.05, .cfmt = "%.2f" })) {
                    state.render.zoom = @max(1.0, state.render.zoom);
                    state.action = .{ .SetCameraZoom = state.render.zoom };
                }
            }

            imgui.spacing();

            {
                imgui.alignTextToFramePadding();
                imgui.text("Tint", .{});

                imgui.sameLine(.{});
                _ = imgui.colorEdit3("##Tint", &state.render.tint, .{ .no_inputs = true });

                imgui.sameLine(.{});
                imgui.text("Intensity", .{});

                imgui.sameLine(.{});
                {
                    imgui.pushItemWidth(-1.0);
                    defer imgui.popItemWidth();

                    _ = imgui.sliderFloat("##Intensity", &state.render.tint_intensity, 0.0, 1.0);
                }
            }
        }

        imgui.spacing();
        imgui.separator();
        imgui.spacing();

        {
            // TODO(paoda): delete this method, opt for picker UI

            imgui.textDisabled("Source Device", .{});
            imgui.sameLine(.{ .spacing = 2 });
            imgui.textDisabled("(?)", .{});
            if (imgui.isItemHovered() and imgui.beginTooltip()) {
                defer imgui.endTooltip();

                imgui.text("Native resolution of the recording device. When set, letterbox/pillarbox\nbars added by streaming apps (e.g. Moblin, Larix) are cropped.", .{});
            }

            if (imgui.beginTable("DeviceResolutionForm", 3)) {
                defer imgui.endTable();

                imgui.tableSetupColumn("Label", .{ .width_fixed = true });
                imgui.tableSetupColumn("Input", .{ .width_fixed = true });
                imgui.tableSetupColumn("Button", .{ .width_stretch = true });

                _ = imgui.tableNextColumn();
                imgui.alignTextToFramePadding();
                imgui.text("Resolution", .{});

                _ = imgui.tableNextColumn();
                {
                    imgui.beginDisabled(state.source.detect_status == .idle or state.source.isSetManually());
                    defer imgui.endDisabled();

                    if (imgui.button("Redetect", .{})) state.action = .Redetect;
                }

                _ = imgui.tableNextColumn();
                {
                    imgui.pushItemWidth(-1.0);
                    defer imgui.popItemWidth();

                    if (imgui.inputInt2("##DeviceResolution", &state.source.resolution)) {
                        state.source.resolution[0] = @max(0, state.source.resolution[0]);
                        state.source.resolution[1] = @max(0, state.source.resolution[1]);
                    }
                }
            }

            if (state.source.isSetManually()) {
                const width, const height = state.source.resolution;
                const aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
                imgui.textDisabled("({d:.3}:1)", .{aspect});
            }

            if (!state.source.isSetManually()) {
                switch (state.source.detect_status) {
                    .idle, .locked => {},
                    .searching => imgui.textDisabled("detecting letterbox\u{2026}", .{}),
                    .validating => imgui.textDisabled("verifying corner markers\u{2026}", .{}),
                    .failed => imgui.textDisabled("auto-detect failed \u{2014} enter device W\u{d7}H above", .{}),
                }
            }

            // the crop that's actually applied to the active video, manual or detected
            if (maybe_frame) |frame| {
                const rect = state.source.effectiveRect(frame);

                if (!rect.frame.eql(frame)) {
                    imgui.textDisabled("{f} of {f}", .{ rect.frame, frame });
                }
            }
        }
    }

    fn drawUploadPanel(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const panel_width = imgui.getContentRegionAvail()[0];

        imgui.text("Scan to upload a video over Wi-Fi", .{});

        if (state.default_path) |path| {
            imgui.sameLine(.{ .spacing = 2 });
            imgui.textDisabled("(?)", .{});
            if (imgui.isItemHovered() and imgui.beginTooltip()) {
                defer imgui.endTooltip();
                imgui.text("files will be uploaded to '{s}'", .{path});
            }
        }

        imgui.spacing();
        imgui.separator();
        imgui.spacing();

        if (state.net.local_addr) |address| {
            const qr_size = @min(panel_width - 16.0, 180.0);

            imgui.setCursorPosX((panel_width - qr_size) / 2.0);
            imgui.image(state.net.qr.tex_id[0], qr_size, qr_size, .{ 0.0, 0.0 }, .{ 1.0, 1.0 }, .{ .nearest = true });

            imgui.spacing();

            const ip_label = blk: {
                const bytes = address.ip4.bytes;

                var buf: [0xF]u8 = undefined;
                break :blk std.fmt.bufPrint(&buf, "{d}.{d}.{d}.{d}", .{ bytes[0], bytes[1], bytes[2], bytes[3] }) catch unreachable;
            };
            const ip_width = imgui.calcTextSize(ip_label)[0];

            imgui.setCursorPosX((panel_width - ip_width) / 2.0);
            imgui.textDisabled("{s}", .{ip_label});
        } else {
            const label = "Waiting for network...";
            const label_width = imgui.calcTextSize(label)[0];

            imgui.setCursorPosX((panel_width - label_width) / 2.0);
            imgui.textDisabled("{s}", .{label});
        }
    }

    fn drawInfoPanel(_: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        imgui.text("TIPS:", .{});
        imgui.bulletText("CTRL+Click to manually input a value", .{});
        imgui.bulletText("Click on the video to toggle fullscreen", .{});
        imgui.bulletText("For YouTube: 3840x2160 @ 60_000kbps", .{});

        const version_label = blk: {
            var buf: [64]u8 = undefined;
            break :blk std.fmt.bufPrintZ(&buf, "rota-stabilizer v{s}", .{version}) catch unreachable;
        };
        const version_width = imgui.calcTextSize(version_label)[0];
        const line_height = imgui.getTextLineHeightWithSpacing();
        const panel_height = imgui.getContentRegionAvail()[1];

        imgui.setCursorPosY(imgui.getCursorPosY() + panel_height - line_height);
        imgui.setCursorPosX((imgui.getContentRegionAvail()[0] - version_width) / 2.0);
        imgui.textDisabled("{s}", .{version_label});
    }

    fn drawVideoWindow(ui: Ui, state: *State, maybe_video: ?VideoContext) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        var pushed: u32 = 1;
        imgui.pushStyleVarImVec2(.window_padding, .{ 0, 0 });
        defer imgui.popStyleVar(.{ .count = @intCast(pushed) });

        const name = if (state.fullscreen) "##FullscreenVideo" else "Video";

        const showing = blk: {
            const overrides: imgui.DockNodeFlags = .{ .no_tab_bar = true };
            var class: imgui.WindowClass = .{ .DockNodeFlagsOverrideSet = @bitCast(overrides) };
            imgui.setNextWindowClass(&class);

            if (state.fullscreen) {
                const viewport = imgui.getMainViewport();
                const pos = viewport.WorkPos;
                const size = viewport.WorkSize;

                imgui.setNextWindowPos(.{ pos.x, pos.y }, .always);
                imgui.setNextWindowSize(.{ size.x, size.y }, .always);

                imgui.pushStyleVar(.window_border_size, 0.0);
                pushed += 1;

                break :blk imgui.begin(name, null, .{
                    .no_title_bar = true,
                    .no_move = true,
                    .no_resize = true,
                    .no_collapse = true,
                    .no_bring_to_front_on_focus = true,
                    .no_nav_focus = true,
                    .no_docking = true,
                    .no_background = true,
                    .no_scrollbar = true,
                });
            }

            break :blk imgui.begin(name, null, .{});
        };
        defer imgui.end();

        if (!showing) return;

        drawVideoContent(ui, state, maybe_video);
    }

    fn drawVideoContent(ui: Ui, state: *State, maybe_video: ?VideoContext) void {
        const vid = maybe_video orelse {
            const text = "DRAG AND DROP A VIDEO FILE ANYWHERE";

            const panel_width, const panel_height = imgui.getContentRegionAvail();
            const text_width, const text_height = imgui.calcTextSize(text);

            imgui.setCursorPosX((panel_width - text_width) / 2.0);
            imgui.setCursorPosY((panel_height - text_height) / 2.0);

            return imgui.textDisabled(text, .{});
        };
        const video_aspect = vid.render_view.aspect();

        const dw, const dh = imgui.getContentRegionAvail();
        if (dw <= 0 or dh <= 0) return;

        const window_aspect = dw / dh;
        const w = if (window_aspect > video_aspect) dh * video_aspect else dw;
        const h = if (window_aspect > video_aspect) dh else dw / video_aspect;

        const pos = imgui.getCursorPos();
        imgui.setCursorPos(.{
            pos[0] + (dw - w) * 0.5,
            pos[1] + (dh - h) * 0.5,
        });

        imgui.image(vid.tex_id, w, h, .{ 0.0, 1.0 }, .{ 1.0, 0.0 }, .{});

        // FIXME(paoda): change drag and drop to only accept in this area
        if (imgui.isItemHovered() and imgui.isMouseClicked(.left)) {
            toggleVideoFullscreen(ui, state, video_aspect);
        }
    }

    fn toggleVideoFullscreen(ui: Ui, state: *State, video_aspect: f32) void {
        state.fullscreen = !state.fullscreen;

        if (state.fullscreen) {
            _, const height = ui.view.get();

            const new_width: c_int = @intFromFloat(@as(f32, @floatFromInt(height)) * video_aspect);
            _ = c.SDL_SetWindowSize(ui.window, new_width, height);
        } else {
            const width, const height = state.init_view.get();
            _ = c.SDL_SetWindowSize(ui.window, width, height);
        }
    }

    fn drawControls(state: *State) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const overrides: imgui.DockNodeFlags = .{ .no_tab_bar = true };
        var class: imgui.WindowClass = .{ .DockNodeFlagsOverrideSet = @bitCast(overrides) };
        imgui.setNextWindowClass(&class);

        const showing = imgui.begin("Controls", null, .{});
        defer imgui.end();

        if (!showing) return;

        if (state.encode_progress > 0.0) {
            imgui.progressBar(state.encode_progress, -1.0, "Encoding...");
        } else {
            imgui.pushItemWidth(-1.0);
            defer imgui.popItemWidth();

            const duration = state.progress.duration orelse 0.0;

            // FIXME: until seeking impl, always disabled
            if (duration == 0.0 or true) imgui.beginDisabled(true);
            defer if (duration == 0.0 or true) imgui.endDisabled();

            if (imgui.sliderFloat("##Progress", &state.progress.timestamp, 0.0, duration)) {
                state.action = .{ .Seek = state.progress.timestamp };
            }
        }

        imgui.spacing();

        if (imgui.beginTable("VolumeControl", 3)) {
            defer imgui.endTable();

            imgui.tableSetupColumn("Label", .{ .width_fixed = true });
            imgui.tableSetupColumn("Slider", .{ .width_stretch = true });
            imgui.tableSetupColumn("Button", .{ .width_fixed = true });

            _ = imgui.tableNextColumn();
            imgui.alignTextToFramePadding();
            imgui.text("Volume", .{});

            _ = imgui.tableNextColumn();
            {
                imgui.pushItemWidth(-1.0);
                defer imgui.popItemWidth();

                if (imgui.sliderFloat("##Volume", &state.volume.value, 0.0, 1.0)) {
                    state.volume.cache = state.volume.value;
                    state.action = .{ .SetVolume = state.volume.value };
                }
            }

            _ = imgui.tableNextColumn();

            const w = imgui.calcTextSize("Unmute")[0] + 2 * imgui.getStyle().inner.FramePadding.x;
            const is_muted = state.volume.value < std.math.floatEps(f32);
            if (imgui.button(if (is_muted) "Unmute" else "Mute", .{ .w = w })) {
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
///
/// TODO(paoda/zig-0.16): `std.posix.socket`/`.connect`/`.getsockname` were removed;
/// that functionality now lives only inside `std.Io`'s private `Threaded`
/// implementation, which has no public "query local address of a socket" API.
/// Needs new per-platform raw-syscall code (or a design change) to restore this.
/// Until then this always returns null, so the QR/local-IP UI silently no-ops.
fn getLocalIpAddress(io: std.Io) ?std.Io.net.IpAddress {
    _ = io;
    return null;
}

pub fn getVideoDirectory(io: std.Io, environ_map: *const std.process.Environ.Map, allocator: std.mem.Allocator) !?[:0]const u8 {
    const base_path = try known_folders.getPath(io, allocator, environ_map, .videos) orelse return null;
    defer allocator.free(base_path);

    const path = try std.fs.path.joinZ(allocator, &.{ base_path, "rota-stabilizer" });
    errdefer allocator.free(path);

    std.Io.Dir.createDirAbsolute(io, path, .default_dir) catch |e| if (e != error.PathAlreadyExists) return e;

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

    pub fn updateTexture(self: *QrCode, allocator: std.mem.Allocator, address: std.Io.net.IpAddress) !void {
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
