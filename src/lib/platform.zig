const std = @import("std");
const builtin = @import("builtin");
const gl = @import("gl");
const tracy = @import("tracy");
const c = @import("../lib.zig").c;

pub var gl_procs: gl.ProcTable = undefined;

pub const Ui = struct {
    window: *c.SDL_Window,
    gl_ctx: c.SDL_GLContext,

    const log = std.log.scoped(.ui);

    pub fn deinit(self: @This()) void {
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

    pub fn toggleFullscreen(self: @This()) !void {
        if (c.SDL_GetWindowFlags(self.window) & c.SDL_WINDOW_FULLSCREEN != 0) {
            try errify(c.SDL_SetWindowFullscreen(self.window, false));
        } else {
            try errify(c.SDL_SetWindowFullscreen(self.window, true));
        }
    }

    pub fn swap(self: @This()) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        try errify(c.SDL_GL_SwapWindow(self.window));
        tracy.frameMark(null);
    }
};

pub fn createWindow(width: u32, height: u32) !Ui {
    return createWindowEx(width, height, false);
}

pub fn createHeadless(width: u32, height: u32) !Ui {
    return createWindowEx(width, height, true);
}

fn createWindowEx(width: u32, height: u32, headless: bool) !Ui {
    c.SDL_SetMainReady();
    try errify(c.SDL_Init(c.SDL_INIT_VIDEO | (if (headless) 0 else c.SDL_INIT_AUDIO)));
    errdefer c.SDL_Quit();

    try errify(c.SDL_SetAppMetadata("Rotaeno Stabilizer", "0.1.0", "moe.paoda.rota-stabilizer"));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_ALPHA_SIZE, 8));

    const window_flags: c.SDL_WindowFlags = c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE | (if (headless) c.SDL_WINDOW_HIDDEN else 0);
    const window: *c.SDL_Window = try errify(c.SDL_CreateWindow("Rotaeno Stabilizer", @intCast(width), @intCast(height), window_flags));
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

pub fn guessHardware() struct { ?c.AVHWDeviceType, ?c.AVHWDeviceType } {
    const vendor = std.mem.span(gl.GetString(gl.VENDOR) orelse return .{ null, null });

    const is_apple = std.mem.indexOf(u8, vendor, "Apple") != null;
    if (is_apple) return .{ c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX, c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX };

    const is_nvidia = std.mem.indexOf(u8, vendor, "NVIDIA") != null;
    if (is_nvidia) return .{ c.AV_HWDEVICE_TYPE_CUDA, c.AV_HWDEVICE_TYPE_CUDA };

    const is_intel = std.mem.indexOf(u8, vendor, "Intel") != null;
    if (is_intel) return .{ c.AV_HWDEVICE_TYPE_QSV, c.AV_HWDEVICE_TYPE_QSV };

    const is_amd = std.mem.indexOf(u8, vendor, "AMD") != null or std.mem.indexOf(u8, vendor, "ATI") != null;
    if (is_amd) return switch (builtin.os.tag) {
        .linux => .{ c.AV_HWDEVICE_TYPE_VAAPI, c.AV_HWDEVICE_TYPE_VAAPI },
        .windows => .{ c.AV_HWDEVICE_TYPE_D3D11VA, c.AV_HWDEVICE_TYPE_AMF },
        else => unreachable,
    };

    return .{ null, null };
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
