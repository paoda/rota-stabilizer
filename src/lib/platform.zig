const gl = @import("gl");
const c = @import("../lib.zig").c;

pub var gl_procs: gl.ProcTable = undefined;

pub const Ui = struct {
    window: *c.SDL_Window,
    gl_ctx: c.SDL_GLContext,

    pub fn deinit(self: @This()) void {
        gl.makeProcTableCurrent(null);
        _ = c.SDL_GL_MakeCurrent(self.window, null);
        _ = c.SDL_GL_DestroyContext(self.gl_ctx);
        c.SDL_DestroyWindow(self.window);
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
        try errify(c.SDL_GL_SwapWindow(self.window));
    }
};

pub fn createWindow(width: u32, height: u32) !Ui {
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

    const window: *c.SDL_Window = try errify(c.SDL_CreateWindow("Rotaeno Stabilizer", @intCast(width), @intCast(height), c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE));
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
