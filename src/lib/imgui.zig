//! dear_bindings zig bindings
const std = @import("std");
const zimgui = @import("zimgui");

fn toVec2(v: [2]f32) zimgui.ImVec2 {
    return .{ .x = v[0], .y = v[1] };
}

fn fromVec2(v: zimgui.ImVec2) [2]f32 {
    return .{ v.x, v.y };
}

// -- context --

var ctx: ?*zimgui.ImGuiContext = null;

pub fn init(shared_font_atlas: ?*FontAtlas) void {
    _ = zimgui.CIMGUI_CHECKVERSION();
    ctx = zimgui.ImGui_CreateContext(shared_font_atlas);
}

pub fn deinit() void {
    zimgui.ImGui_DestroyContext(ctx);
}

// -- style & fonts --

pub const FontAtlas = zimgui.ImFontAtlas;

pub fn styleColourDark(dst: ?*zimgui.ImGuiStyle) void {
    return zimgui.ImGui_StyleColorsDark(dst);
}

// https://github.com/zig-gamedev/zgui/blob/bfbebed372723d1f585f86fc0a550232b3427f4d/src/gui.zig#L173
pub const ConfigFlags = packed struct(c_int) {
    nav_enable_keyboard: bool = false,
    nav_enable_gamepad: bool = false,
    nav_enable_set_mouse_pos: bool = false,
    nav_no_capture_keyboard: bool = false,
    no_mouse: bool = false,
    no_mouse_cursor_change: bool = false,
    no_keyboard: bool = false,
    dock_enable: bool = false,
    _8: u2 = 0,
    viewport_enable: bool = false,
    _11: u3 = 0,
    dpi_enable_scale_viewport: bool = false,
    dpi_enable_scale_fonts: bool = false,
    user_storage: u4 = 0,
    is_srgb: bool = false,
    is_touch_screen: bool = false,
    _22: u10 = 0,
};

pub const FontConfig = struct {
    inner: zimgui.ImFontConfig,

    pub fn init(self: *FontConfig, name: []const u8) void {
        self.* = .{
            .inner = .{
                .FontDataOwnedByAtlas = false,
                .MergeMode = false,
                .PixelSnapH = false,
                .GlyphMaxAdvanceX = std.math.floatMax(f32),
                .RasterizerMultiply = 1.0,
                .RasterizerDensity = 1.0,
                .ExtraSizeScale = 1.0,
            },
        };

        @memcpy(self.inner.Name[0..name.len], name);
    }
};

const Font = zimgui.ImFont;

const Io = struct {
    inner: *zimgui.ImGuiIO,

    pub fn addFontFromMemoryTTF(self: Io, buf: []const u8, size: f32, config: *FontConfig) *Font {
        return zimgui.ImFontAtlas_AddFontFromMemoryTTF(
            self.inner.Fonts,
            @ptrCast(@constCast(buf.ptr)),
            @intCast(buf.len),
            size,
            &config.inner,
            null,
        );
    }
};

pub fn getIo() Io {
    const ptr: ?*zimgui.ImGuiIO = zimgui.ImGui_GetIO();
    return .{ .inner = ptr.? };
}

const Style = struct {
    inner: *zimgui.ImGuiStyle,

    pub fn scaleAllSizes(self: Style, scale_factor: f32) void {
        return zimgui.ImGuiStyle_ScaleAllSizes(self.inner, scale_factor);
    }
};

pub fn getStyle() Style {
    const ptr: ?*zimgui.ImGuiStyle = zimgui.ImGui_GetStyle();
    return .{ .inner = ptr.? };
}

// -- viewport & windows --

pub const Viewport = zimgui.ImGuiViewport;

pub fn getMainViewport() *Viewport {
    return zimgui.ImGui_GetMainViewport();
}

pub fn getCenter(viewport: *const Viewport) [2]f32 {
    return fromVec2(zimgui.ImGuiViewport_GetCenter(viewport));
}

const Condition = enum(c_int) {
    always = zimgui.ImGuiCond_Always,
    appearing = zimgui.ImGuiCond_Appearing,
};

pub const WindowClass = zimgui.ImGuiWindowClass;

pub fn setNextWindowClass(class: *WindowClass) void {
    zimgui.ImGui_SetNextWindowClass(class);
}

pub fn setNextWindowPos(pos: [2]f32, cond: Condition) void {
    zimgui.ImGui_SetNextWindowPos(toVec2(pos), @intFromEnum(cond));
}

pub fn setNextWindowPosCentered(pos: [2]f32, cond: Condition) void {
    zimgui.ImGui_SetNextWindowPosEx(toVec2(pos), @intFromEnum(cond), toVec2(.{ 0.5, 0.5 }));
}

pub fn setNextWindowSize(size: [2]f32, cond: Condition) void {
    zimgui.ImGui_SetNextWindowSize(toVec2(size), @intFromEnum(cond));
}

const StyleVar = enum(c_int) {
    window_rounding = zimgui.ImGuiStyleVar_WindowRounding,
    window_border_size = zimgui.ImGuiStyleVar_WindowBorderSize,
    window_padding = zimgui.ImGuiStyleVar_WindowPadding,
    frame_rounding = zimgui.ImGuiStyleVar_FrameRounding,
};

pub fn pushStyleVar(idx: StyleVar, val: f32) void {
    zimgui.ImGui_PushStyleVar(@intFromEnum(idx), val);
}

pub fn pushStyleVarImVec2(idx: StyleVar, val: [2]f32) void {
    zimgui.ImGui_PushStyleVarImVec2(@intFromEnum(idx), toVec2(val));
}

pub fn popStyleVar(opt: struct { count: c_int = 1 }) void {
    zimgui.ImGui_PopStyleVarEx(opt.count);
}

// https://github.com/zig-gamedev/zgui/blob/bfbebed372723d1f585f86fc0a550232b3427f4d/src/gui.zig#L656
const WindowFlags = packed struct(c_int) {
    no_title_bar: bool = false,
    no_resize: bool = false,
    no_move: bool = false,
    no_scrollbar: bool = false,
    no_scroll_with_mouse: bool = false,
    no_collapse: bool = false,
    always_auto_resize: bool = false,
    no_background: bool = false,
    no_saved_settings: bool = false,
    no_mouse_inputs: bool = false,
    menu_bar: bool = false,
    horizontal_scrollbar: bool = false,
    no_focus_on_appearing: bool = false,
    no_bring_to_front_on_focus: bool = false,
    always_vertical_scrollbar: bool = false,
    always_horizontal_scrollbar: bool = false,
    no_nav_inputs: bool = false,
    no_nav_focus: bool = false,
    unsaved_document: bool = false,
    no_docking: bool = false,
    _20: u12 = 0,
};

pub fn begin(name: []const u8, p_open: ?*bool, flags: WindowFlags) bool {
    return zimgui.ImGui_Begin(name.ptr, p_open, @bitCast(flags));
}

pub fn end() void {
    zimgui.ImGui_End();
}

pub fn showDemoWindow(p_open: ?*bool) void {
    zimgui.ImGui_ShowDemoWindow(p_open);
}

pub const Id = enum(zimgui.ImGuiID) { _ };

pub fn getIDFromString(str: []const u8) Id {
    return @enumFromInt(zimgui.ImGui_GetID(str.ptr));
}

// -- layout / cursor --

pub fn spacing() void {
    zimgui.ImGui_Spacing();
}

pub fn separator() void {
    zimgui.ImGui_Separator();
}

pub fn sameLine(opt: struct { offset: f32 = 0.0, spacing: f32 = -1.0 }) void {
    zimgui.ImGui_SameLineEx(opt.offset, opt.spacing);
}

pub fn alignTextToFramePadding() void {
    zimgui.ImGui_AlignTextToFramePadding();
}

pub fn getTextLineHeightWithSpacing() f32 {
    return zimgui.ImGui_GetTextLineHeightWithSpacing();
}

pub fn pushItemWidth(width: f32) void {
    zimgui.ImGui_PushItemWidth(width);
}

pub fn popItemWidth() void {
    zimgui.ImGui_PopItemWidth();
}

pub fn getContentRegionAvail() [2]f32 {
    return fromVec2(zimgui.ImGui_GetContentRegionAvail());
}

pub fn getCursorPos() [2]f32 {
    return fromVec2(zimgui.ImGui_GetCursorPos());
}

pub fn setCursorPos(pos: [2]f32) void {
    zimgui.ImGui_SetCursorPos(toVec2(pos));
}

pub fn getCursorPosX() f32 {
    return zimgui.ImGui_GetCursorPosX();
}

pub fn setCursorPosX(x: f32) void {
    zimgui.ImGui_SetCursorPosX(x);
}

pub fn getCursorPosY() f32 {
    return zimgui.ImGui_GetCursorPosY();
}

pub fn setCursorPosY(y: f32) void {
    zimgui.ImGui_SetCursorPosY(y);
}

pub fn calcTextSize(str: []const u8) [2]f32 {
    return fromVec2(zimgui.ImGui_CalcTextSizeEx(str.ptr, str.ptr + str.len, false, -1.0));
}

// -- text --

threadlocal var format_buf: [4096]u8 = undefined;

fn format(comptime fmt: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(&format_buf, fmt, args) catch format_buf[0..];
}

fn textUnformatted(str: []const u8) void {
    zimgui.ImGui_TextUnformattedEx(str.ptr, str.ptr + str.len);
}

pub fn text(comptime fmt: []const u8, args: anytype) void {
    textUnformatted(format(fmt, args));
}

pub fn textDisabled(comptime fmt: []const u8, args: anytype) void {
    const str = format(fmt, args);

    zimgui.ImGui_PushStyleColorImVec4(zimgui.ImGuiCol_Text, zimgui.ImGui_GetStyleColorVec4(zimgui.ImGuiCol_TextDisabled).*);
    defer zimgui.ImGui_PopStyleColor();

    textUnformatted(str);
}

pub fn bulletText(comptime fmt: []const u8, args: anytype) void {
    zimgui.ImGui_Bullet();
    textUnformatted(format(fmt, args));
}

// -- widgets --

pub fn button(label: []const u8, opt: struct { w: f32 = 0.0, h: f32 = 0.0 }) bool {
    return zimgui.ImGui_ButtonEx(label.ptr, toVec2(.{ opt.w, opt.h }));
}

pub fn checkbox(label: []const u8, v: *bool) bool {
    return zimgui.ImGui_Checkbox(label.ptr, v);
}

pub fn sliderFloat(label: []const u8, v: *f32, min: f32, max: f32) bool {
    return zimgui.ImGui_SliderFloat(label.ptr, v, min, max);
}

pub fn inputFloat(label: []const u8, v: *f32, opt: struct { step: f32 = 0.0, cfmt: [*:0]const u8 = "%.3f" }) bool {
    return zimgui.ImGui_InputFloatEx(label.ptr, v, opt.step, 0.0, opt.cfmt, 0);
}

pub fn inputInt(label: []const u8, v: *i32) bool {
    return zimgui.ImGui_InputIntEx(label.ptr, v, 1, 100, 0);
}

pub fn inputInt2(label: []const u8, v: *[2]i32) bool {
    return zimgui.ImGui_InputInt2(label.ptr, v, 0);
}

pub fn inputTextWithHint(label: []const u8, hint: []const u8, buf: []u8) bool {
    return zimgui.ImGui_InputTextWithHint(label.ptr, hint.ptr, buf.ptr, buf.len, 0);
}

pub const ColorEditFlags = packed struct(c_int) {
    _0: u5 = 0,
    no_inputs: bool = false,
    _6: u26 = 0,
};

pub fn colorEdit3(label: []const u8, col: *[3]f32, flags: ColorEditFlags) bool {
    return zimgui.ImGui_ColorEdit3(label.ptr, col, @bitCast(flags));
}

pub fn progressBar(fraction: f32, w: f32, overlay: ?[]const u8) void {
    zimgui.ImGui_ProgressBar(fraction, toVec2(.{ w, 0.0 }), if (overlay) |o| o.ptr else null);
}

pub fn image(tex_id: c_uint, w: f32, h: f32, uv0: [2]f32, uv1: [2]f32, opt: struct { nearest: bool = false }) void {
    const tex_ref: zimgui.ImTextureRef = .{ ._TexData = null, ._TexID = tex_id };

    // GL 3.3+ sampler objects take priority over a texture's own filter params, and the
    // OpenGL3 backend always binds one: without this, every image is force-sampled Linear.
    const draw_list = if (opt.nearest) zimgui.ImGui_GetWindowDrawList() else null;
    const platform_io = if (opt.nearest) zimgui.ImGui_GetPlatformIO() else null;

    if (draw_list) |dl| zimgui.ImDrawList_AddCallback(dl, platform_io.?.*.DrawCallback_SetSamplerNearest);
    zimgui.ImGui_ImageEx(tex_ref, toVec2(.{ w, h }), toVec2(uv0), toVec2(uv1));
    if (draw_list) |dl| zimgui.ImDrawList_AddCallback(dl, platform_io.?.*.DrawCallback_SetSamplerLinear);
}

pub fn comboFromEnum(label: []const u8, ptr: anytype) bool {
    const Enum = @typeInfo(@TypeOf(ptr)).pointer.child;
    const fields = @typeInfo(Enum).@"enum".fields;

    const Impl = struct {
        const names: [fields.len][*:0]const u8 = blk: {
            var arr: [fields.len][*:0]const u8 = undefined;
            for (fields, 0..) |f, i| arr[i] = f.name.ptr;
            break :blk arr;
        };
    };

    var idx: c_int = 0;
    inline for (fields, 0..) |f, i| {
        if (@intFromEnum(ptr.*) == f.value) idx = i;
    }

    const changed = zimgui.ImGui_ComboChar(label.ptr, &idx, @ptrCast(&Impl.names), fields.len);

    if (changed) {
        inline for (fields, 0..) |f, i| {
            if (i == idx) ptr.* = @enumFromInt(f.value);
        }
    }

    return changed;
}

// -- interaction --

pub const MouseButton = enum(c_int) {
    left = zimgui.ImGuiMouseButton_Left,
    right = zimgui.ImGuiMouseButton_Right,
    middle = zimgui.ImGuiMouseButton_Middle,
};

pub fn isItemHovered() bool {
    return zimgui.ImGui_IsItemHovered(0);
}

pub fn isMouseClicked(button_: MouseButton) bool {
    return zimgui.ImGui_IsMouseClicked(@intFromEnum(button_));
}

pub fn beginDisabled(disabled: bool) void {
    zimgui.ImGui_BeginDisabled(disabled);
}

pub fn endDisabled() void {
    zimgui.ImGui_EndDisabled();
}

// -- tooltips & popups --

pub fn beginTooltip() bool {
    return zimgui.ImGui_BeginTooltip();
}

pub fn endTooltip() void {
    zimgui.ImGui_EndTooltip();
}

pub fn beginPopupModal(name: []const u8, flags: WindowFlags) bool {
    return zimgui.ImGui_BeginPopupModal(name.ptr, null, @bitCast(flags));
}

pub fn endPopup() void {
    zimgui.ImGui_EndPopup();
}

pub fn openPopup(str_id: []const u8) void {
    zimgui.ImGui_OpenPopup(str_id.ptr, 0);
}

pub fn isPopupOpen(str_id: []const u8) bool {
    return zimgui.ImGui_IsPopupOpen(str_id.ptr, 0);
}

pub fn closeCurrentPopup() void {
    zimgui.ImGui_CloseCurrentPopup();
}

// -- tabs --

pub fn beginTabBar(str_id: []const u8) bool {
    return zimgui.ImGui_BeginTabBar(str_id.ptr, 0);
}

pub fn endTabBar() void {
    zimgui.ImGui_EndTabBar();
}

pub fn beginTabItem(label: []const u8) bool {
    return zimgui.ImGui_BeginTabItem(label.ptr, null, 0);
}

pub fn endTabItem() void {
    zimgui.ImGui_EndTabItem();
}

// -- tables --

pub const TableColumnFlags = packed struct(c_int) {
    _0: u3 = 0,
    width_stretch: bool = false,
    width_fixed: bool = false,
    _5: u27 = 0,
};

pub fn beginTable(str_id: []const u8, columns: c_int) bool {
    return zimgui.ImGui_BeginTable(str_id.ptr, columns, 0);
}

pub fn endTable() void {
    zimgui.ImGui_EndTable();
}

pub fn tableSetupColumn(label: []const u8, flags: TableColumnFlags) void {
    zimgui.ImGui_TableSetupColumn(label.ptr, @bitCast(flags));
}

pub fn tableNextColumn() bool {
    return zimgui.ImGui_TableNextColumn();
}

// -- docking related --

const DockNode = zimgui.ImGuiDockNode;

// https://github.com/zig-gamedev/zgui/blob/bfbebed372723d1f585f86fc0a550232b3427f4d/src/gui.zig#L1021
pub const DockNodeFlags = packed struct(c_int) {
    keep_alive_only: bool = false,
    _1: u1 = 0,
    no_docking_over_central_node: bool = false,
    passthru_central_node: bool = false,
    no_docking_split: bool = false,
    no_resize: bool = false,
    auto_hide_tab_bar: bool = false,
    no_undocking: bool = false,
    _8: u2 = 0,

    // Extended enum entries from imgui_internal (unstable, subject to change, use at own risk)
    dock_space: bool = false,
    central_node: bool = false,
    no_tab_bar: bool = false,
    hidden_tab_bar: bool = false,
    no_window_menu_button: bool = false,
    no_close_button: bool = false,
    no_resize_x: bool = false,
    no_resize_y: bool = false,
    docked_windows_in_focus_route: bool = false,
    no_docking_split_other: bool = false,
    no_docking_over_me: bool = false,
    no_docking_over_other: bool = false,
    no_docking_over_empty: bool = false,
    _23: u9 = 0,
};

pub fn dockSpaceOverViewport(id: Id, viewport: *Viewport, flags: DockNodeFlags) Id {
    return @enumFromInt(zimgui.ImGui_DockSpaceOverViewportEx(@intFromEnum(id), viewport, @bitCast(flags), null));
}

pub fn dockBuilderGetNode(id: Id) ?*DockNode {
    return zimgui.ImGui_DockBuilderGetNode(@intFromEnum(id));
}

pub fn dockBuilderAddNode(id: Id, flags: DockNodeFlags) Id {
    return @enumFromInt(zimgui.ImGui_DockBuilderAddNodeEx(@intFromEnum(id), @bitCast(flags)));
}

pub fn dockBuilderSetNodeSize(id: Id, size: [2]f32) void {
    zimgui.ImGui_DockBuilderSetNodeSize(@intFromEnum(id), toVec2(size));
}

const Direction = enum(c_int) {
    none = zimgui.ImGuiDir_None,
    left = zimgui.ImGuiDir_Left,
    right = zimgui.ImGuiDir_Right,
    up = zimgui.ImGuiDir_Up,
    down = zimgui.ImGuiDir_Down,
    count = zimgui.ImGuiDir_COUNT,
};

pub fn dockBuilderSplitNode(
    target: Id,
    split_dir: Direction,
    size_ratio_for_node_at_dir: f32,
    out_id_at_dir: *Id,
    out_id_at_opposite_dir: *Id,
) Id {
    var left: zimgui.ImGuiID = undefined;
    var right: zimgui.ImGuiID = undefined;

    const id: Id = @enumFromInt(zimgui.ImGui_DockBuilderSplitNode(
        @intFromEnum(target),
        @intFromEnum(split_dir),
        size_ratio_for_node_at_dir,
        &left,
        &right,
    ));

    out_id_at_dir.* = @enumFromInt(left);
    out_id_at_opposite_dir.* = @enumFromInt(right);

    return id;
}

pub fn dockBuilderDockWindow(name: []const u8, id: Id) void {
    zimgui.ImGui_DockBuilderDockWindow(name.ptr, @intFromEnum(id));
}

pub fn dockBuilderFinish(id: Id) void {
    zimgui.ImGui_DockBuilderFinish(@intFromEnum(id));
}

// -- backend --

pub const backend = struct {
    const c = @import("../lib.zig").c;

    pub fn init(
        window: *c.SDL_Window,
        gl_ctx: c.SDL_GLContext,
    ) void {
        _ = zimgui.cImGui_ImplSDL3_InitForOpenGL(@ptrCast(window), gl_ctx);
        _ = zimgui.cImGui_ImplOpenGL3_Init();
    }

    pub fn deinit() void {
        zimgui.cImGui_ImplOpenGL3_Shutdown();
        zimgui.cImGui_ImplSDL3_Shutdown();
    }

    pub fn processEvent(event: *c.SDL_Event) bool {
        return zimgui.cImGui_ImplSDL3_ProcessEvent(@ptrCast(event));
    }

    pub fn newFrame() void {
        zimgui.cImGui_ImplOpenGL3_NewFrame();
        zimgui.cImGui_ImplSDL3_NewFrame();
        zimgui.ImGui_NewFrame();
    }

    pub fn render() void {
        zimgui.ImGui_Render();
        zimgui.cImGui_ImplOpenGL3_RenderDrawData(zimgui.ImGui_GetDrawData());
    }
};
