const std = @import("std");
const gl = @import("gl");

const c = @import("librota").c;

const libavError = @import("librota").libavError;
const FrameQueue = @import("librota").FrameQueue;

var gl_procs: gl.ProcTable = undefined;

pub fn main() !void {
    const log = std.log.scoped(.ui);
    errdefer |err| if (err == error.sdl_error) log.err("SDL Error: {s}", .{c.SDL_GetError()});

    var gpa: std.heap.DebugAllocator(.{}) = .{ .backing_allocator = std.heap.c_allocator };
    defer std.debug.assert(gpa.deinit() == .ok);

    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.skip(); // self
    const path = args.next() orelse return error.missing_file;

    var maybe_fmt_ctx: ?*c.AVFormatContext = c.avformat_alloc_context();
    defer c.avformat_close_input(&maybe_fmt_ctx); // SAFETY: only okay 'cause we're cleaning up

    _ = try libavError(c.avformat_open_input(&maybe_fmt_ctx, path, null, null));
    const fmt_ctx = maybe_fmt_ctx orelse return error.out_of_memory;

    _ = try libavError(c.avformat_find_stream_info(fmt_ctx, null));

    var maybe_vid_codec_ctx: ?*c.AVCodecContext, const vid_stream: usize = blk: {
        var codec_ptr: ?*const c.AVCodec = null;
        const stream = try libavError(c.av_find_best_stream(fmt_ctx, c.AVMEDIA_TYPE_VIDEO, -1, -1, &codec_ptr, 0));
        if (codec_ptr == null) return error.unsupported_codec;

        const ctx_ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec_ptr.?);

        _ = try libavError(c.avcodec_parameters_to_context(ctx_ptr, fmt_ctx.streams[@intCast(stream)].*.codecpar));
        _ = try libavError(c.avcodec_open2(ctx_ptr, codec_ptr.?, null));

        break :blk .{ ctx_ptr, @intCast(stream) };
    };
    defer c.avcodec_free_context(&maybe_vid_codec_ctx);

    var maybe_aud_codec_ctx: ?*c.AVCodecContext, const aud_stream: usize = blk: {
        var codec_ptr: ?*const c.AVCodec = null;
        const stream = try libavError(c.av_find_best_stream(fmt_ctx, c.AVMEDIA_TYPE_AUDIO, -1, -1, &codec_ptr, 0));
        if (codec_ptr == null) return error.unsupported_codec;

        const ctx_ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec_ptr.?);

        _ = try libavError(c.avcodec_parameters_to_context(ctx_ptr, fmt_ctx.streams[@intCast(stream)].*.codecpar));
        _ = try libavError(c.avcodec_open2(ctx_ptr, codec_ptr.?, null));

        break :blk .{ ctx_ptr, @intCast(stream) };
    };
    defer c.avcodec_free_context(&maybe_aud_codec_ctx);

    const vid_codec_ctx = maybe_vid_codec_ctx orelse return error.out_of_memory;
    const aud_codec_ctx = maybe_aud_codec_ctx orelse return error.out_of_memory;

    _ = aud_codec_ctx;
    _ = aud_stream;

    // lmao just put SDL stuff after this
    c.SDL_SetMainReady();

    try errify(c.SDL_Init(c.SDL_INIT_AUDIO | c.SDL_INIT_VIDEO));

    try errify(c.SDL_SetAppMetadata("Rotaeno Stabilizer", "0.1.0", "moe.paoda.rota-stabilizer"));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));

    const sz = @max(vid_codec_ctx.width, vid_codec_ctx.height);

    const window: *c.SDL_Window = try errify(c.SDL_CreateWindow("Rotaeno Stabilizer", sz >> 1, sz >> 1, c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE));
    defer c.SDL_DestroyWindow(window);

    const gl_ctx = try errify(c.SDL_GL_CreateContext(window));
    defer errify(c.SDL_GL_DestroyContext(gl_ctx)) catch {};

    try errify(c.SDL_GL_MakeCurrent(window, gl_ctx));
    defer errify(c.SDL_GL_MakeCurrent(window, null)) catch {};

    if (!gl_procs.init(c.SDL_GL_GetProcAddress)) return error.gl_init_failed;

    gl.makeProcTableCurrent(&gl_procs);
    defer gl.makeProcTableCurrent(null);

    _ = c.SDL_GL_SetSwapInterval(0);

    // zig fmt: off
    const vertices: [16]f32 = .{
        // pos      // uv
        -1.0, -1.0, 0.0, 1.0, // bottom left
         1.0, -1.0, 1.0, 1.0, // bottom right
        -1.0,  1.0, 0.0, 0.0, // top left
         1.0,  1.0, 1.0, 0.0, // top right
    };
    // zig fmt: on

    // -- opengl --
    var vao_id = opengl_impl.vao();
    defer gl.DeleteVertexArrays(1, vao_id[0..]);

    var vbo_id = opengl_impl.vbo();
    defer gl.DeleteBuffers(1, vbo_id[0..]);

    {
        gl.BindVertexArray(vao_id[0]);
        defer gl.BindVertexArray(0);

        gl.BindBuffer(gl.ARRAY_BUFFER, vbo_id[0]);
        defer gl.BindBuffer(gl.ARRAY_BUFFER, 0);

        gl.BufferData(gl.ARRAY_BUFFER, @sizeOf(@TypeOf(vertices)), vertices[0..].ptr, gl.STATIC_DRAW);

        gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 0);
        gl.EnableVertexAttribArray(0);

        gl.VertexAttribPointer(1, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 2 * @sizeOf(f32));
        gl.EnableVertexAttribArray(1);
    }

    var tex_id = opengl_impl.vidTex(vid_codec_ctx.width, vid_codec_ctx.height);
    defer gl.DeleteTextures(1, tex_id[0..]);

    const prog_id = try opengl_impl.program();
    defer gl.DeleteProgram(prog_id);

    // -- opengl end --
    var queue = try FrameQueue.init(allocator, 0x40);
    defer queue.deinit(allocator);

    var should_quit: std.atomic.Value(bool) = .init(false);

    const decode_thread = try std.Thread.spawn(.{}, decode, .{ fmt_ctx, vid_codec_ctx, vid_stream, &queue, &should_quit });
    defer decode_thread.join();

    var start_time: ?u64 = null; // set to null to disable vsync
    var current_frame: ?*c.AVFrame = null;

    while (!should_quit.load(.monotonic)) {
        var event: c.SDL_Event = undefined;

        while (c.SDL_PollEvent(&event)) {
            switch (event.type) {
                c.SDL_EVENT_QUIT => should_quit.store(true, .monotonic),
                else => {},
            }
        }

        // Now do your regular OpenGL render pass.
        var w: c_int, var h: c_int = .{ undefined, undefined };
        try errify(c.SDL_GetWindowSizeInPixels(window, &w, &h));

        gl.Viewport(0, 0, w, h);

        // gl.ClearColor(1, 1, 1, 1);
        gl.Clear(gl.COLOR_BUFFER_BIT);

        blocking: while (true) {
            current_frame = queue.pop() orelse continue :blocking;
            break :blocking;
        }

        // if (queue.pop()) |frame| current_frame = frame;

        const frame = current_frame orelse {
            log.debug("queue empty, repeated frame", .{});

            try errify(c.SDL_GL_SwapWindow(window));
            continue;
        };

        video_sync: {
            const time_base: f64 = c.av_q2d(fmt_ctx.streams[vid_stream].*.time_base);
            const pt_in_seconds = @as(f64, @floatFromInt(frame.pts)) * time_base;

            if (start_time == null) break :video_sync;
            if (start_time == 0) start_time = c.SDL_GetPerformanceCounter();

            while (true) {
                const elapsed_seconds = @as(f64, @floatFromInt(c.SDL_GetPerformanceCounter() - start_time.?)) / @as(f64, @floatFromInt(c.SDL_GetPerformanceFrequency()));
                if (@abs(pt_in_seconds - elapsed_seconds) < std.math.floatEps(f64) or pt_in_seconds < elapsed_seconds) break;

                std.atomic.spinLoopHint(); // TODO: less resource intensive
            }
        }

        gl.UseProgram(prog_id);
        defer gl.UseProgram(0);

        gl.BindVertexArray(vao_id[0]);
        defer gl.BindVertexArray(0);

        gl.ActiveTexture(gl.TEXTURE0);
        defer gl.ActiveTexture(0);

        gl.BindTexture(gl.TEXTURE_2D, tex_id[0]);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.PixelStorei(gl.UNPACK_ROW_LENGTH, @divTrunc(frame.linesize[0], 3)); // FIXME: is necesary becaues frame.width or frame.height can be wrong?

        gl.TexSubImage2D(
            gl.TEXTURE_2D,
            0,
            0,
            0,
            frame.width,
            frame.height,
            gl.RGB, // match the format of frame data
            gl.UNSIGNED_BYTE, // since RGB24 uses one byte per channel
            frame.data[0][0..],
        );

        { // calcualte uniforms
            const rad = -angle(frame, @intCast(frame.width), @intCast(frame.height)) * std.math.rad_per_deg;
            const aspect = @as(f32, @floatFromInt(frame.width)) / @as(f32, @floatFromInt(frame.height));
            const ratio: [2]f32 = if (aspect > 1.0) .{ 1.0, 1.0 / aspect } else .{ aspect, 1.0 };
            const scale = 1.0 / std.math.sqrt(ratio[0] * ratio[0] + ratio[1] * ratio[1]); // factor allows for ration w/out clipping

            gl.Uniform2f(gl.GetUniformLocation(prog_id, "u_aspect"), ratio[0], ratio[1]);
            gl.Uniform1f(gl.GetUniformLocation(prog_id, "u_scale"), scale);
            gl.Uniform2f(gl.GetUniformLocation(prog_id, "u_rotation"), @sin(rad), @cos(rad));
            gl.Uniform1i(gl.GetUniformLocation(prog_id, "u_screen"), 0);
        }

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);

        try errify(c.SDL_GL_SwapWindow(window));
    }
}

fn decode(fmt_ctx: *c.AVFormatContext, codec_ctx: *c.AVCodecContext, vid_stream: usize, queue: *FrameQueue, should_quit: *std.atomic.Value(bool)) !void {
    const log = std.log.scoped(.decode);

    log.info("decode thread start", .{});
    defer log.info("decode thread end", .{});

    var maybe_pkt: ?*c.AVPacket = c.av_packet_alloc();
    defer c.av_packet_free(&maybe_pkt);

    var maybe_src_frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(&maybe_src_frame);

    var maybe_dst_frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(&maybe_dst_frame);

    const pkt = maybe_pkt orelse return error.out_of_memory;
    const src_frame = maybe_src_frame orelse return error.out_of_memory;
    const dst_frame = maybe_dst_frame orelse return error.out_of_memory;

    dst_frame.width = codec_ctx.width;
    dst_frame.height = codec_ctx.height;
    dst_frame.format = c.AV_PIX_FMT_RGB24;

    _ = try libavError(c.av_frame_get_buffer(dst_frame, 32));

    const sws_ctx = blk: {
        const ptr = c.sws_getContext(
            codec_ctx.width,
            codec_ctx.height,
            codec_ctx.pix_fmt,
            codec_ctx.width,
            codec_ctx.height,
            c.AV_PIX_FMT_RGB24,
            c.SWS_BILINEAR,
            null,
            null,
            null,
        );
        if (ptr == null) return error.out_of_memory;

        break :blk ptr.?;
    };
    defer c.sws_freeContext(sws_ctx);

    while (c.av_read_frame(fmt_ctx, pkt) >= 0) {
        defer c.av_packet_unref(pkt);

        if (should_quit.load(.monotonic)) return;
        if (pkt.stream_index != vid_stream) continue;

        const send_ret = c.avcodec_send_packet(codec_ctx, pkt);
        if (send_ret == c.AVERROR_EOF) return;
        if (send_ret == c.AVERROR(c.EAGAIN)) @panic("TODO: handle EAGAIN");
        if (send_ret != 0) @panic("TODO: unrecoverable error in avcodec_send_packet");

        const recv_ret = c.avcodec_receive_frame(codec_ctx, src_frame);
        if (recv_ret == c.AVERROR_EOF) return;
        if (recv_ret == c.AVERROR(c.EAGAIN)) @panic("TODO: handle EAGAIN");
        if (recv_ret != 0) @panic("TODO: unrecoverable error in avcodec_receive_frame");

        dst_frame.pts = src_frame.pts; // for timing

        _ = c.sws_scale(
            sws_ctx,
            src_frame.data[0..],
            src_frame.linesize[0..],
            0,
            src_frame.height, // TODO: this should be src_frame
            dst_frame.data[0..],
            dst_frame.linesize[0..],
        );

        blocking: while (true) {
            queue.push(dst_frame) catch |e| {
                if (e == error.out_of_memory) continue :blocking;
                std.debug.panic("error: {}", .{e});
            };

            break :blocking;
        }
    }
}

fn angle(frame: *const c.AVFrame, width: usize, height: usize) f32 {
    const ofs = 5;
    const size = 3;

    const btm_left = sample(size, frame, ofs, height - ofs);
    const top_left = sample(size, frame, ofs, ofs);
    const btm_right = sample(size, frame, width - ofs, height - ofs);
    const top_right = sample(size, frame, width - ofs, ofs);

    var out: u16 = 0;
    out |= @as(u16, @intFromBool(top_left[0] >= 128)) << 11;
    out |= @as(u16, @intFromBool(top_left[1] >= 128)) << 10;
    out |= @as(u16, @intFromBool(top_left[2] >= 128)) << 9;
    out |= @as(u16, @intFromBool(top_right[0] >= 128)) << 8;
    out |= @as(u16, @intFromBool(top_right[1] >= 128)) << 7;
    out |= @as(u16, @intFromBool(top_right[2] >= 128)) << 6;
    out |= @as(u16, @intFromBool(btm_left[0] >= 128)) << 5;
    out |= @as(u16, @intFromBool(btm_left[1] >= 128)) << 4;
    out |= @as(u16, @intFromBool(btm_left[2] >= 128)) << 3;
    out |= @as(u16, @intFromBool(btm_right[0] >= 128)) << 2;
    out |= @as(u16, @intFromBool(btm_right[1] >= 128)) << 1;
    out |= @as(u16, @intFromBool(btm_right[2] >= 128)) << 0;

    return 360.0 * @as(f32, @floatFromInt(out)) / 4096.0;
}

fn sample(comptime size: usize, frame: *const c.AVFrame, x: usize, y: usize) [3]usize {
    var sum = [_]usize{0} ** 3;

    for (0..size) |dy| {
        for (0..size) |dx| {
            const _y = y + dy;
            const _x = x + dx;

            const row = _y * @as(usize, @intCast(frame.linesize[0]));
            const px = frame.data[0][row + _x * 3 ..][0..3];

            for (&sum, px) |*a, val| a.* += val;
        }
    }

    for (&sum) |*val| val.* /= size * size; // average values

    return sum;
}

const opengl_impl = struct {
    fn vao() [1]c_uint {
        var vao_id: [1]c_uint = undefined;
        gl.GenVertexArrays(1, vao_id[0..]);

        return vao_id;
    }

    fn vbo() [1]c_uint {
        var vbo_id: [1]c_uint = undefined;
        gl.GenBuffers(1, vbo_id[0..]);

        return vbo_id;
    }

    fn vidTex(width: c_int, height: c_int) [1]c_uint {
        var tex_id: [1]c_uint = undefined;
        gl.GenTextures(1, tex_id[0..]);

        gl.BindTexture(gl.TEXTURE_2D, tex_id[0]);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        gl.TexImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            width,
            height,
            0,
            gl.RGB,
            gl.UNSIGNED_BYTE,
            null,
        );

        return tex_id;
    }

    fn program() !c_uint {
        const vert_shader: [1][*]const u8 = .{@embedFile("shader/texture.vert")[0..].ptr};
        const frag_shader: [1][*]const u8 = .{@embedFile("shader/texture.frag")[0..].ptr};

        const vs = gl.CreateShader(gl.VERTEX_SHADER);
        defer gl.DeleteShader(vs);

        gl.ShaderSource(vs, 1, vert_shader[0..], null);
        gl.CompileShader(vs);

        if (!shader.didCompile(vs)) return error.VertexCompileError;

        const fs = gl.CreateShader(gl.FRAGMENT_SHADER);
        defer gl.DeleteShader(fs);

        gl.ShaderSource(fs, 1, frag_shader[0..], null);
        gl.CompileShader(fs);

        if (!shader.didCompile(fs)) return error.FragmentCompileError;

        const prog = gl.CreateProgram();
        gl.AttachShader(prog, vs);
        gl.AttachShader(prog, fs);
        gl.LinkProgram(prog);

        return prog;
    }

    const shader = struct {
        const log = std.log.scoped(.shader);

        fn didCompile(id: c_uint) bool {
            var success: c_int = undefined;
            gl.GetShaderiv(id, gl.COMPILE_STATUS, &success);

            if (success == 0) err(id);

            return success == 1;
        }

        fn err(id: c_uint) void {
            var error_msg: [512]u8 = undefined;

            gl.GetShaderInfoLog(id, error_msg.len, null, &error_msg);
            log.err("{s}", .{std.mem.sliceTo(&error_msg, 0)});
        }
    };
};

// https://github.com/castholm/zig-examples/blob/77a829c85b5ddbad673026d504626015db4093ac/opengl-sdl/main.zig#L200-L219
inline fn errify(value: anytype) error{sdl_error}!switch (@typeInfo(@TypeOf(value))) {
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
