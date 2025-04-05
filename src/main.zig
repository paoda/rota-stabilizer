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
    defer c.avformat_close_input(&maybe_fmt_ctx);

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

    // lmao just put SDL stuff after this
    c.SDL_SetMainReady();

    try errify(c.SDL_Init(c.SDL_INIT_AUDIO | c.SDL_INIT_VIDEO | c.SDL_INIT_AUDIO));

    try errify(c.SDL_SetAppMetadata("Rotaeno Stabilizer", "0.1.0", "moe.paoda.rota-stabilizer"));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));

    const sdl_stream = blk: {
        var desired: c.SDL_AudioSpec = std.mem.zeroes(c.SDL_AudioSpec);
        desired.freq = aud_codec_ctx.sample_rate;
        desired.format = c.SDL_AUDIO_F32;
        desired.channels = aud_codec_ctx.ch_layout.nb_channels;

        break :blk c.SDL_OpenAudioDeviceStream(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired, null, null).?;
    };
    defer c.SDL_DestroyAudioStream(sdl_stream);

    _ = c.SDL_ResumeAudioStreamDevice(sdl_stream);

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
    var tex_vao_id = opengl_impl.vao();
    defer gl.DeleteVertexArrays(1, tex_vao_id[0..]);

    var tex_vbo_id = opengl_impl.vbo();
    defer gl.DeleteBuffers(1, tex_vbo_id[0..]);

    var ring_vao_id = opengl_impl.vao();
    defer gl.DeleteVertexArrays(1, ring_vao_id[0..]);

    var ring_vbo_id = opengl_impl.vbo();
    defer gl.DeleteBuffers(1, ring_vbo_id[0..]);

    {
        gl.BindVertexArray(tex_vao_id[0]);
        defer gl.BindVertexArray(0);

        gl.BindBuffer(gl.ARRAY_BUFFER, tex_vbo_id[0]);
        defer gl.BindBuffer(gl.ARRAY_BUFFER, 0);

        gl.BufferData(gl.ARRAY_BUFFER, @sizeOf(@TypeOf(vertices)), vertices[0..].ptr, gl.STATIC_DRAW);

        gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 0);
        gl.EnableVertexAttribArray(0);

        gl.VertexAttribPointer(1, 2, gl.FLOAT, gl.FALSE, 4 * @sizeOf(f32), 2 * @sizeOf(f32));
        gl.EnableVertexAttribArray(1);
    }

    // TODO: Can I calculate these values or something?

    const radius = 0.749; // TODO: come back later and see if off-by-one looks better still
    const ring_vertices = try ring(allocator, radius - 0.015, radius, 0x400);

    defer ring_vertices.deinit();
    {
        gl.BindVertexArray(ring_vao_id[0]);
        defer gl.BindVertexArray(0);

        gl.BindBuffer(gl.ARRAY_BUFFER, ring_vbo_id[0]);
        defer gl.BindBuffer(gl.ARRAY_BUFFER, 0);

        gl.BufferData(gl.ARRAY_BUFFER, @intCast(ring_vertices.items.len * @sizeOf(f32)), ring_vertices.items[0..].ptr, gl.STATIC_DRAW);
        gl.VertexAttribPointer(0, 2, gl.FLOAT, gl.FALSE, 2 * @sizeOf(f32), 0);
        gl.EnableVertexAttribArray(0);
    }

    var tex_id = opengl_impl.vidTex(vid_codec_ctx.width, vid_codec_ctx.height);
    defer gl.DeleteTextures(1, tex_id[0..]);

    const tex_prog = try opengl_impl.program("shader/texture.vert", "shader/texture.frag");
    defer gl.DeleteProgram(tex_prog);

    const ring_prog = try opengl_impl.program("shader/ring.vert", "shader/ring.frag");
    defer gl.DeleteProgram(ring_prog);

    // -- opengl end --
    var queue = try FrameQueue.init(allocator, 0x40);
    defer queue.deinit(allocator);

    var should_quit: std.atomic.Value(bool) = .init(false);

    const decode_thread = try std.Thread.spawn(.{}, decode, .{
        fmt_ctx,
        AVBundle{ .codec_ctx = vid_codec_ctx, .stream = vid_stream },
        AVBundle{ .codec_ctx = aud_codec_ctx, .stream = aud_stream },
        sdl_stream,
        &queue,
        &should_quit,
    });
    defer decode_thread.join();

    var start_time: ?u64 = 0; // set to null to disable vsync
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

        gl.ClearColor(0, 0, 0, 0);
        gl.Clear(gl.COLOR_BUFFER_BIT);

        // blocking: while (true) {
        //     current_frame = queue.pop() orelse continue :blocking;
        //     break :blocking;
        // }

        if (queue.pop()) |frame| current_frame = frame;

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

        const aspect = @as(f32, @floatFromInt(frame.width)) / @as(f32, @floatFromInt(frame.height));
        const ratio: [2]f32 = if (aspect > 1.0) .{ 1.0, 1.0 / aspect } else .{ aspect, 1.0 };
        // const scale = 1.0 / std.math.sqrt(ratio[0] * ratio[0] + ratio[1] * ratio[1]); // factor allows for ration w/out clipping
        const scale = 1.0;

        {
            gl.UseProgram(ring_prog);
            defer gl.UseProgram(0);

            gl.BindVertexArray(ring_vao_id[0]);
            defer gl.BindVertexArray(0);

            gl.Uniform1f(gl.GetUniformLocation(ring_prog, "u_scale"), scale);

            gl.DrawArrays(gl.TRIANGLE_STRIP, 0, @intCast(ring_vertices.items.len));
        }

        {
            gl.UseProgram(tex_prog);
            defer gl.UseProgram(0);

            gl.BindVertexArray(tex_vao_id[0]);
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

            const rad = -angle(frame, @intCast(frame.width), @intCast(frame.height)) * std.math.rad_per_deg;

            gl.Uniform2f(gl.GetUniformLocation(tex_prog, "u_aspect"), ratio[0], ratio[1]);
            gl.Uniform1f(gl.GetUniformLocation(tex_prog, "u_scale"), scale);
            gl.Uniform2f(gl.GetUniformLocation(tex_prog, "u_rotation"), @sin(rad), @cos(rad));
            gl.Uniform1i(gl.GetUniformLocation(tex_prog, "u_screen"), 0);

            gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }

        try errify(c.SDL_GL_SwapWindow(window));
    }
}

const AVBundle = struct {
    codec_ctx: *c.AVCodecContext,
    stream: usize,
};

fn decode(fmt_ctx: *c.AVFormatContext, vid: AVBundle, aud: AVBundle, sdl_stream: *c.SDL_AudioStream, queue: *FrameQueue, should_quit: *std.atomic.Value(bool)) !void {
    const log = std.log.scoped(.decode);

    log.info("decode thread start", .{});
    defer log.info("decode thread end", .{});

    var maybe_pkt: ?*c.AVPacket = c.av_packet_alloc();
    defer c.av_packet_free(&maybe_pkt);

    var maybe_src_frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(&maybe_src_frame);

    var maybe_dst_frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(&maybe_dst_frame);

    var maybe_aud_frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(&maybe_aud_frame);

    const pkt = maybe_pkt orelse return error.out_of_memory;
    const src_frame = maybe_src_frame orelse return error.out_of_memory;
    const dst_frame = maybe_dst_frame orelse return error.out_of_memory;
    const aud_frame = maybe_aud_frame orelse return error.out_of_memory;

    _ = try libavError(c.av_channel_layout_copy(&aud_frame.ch_layout, &aud.codec_ctx.ch_layout));
    aud_frame.sample_rate = aud.codec_ctx.sample_rate;
    aud_frame.format = c.AV_SAMPLE_FMT_FLT;

    dst_frame.width = vid.codec_ctx.width;
    dst_frame.height = vid.codec_ctx.height;
    dst_frame.format = c.AV_PIX_FMT_RGB24;

    _ = try libavError(c.av_frame_get_buffer(dst_frame, 32));

    const maybe_sws = c.sws_getContext(
        vid.codec_ctx.width,
        vid.codec_ctx.height,
        vid.codec_ctx.pix_fmt,
        vid.codec_ctx.width,
        vid.codec_ctx.height,
        c.AV_PIX_FMT_RGB24,
        c.SWS_BILINEAR,
        null,
        null,
        null,
    );
    defer c.sws_freeContext(maybe_sws);

    var maybe_swr = c.swr_alloc();
    defer c.swr_free(&maybe_swr);

    const sws = maybe_sws orelse return error.out_of_memory;
    const swr = maybe_swr orelse return error.out_of_memory;

    while (c.av_read_frame(fmt_ctx, pkt) >= 0) {
        defer c.av_packet_unref(pkt);
        if (should_quit.load(.monotonic)) return;

        // TODO: cleanup this code

        if (pkt.stream_index == aud.stream) {
            const send_ret = c.avcodec_send_packet(aud.codec_ctx, pkt);
            if (send_ret == c.AVERROR_EOF) return;
            if (send_ret == c.AVERROR(c.EAGAIN)) @panic("TODO: handle EAGAIN");
            if (send_ret != 0) @panic("TODO: unrecoverable error in avcodec_send_packet");

            const recv_ret = c.avcodec_receive_frame(aud.codec_ctx, src_frame);
            if (recv_ret == c.AVERROR_EOF) return;
            if (recv_ret == c.AVERROR(c.EAGAIN)) @panic("TODO: handle EAGAIN");
            if (recv_ret != 0) @panic("TODO: unrecoverable error in avcodec_send_packet");

            _ = c.swr_convert_frame(swr, aud_frame, src_frame);

            // std.debug.print("src sample_rate: {}Hz\n", .{src_frame.sample_rate});
            // std.debug.print("src nb_channels: {}\n", .{src_frame.ch_layout.nb_channels});
            // std.debug.print("src format: {s}\n", .{c.av_get_sample_fmt_name(src_frame.format)});
            // std.debug.print("dst sample_rate: {}Hz\n", .{aud_frame.sample_rate});
            // std.debug.print("dst format: {s}\n", .{c.av_get_sample_fmt_name(aud_frame.format)});
            // std.debug.print("dst nb_channels: {}\n", .{aud_frame.ch_layout.nb_channels});
            // std.debug.print("delay (ms): {}\n\n", .{c.swr_get_delay(swr, 1000)});

            _ = c.SDL_PutAudioStreamData(sdl_stream, aud_frame.data[0], src_frame.linesize[0]);
            continue;
        } else if (pkt.stream_index == vid.stream) {
            const send_ret = c.avcodec_send_packet(vid.codec_ctx, pkt);
            if (send_ret == c.AVERROR_EOF) return;
            if (send_ret == c.AVERROR(c.EAGAIN)) @panic("TODO: handle EAGAIN");
            if (send_ret != 0) @panic("TODO: unrecoverable error in avcodec_send_packet");

            const recv_ret = c.avcodec_receive_frame(vid.codec_ctx, src_frame);
            if (recv_ret == c.AVERROR_EOF) return;
            if (recv_ret == c.AVERROR(c.EAGAIN)) @panic("TODO: handle EAGAIN");
            if (recv_ret != 0) @panic("TODO: unrecoverable error in avcodec_receive_frame");

            dst_frame.pts = src_frame.pts; // for timing

            _ = c.sws_scale(
                sws,
                src_frame.data[0..],
                src_frame.linesize[0..],
                0,
                src_frame.height, // TODO: this should be src_frame
                dst_frame.data[0..],
                dst_frame.linesize[0..],
            );

            blocking: while (true) {
                queue.push(dst_frame) catch |e| {
                    if (e == error.full) continue :blocking;
                    std.debug.panic("error: {}", .{e});
                };

                break :blocking;
            }
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

    fn program(comptime vert_path: []const u8, comptime frag_path: []const u8) !c_uint {
        const vert_shader: [1][*]const u8 = .{@embedFile(vert_path)[0..].ptr};
        const frag_shader: [1][*]const u8 = .{@embedFile(frag_path)[0..].ptr};

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

fn ring(allocator: std.mem.Allocator, inner_radius: f32, outer_radius: f32, len: usize) !std.ArrayList(f32) {
    var list = std.ArrayList(f32).init(allocator);
    errdefer list.deinit();

    const _len: f32 = @floatFromInt(len);

    for (0..len) |i| {
        const _angle = @as(f32, @floatFromInt(i)) * 2.0 * std.math.pi / _len;
        const x = @cos(_angle);
        const y = @sin(_angle);

        try list.append(x * outer_radius);
        try list.append(y * outer_radius);

        try list.append(x * inner_radius);
        try list.append(y * inner_radius);
    }

    try list.appendSlice(list.items[0..4]); // complete the loop

    return list;
}
