const std = @import("std");
const lib = @import("rota_stabilizer_lib");
const gl = @import("gl");

const c = @cImport({
    @cDefine("SDL_DISABLE_OLD_NAMES", {});
    @cDefine("SDL_MAIN_HANDLED", {});

    @cInclude("SDL3/SDL.h");
    @cInclude("SDL3/SDL_main.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libswscale/swscale.h");
    @cInclude("libavutil/imgutils.h");
});

var gl_procs: gl.ProcTable = undefined;

pub fn main() !void {
    errdefer |err| if (err == error.sdl_error) std.log.err("SDL Error: {s}", .{c.SDL_GetError()});

    var args = std.process.args();
    _ = args.skip(); // self
    const path = args.next() orelse return error.missing_file;

    const format_ctx: *c.struct_AVFormatContext = blk: {
        var ptr: ?*c.struct_AVFormatContext = c.avformat_alloc_context();
        if (ptr == null) return error.out_of_memory;

        const ret = c.avformat_open_input(&ptr, path, null, null);
        if (ret != 0) return error.file_open_failed;

        break :blk ptr.?;
    };
    defer c.avformat_close_input(@constCast(@ptrCast(&format_ctx))); // SAFETY: only okay 'cause we're cleaning up

    if (c.avformat_find_stream_info(format_ctx, null) < 0) return error.find_stream_info_failed;

    const found_idx: usize = blk: {
        var found: ?usize = null;
        for (0..format_ctx.nb_streams) |i| {
            if (format_ctx.streams[i].*.codecpar.*.codec_type == c.AVMEDIA_TYPE_VIDEO) {
                found = i;
                break;
            }
        }

        break :blk found orelse return error.missing_video_stream;
    };

    const codec = blk: {
        const ptr: ?*const c.AVCodec = c.avcodec_find_decoder(format_ctx.streams[found_idx].*.codecpar.*.codec_id);
        if (ptr == null) return error.unsupported_codec;

        break :blk ptr.?;
    };

    const codec_ctx: *c.AVCodecContext = blk: {
        const ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec);
        if (ptr == null) return error.out_of_memory;

        if (c.avcodec_parameters_to_context(ptr, format_ctx.streams[found_idx].*.codecpar) < 0) return error.codec_copy_failed;
        break :blk ptr.?;
    };
    defer c.avcodec_free_context(@constCast(@ptrCast(&codec_ctx))); // SAFETY: only okay 'cause we're cleaning up

    if (c.avcodec_open2(codec_ctx, codec, null) < 0) return error.codec_open_failed;

    const frame: *c.AVFrame = blk: {
        const ptr: ?*c.AVFrame = c.av_frame_alloc();
        if (ptr == null) return error.out_of_memory;

        break :blk ptr.?;
    };
    defer c.av_frame_free(@constCast(@ptrCast(&frame))); // SAFETY: only okay 'cause we're cleaning up

    const out_frame: *c.AVFrame = blk: {
        const ptr: ?*c.AVFrame = c.av_frame_alloc();
        if (ptr == null) return error.out_of_memory;

        break :blk ptr.?;
    };
    defer c.av_frame_free(@constCast(@ptrCast(&out_frame))); // SAFETY: only okay 'cause we're cleaning up

    const pkt: *c.AVPacket = blk: {
        const ptr: ?*c.AVPacket = c.av_packet_alloc();
        if (ptr == null) return error.out_of_memory;

        break :blk ptr.?;
    };
    defer c.av_packet_free(@constCast(@ptrCast(&pkt))); // SAFETY: only okay 'cause we're cleaning up

    const byte_count: usize = @intCast(c.av_image_get_buffer_size(c.AV_PIX_FMT_RGB24, codec_ctx.width, codec_ctx.height, 32)); // FIXME: why 32?

    const buffer = blk: {
        const ptr: [*]u8 = @ptrCast(c.av_malloc(byte_count * @sizeOf(u8)));
        break :blk ptr[0 .. byte_count / @sizeOf(u8)];
    };

    _ = c.av_image_fill_arrays(
        @ptrCast(&out_frame.data),
        @ptrCast(&out_frame.linesize),
        buffer.ptr,
        c.AV_PIX_FMT_RGB24,
        codec_ctx.width,
        codec_ctx.height,
        32,
    );

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

    var frames_to_decode: i32 = 8;

    while (c.av_read_frame(format_ctx, pkt) >= 0) {
        if (pkt.stream_index != found_idx) continue;

        var ret = c.avcodec_send_packet(codec_ctx, pkt);
        if (ret < 0) return error.packet_decode_send_fail;

        while (ret >= 0) {
            ret = c.avcodec_receive_frame(codec_ctx, frame);
            if (ret == c.AVERROR(c.EAGAIN) or ret == c.AVERROR_EOF) break;
            if (ret < 0) return error.decode_fail;

            _ = c.sws_scale(
                sws_ctx,
                @ptrCast(&frame.data),
                @ptrCast(&frame.linesize),
                0,
                codec_ctx.height,
                @ptrCast(&out_frame.data),
                @ptrCast(&out_frame.linesize),
            );

            try saveFrame(out_frame, @intCast(codec_ctx.width), @intCast(codec_ctx.height), codec_ctx.frame_num);
        }

        frames_to_decode -= 1;
        if (frames_to_decode <= 0) break;
    }

    // lmao just put SDL stuff after this
    c.SDL_SetMainReady();

    try errify(c.SDL_Init(c.SDL_INIT_AUDIO | c.SDL_INIT_VIDEO));

    try errify(c.SDL_SetAppMetadata("Rotaeno Stabilizer", "0.1.0", "moe.paoda.rota-stabilizer"));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));

    const window: *c.SDL_Window = try errify(c.SDL_CreateWindow("Rotaeno Stabilizer", codec_ctx.width, codec_ctx.height, c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE));
    defer c.SDL_DestroyWindow(window);

    const gl_ctx = try errify(c.SDL_GL_CreateContext(window));
    defer errify(c.SDL_GL_DestroyContext(gl_ctx)) catch {};

    try errify(c.SDL_GL_MakeCurrent(window, gl_ctx));
    defer errify(c.SDL_GL_MakeCurrent(window, null)) catch {};

    if (!gl_procs.init(c.SDL_GL_GetProcAddress)) return error.gl_init_failed;

    gl.makeProcTableCurrent(&gl_procs);
    defer gl.makeProcTableCurrent(null);

    // -- opengl --
    var vao_id = opengl_impl.vao();
    defer gl.DeleteVertexArrays(1, vao_id[0..]);

    var tex: [2]c_uint = .{
        opengl_impl.vidTex(out_frame, codec_ctx.width, codec_ctx.height),
        opengl_impl.outTex(codec_ctx.width, codec_ctx.height),
    };
    defer gl.DeleteBuffers(2, tex[0..]);

    // var fbo_id = try opengl_impl.frameBuffer(tex[1]);
    // defer gl.DeleteFramebuffers(1, fbo_id[0..]);

    const prog_id = try opengl_impl.program();
    defer gl.DeleteProgram(prog_id);

    // -- opengl end --

    win_loop: while (true) {
        var event: c.SDL_Event = undefined;
        var timeout: i32 = -1;

        while (c.SDL_WaitEventTimeout(&event, timeout)) : (timeout = 0) {
            if (event.type == c.SDL_EVENT_QUIT) break :win_loop;
        }

        var w: c_int, var h: c_int = .{ undefined, undefined };
        try errify(c.SDL_GetWindowSizeInPixels(window, &w, &h));

        gl.Viewport(0, 0, w, h);
        gl.ClearBufferfv(gl.COLOR, 0, &.{ 1, 1, 1, 1 });

        {
            // gl.BindFramebuffer(gl.FRAMEBUFFER, fbo_id[1]);
            // defer gl.BindFramebuffer(.gl.FRAMEBUFFER, 0);

            gl.BindTexture(gl.TEXTURE_2D, tex[0]);
            defer gl.BindTexture(gl.TEXTURE_2D, 0);

            gl.TexSubImage2D(
                gl.TEXTURE_2D,
                0,
                0,
                0,
                codec_ctx.width,
                codec_ctx.height,
                gl.RGBA,
                gl.UNSIGNED_INT_8_8_8_8,
                frame.data[0][0..],
            );

            gl.BindVertexArray(vao_id[0]);
            defer gl.BindVertexArray(0);

            gl.UseProgram(prog_id);
            defer gl.UseProgram(0);

            gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 3);
        }

        try errify(c.SDL_GL_SwapWindow(window));
    }
}

fn saveFrame(frame: *c.AVFrame, width: usize, height: usize, frame_idx: i64) !void {
    const path = blk: {
        var buf: [0x100]u8 = undefined;
        break :blk try std.fmt.bufPrint(&buf, "frame-{}.ppm", .{frame_idx});
    };

    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    var w = file.writer();
    try w.print("P6\n{} {}\n255\n", .{ width, height });

    for (0..height) |i| {
        const offset = i * @as(usize, @intCast(frame.linesize[0]));

        try file.writeAll(frame.data[0][offset .. offset + (width * 3)]);
    }
}

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

const opengl_impl = struct {
    fn vao() [1]c_uint {
        var vao_id: [1]c_uint = undefined;
        gl.GenVertexArrays(1, vao_id[0..]);

        return vao_id;
    }

    fn vidTex(frame: *c.AVFrame, width: c_int, height: c_int) c_uint {
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
            gl.RGBA,
            gl.UNSIGNED_INT_8_8_8_8,
            frame.data[0][0..],
        );

        return tex_id[0];
    }

    fn outTex(width: c_int, height: c_int) c_uint {
        var tex_id: [1]c_uint = undefined;
        gl.GenTextures(1, tex_id[0..]);

        gl.BindTexture(gl.TEXTURE_2D, tex_id[0]);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.TexParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

        gl.TexImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_INT_8_8_8_8, null);

        return tex_id[0];
    }

    fn frameBuffer(tex_id: c_uint) ![1]c_uint {
        var fbo_id: [1]c_uint = undefined;
        gl.GenFramebuffers(1, fbo_id[0..]);

        gl.BindFramebuffer(gl.FRAMEBUFFER, fbo_id[0]);
        defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        gl.FramebufferTexture(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, tex_id, 0);
        gl.DrawBuffers(1, &@as(c_uint, gl.COLOR_ATTACHMENT0));

        if (gl.CheckFramebufferStatus(gl.FRAMEBUFFER) != gl.FRAMEBUFFER_COMPLETE)
            return error.FrameBufferObejctInitFailed;

        return fbo_id;
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
