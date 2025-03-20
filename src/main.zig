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
    // @cInclude("libswscale/swscale.h");
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

    if (c.avformat_find_stream_info(format_ctx, null) < 0)
        return error.find_stream_info_failed;

    var found: ?VideoStream = null;

    for (0..format_ctx.nb_streams) |i| {
        std.debug.print("AVStream->time_base before open coded {}/{}\n", .{ format_ctx.streams[i].*.time_base.num, format_ctx.streams[i].*.time_base.den });
        std.debug.print("AVStream->r_frame_rate before open coded {}/{}\n", .{ format_ctx.streams[i].*.r_frame_rate.num, format_ctx.streams[i].*.r_frame_rate.den });
        std.debug.print("AVStream->start_time {}\n", .{format_ctx.streams[i].*.start_time});
        std.debug.print("AVStream->duration {}\n", .{format_ctx.streams[i].*.duration});

        const codec_params: *c.AVCodecParameters = format_ctx.streams[i].*.codecpar;

        const codec = blk: {
            const ptr: ?*const c.AVCodec = c.avcodec_find_decoder(codec_params.codec_id);
            if (ptr == null) continue; // skip unmatched codecs

            break :blk ptr.?;
        };

        switch (codec.type) {
            c.AVMEDIA_TYPE_VIDEO => {
                std.debug.print("Video Codec: resolution {} x {}\n", .{ codec_params.width, codec_params.height });
                found = .{ .codec = codec, .params = codec_params, .idx = @intCast(i) };
            },
            c.AVMEDIA_TYPE_AUDIO => {
                std.debug.print("Audio Codec: {} channels, sample rate: {}\n", .{ codec_params.ch_layout.nb_channels, codec_params.sample_rate });
            },
            else => {},
        }

        std.debug.print("Codec: {s} ID: {} bit_rate: {}\n", .{ codec.name, codec.id, codec_params.bit_rate });
        std.debug.print("\n", .{});
    }
    std.debug.print("format {s}, duration: {} us, bit_rate: {}\n", .{ format_ctx.iformat.*.name, format_ctx.duration, format_ctx.bit_rate });

    if (found == null) return error.no_video_stream_found;

    const codec_ctx: *c.AVCodecContext = blk: {
        const ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(found.?.codec);
        if (ptr == null) return error.out_of_memory;

        if (c.avcodec_parameters_to_context(ptr, found.?.params) < 0)
            return error.codec_copy_failed;

        break :blk ptr.?;
    };
    defer c.avcodec_free_context(@constCast(@ptrCast(&codec_ctx))); // SAFETY: only okay 'cause we're cleaning up

    if (c.avcodec_open2(codec_ctx, found.?.codec, null) < 0)
        return error.codec_open_failed;

    const frame: ?*c.AVFrame = c.av_frame_alloc();
    defer c.av_frame_free(@constCast(@ptrCast(&frame))); // SAFETY: only okay 'cause we're cleaning up

    const pkt: ?*c.AVPacket = c.av_packet_alloc();
    defer c.av_packet_free(@constCast(@ptrCast(&pkt))); // SAFETY: only okay 'cause we're cleaning up

    if (frame == null) return error.out_of_memory;
    if (pkt == null) return error.out_of_memory;

    var how_many_packets_to_process: i32 = 8;

    while (c.av_read_frame(format_ctx, pkt) >= 0) {
        defer c.av_packet_unref(pkt);
        if (pkt.?.stream_index != found.?.idx) continue;

        std.debug.print("AVPacket->pts {}\n", .{pkt.?.pts});
        decodePacket(pkt.?, codec_ctx, frame.?) catch break;

        // don't process too many for this dmeo
        how_many_packets_to_process -= 1;
        if (how_many_packets_to_process <= 0) break;
    }

    // lmao just put SDL stuff after this
    c.SDL_SetMainReady();

    try errify(c.SDL_Init(c.SDL_INIT_AUDIO | c.SDL_INIT_VIDEO));

    try errify(c.SDL_SetAppMetadata("Rotaeno Stabilizer", "0.1.0", "moe.paoda.rota-stabilizer"));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MAJOR_VERSION, gl.info.version_major));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_MINOR_VERSION, gl.info.version_minor));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_PROFILE_MASK, c.SDL_GL_CONTEXT_PROFILE_CORE));
    try errify(c.SDL_GL_SetAttribute(c.SDL_GL_CONTEXT_FLAGS, c.SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG));

    const window: *c.SDL_Window = try errify(c.SDL_CreateWindow("Rotaeno Stabilizer", 640, 480, c.SDL_WINDOW_OPENGL | c.SDL_WINDOW_RESIZABLE));
    defer c.SDL_DestroyWindow(window);

    const gl_ctx = try errify(c.SDL_GL_CreateContext(window));
    defer errify(c.SDL_GL_DestroyContext(gl_ctx)) catch {};

    try errify(c.SDL_GL_MakeCurrent(window, gl_ctx));
    defer errify(c.SDL_GL_MakeCurrent(window, null)) catch {};

    if (!gl_procs.init(c.SDL_GL_GetProcAddress)) return error.gl_init_failed;

    gl.makeProcTableCurrent(&gl_procs);
    defer gl.makeProcTableCurrent(null);

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

        try errify(c.SDL_GL_SwapWindow(window));
    }
}

fn decodePacket(pkt: *c.AVPacket, codec_ctx: *c.AVCodecContext, frame: *c.AVFrame) !void {
    var response = c.avcodec_send_packet(codec_ctx, pkt);
    if (response < 0) return error.packet_send_failure;

    while (response >= 0) {
        response = c.avcodec_receive_frame(codec_ctx, frame);
        if (response == c.AVERROR(c.EAGAIN) or response == c.AVERROR_EOF) break;
        if (response < 0) return error.decoding_failure;

        if (response >= 0) {
            std.debug.print("Frame: {} (type=?, size={} bytes, format={}) pts {} key_frame {}\n", .{
                codec_ctx.frame_num,
                // &[_]u8{c.av_get_picture_type_char(frame.pict_type)},
                frame.pkt_size,
                frame.format,
                frame.pts,
                frame.key_frame,
            });

            var buf: [1024]u8 = undefined;
            const path = try std.fmt.bufPrint(&buf, "{s}-{}.pgm", .{ "frame", codec_ctx.frame_num });

            try saveGrayFrame(frame.data[0], @intCast(frame.linesize[0]), @intCast(frame.width), @intCast(frame.height), path);
        }
    }
}

pub fn saveGrayFrame(buf: [*]const u8, wrap: usize, width: usize, height: usize, path: []const u8) !void {
    var file = try std.fs.cwd().createFile(path, .{ .truncate = true });
    defer file.close();

    // Prepare the minimal required header for a PGM file
    // Format: "P5\n<xsize> <ysize>\n255\n"
    var w = file.writer();
    try w.print("P5\n{} {}\n255\n", .{ width, height });

    // Write image rows: for each row, write xsize bytes from buf starting at offset (i * wrap)
    for (0..height) |i| {
        const offset = i * wrap;
        try file.writeAll(buf[offset .. offset + width]);
    }
}

const VideoStream = struct {
    codec: *const c.AVCodec,
    params: *const c.struct_AVCodecParameters,
    idx: c_int,
};

// https://github.com/castholm/zig-examples/blob/77a829c85b5ddbad673026d504626015db4093ac/opengl-sdl/main.zig#L200-L219
inline fn errify(value: anytype) error{SdlError}!switch (@typeInfo(@TypeOf(value))) {
    .bool => void,
    .pointer, .optional => @TypeOf(value.?),
    .int => |info| switch (info.signedness) {
        .signed => @TypeOf(@max(0, value)),
        .unsigned => @TypeOf(value),
    },
    else => @compileError("unerrifiable type: " ++ @typeName(@TypeOf(value))),
} {
    return switch (@typeInfo(@TypeOf(value))) {
        .bool => if (!value) error.SdlError,
        .pointer, .optional => value orelse error.SdlError,
        .int => |info| switch (info.signedness) {
            .signed => if (value >= 0) @max(0, value) else error.SdlError,
            .unsigned => if (value != 0) value else error.SdlError,
        },
        else => comptime unreachable,
    };
}
