//! FFmpeg bindings
const std = @import("std");
const c = @import("../lib.zig").c;

pub const AvPacket = struct {
    inner: ?*c.AVPacket,

    pub fn init() AvPacket {
        const p: ?*c.AVPacket = c.av_packet_alloc();
        errdefer c.av_packet_free(p);

        return .{ .inner = p };
    }

    pub fn deinit(self: *@This()) void {
        c.av_packet_free(&self.inner);
    }

    pub inline fn ptr(self: @This()) *c.AVPacket {
        return self.inner.?;
    }
};

pub const AvCodecContext = struct {
    const Kind = enum { video, audio };
    const Options = struct { dev_type: ?c.AVHWDeviceType = null };

    inner: ?*c.AVCodecContext,
    stream: c_int,

    device: ?Device = null, //TODO: move Device to inside this struct

    const log = std.log.scoped(.codec);

    pub fn init(allocator: std.mem.Allocator, comptime kind: Kind, fmt_ctx: AvFormatContext, opt: Options) !*AvCodecContext {
        const self = try allocator.create(@This());

        if (kind == .video) blk: {
            const device_type = opt.dev_type orelse break :blk;

            self.initHardware(fmt_ctx, device_type) catch |e| {
                log.err("failed to set up hardware context: {}", .{e});
                break :blk log.warn("defaulting to software", .{});
            };

            self.device.?.configure();
            return self;
        }

        try self.initSoftware(kind, fmt_ctx);
        return self;
    }

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        if (self.device) |*dev| dev.deinit();
        c.avcodec_free_context(&self.inner);

        allocator.destroy(self);
    }

    fn initSoftware(self: *@This(), comptime kind: Kind, fmt_ctx: AvFormatContext) !void {
        const media_type = switch (kind) {
            .video => c.AVMEDIA_TYPE_VIDEO,
            .audio => c.AVMEDIA_TYPE_AUDIO,
        };

        var codec_ptr: ?*const c.AVCodec = null;
        const stream = try err(c.av_find_best_stream(fmt_ctx.ptr(), media_type, -1, -1, &codec_ptr, 0));

        var ctx_ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec_ptr);
        errdefer c.avcodec_free_context(&ctx_ptr);

        _ = try err(c.avcodec_parameters_to_context(ctx_ptr, fmt_ctx.ptr().streams[@intCast(stream)].*.codecpar));
        _ = try err(c.avcodec_open2(ctx_ptr, codec_ptr, null));

        self.* = .{ .inner = ctx_ptr, .stream = stream };
    }

    fn initHardware(self: *@This(), fmt_ctx: AvFormatContext, device_type: c.AVHWDeviceType) !void {
        var codec_ptr: ?*const c.AVCodec = null;
        const stream = try err(c.av_find_best_stream(fmt_ctx.ptr(), c.AVMEDIA_TYPE_VIDEO, -1, -1, &codec_ptr, 0));

        var ctx_ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec_ptr);
        errdefer c.avcodec_free_context(&ctx_ptr);

        _ = try err(c.avcodec_parameters_to_context(ctx_ptr, fmt_ctx.ptr().streams[@intCast(stream)].*.codecpar));

        var hw_device_ctx: ?*c.AVBufferRef = null;
        var hw_pix_fmt: c.AVPixelFormat = undefined;

        {
            var i: c_int = 0;
            while (true) {
                defer i += 1;

                const config: ?*const c.AVCodecHWConfig = c.avcodec_get_hw_config(codec_ptr.?, i);
                if (config == null) {
                    log.err("no hardware {s} decoder found in {s} device", .{ codec_ptr.?.name, c.av_hwdevice_get_type_name(device_type) });
                    return error.unsupported_codec;
                }

                if (config.?.methods & c.AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX == 0) continue;
                if (config.?.device_type != device_type) continue;

                hw_pix_fmt = config.?.pix_fmt;
                log.info("found {s} hwdevice for {s} decoder", .{ c.av_hwdevice_get_type_name(device_type), codec_ptr.?.name });

                // FIXME: does this need to be deallocated????
                _ = try err(c.av_hwdevice_ctx_create(&hw_device_ctx, device_type, null, null, 0));
                ctx_ptr.?.hw_device_ctx = c.av_buffer_ref(hw_device_ctx);

                {
                    var constraints: ?*c.AVHWFramesConstraints = c.av_hwdevice_get_hwframe_constraints(hw_device_ctx, null);
                    defer c.av_hwframe_constraints_free(&constraints);

                    log.debug("valid sw formats:", .{});
                    const valid_sw = std.mem.sliceTo(constraints.?.valid_sw_formats, c.AV_SAMPLE_FMT_NONE);
                    for (valid_sw) |fmt| log.debug("\t{s}", .{c.av_get_pix_fmt_name(fmt)});
                }

                break;
            }
        }

        _ = try err(c.avcodec_open2(ctx_ptr, codec_ptr, null));

        self.* = .{
            .inner = ctx_ptr,
            .stream = stream,
            .device = .{
                .pix_fmt = hw_pix_fmt,
                .ctx = hw_device_ctx,
            },
        };
    }
};

const Device = struct {
    ctx: ?*c.AVBufferRef,
    pix_fmt: c.AVPixelFormat,

    /// Must be called immediately after creation, sets a self-referential ptr in self.codec_ctx.@"opaque"
    ///
    /// Also sets the get_format fn pointer in AVCodecContext
    fn configure(self: *@This()) void {
        const super: *AvCodecContext = @fieldParentPtr("device", @as(*?Device, @ptrCast(self)));

        super.inner.?.@"opaque" = @ptrCast(self);
        super.inner.?.get_format = @This().getFormat;
    }

    fn deinit(self: *@This()) void {
        c.av_buffer_unref(&self.ctx);
    }

    fn getFormat(ctx: ?*c.AVCodecContext, formats: [*c]const c.AVPixelFormat) callconv(.c) c.AVPixelFormat {
        const self: *const @This() = @alignCast(@ptrCast(ctx.?.@"opaque"));

        const fmts = std.mem.sliceTo(formats, c.AV_PIX_FMT_NONE);
        for (fmts) |fmt| {
            // if (hw_pix_fmt == null) break; TODO: maybe i still need this?
            if (fmt == self.pix_fmt) return self.pix_fmt;
        }

        return c.AV_PIX_FMT_NONE;
    }
};

pub const AvFormatContext = struct {
    inner: ?*c.AVFormatContext,

    pub fn init(path: []const u8) !AvFormatContext {
        var p: ?*c.AVFormatContext = c.avformat_alloc_context();
        errdefer c.avformat_close_input(&p);

        _ = try err(c.avformat_open_input(&p, path.ptr, null, null));
        _ = try err(c.avformat_find_stream_info(p, null));

        return .{ .inner = p };
    }

    pub inline fn ptr(self: @This()) *c.AVFormatContext {
        return self.inner.?;
    }

    pub fn deinit(self: *@This()) void {
        c.avformat_close_input(&self.inner);
    }
};

pub const AvFrame = struct {
    inner: ?*c.AVFrame,

    pub fn init() !AvFrame {
        const p: ?*c.AVFrame = c.av_frame_alloc();
        if (p == null) return error.out_of_memory;

        return .{ .inner = p };
    }

    pub fn setup(self: *@This(), width: c_int, height: c_int, fmt: c.AVPixelFormat) !void {
        self.inner.?.width = width;
        self.inner.?.height = height;
        self.inner.?.format = fmt;

        _ = try err(c.av_frame_get_buffer(self.inner, 32));
    }

    pub fn deinit(self: *@This()) void {
        c.av_frame_free(&self.inner);
    }

    pub fn ptr(self: @This()) *c.AVFrame {
        return self.inner.?;
    }
};

pub inline fn err(value: c_int) error{ffmpeg_error}!c_int {
    if (value >= 0) return value;
    var buf: [0x100]u8 = undefined;
    const ret = c.av_strerror(value, &buf, buf.len);

    if (ret < 0) std.debug.panic("paniced in ffmpeg error handler: {}", .{value});

    std.debug.print("ffmpeg err: {s}\n", .{std.mem.sliceTo(&buf, 0)});
    return error.ffmpeg_error;
}
