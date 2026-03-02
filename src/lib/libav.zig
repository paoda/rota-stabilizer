//! FFmpeg bindings
const std = @import("std");
const c = @import("../lib.zig").c;

pub const enc = struct {
    const log = std.log.scoped(.encode);

    pub const AvCodec = struct {
        inner: ?*const c.AVCodec,

        pix_fmt: c.AVPixelFormat,

        hw: ?struct { dev_type: c.AVHWDeviceType } = null,

        pub fn findSoftware(codec_id: c.AVCodecID) AvCodec {
            const codec = c.avcodec_find_encoder(codec_id);
            const ideal_fmt = c.AV_PIX_FMT_RGB24;

            const pix_fmts: [*]const c.AVPixelFormat = @as(?[*]const c.AVPixelFormat, codec.*.pix_fmts) orelse @panic("FIXME: no supported sw encode pix fmt?");

            var i: usize = 0;
            while (pix_fmts[i] != c.AV_PIX_FMT_NONE) : (i += 1) {
                if (pix_fmts[i] == ideal_fmt) return .{ .inner = codec, .pix_fmt = ideal_fmt };
            }

            return .{ .inner = codec, .pix_fmt = pix_fmts[0] };
        }

        pub fn findHardware(dev_type: c.AVHWDeviceType, codec_id: c.AVCodecID) ?AvCodec {
            log.debug("search for hardware encoder", .{});
            const av_codec_iterate = struct {
                fn inner(idx: *?*anyopaque) ?*const c.AVCodec {
                    return c.av_codec_iterate(idx);
                }
            }.inner;

            var i: ?*anyopaque = null;
            while (av_codec_iterate(&i)) |codec| {
                if (c.av_codec_is_encoder(codec) == 0) continue;
                if (codec.id != codec_id) continue;

                for (0..0x100) |j| { // FIXME: some arbitrary limit
                    const config: *const c.AVCodecHWConfig = c.avcodec_get_hw_config(codec, @intCast(j)) orelse break;

                    if (config.methods & c.AV_CODEC_HW_CONFIG_METHOD_HW_FRAMES_CTX == 0) continue;
                    if (config.device_type != dev_type) continue;

                    log.info("found {s} encoder for {s} ", .{ c.av_hwdevice_get_type_name(dev_type), codec.name });
                    return .{ .inner = codec, .pix_fmt = config.pix_fmt, .hw = .{ .dev_type = dev_type } };
                }
            }

            return null; // try for software encoder
        }

        pub fn ptr(self: AvCodec) *const c.AVCodec {
            return self.inner.?;
        }
    };

    pub const AvFormatContext = struct {
        inner: ?*c.AVFormatContext,

        pub fn init(path: []const u8) !AvFormatContext {
            var p: ?*c.AVFormatContext = null;
            _ = try err(c.avformat_alloc_output_context2(&p, null, null, path.ptr));

            return .{ .inner = p };
        }

        pub fn ptr(self: AvFormatContext) *c.AVFormatContext {
            return self.inner.?;
        }

        pub fn deinit(self: AvFormatContext) void {
            if (self.inner.?.oformat.*.flags & c.AVFMT_NOFILE == 0) {
                _ = c.avio_closep(&self.inner.?.pb);
            }

            c.avformat_free_context(self.inner);
        }
    };

    pub const AvStream = struct {
        inner: ?*c.AVStream,

        pub fn init(fmt_ctx: AvFormatContext) !AvStream {
            const p: ?*c.AVStream = c.avformat_new_stream(fmt_ctx.ptr(), null);
            if (p == null) return error.ffmpeg_error;

            return .{ .inner = p };
        }

        pub fn ptr(self: AvStream) *c.AVStream {
            return self.inner.?;
        }
    };

    const AvHwDeviceContext = struct {
        inner: ?*c.AVBufferRef,

        fn init(hw_device: c.AVHWDeviceType) !AvHwDeviceContext {
            var p: ?*c.AVBufferRef = null;
            _ = try err(c.av_hwdevice_ctx_create(&p, hw_device, null, null, 0));

            return .{ .inner = p };
        }

        fn ptr(self: AvHwDeviceContext) *c.AVBufferRef {
            return self.inner.?;
        }

        fn deinit(self: *AvHwDeviceContext) void {
            c.av_buffer_unref(&self.inner);
        }
    };

    const AvHwFrameRef = struct {
        inner: ?*c.AVBufferRef,

        fn init(device_ctx: AvHwDeviceContext) !AvHwFrameRef {
            const p: ?*c.AVBufferRef = c.av_hwframe_ctx_alloc(device_ctx.ptr());
            if (p == null) return error.ffmpeg_error;

            return .{ .inner = p };
        }

        fn ptr(self: AvHwFrameRef) *c.AVBufferRef {
            return self.inner.?;
        }

        fn deinit(self: *AvHwFrameRef) void {
            c.av_buffer_unref(&self.inner);
        }
    };

    pub const AvCodecContext = struct {
        inner: ?*c.AVCodecContext,

        hw: ?struct { dev_ctx: AvHwDeviceContext } = null,

        const Options = struct {
            // Viewport of the video that will be encoded
            width: c_int,
            height: c_int,

            input: Input,

            const Input = struct {
                fmt_ctx: dec.AvFormatContext,
                video_ctx: *const dec.AvCodecContext,
            };
        };

        pub fn init(codec: AvCodec, fmt: AvFormatContext, opt: Options) !AvCodecContext {
            const KBPS_TO_BPS = 1000;

            var p: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec.ptr());
            errdefer c.avcodec_free_context(&p);

            const ctx = p.?;

            const inpt_vid = opt.input.fmt_ctx.ptr().streams[@intCast(opt.input.video_ctx.stream)];

            ctx.width = opt.width;
            ctx.height = opt.height;
            ctx.time_base = inpt_vid.*.time_base;
            ctx.framerate = inpt_vid.*.avg_frame_rate;
            ctx.sample_aspect_ratio = .{ .num = 1, .den = 1 };
            ctx.pix_fmt = codec.pix_fmt;
            ctx.bit_rate = 30_000 * KBPS_TO_BPS; // FIXME: configurable?
            ctx.gop_size = @intFromFloat(c.av_q2d(ctx.framerate) / 2);

            ctx.color_range = c.AVCOL_RANGE_MPEG;
            ctx.color_primaries = c.AVCOL_PRI_BT709;
            ctx.color_trc = c.AVCOL_TRC_BT709;
            ctx.colorspace = c.AVCOL_SPC_BT709;

            if (fmt.ptr().oformat.*.flags & c.AVFMT_GLOBALHEADER != 0) {
                ctx.flags |= c.AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            if (codec.hw) |hw| {
                var dev_ctx = try AvHwDeviceContext.init(hw.dev_type);
                errdefer dev_ctx.deinit();

                try setHwframeContext(p.?, dev_ctx, codec.pix_fmt);

                return .{ .inner = p, .hw = .{ .dev_ctx = dev_ctx } };
            }

            return .{ .inner = p };
        }

        pub fn ptr(self: AvCodecContext) *c.AVCodecContext {
            return self.inner.?;
        }

        fn setHwframeContext(codec_ctx: *c.AVCodecContext, device_ctx: AvHwDeviceContext, pix_fmt: c.AVPixelFormat) !void {
            var hw_frame_ref = try AvHwFrameRef.init(device_ctx);
            defer hw_frame_ref.deinit(); // FIXME: this might cause issues?

            var ctx: *c.AVHWFramesContext = @ptrCast(@alignCast(hw_frame_ref.ptr().data));
            ctx.format = pix_fmt;
            ctx.sw_format = c.AV_PIX_FMT_NV12;
            ctx.width = codec_ctx.width;
            ctx.height = codec_ctx.height;
            ctx.initial_pool_size = 20; // FIXME: magic number, why?

            _ = try err(c.av_hwframe_ctx_init(hw_frame_ref.ptr()));
            codec_ctx.hw_frames_ctx = c.av_buffer_ref(hw_frame_ref.ptr());
        }

        pub fn deinit(self: *AvCodecContext) void {
            // FIXME: do we need to call av_buffer_unref on codec.hw_frames_ctx?

            if (self.hw) |*hw| hw.dev_ctx.deinit();
            c.avcodec_free_context(&self.inner);
        }
    };
};

pub const dec = struct {
    const log = std.log.scoped(.decode);

    pub const AvCodecContext = struct {
        const Kind = enum { video, audio };
        const Options = struct { dev_type: ?c.AVHWDeviceType = null };

        inner: ?*c.AVCodecContext,
        stream: c_int,

        device: ?Device = null, //TODO: move Device to inside this struct

        // FIXME: why is this heap allocated?
        pub fn init(allocator: std.mem.Allocator, comptime kind: Kind, fmt_ctx: AvFormatContext, opt: Options) !*AvCodecContext {
            const self = try allocator.create(@This());

            if (kind == .video) blk: {
                const device_type = opt.dev_type orelse break :blk;

                self.initHardware(fmt_ctx, device_type) catch {
                    break :blk log.err("failed to set up hardware device, defaulting to software", .{});
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
                log.debug("search for hardware decoder", .{});

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
                    log.info("found {s} decoder for {s} ", .{ c.av_hwdevice_get_type_name(device_type), codec_ptr.?.name });

                    // FIXME: does this need to be deallocated????
                    _ = try err(c.av_hwdevice_ctx_create(&hw_device_ctx, device_type, null, null, 0));
                    ctx_ptr.?.hw_device_ctx = c.av_buffer_ref(hw_device_ctx);

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
            const self: *const @This() = @ptrCast(@alignCast(ctx.?.@"opaque"));

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
};

pub const AvPacket = struct {
    inner: ?*c.AVPacket,

    // FIXME: remove and then rename try_init to init
    pub fn init() AvPacket {
        const p: ?*c.AVPacket = c.av_packet_alloc();
        errdefer c.av_packet_free(p);

        return .{ .inner = p };
    }

    pub fn try_init() !AvPacket {
        const p: ?*c.AVPacket = c.av_packet_alloc();
        if (p == null) return error.ffmpeg_error;

        return .{ .inner = p };
    }

    pub fn deinit(self: *@This()) void {
        c.av_packet_free(&self.inner);
    }

    pub inline fn ptr(self: @This()) *c.AVPacket {
        return self.inner.?;
    }
};

pub const AvFrame = struct {
    inner: ?*c.AVFrame,

    pub fn init() !AvFrame {
        const p: ?*c.AVFrame = c.av_frame_alloc();
        if (p == null) return error.ffmpeg_error;

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
