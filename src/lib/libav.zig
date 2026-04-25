//! FFmpeg bindings
const std = @import("std");
const tracy = @import("tracy");
const c = @import("../lib.zig").c;

const getPixelFormatName = @import("../lib.zig").getPixelFormatName;
const KBPS_TO_BPS = 1000;

const Resolution = @import("../lib.zig").Resolution;

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
            resolution: Resolution,
            bit_rate: i64,

            input: Input,

            const Input = struct {
                fmt_ctx: dec.AvFormatContext,
                video_ctx: *const dec.AvCodecContext,
            };
        };

        pub fn init(codec: AvCodec, fmt: AvFormatContext, opt: Options) !AvCodecContext {
            var p: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec.ptr());
            errdefer c.avcodec_free_context(&p);

            const ctx = p.?;

            const inpt_vid = opt.input.fmt_ctx.ptr().streams[@intCast(opt.input.video_ctx.stream)];

            ctx.width = opt.resolution.width;
            ctx.height = opt.resolution.height;
            ctx.time_base = inpt_vid.*.time_base;
            ctx.framerate = inpt_vid.*.avg_frame_rate;
            ctx.sample_aspect_ratio = .{ .num = 1, .den = 1 };
            ctx.pix_fmt = codec.pix_fmt;
            ctx.bit_rate = opt.bit_rate * KBPS_TO_BPS;
            ctx.gop_size = @intFromFloat(c.av_q2d(ctx.framerate) / 2);

            ctx.thread_count = 0;
            ctx.thread_type = c.FF_THREAD_FRAME | c.FF_THREAD_SLICE;

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
    pub const AvCodecContext = struct {
        const Kind = enum { video, audio };
        const Options = struct { dev_type: c.AVHWDeviceType = c.AV_HWDEVICE_TYPE_NONE };

        inner: ?*c.AVCodecContext,
        stream: c_int,

        device: ?Device = null, //TODO: move Device to inside this struct

        const log = std.log.scoped(.dec_codec);

        // FIXME: why is this heap allocated?
        pub fn init(allocator: std.mem.Allocator, comptime kind: Kind, fmt_ctx: AvFormatContext, opt: Options) !*AvCodecContext {
            const self = try allocator.create(@This());

            if (kind == .video) blk: {
                if (opt.dev_type == c.AV_HWDEVICE_TYPE_NONE) break :blk;

                self.initHardware(fmt_ctx, opt.dev_type) catch {
                    break :blk log.err("failed to set up hardware device, defaulting to software", .{});
                };

                return self;
            }

            try self.initSoftware(kind, fmt_ctx);
            return self;
        }

        // TODO: dump audio information as well?
        pub fn dump(self: @This()) void {
            const inner = self.inner.?;

            const width: u32 = @intCast(inner.width);
            const height: u32 = @intCast(inner.height);

            const gcd = std.math.gcd(width, height);
            const aspect_ratio = @as(f32, @floatFromInt(inner.width)) / @as(f32, @floatFromInt(inner.height));

            log.debug("bit_rate: {d:.2}kbps", .{@as(f32, @floatFromInt(inner.bit_rate)) / KBPS_TO_BPS});
            log.debug("resolution: {}x{}", .{ inner.width, inner.height });
            log.debug("pix_fmt: {s}", .{getPixelFormatName(inner.pix_fmt)});
            log.debug("sw_pix_fmt: {s}", .{getPixelFormatName(inner.sw_pix_fmt)});
            log.debug("aspect ratio: {}:{} | {d:.3}", .{ width / gcd, height / gcd, aspect_ratio });
            log.debug("colour primaries: {s}", .{c.av_color_primaries_name(inner.color_primaries)});
            log.debug("colour transfer: {s}", .{c.av_color_transfer_name(inner.color_trc)});
            log.debug("colour space: {s}", .{c.av_color_space_name(inner.colorspace)});
            log.debug("colour range: {s}", .{c.av_color_range_name(inner.color_range)});
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

            ctx_ptr.?.thread_count = 0;
            ctx_ptr.?.thread_type = c.FF_THREAD_FRAME | c.FF_THREAD_SLICE;

            _ = try err(c.avcodec_parameters_to_context(ctx_ptr, fmt_ctx.ptr().streams[@intCast(stream)].*.codecpar));
            _ = try err(c.avcodec_open2(ctx_ptr, codec_ptr, null));

            self.* = .{ .inner = ctx_ptr, .stream = stream };
        }

        fn initHardware(self: *@This(), fmt_ctx: AvFormatContext, device_type: c.AVHWDeviceType) !void {
            var codec_ptr: ?*const c.AVCodec = null;
            const stream = try err(c.av_find_best_stream(fmt_ctx.ptr(), c.AVMEDIA_TYPE_VIDEO, -1, -1, &codec_ptr, 0));

            var ctx_ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec_ptr);
            errdefer c.avcodec_free_context(&ctx_ptr);

            ctx_ptr.?.thread_count = 0;
            ctx_ptr.?.thread_type = c.FF_THREAD_FRAME | c.FF_THREAD_SLICE;

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

                    _ = try err(c.av_hwdevice_ctx_create(&hw_device_ctx, device_type, null, null, 0));
                    errdefer c.av_buffer_unref(&hw_device_ctx);

                    ctx_ptr.?.hw_device_ctx = c.av_buffer_ref(hw_device_ctx);

                    self.device = .{ .ctx = hw_device_ctx, .pix_fmt = hw_pix_fmt };

                    ctx_ptr.?.@"opaque" = &self.device.?;
                    ctx_ptr.?.get_format = Device.getFormat;

                    break;
                }
            }

            _ = try err(c.avcodec_open2(ctx_ptr, codec_ptr, null));

            self.inner = ctx_ptr;
            self.stream = stream;
        }
    };

    const Device = struct {
        ctx: ?*c.AVBufferRef,
        pix_fmt: c.AVPixelFormat,

        const log = std.log.scoped(.dec_device);

        fn deinit(self: *@This()) void {
            c.av_buffer_unref(&self.ctx);
        }

        fn getFormat(ctx: ?*c.AVCodecContext, formats: [*c]const c.AVPixelFormat) callconv(.c) c.AVPixelFormat {
            const self: *const @This() = @ptrCast(@alignCast(ctx.?.@"opaque"));
            const fmts = std.mem.sliceTo(formats, c.AV_PIX_FMT_NONE);

            const sw = fmts[fmts.len - 1];
            std.debug.assert(c.av_pix_fmt_desc_get(sw).*.flags & c.AV_PIX_FMT_FLAG_HWACCEL == 0);

            for (fmts) |fmt| {
                if (fmt == self.pix_fmt) {
                    log.debug("match for {s} found", .{c.av_get_pix_fmt_name(fmt)});
                    return self.pix_fmt;
                }
            }

            log.err("failed to match hw pix fmt. fallback to {s}", .{getPixelFormatName(sw)});
            return sw;
        }
    };

    pub const AvFormatContext = struct {
        inner: ?*c.AVFormatContext,

        const log = std.log.scoped(.dec_fmt);

        pub fn init(path: []const u8) !AvFormatContext {
            var p: ?*c.AVFormatContext = c.avformat_alloc_context();
            errdefer c.avformat_close_input(&p);

            _ = try err(c.avformat_open_input(&p, path.ptr, null, null));
            _ = try err(c.avformat_find_stream_info(p, null));

            return .{ .inner = p };
        }

        pub fn dump(self: @This()) void {
            const inner = self.inner.?;

            // bit_rate exists here too

            log.debug("format: {s} ", .{inner.iformat.*.long_name});
            log.debug("stream count: {}", .{inner.nb_streams});
            log.debug("bit_rate: {d:.2}kbps", .{@as(f32, @floatFromInt(inner.bit_rate)) / KBPS_TO_BPS});
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

    pub fn deinit(self: *@This()) void {
        c.av_frame_free(&self.inner);
    }

    pub fn setYData(self: *AvFrame, buf: []const u8) !void {
        const frame = self.ptr();

        const stride = alignUp(frame.width, 32);
        const len: usize = @intCast(stride * frame.height);
        std.debug.assert(buf.len == len);

        frame.linesize[0] = stride;
        frame.data[0] = @ptrCast(@constCast(buf));
        frame.buf[0] = c.av_buffer_create(frame.data[0], len, &bufferDummyFree, null, 0);

        if (frame.buf[0] == null) return error.ffmpeg_error;
    }

    pub fn setUvData(self: *AvFrame, buf: []const u8) !void {
        const frame = self.ptr();

        const stride = alignUp(frame.width, 32);
        const len: usize = @intCast(stride * @divTrunc(frame.height, 2));
        std.debug.assert(buf.len == len);

        frame.linesize[1] = stride;
        frame.data[1] = @ptrCast(@constCast(buf));
        frame.buf[1] = c.av_buffer_create(frame.data[0], len, &bufferDummyFree, null, 0);

        if (frame.buf[1] == null) return error.ffmpeg_error;
    }

    fn bufferDummyFree(_: ?*anyopaque, _: [*c]u8) callconv(.c) void {}

    pub fn ptr(self: @This()) *c.AVFrame {
        return self.inner.?;
    }

    pub fn alignUp(value: c_int, alignment: c_int) c_int {
        return (value + alignment - 1) & ~(alignment - 1);
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
