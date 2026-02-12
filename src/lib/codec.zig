const std = @import("std");
const c = @import("../lib.zig").c;
const libav = @import("libav.zig");

const errify = @import("platform.zig").errify;

const LinearFifo = @import("fifo.zig").LinearFifo(*c.AVPacket, .dynamic);
const AvFormatContext = @import("libav.zig").AvFormatContext;
const AvCodecContext = @import("libav.zig").AvCodecContext;

// TODO: some universal thread sync primitive

pub const packet = struct {
    const AvPacket = libav.AvPacket;

    pub const Queue = struct { // FIXME: is there any point to rolling my own?
        list: LinearFifo,

        mutex: std.Thread.Mutex = .{},
        cond: std.Thread.Condition = .{},

        end_of_stream: std.atomic.Value(bool) = .init(false),

        const log = std.log.scoped(.packet_queue);

        pub fn init(allocator: std.mem.Allocator) Queue {
            return .{
                .list = LinearFifo.init(allocator),
            };
        }

        pub fn deinit(self: *@This(), _: std.mem.Allocator) void {
            while (self.list.readItem()) |pkt| {
                var p: ?*c.AVPacket = pkt;
                c.av_packet_free(&p);
            }

            self.list.deinit();
        }

        pub fn push(self: *@This(), pkt: *c.AVPacket) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            const copy: ?*c.AVPacket = c.av_packet_alloc();
            _ = try libav.err(c.av_packet_ref(copy, pkt));

            try self.list.writeItem(copy.?);
            self.cond.signal();
        }

        pub fn pop(self: *@This(), should_quit: *std.atomic.Value(bool)) ?*c.AVPacket {
            self.mutex.lock();
            defer self.mutex.unlock();

            while (self.list.readableLength() == 0) {
                if (self.end_of_stream.load(.monotonic)) return null;
                if (should_quit.load(.monotonic)) return null;
                self.cond.wait(&self.mutex);
            }

            return self.list.readItem();
        }

        /// Non-blocking pop - returns null immediately if queue is empty
        pub fn tryPop(self: *@This()) ?*c.AVPacket {
            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.list.readableLength() == 0) return null;
            return self.list.readItem();
        }

        /// Signal any threads waiting on this queue to wake up and check their quit flag
        pub fn quit(self: *@This()) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.cond.broadcast();
        }
    };

    pub const ReadOptions = struct {
        video_stream: c_int,
        audio_stream: c_int,
        should_quit: *std.atomic.Value(bool),
    };

    pub fn read(video_queue: *Queue, audio_queue: *Queue, fmt_ctx: AvFormatContext, opt: ReadOptions) !void {
        const log = std.log.scoped(.packet_read);

        log.info("packet read thread start", .{});
        defer log.info("packet read thread end", .{});

        var pkt = AvPacket.init();
        defer pkt.deinit();

        while (!opt.should_quit.load(.monotonic)) {
            switch (c.av_read_frame(fmt_ctx.ptr(), pkt.ptr())) {
                0...std.math.maxInt(c_int) => {
                    defer c.av_packet_unref(pkt.ptr());

                    if (pkt.ptr().stream_index == opt.video_stream) {
                        try video_queue.push(pkt.ptr());
                    } else if (pkt.ptr().stream_index == opt.audio_stream) {
                        try audio_queue.push(pkt.ptr());
                    }
                },
                c.AVERROR_EOF => {
                    audio_queue.end_of_stream.store(true, .monotonic);
                    video_queue.end_of_stream.store(true, .monotonic);
                    return;
                },
                else => |e| _ = try libav.err(e),
            }
        }
    }
};

pub const DecodeContext = struct {
    codec_ctx: *AvCodecContext,
    fmt_ctx: AvFormatContext,
    should_quit: *std.atomic.Value(bool),
};

pub const audio = struct {
    pub const Clock = struct {
        bytes_sent: std.atomic.Value(u64) = .init(0),
        start_time: u64,

        sample_rate: u16,
        channels: u8,
        bytes_per_sample: u32,

        hw_latency_secs: f64,

        /// Offset to align audio time with video stream time (set from first video PTS)
        /// This accounts for videos that don't start at PTS=0
        stream_start_offset: f64 = 0.0,

        is_muted: bool = true,

        stream: *c.SDL_AudioStream,

        const log = std.log.scoped(.audio);

        pub fn init(ctx: *const AvCodecContext) !@This() {
            var desired: c.SDL_AudioSpec = std.mem.zeroes(c.SDL_AudioSpec);
            desired.freq = ctx.inner.?.sample_rate;
            desired.format = c.SDL_AUDIO_F32;
            desired.channels = ctx.inner.?.ch_layout.nb_channels;

            const stream = try errify(c.SDL_OpenAudioDeviceStream(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired, null, null));
            errdefer c.SDL_DestroyAudioStream(stream);

            try errify(c.SDL_SetAudioStreamGain(stream, 0));

            var actual: c.SDL_AudioSpec = std.mem.zeroes(c.SDL_AudioSpec);
            var sample_frames: c_int = undefined;

            try errify(c.SDL_GetAudioDeviceFormat(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &actual, &sample_frames));
            const offset = @as(f64, @floatFromInt(sample_frames)) / @as(f64, @floatFromInt(actual.freq));

            log.debug("{d:.2}ms ({} frames) @ {}Hz", .{ offset * std.time.ms_per_s, sample_frames, actual.freq });

            return .{
                .start_time = c.SDL_GetPerformanceCounter(),
                .sample_rate = @intCast(desired.freq),
                .channels = @intCast(desired.channels),
                .hw_latency_secs = offset,
                .bytes_per_sample = @intCast(c.av_get_bytes_per_sample(c.AV_SAMPLE_FMT_FLT)),
                .stream = stream,
            };
        }

        pub fn mute(self: *@This()) !void {
            self.is_muted = true;
            try errify(c.SDL_SetAudioStreamGain(self.stream, 0.0));
        }

        pub fn unmute(self: *@This()) !void {
            self.is_muted = false;
            try errify(c.SDL_SetAudioStreamGain(self.stream, 1.0));
        }

        pub fn deinit(self: @This()) void {
            c.SDL_DestroyAudioStream(self.stream);
        }

        /// Start audio playback (called when video is ready for initial sync)
        /// first_video_pts: The PTS of the first video frame, to sync audio time to video time
        pub fn start(self: *@This(), first_video_pts: f64) !void {
            if (c.SDL_AudioStreamDevicePaused(self.stream)) {
                self.stream_start_offset = first_video_pts;
                try errify(c.SDL_ResumeAudioStreamDevice(self.stream));
            }
        }

        pub fn isPaused(self: *const @This()) bool {
            return c.SDL_AudioStreamDevicePaused(self.stream);
        }

        pub fn seconds_passed(self: @This()) f64 {
            const bytes_per_sec: f64 = @floatFromInt(self.bytes_per_sample * self.sample_rate * self.channels);

            const queued: f64 = @floatFromInt(c.SDL_GetAudioStreamQueued(self.stream));
            const bytes_sent: f64 = @floatFromInt(self.bytes_sent.load(.monotonic));
            const hw_latency_bytes = self.hw_latency_secs * bytes_per_sec;

            const pos = bytes_sent - queued - hw_latency_bytes;
            // Add stream_start_offset to align with video PTS
            return (pos / bytes_per_sec) + self.stream_start_offset;
        }

        /// Debug: Get detailed clock state for A/V sync debugging
        pub fn debug_state(self: @This()) struct { bytes_sent: f64, queued: f64, hw_latency_bytes: f64, bytes_per_sec: f64, position_secs: f64, stream_start_offset: f64 } {
            const bytes_per_sec: f64 = @floatFromInt(self.bytes_per_sample * self.sample_rate * self.channels);
            const queued: f64 = @floatFromInt(c.SDL_GetAudioStreamQueued(self.stream));
            const bytes_sent: f64 = @floatFromInt(self.bytes_sent.load(.monotonic));
            const hw_latency_bytes = self.hw_latency_secs * bytes_per_sec;
            const pos = bytes_sent - queued - hw_latency_bytes;

            return .{
                .bytes_sent = bytes_sent,
                .queued = queued,
                .hw_latency_bytes = hw_latency_bytes,
                .bytes_per_sec = bytes_per_sec,
                .position_secs = (pos / bytes_per_sec) + self.stream_start_offset,
                .stream_start_offset = self.stream_start_offset,
            };
        }
    };

    // A/V Sync Debug: Set to true to enable audio-side logging

    pub fn decode(clock: *Clock, pkt_queue: *packet.Queue, decode_ctx: DecodeContext) !void {
        const log = std.log.scoped(.audio_decode);

        log.info("audio decode thread start", .{});
        log.debug("sample_rate: {} channels: {} bytes_per_sample: {}", .{ clock.sample_rate, clock.channels, clock.bytes_per_sample });
        defer log.info("audio decode thread end", .{});

        var maybe_src_frame: ?*c.AVFrame = c.av_frame_alloc();
        defer c.av_frame_free(&maybe_src_frame);

        var maybe_dst_frame: ?*c.AVFrame = c.av_frame_alloc();
        defer c.av_frame_free(&maybe_dst_frame);

        const src_frame = maybe_src_frame orelse return error.out_of_memory;
        const dst_frame = maybe_dst_frame orelse return error.out_of_memory;

        const audio_ctx = decode_ctx.codec_ctx.inner.?;

        _ = try libav.err(c.av_channel_layout_copy(&dst_frame.ch_layout, &audio_ctx.ch_layout));
        dst_frame.sample_rate = audio_ctx.sample_rate;
        dst_frame.format = c.AV_SAMPLE_FMT_FLT;

        var maybe_swr = c.swr_alloc();
        defer c.swr_free(&maybe_swr);

        const swr = maybe_swr orelse return error.out_of_memory;

        // debug
        var frame_count: u64 = 0;

        var end_of_file = false;
        while (!decode_ctx.should_quit.load(.monotonic)) {
            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                switch (c.avcodec_receive_frame(audio_ctx, src_frame)) {
                    0 => { // got a frame
                        defer c.av_frame_unref(src_frame);

                        frame_count += 1;

                        const bytes_per_sec = clock.sample_rate * clock.bytes_per_sample * clock.channels;
                        // Reduced from 1 second to 150ms for lower latency
                        const max_queue_secs = 0.150;
                        const max_len: c_int = @intFromFloat(@as(f64, @floatFromInt(bytes_per_sec)) * max_queue_secs);

                        _ = try libav.err(c.swr_convert_frame(swr, dst_frame, src_frame));

                        // Track stalls when audio buffer is full
                        var stall_count: u32 = 0;
                        while (true) {
                            if (c.SDL_GetAudioStreamQueued(clock.stream) < max_len) break;
                            defer stall_count += 1;

                            std.Thread.sleep(5 * std.time.ns_per_ms);
                        }

                        if (stall_count > 0) {
                            log.debug("STALL frame #{} waited {}x5ms", .{ frame_count, stall_count });
                        }

                        const len = c.av_samples_get_buffer_size(null, dst_frame.ch_layout.nb_channels, dst_frame.nb_samples, dst_frame.format, 0);
                        _ = c.SDL_PutAudioStreamData(clock.stream, dst_frame.data[0], len);
                        _ = clock.bytes_sent.fetchAdd(@intCast(len), .monotonic);
                    },
                    c.AVERROR(c.EAGAIN) => break :recv_loop,
                    c.AVERROR_EOF => return,
                    else => |e| _ = try libav.err(e),
                }
            }

            var pkt: ?*c.AVPacket = pkt_queue.pop(decode_ctx.should_quit) orelse {
                if (decode_ctx.should_quit.load(.monotonic)) return;
                const flush_ret = c.avcodec_send_packet(audio_ctx, null);
                if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) _ = try libav.err(flush_ret);
                end_of_file = true;
                continue;
            };
            defer c.av_packet_free(&pkt);

            const send_ret = c.avcodec_send_packet(audio_ctx, pkt);
            if (send_ret == c.AVERROR_EOF) return;
            if (send_ret == c.AVERROR(c.EAGAIN)) continue;
            if (send_ret < 0) _ = try libav.err(send_ret);
        }
    }
};

pub const video = struct {
    const AvFrame = libav.AvFrame;

    pub fn decode(frame_queue: *FrameQueue, pkt_queue: *packet.Queue, decode_ctx: DecodeContext) !void {
        const log = std.log.scoped(.video_decode);

        log.info("video decode thread start", .{});
        defer {
            frame_queue.end_of_stream.store(true, .monotonic);
            log.info("video decode thread end", .{});
        }

        var src_frame: AvFrame = try .init();
        defer src_frame.deinit();

        var dst_frame: AvFrame = try .init();
        defer dst_frame.deinit();

        const video_ctx = decode_ctx.codec_ctx.inner.?;

        try dst_frame.setup(video_ctx.width, video_ctx.height, c.AV_PIX_FMT_NV12);

        const maybe_sws = c.sws_getContext(
            video_ctx.width,
            video_ctx.height,
            video_ctx.pix_fmt,
            video_ctx.width,
            video_ctx.height,
            c.AV_PIX_FMT_NV12,
            c.SWS_FAST_BILINEAR,
            null,
            null,
            null,
        );
        defer c.sws_freeContext(maybe_sws);

        const sws = maybe_sws orelse return error.out_of_memory;

        // TODO: using a fn ptr I think we can deduplicate this massive loop
        var end_of_file = false;
        while (!decode_ctx.should_quit.load(.monotonic)) {
            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                switch (c.avcodec_receive_frame(video_ctx, src_frame.ptr())) {
                    0 => {
                        defer c.av_frame_unref(dst_frame.ptr());

                        try convert(decode_ctx.codec_ctx, src_frame.ptr(), dst_frame.ptr(), sws);
                        try frame_queue.push(dst_frame.ptr());
                    },
                    c.AVERROR(c.EAGAIN) => break :recv_loop,
                    c.AVERROR_EOF => return,
                    else => |e| _ = try libav.err(e),
                }
            }

            var pkt: ?*c.AVPacket = pkt_queue.pop(decode_ctx.should_quit) orelse {
                if (decode_ctx.should_quit.load(.monotonic)) return;
                const flush_ret = c.avcodec_send_packet(video_ctx, null);
                if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) _ = try libav.err(flush_ret);
                end_of_file = true;
                continue;
            };
            defer c.av_packet_free(&pkt);

            const send_ret = c.avcodec_send_packet(video_ctx, pkt);
            if (send_ret == c.AVERROR_EOF) return;
            if (send_ret == c.AVERROR(c.EAGAIN)) continue;
            if (send_ret < 0) _ = try libav.err(send_ret);
        }
    }

    fn convert(codec_ctx: *const AvCodecContext, src_frame: *c.AVFrame, dst_frame: *c.AVFrame, sws: *c.SwsContext) !void {
        @setRuntimeSafety(false);

        if (codec_ctx.device) |dev| {
            std.debug.assert(src_frame.format == dev.pix_fmt);
            _ = try libav.err(c.av_hwframe_transfer_data(dst_frame, src_frame, 0));
        } else {
            _ = try libav.err(c.sws_scale_frame(sws, dst_frame, src_frame));
        }

        // timing
        dst_frame.pts = src_frame.pts;
        dst_frame.pkt_dts = src_frame.pkt_dts;
        dst_frame.best_effort_timestamp = src_frame.best_effort_timestamp;

        dst_frame.colorspace = src_frame.colorspace;
        dst_frame.color_range = src_frame.color_range;
        dst_frame.color_primaries = src_frame.color_primaries;
        dst_frame.color_trc = src_frame.color_trc;
    }
};

pub const FrameQueue = struct {
    slot: Slot,
    read_idx: usize,
    write_idx: usize,
    end_of_stream: std.atomic.Value(bool) = .init(false),

    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},

    const Slot = struct { frame: []c.AVFrame, state: []State };
    const State = enum { empty, in_use, ready_to_reuse };
    const Error = error{ ffmpeg_error, invalid_size } || std.mem.Allocator.Error;

    pub fn init(allocator: std.mem.Allocator, count: usize) Error!FrameQueue {
        if (!std.math.isPowerOfTwo(count)) return error.invalid_size;

        const frames = try allocator.alloc(c.AVFrame, count);
        const states = try allocator.alloc(State, count);

        for (frames, states) |*frame, *state| {
            frame.* = std.mem.zeroes(c.AVFrame);
            c.av_frame_unref(frame);

            state.* = .empty;
        }

        return .{
            .slot = .{ .frame = frames, .state = states },
            .read_idx = 0,
            .write_idx = 0,
        };
    }

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        for (self.slot.frame) |*frame| c.av_frame_unref(frame);

        allocator.free(self.slot.frame);
        allocator.free(self.slot.state);
    }

    pub fn push(self: *@This(), new_frame: *const c.AVFrame) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const idx = self.mask(self.write_idx);

        while (self.slot.state[idx] != .empty) self.cond.wait(&self.mutex);
        defer self.cond.signal();
        defer self.write_idx += 1;

        const frame = &self.slot.frame[idx];
        self.slot.state[idx] = .in_use;

        _ = try libav.err(c.av_frame_ref(frame, new_frame));
    }

    pub fn pop(self: *@This()) ?*c.AVFrame {
        self.mutex.lock();
        defer self.mutex.unlock();

        const idx = self.mask(self.read_idx);

        if (self.slot.state[idx] != .in_use) return null;
        defer self.cond.broadcast();
        defer self.read_idx += 1;

        const frame = &self.slot.frame[idx];
        self.slot.state[idx] = .ready_to_reuse;

        return frame;
    }

    pub fn recycle(self: *@This(), used_frame: *c.AVFrame) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.slot.frame, self.slot.state) |*frame, *state| {
            if (frame != used_frame) continue;
            std.debug.assert(state.* == .ready_to_reuse);

            state.* = .empty;
            c.av_frame_unref(frame);
            self.cond.signal();
            return;
        }

        std.debug.panic("attempted to recycle a frame that was not in the queue", .{});
    }

    pub fn isEmpty(self: *@This()) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        const idx = self.mask(self.read_idx);
        return self.slot.state[idx] != .in_use;
    }

    inline fn mask(self: @This(), idx: usize) usize {
        return idx & (self.slot.frame.len - 1);
    }
};

pub const Encoder = struct {
    width: u32,
    height: u32,

    hw_device_ctx: *c.AVBufferRef,
    av_codec_ctx: *c.AVCodecContext,
    sws_ctx: *c.SwsContext,

    fmt_ctx: *c.AVFormatContext,

    audio_stream: *c.AVStream,
    video_stream: *c.AVStream,

    /// for encoding video
    _pkt: *c.AVPacket,

    _sw_frame: *c.AVFrame,
    _hw_frame: *c.AVFrame,

    const log = std.log.scoped(.encode);

    pub const Options = struct {
        width: u32,
        height: u32,
        fps: c.AVRational,
        input_fmt_ctx: *c.AVFormatContext,
        input_video_stream: c_int,
        input_audio_stream: c_int,
    };

    pub fn init(opt: Options) !Encoder {
        return initHardware(opt);
    }

    pub fn initHardware(opt: Options) !Encoder {
        const path = "output.mp4";

        const fmt_ctx = blk: {
            var ptr: ?*c.AVFormatContext = null;
            _ = try libav.err(c.avformat_alloc_output_context2(&ptr, null, null, path));
            errdefer c.avformat_free_context(ptr);

            break :blk ptr.?;
        };

        const hw_device_ctx = blk: {
            var ptr: ?*c.AVBufferRef = null;
            _ = try libav.err(c.av_hwdevice_ctx_create(&ptr, c.AV_HWDEVICE_TYPE_VAAPI, null, null, 0));
            errdefer c.av_buffer_unref(&ptr);

            break :blk ptr.?;
        };

        // FIXME: figure out how to list all hardware encoders
        const codec = blk: {
            // TODO: can we free this?
            const ptr: ?*const c.AVCodec = c.avcodec_find_encoder_by_name("h264_vaapi");
            break :blk ptr.?;
        };

        const input_audio = opt.input_fmt_ctx.streams[@intCast(opt.input_audio_stream)];
        const input_video = opt.input_fmt_ctx.streams[@intCast(opt.input_video_stream)];

        const av_codec_ctx = blk: {
            var ptr: ?*c.AVCodecContext = c.avcodec_alloc_context3(codec);
            errdefer c.avcodec_free_context(&ptr);

            const ctx = ptr.?;
            // TODO: AI added some extra params idk if i need 'em

            ctx.width = @intCast(opt.width);
            ctx.height = @intCast(opt.height);
            ctx.time_base = input_video.*.time_base;
            ctx.framerate = opt.fps;
            ctx.sample_aspect_ratio = .{ .num = 1, .den = 1 };
            ctx.pix_fmt = c.AV_PIX_FMT_VAAPI;

            // Set global header flag BEFORE opening codec if muxer needs it
            if (fmt_ctx.oformat.*.flags & c.AVFMT_GLOBALHEADER != 0) {
                ctx.flags |= c.AV_CODEC_FLAG_GLOBAL_HEADER;
            }

            break :blk ctx;
        };

        try setHwframeContext(opt, av_codec_ctx, hw_device_ctx);

        _ = try libav.err(c.avcodec_open2(av_codec_ctx, codec, null));

        const audio_stream = blk: {
            const ptr: ?*c.AVStream = c.avformat_new_stream(fmt_ctx, null);
            const stream = ptr.?;

            _ = try libav.err(c.avcodec_parameters_copy(stream.codecpar, input_audio.*.codecpar));
            // stream.time_base = input_audio.*.time_base;
            stream.codecpar.*.codec_tag = 0;

            break :blk stream;
        };

        const video_stream = blk: {
            const ptr: ?*c.AVStream = c.avformat_new_stream(fmt_ctx, null);
            const stream = ptr.?;

            _ = try libav.err(c.avcodec_parameters_from_context(stream.codecpar, av_codec_ctx));
            stream.time_base = av_codec_ctx.time_base;

            break :blk stream;
        };

        c.av_dump_format(fmt_ctx, 0, path, 1);

        _ = try libav.err(c.avio_open(&fmt_ctx.pb, path, c.AVIO_FLAG_WRITE));

        _ = try libav.err(c.avformat_write_header(fmt_ctx, null));

        return .{
            .width = opt.width,
            .height = opt.height,
            .hw_device_ctx = hw_device_ctx,
            .av_codec_ctx = av_codec_ctx,
            .fmt_ctx = fmt_ctx,

            .audio_stream = audio_stream,
            .video_stream = video_stream,

            ._pkt = blk: {
                const ptr: ?*c.AVPacket = c.av_packet_alloc();
                break :blk ptr.?;
            },
            ._sw_frame = blk: {
                const ptr: ?*c.AVFrame = c.av_frame_alloc();
                const frame = ptr.?;

                frame.format = c.AV_PIX_FMT_NV12;
                frame.width = @intCast(opt.width);
                frame.height = @intCast(opt.height);
                _ = try libav.err(c.av_frame_get_buffer(frame, 0));

                break :blk frame;
            },
            ._hw_frame = blk: {
                const ptr: ?*c.AVFrame = c.av_frame_alloc();
                break :blk ptr.?;
            },
            .sws_ctx = blk: {
                const ptr: ?*c.SwsContext = c.sws_getContext(
                    @intCast(opt.width),
                    @intCast(opt.height),
                    c.AV_PIX_FMT_RGB24,
                    @intCast(opt.width),
                    @intCast(opt.height),
                    c.AV_PIX_FMT_NV12,
                    c.SWS_POINT,
                    null,
                    null,
                    null,
                );

                break :blk ptr.?;
            },
        };
    }

    fn setHwframeContext(opt: Options, ctx: *c.AVCodecContext, hw_device_ctx: *c.AVBufferRef) !void {
        const hw_frames_ref = blk: {
            const ptr: ?*c.AVBufferRef = c.av_hwframe_ctx_alloc(hw_device_ctx);
            // TODO: errdefer free

            break :blk ptr.?;
        };
        // TODO: re-enable this free
        // defer c.av_buffer_unref(&hw_frames_ref);

        var frames_ctx: *c.AVHWFramesContext = @ptrCast(@alignCast(hw_frames_ref.data));
        frames_ctx.format = c.AV_PIX_FMT_VAAPI;
        frames_ctx.sw_format = c.AV_PIX_FMT_NV12;
        frames_ctx.width = @intCast(opt.width);
        frames_ctx.height = @intCast(opt.height);
        frames_ctx.initial_pool_size = 20; // FIXME: why?

        _ = try libav.err(c.av_hwframe_ctx_init(hw_frames_ref));
        ctx.hw_frames_ctx = c.av_buffer_ref(hw_frames_ref);
    }

    pub fn encodeRgbFrame(self: *Encoder, buf: []const u8, frame_pts: i64) !void {
        const RGB_BPP = 3;
        _ = try libav.err(c.av_frame_make_writable(self._sw_frame));

        // OpenGL gives us bottom-to-top RGB, swscale expects top-to-bottom
        // Use negative source stride to flip during conversion
        const stride = self.width * RGB_BPP;

        // Point to last row (first row of OpenGL image) with negative stride to read bottom-up
        const last_row_ptr: [*]const u8 = @ptrCast(buf.ptr + (@as(usize, self.height - 1) * @as(usize, stride)));
        const src_data: [4][*]const u8 = .{ last_row_ptr, undefined, undefined, undefined };
        const src_stride: [4]c_int = .{ -@as(c_int, @intCast(stride)), 0, 0, 0 };

        _ = c.sws_scale(
            self.sws_ctx,
            @ptrCast(&src_data),
            @ptrCast(&src_stride),
            0,
            @intCast(self.height),
            @ptrCast(&self._sw_frame.data),
            &self._sw_frame.linesize,
        );

        // Use PTS directly since we're using the same time_base as input
        self._sw_frame.pts = frame_pts;

        c.av_frame_unref(self._hw_frame);
        _ = try libav.err(c.av_hwframe_get_buffer(self.av_codec_ctx.hw_frames_ctx, self._hw_frame, 0));
        _ = try libav.err(c.av_hwframe_transfer_data(self._hw_frame, self._sw_frame, 0));
        _ = try libav.err(c.av_frame_copy_props(self._hw_frame, self._sw_frame));

        try self.writeVideoFrame(self._hw_frame);
    }

    pub fn writeAudioPacket(self: *Encoder, in: *c.AVStream, pkt: *c.AVPacket) !void {
        const UNKN_POS = -1;

        c.av_packet_rescale_ts(pkt, in.time_base, self.audio_stream.time_base);
        pkt.pos = UNKN_POS;
        pkt.stream_index = self.audio_stream.index;

        _ = try libav.err(c.av_interleaved_write_frame(self.fmt_ctx, pkt));
    }

    fn writeVideoFrame(self: *Encoder, frame: ?*c.AVFrame) !void {
        _ = try libav.err(c.avcodec_send_frame(self.av_codec_ctx, frame));

        while (true) {
            const ret = c.avcodec_receive_packet(self.av_codec_ctx, self._pkt);
            if (ret == c.AVERROR(c.EAGAIN) or ret == c.AVERROR_EOF) break;
            _ = try libav.err(ret);

            c.av_packet_rescale_ts(self._pkt, self.av_codec_ctx.time_base, self.video_stream.time_base);
            self._pkt.stream_index = self.video_stream.index;

            _ = try libav.err(c.av_interleaved_write_frame(self.fmt_ctx, self._pkt));
        }
    }

    pub fn deinit(self: *Encoder) void {
        self.writeVideoFrame(null) catch @panic("failed to flush encoder");
        _ = c.av_write_trailer(self.fmt_ctx);

        {
            var tmp: ?*c.AVFrame = self._sw_frame;
            c.av_frame_free(&tmp);

            tmp = self._hw_frame;
            c.av_frame_free(&tmp);
        }
        {
            var tmp: ?*c.AVPacket = self._pkt;
            c.av_packet_free(&tmp);
        }
        {
            var tmp: ?*c.AVCodecContext = self.av_codec_ctx;
            c.avcodec_free_context(&tmp);
        }
        {
            var tmp: ?*c.AVBufferRef = self.hw_device_ctx;
            c.av_buffer_unref(&tmp);
        }

        if (self.fmt_ctx.oformat.*.flags & c.AVFMT_NOFILE == 0) {
            _ = c.avio_closep(&self.fmt_ctx.pb);
        }

        // var tmp: ?*c.AVFormatContext = self.fmt_ctx;
        c.avformat_free_context(self.fmt_ctx);

        self.* = undefined;
    }
};
