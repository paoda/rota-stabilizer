const std = @import("std");
const c = @import("../lib.zig").c;
const libav = @import("libav.zig");

const errify = @import("platform.zig").errify;

const AvFormatContext = @import("libav.zig").AvFormatContext;
const AvCodecContext = @import("libav.zig").AvCodecContext;

// TODO: some universal thread sync primitive

pub const packet = struct {
    const AvPacket = libav.AvPacket;

    pub const Queue = struct { // FIXME: is there any point to rolling my own?
        list: std.fifo.LinearFifo(*c.AVPacket, .Dynamic),

        mutex: std.Thread.Mutex = .{},
        cond: std.Thread.Condition = .{},

        end_of_stream: std.atomic.Value(bool) = .init(false),

        const log = std.log.scoped(.packet_queue);

        pub fn init(allocator: std.mem.Allocator) Queue {
            return .{
                .list = std.fifo.LinearFifo(*c.AVPacket, .Dynamic).init(allocator),
            };
        }

        pub fn deinit(self: @This(), _: std.mem.Allocator) void {
            for (self.list.writableSlice(0)) |pkt| {
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

        pub fn pop(self: *@This()) ?*c.AVPacket {
            self.mutex.lock();
            defer self.mutex.unlock();

            while (self.list.readableLength() == 0) {
                if (self.end_of_stream.load(.monotonic)) return null;
                self.cond.wait(&self.mutex);
            }

            return self.list.readItem();
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
            try errify(c.SDL_ResumeAudioStreamDevice(stream));

            return .{
                .start_time = c.SDL_GetPerformanceCounter(),
                .sample_rate = @intCast(desired.freq),
                .channels = @intCast(desired.channels),
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

        pub fn seconds_passed(self: @This()) f64 {
            const bytes_per_sec: f64 = @floatFromInt(self.bytes_per_sample * self.sample_rate * self.channels);

            const queued: f64 = @floatFromInt(c.SDL_GetAudioStreamQueued(self.stream));
            const bytes_sent: f64 = @floatFromInt(self.bytes_sent.load(.monotonic));

            const pos = bytes_sent - queued;
            return pos / bytes_per_sec;
        }
    };

    pub fn decode(clock: *Clock, pkt_queue: *packet.Queue, decode_ctx: DecodeContext) !void {
        const log = std.log.scoped(.audio_decode);

        log.info("audio decode thread start", .{});
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

        var end_of_file = false;
        while (!decode_ctx.should_quit.load(.monotonic)) {
            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                switch (c.avcodec_receive_frame(audio_ctx, src_frame)) {
                    0 => { // got a frame
                        defer c.av_frame_unref(src_frame);

                        const bytes_per_sec = clock.sample_rate * clock.bytes_per_sample * clock.channels;
                        const max_len = 1 * bytes_per_sec;

                        _ = try libav.err(c.swr_convert_frame(swr, dst_frame, src_frame));

                        while (true) {
                            const queued_len = c.SDL_GetAudioStreamQueued(clock.stream);
                            if (queued_len < max_len) break;

                            std.Thread.sleep(5 * std.time.ns_per_ms);
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

            var pkt: ?*c.AVPacket = pkt_queue.pop() orelse {
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
        defer log.info("video decode thread end", .{});

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

            var pkt: ?*c.AVPacket = pkt_queue.pop() orelse {
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

    inline fn mask(self: @This(), idx: usize) usize {
        return idx & (self.slot.frame.len - 1);
    }
};
