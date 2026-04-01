const std = @import("std");
const libav = @import("libav.zig");
const ztracy = @import("ztracy");
const c = @import("../lib.zig").c;

const errify = @import("platform.zig").errify;

const LinearFifo = @import("fifo.zig").LinearFifo(*c.AVPacket, .dynamic);

const enc = @import("libav.zig").enc;
const dec = @import("libav.zig").dec;

const AvPacket = @import("libav.zig").AvPacket;
const AvFrame = @import("libav.zig").AvFrame;

// TODO: some universal thread sync primitive

pub const packet = struct {
    pub const Queue = struct { // FIXME: is there any point to rolling my own?
        list: LinearFifo,

        mutex: std.Thread.Mutex = .{},
        cond: std.Thread.Condition = .{},

        end_of_stream: std.atomic.Value(bool) = .init(false),
        should_quit: *const std.atomic.Value(bool),

        const log = std.log.scoped(.packet_queue);

        // INVARIANT: this is an SPSC Queue

        pub fn init(allocator: std.mem.Allocator, should_quit: *const std.atomic.Value(bool)) Queue {
            return .{
                .list = LinearFifo.init(allocator),
                .should_quit = should_quit,
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
            const zone = ztracy.ZoneN(@src(), "PacketQueue.push");
            defer zone.End();

            var copy: ?*c.AVPacket = c.av_packet_alloc() orelse return error.out_of_memory;
            errdefer c.av_packet_free(&copy);

            _ = try libav.err(c.av_packet_ref(copy, pkt));

            {
                self.mutex.lock();
                defer self.mutex.unlock();

                try self.list.writeItem(copy.?);
            }

            self.cond.signal();
        }

        pub fn pop(self: *@This()) ?*c.AVPacket {
            const zone = ztracy.ZoneN(@src(), "PacketQueue.pop");
            defer zone.End();

            self.mutex.lock();
            defer self.mutex.unlock();

            {
                const z = ztracy.ZoneNC(@src(), "wait for packet", 0x3b3b3b);
                defer z.End();

                while (self.list.readableLength() == 0) {
                    if (self.end_of_stream.load(.monotonic)) return null;
                    if (self.should_quit.load(.monotonic)) return null;

                    self.cond.wait(&self.mutex);
                }
            }

            return self.list.readItem();
        }

        /// Non-blocking pop - returns null immediately if queue is empty
        pub fn tryPop(self: *@This()) ?*c.AVPacket {
            const zone = ztracy.ZoneN(@src(), "PacketQueue.tryPop");
            defer zone.End();

            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.list.readableLength() == 0) return null;
            return self.list.readItem();
        }

        pub fn complete(self: *@This()) void {
            self.end_of_stream.store(true, .monotonic);

            self.mutex.lock();
            defer self.mutex.unlock();

            self.cond.broadcast();
        }

        pub fn interrupt(self: *@This()) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.cond.broadcast();
        }
    };

    pub fn read(decode: *Decoder) !void {
        // const log = std.log.scoped(.packet_read);
        ztracy.SetThreadName("packet read");

        var pkt = AvPacket.init();
        defer pkt.deinit();

        const audio_queue = &decode.queue.pkt.audio;
        const video_queue = &decode.queue.pkt.video;
        const fmt_ctx = &decode.fmt_ctx;

        while (!decode.should_quit.load(.monotonic)) {
            const zone = ztracy.ZoneN(@src(), "read loop");
            defer zone.End();

            switch (c.av_read_frame(fmt_ctx.ptr(), pkt.ptr())) {
                0...std.math.maxInt(c_int) => {
                    defer c.av_packet_unref(pkt.ptr());

                    const z = ztracy.ZoneN(@src(), "read frame");
                    defer z.End();

                    if (pkt.ptr().stream_index == decode.video_ctx.stream) {
                        try video_queue.push(pkt.ptr());
                    } else if (pkt.ptr().stream_index == decode.audio_ctx.stream) {
                        try audio_queue.push(pkt.ptr());
                    }
                },
                c.AVERROR_EOF => {
                    audio_queue.complete();
                    video_queue.complete();
                    return;
                },
                else => |e| _ = try libav.err(e),
            }
        }
    }
};

pub const audio = struct {
    pub const Clock = struct {
        bytes_sent: std.atomic.Value(u64) = .init(0),
        start_time: u64,

        sample_rate: u16,
        channels: u8,
        bytes_per_sample: u32,

        hw_latency_secs: f64,

        last_hw_pos: f64 = 0.0,
        last_hw_update_time_ns: u64 = 0,

        ema: MonoExpoMovingAvg = .{},

        /// Offset to align audio time with video stream time (set from first video PTS)
        /// This accounts for videos that don't start at PTS=0
        stream_start_offset: f64 = 0.0,

        is_muted: bool = true,

        stream: *c.SDL_AudioStream,

        const log = std.log.scoped(.audio);

        const MonoExpoMovingAvg = struct {
            last_called: u64 = 0,

            last_returned: f64 = 0.0,
            smoothed: f64 = 0.0,

            const factor = 25.0; // higher = more accurate
            const desync_threshold = 0.080; // s

            pub fn push(self: *@This(), s: f64) f64 {
                const now_ns = c.SDL_GetTicksNS();

                if (self.last_called == 0) { // init
                    self.last_called = now_ns;
                    self.smoothed = s;
                    self.last_returned = s;

                    return s;
                }

                const delta_time_s = @as(f64, @floatFromInt(now_ns - self.last_called)) / std.time.ns_per_s;
                self.last_called = now_ns;

                if (@abs(s - self.smoothed) > desync_threshold) {
                    self.smoothed = s; // snap to reality
                } else {
                    const alpha = 1.0 - std.math.exp(-factor * delta_time_s);
                    self.smoothed += (s - self.smoothed) * alpha;
                }

                const monotonic = @max(self.smoothed, self.last_returned);
                self.last_returned = monotonic;

                return monotonic;
            }
        };

        pub fn init(ctx: *const dec.AvCodecContext) !@This() {
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

            log.info("hw latency: {d:.2}ms ({} frames) @ {}Hz", .{ offset * std.time.ms_per_s, sample_frames, actual.freq });

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

        pub fn seconds_passed(self: *@This()) f64 {
            const zone = ztracy.ZoneN(@src(), "AudioClock.seconds_passed");
            defer zone.End();

            const bytes_per_sec: f64 = @floatFromInt(self.bytes_per_sample * self.sample_rate * self.channels);
            const queued: f64 = @floatFromInt(c.SDL_GetAudioStreamQueued(self.stream));
            const bytes_sent: f64 = @floatFromInt(self.bytes_sent.load(.monotonic));
            const hw_latency_bytes = self.hw_latency_secs * bytes_per_sec;

            const hw_pos = bytes_sent - queued - hw_latency_bytes;
            const hw_secs = (hw_pos / bytes_per_sec) + self.stream_start_offset;

            const now_ns = c.SDL_GetTicksNS();

            if (hw_pos != self.last_hw_pos) {
                self.last_hw_pos = hw_pos;
                self.last_hw_update_time_ns = now_ns;

                return self.ema.push(hw_secs);
            } else {
                const duration_s: f64 = @as(f64, @floatFromInt(now_ns - self.last_hw_update_time_ns)) / std.time.ns_per_s;
                return self.ema.push(hw_secs + duration_s);
            }
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

    pub fn decode(decoder: *Decoder) !void {
        const log = std.log.scoped(.audio_decode);
        ztracy.SetThreadName("audio decode");

        const clock = &(decoder.audio_clock orelse return error.uninitialized_audio_clock);
        const codec_ctx = &decoder.audio_ctx;
        const pkt_queue = &decoder.queue.pkt.audio;

        const max_queue_sec = clock.hw_latency_secs * 1.5;
        const bytes_per_sec: f64 = @floatFromInt(clock.sample_rate * clock.bytes_per_sample * clock.channels);
        const max_len: c_int = @intFromFloat(bytes_per_sec * max_queue_sec);

        log.info("rate: {} channels: {} bytes_per_sample: {}", .{ clock.sample_rate, clock.channels, clock.bytes_per_sample });
        log.info("queue: {d:.2}ms ({d} bytes)", .{ max_queue_sec * std.time.ms_per_s, @floor(max_queue_sec * bytes_per_sec) });

        var maybe_src_frame: ?*c.AVFrame = c.av_frame_alloc();
        defer c.av_frame_free(&maybe_src_frame);

        var maybe_dst_frame: ?*c.AVFrame = c.av_frame_alloc();
        defer c.av_frame_free(&maybe_dst_frame);

        const src_frame = maybe_src_frame orelse return error.out_of_memory;
        const dst_frame = maybe_dst_frame orelse return error.out_of_memory;

        const audio_ctx = codec_ctx.*.inner.?;

        _ = try libav.err(c.av_channel_layout_copy(&dst_frame.ch_layout, &audio_ctx.ch_layout));
        dst_frame.sample_rate = audio_ctx.sample_rate;
        dst_frame.format = c.AV_SAMPLE_FMT_FLT;

        var maybe_swr = c.swr_alloc();
        defer c.swr_free(&maybe_swr);

        const swr = maybe_swr orelse return error.out_of_memory;

        // debug
        var frame_count: u64 = 0;

        while (!decoder.should_quit.load(.monotonic)) {
            const zone = ztracy.ZoneN(@src(), "decode loop");
            defer zone.End();

            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                const recv_z = ztracy.ZoneN(@src(), "avcodec_receive_frame");
                defer recv_z.End();

                switch (c.avcodec_receive_frame(audio_ctx, src_frame)) {
                    0 => { // got a frame
                        defer c.av_frame_unref(src_frame);

                        const z = ztracy.ZoneN(@src(), "process frame");
                        defer z.End();

                        frame_count += 1;

                        {
                            const swr_z = ztracy.ZoneN(@src(), "swr_convert_frame");
                            defer swr_z.End();

                            _ = try libav.err(c.swr_convert_frame(swr, dst_frame, src_frame));
                        }

                        // Track stalls when audio buffer is full
                        {
                            const wait_z = ztracy.ZoneNC(@src(), "wait for audio buffer", 0x3b3b3b);
                            defer wait_z.End();

                            while (true) {
                                if (c.SDL_GetAudioStreamQueued(clock.stream) < max_len) break;
                                std.Thread.sleep(5 * std.time.ns_per_ms);
                            }
                        }

                        {
                            const put_z = ztracy.ZoneN(@src(), "SDL_PutAudioStreamData");
                            defer put_z.End();

                            const len = c.av_samples_get_buffer_size(null, dst_frame.ch_layout.nb_channels, dst_frame.nb_samples, dst_frame.format, 0);
                            _ = c.SDL_PutAudioStreamData(clock.stream, dst_frame.data[0], len);
                            _ = clock.bytes_sent.fetchAdd(@intCast(len), .monotonic);
                        }
                    },
                    c.AVERROR(c.EAGAIN) => break :recv_loop,
                    c.AVERROR_EOF => return,
                    else => |e| _ = try libav.err(e),
                }
            }

            var pkt: ?*c.AVPacket = pkt_queue.pop() orelse {
                const flush_ret = c.avcodec_send_packet(audio_ctx, null);
                if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) _ = try libav.err(flush_ret);

                continue;
            };
            defer c.av_packet_free(&pkt);

            {
                const send_z = ztracy.ZoneN(@src(), "avcodec_send_packet");
                defer send_z.End();

                const send_ret = c.avcodec_send_packet(audio_ctx, pkt);
                if (send_ret == c.AVERROR_EOF) return;
                if (send_ret == c.AVERROR(c.EAGAIN)) continue;
                if (send_ret < 0) _ = try libav.err(send_ret);
            }
        }
    }
};

pub const video = struct {
    pub fn decode(decoder: *Decoder) !void {
        //  const log = std.log.scoped(.video_decode);
        ztracy.SetThreadName("video decode");

        const frame_queue = &decoder.queue.frame;
        defer frame_queue.end_of_stream.store(true, .monotonic);

        const pkt_queue = &decoder.queue.pkt.video;
        const codec_ctx = &decoder.video_ctx;

        var src_frame: AvFrame = try .init();
        defer src_frame.deinit();

        var dst_frame: AvFrame = try .init();
        defer dst_frame.deinit();

        const video_ctx = codec_ctx.*.inner.?;

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
        while (!decoder.should_quit.load(.monotonic)) {
            const z = ztracy.ZoneN(@src(), "decode loop");
            defer z.End();

            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                const recv_z = ztracy.ZoneN(@src(), "avcodec_receive_frame");
                defer recv_z.End();

                switch (c.avcodec_receive_frame(video_ctx, src_frame.ptr())) {
                    0 => {
                        defer c.av_frame_unref(dst_frame.ptr());

                        try convert(decoder.video_ctx, src_frame.ptr(), dst_frame.ptr(), sws);
                        frame_queue.push(dst_frame.ptr()) catch |e| if (e != error.early_exit) return e;
                    },
                    c.AVERROR(c.EAGAIN) => break :recv_loop,
                    c.AVERROR_EOF => return,
                    else => |e| _ = try libav.err(e),
                }
            }

            var pkt: ?*c.AVPacket = pkt_queue.pop() orelse {
                const flush_ret = c.avcodec_send_packet(video_ctx, null);
                if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) _ = try libav.err(flush_ret);

                continue;
            };
            defer c.av_packet_free(&pkt);

            {
                const send_z = ztracy.ZoneN(@src(), "avcodec_send_packet");
                defer send_z.End();

                const send_ret = c.avcodec_send_packet(video_ctx, pkt);
                if (send_ret == c.AVERROR_EOF) return;
                if (send_ret == c.AVERROR(c.EAGAIN)) continue;
                if (send_ret < 0) _ = try libav.err(send_ret);
            }
        }
    }

    fn convert(codec_ctx: *const dec.AvCodecContext, src_frame: *c.AVFrame, dst_frame: *c.AVFrame, sws: *c.SwsContext) !void {
        @setRuntimeSafety(false);
        const zone = ztracy.Zone(@src());
        defer zone.End();

        if (codec_ctx.device) |_| {
            const z = ztracy.ZoneN(@src(), "hwframe_transfer_data");
            defer z.End();

            _ = try libav.err(c.av_hwframe_transfer_data(dst_frame, src_frame, 0));
        } else {
            const z = ztracy.ZoneN(@src(), "sws_scale_frame");
            defer z.End();

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
    should_quit: *const std.atomic.Value(bool),

    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},

    const Slot = struct { frame: []c.AVFrame, state: []State };
    const State = enum { empty, in_use, ready_to_reuse };
    const Error = error{ ffmpeg_error, invalid_size, early_exit } || std.mem.Allocator.Error;

    // INVARIANT: This is an SPSC Queue

    pub fn init(allocator: std.mem.Allocator, should_quit: *const std.atomic.Value(bool), count: usize) Error!FrameQueue {
        if (!std.math.isPowerOfTwo(count)) return error.invalid_size;

        const frames = try allocator.alloc(c.AVFrame, count);
        const states = try allocator.alloc(State, count);

        for (frames, states) |*frame, *state| {
            frame.* = std.mem.zeroes(c.AVFrame);
            c.av_frame_unref(frame);

            state.* = .empty;
        }

        return .{
            .should_quit = should_quit,
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
        const zone = ztracy.ZoneN(@src(), "FrameQueue.push");
        defer zone.End();

        const idx = self.mask(self.write_idx);
        const target_frame = &self.slot.frame[idx];

        {
            self.mutex.lock();
            defer self.mutex.unlock();

            const z = ztracy.ZoneNC(@src(), "wait for empty slot", 0x3b3b3b);
            defer z.End();

            while (self.slot.state[idx] != .empty) {
                if (self.should_quit.load(.monotonic)) return error.early_exit;
                self.cond.wait(&self.mutex);
            }
        }

        // don't lock on this
        _ = try libav.err(c.av_frame_ref(target_frame, new_frame));

        {
            self.mutex.lock();
            defer self.mutex.unlock();

            self.slot.state[idx] = .in_use;
            self.write_idx += 1;
            self.cond.signal();
        }
    }

    pub fn pop(self: *@This()) ?*c.AVFrame {
        const zone = ztracy.ZoneN(@src(), "FrameQueue.pop");
        defer zone.End();

        self.mutex.lock();
        defer self.mutex.unlock();

        const idx = self.mask(self.read_idx);
        if (self.slot.state[idx] != .in_use) return null;

        self.slot.state[idx] = .ready_to_reuse;
        self.read_idx += 1;
        self.cond.signal();

        return &self.slot.frame[idx];
    }

    pub fn recycle(self: *@This(), used_frame: *c.AVFrame) void {
        const zone = ztracy.ZoneN(@src(), "FrameQueue.recycle");
        defer zone.End();

        // SAFETY: we know the producer won't touch this because this ptr is .ready_to_reuse
        c.av_frame_unref(used_frame);

        self.mutex.lock();
        defer self.mutex.unlock();

        {
            const z = ztracy.ZoneNC(@src(), "frame search", 0x3b3b3b);
            defer z.End();

            for (self.slot.frame, self.slot.state) |*frame, *state| {
                if (frame != used_frame) continue;
                std.debug.assert(state.* == .ready_to_reuse);

                state.* = .empty;
                return self.cond.signal();
            }
        }

        std.debug.panic("attempted to recycle a frame that was not in the queue", .{});
    }

    inline fn mask(self: @This(), idx: usize) usize {
        return idx & (self.slot.frame.len - 1);
    }
};

pub const Decoder = struct {
    const PacketQueue = packet.Queue;
    const AudioClock = audio.Clock;

    fmt_ctx: dec.AvFormatContext,

    video_ctx: *dec.AvCodecContext,
    audio_ctx: *dec.AvCodecContext,

    queue: Queues,

    audio_clock: ?AudioClock,

    should_quit: *const std.atomic.Value(bool),

    // Important Information
    colour_space: c.AVColorSpace,
    dimensions: struct { u32, u32 },

    const Queues = struct {
        frame: FrameQueue,
        pkt: struct { audio: PacketQueue, video: PacketQueue },

        fn deinit(self: *Queues, allocator: std.mem.Allocator) void {
            self.pkt.audio.interrupt();
            self.pkt.audio.interrupt();

            self.pkt.audio.deinit(allocator);
            self.pkt.video.deinit(allocator);
            self.frame.deinit(allocator);
        }
    };

    const Handles = struct {
        pkt: std.Thread,
        audio: ?std.Thread,
        video: std.Thread,

        pub fn deinit(self: Handles) void {
            self.pkt.join();
            self.video.join();

            if (self.audio) |a| a.join();
        }
    };

    const log = std.log.scoped(.decode);

    pub fn init(allocator: std.mem.Allocator, should_quit: *const std.atomic.Value(bool), hw_device: ?c.AVHWDeviceType, path: []const u8, headless: bool) !Decoder {
        var fmt_ctx = try dec.AvFormatContext.init(path);
        errdefer fmt_ctx.deinit();

        var video_ctx = try dec.AvCodecContext.init(allocator, .video, fmt_ctx, .{ .dev_type = hw_device });
        errdefer video_ctx.deinit(allocator);

        log.debug("colour space: {s}", .{c.av_color_space_name(video_ctx.inner.?.colorspace)});

        var audio_ctx = try dec.AvCodecContext.init(allocator, .audio, fmt_ctx, .{});
        errdefer audio_ctx.deinit(allocator);

        var frame_queue = try FrameQueue.init(allocator, should_quit, 0x80); // 1s at 120fps?
        errdefer frame_queue.deinit(allocator);

        var video_queue = PacketQueue.init(allocator, should_quit);
        errdefer video_queue.deinit(allocator);

        var audio_queue = PacketQueue.init(allocator, should_quit);
        errdefer audio_queue.deinit(allocator);

        const audio_clock: ?AudioClock = if (headless) null else try AudioClock.init(audio_ctx);
        errdefer if (audio_clock) |clock| clock.deinit();

        return .{
            .should_quit = should_quit,

            .fmt_ctx = fmt_ctx,
            .video_ctx = video_ctx,
            .audio_ctx = audio_ctx,
            .queue = .{
                .frame = frame_queue,
                .pkt = .{
                    .video = video_queue,
                    .audio = audio_queue,
                },
            },
            .audio_clock = audio_clock,
            .colour_space = video_ctx.inner.?.colorspace,
            .dimensions = .{ @intCast(video_ctx.inner.?.width), @intCast(video_ctx.inner.?.height) },
        };
    }

    pub fn framerate(self: Decoder) c.AVRational {
        return self.fmt_ctx.ptr().streams[@intCast(self.video_ctx.stream)].*.avg_frame_rate;
    }

    const StreamKind = enum { audio, video };

    pub fn stream(self: Decoder, comptime kind: StreamKind) *c.AVStream {
        return self.fmt_ctx.ptr().streams[
            switch (kind) {
                .audio => @intCast(self.audio_ctx.stream),
                .video => @intCast(self.video_ctx.stream),
            }
        ];
    }

    pub fn deinit(self: *Decoder, allocator: std.mem.Allocator) void {
        if (self.audio_clock) |clock| clock.deinit();
        self.queue.deinit(allocator);

        self.audio_ctx.deinit(allocator);
        self.video_ctx.deinit(allocator);

        self.fmt_ctx.deinit();
    }

    pub fn spawn(self: *Decoder, render: ?[]const u8) !Handles {
        const pkt_handle = try std.Thread.spawn(.{}, packet.read, .{self});
        const video_handle = try std.Thread.spawn(.{}, video.decode, .{self});
        const audio_handle = if (render) |_| null else try std.Thread.spawn(.{}, audio.decode, .{self});

        return .{
            .pkt = pkt_handle,
            .video = video_handle,
            .audio = audio_handle,
        };
    }
};

pub const Encoder = struct {
    width: u32,
    height: u32,

    codec_ctx: enc.AvCodecContext,
    fmt_ctx: enc.AvFormatContext,

    sws_ctx: ?*c.SwsContext,

    audio_stream: enc.AvStream,
    video_stream: enc.AvStream,

    _pkt: AvPacket,
    _frame: AvFrame,
    _hw: ?struct { frame: AvFrame } = null,

    const log = std.log.scoped(.encode);

    pub const Options = struct {
        width: u32,
        height: u32,

        decoder: *const Decoder,
    };

    pub fn init(opt: Options, device_type: ?c.AVHWDeviceType, path: []const u8) !Encoder {
        const codec_id = c.AV_CODEC_ID_HEVC;

        if (device_type) |dev| {
            return initHardware(opt, dev, codec_id, path) catch blk: {
                log.err("failed to set up hardware device, defaulting to software", .{});
                break :blk initSoftware(opt, codec_id, path);
            };
        }

        return initSoftware(opt, codec_id, path);
    }

    fn initShared(opt: Options, codec: enc.AvCodec, sw_pix_fmt: c.AVPixelFormat, path: []const u8) !Encoder {
        var fmt_ctx = try enc.AvFormatContext.init(path);
        errdefer fmt_ctx.deinit();

        var codec_ctx = try enc.AvCodecContext.init(codec, fmt_ctx, .{
            .width = @intCast(opt.width),
            .height = @intCast(opt.height),
            .input = .{
                .fmt_ctx = opt.decoder.fmt_ctx,
                .video_ctx = opt.decoder.video_ctx,
            },
        });
        errdefer codec_ctx.deinit();

        _ = try libav.err(c.avcodec_open2(codec_ctx.ptr(), codec.ptr(), null));

        const input_audio = opt.decoder.stream(.audio);

        const audio_stream = try enc.AvStream.init(fmt_ctx);
        _ = try libav.err(c.avcodec_parameters_copy(audio_stream.ptr().codecpar, input_audio.*.codecpar));
        audio_stream.ptr().codecpar.*.codec_tag = 0; // reset

        const video_stream = try enc.AvStream.init(fmt_ctx);
        _ = try libav.err(c.avcodec_parameters_from_context(video_stream.ptr().codecpar, codec_ctx.ptr()));
        video_stream.ptr().time_base = codec_ctx.ptr().time_base;

        c.av_dump_format(fmt_ctx.ptr(), 0, path.ptr, 1);

        _ = try libav.err(c.avio_open(&fmt_ctx.ptr().pb, path.ptr, c.AVIO_FLAG_WRITE));
        _ = try libav.err(c.avformat_write_header(fmt_ctx.ptr(), null));

        return .{
            .width = opt.width,
            .height = opt.height,
            .codec_ctx = codec_ctx,
            .fmt_ctx = fmt_ctx,

            .audio_stream = audio_stream,
            .video_stream = video_stream,

            ._pkt = try AvPacket.try_init(),
            ._frame = blk: {
                var frame = try AvFrame.init();
                errdefer frame.deinit();

                // additional setup
                const ptr = frame.ptr();
                ptr.color_range = c.AVCOL_RANGE_MPEG;
                ptr.color_primaries = c.AVCOL_PRI_BT709;
                ptr.color_trc = c.AVCOL_TRC_BT709;
                ptr.colorspace = c.AVCOL_SPC_BT709;

                try frame.setup(@intCast(opt.width), @intCast(opt.height), sw_pix_fmt);

                break :blk frame;
            },
            ._hw = if (codec.hw) |_| .{ .frame = try AvFrame.init() } else null,
            .sws_ctx = blk: {
                const ptr: ?*c.SwsContext = c.sws_getContext(
                    @intCast(opt.width),
                    @intCast(opt.height),
                    c.AV_PIX_FMT_RGB24,
                    @intCast(opt.width),
                    @intCast(opt.height),
                    sw_pix_fmt,
                    c.SWS_POINT,
                    null,
                    null,
                    null,
                );

                break :blk ptr.?;
            },
        };
    }

    fn initHardware(opt: Options, device_type: c.AVHWDeviceType, codec_id: c.AVCodecID, path: []const u8) !Encoder {
        const codec = enc.AvCodec.findHardware(device_type, codec_id) orelse return initSoftware(opt, codec_id, path);
        return initShared(opt, codec, c.AV_PIX_FMT_NV12, path);
    }

    fn initSoftware(opt: Options, codec_id: c.AVCodecID, path: []const u8) !Encoder {
        const codec = enc.AvCodec.findSoftware(codec_id);
        return initShared(opt, codec, codec.pix_fmt, path);
    }

    pub fn encodeRgbFrame(self: *Encoder, buf: []const u8, frame_pts: i64) !void {
        @setRuntimeSafety(false);

        const zone = ztracy.Zone(@src());
        defer zone.End();

        const sw_frame = self._frame.ptr();
        _ = try libav.err(c.av_frame_make_writable(sw_frame));

        const stride: usize = self.width * 3;

        var src_frame: c.AVFrame = .{
            .format = c.AV_PIX_FMT_RGB24,
            .width = @intCast(self.width),
            .height = @intCast(self.height),
        };

        src_frame.data[0] = @constCast(buf.ptr + @as(usize, self.height - 1) * stride);
        src_frame.linesize[0] = -@as(c_int, @intCast(stride)); // negative to flip vertically

        {
            const z = ztracy.ZoneN(@src(), "sws_scale_frame");
            defer z.End();

            _ = try libav.err(c.sws_scale_frame(self.sws_ctx, sw_frame, &src_frame));
            sw_frame.pts = frame_pts;
        }

        if (self._hw) |hw| {
            const hw_frame = hw.frame.ptr();

            {
                const z = ztracy.ZoneN(@src(), "hw upload");
                defer z.End();

                c.av_frame_unref(hw_frame);
                _ = try libav.err(c.av_hwframe_get_buffer(self.codec_ctx.ptr().hw_frames_ctx, hw_frame, 0));
                _ = try libav.err(c.av_hwframe_transfer_data(hw_frame, sw_frame, 0));
                _ = try libav.err(c.av_frame_copy_props(hw_frame, sw_frame));
            }

            try self.writeVideoFrame(hw_frame);
        } else {
            try self.writeVideoFrame(sw_frame);
        }
    }

    pub fn writeAudioPacket(self: *Encoder, in: *c.AVStream, pkt: *c.AVPacket) !void {
        const zone = ztracy.Zone(@src());
        defer zone.End();

        const UNKN_POS = -1;
        const audio_stream = self.audio_stream.ptr();

        c.av_packet_rescale_ts(pkt, in.time_base, audio_stream.time_base);
        pkt.pos = UNKN_POS;
        pkt.stream_index = audio_stream.index;

        {
            const z = ztracy.ZoneN(@src(), "av_interleaved_write_frame");
            defer z.End();

            _ = try libav.err(c.av_interleaved_write_frame(self.fmt_ctx.ptr(), pkt));
        }
    }

    fn writeVideoFrame(self: *Encoder, frame: ?*c.AVFrame) !void {
        const zone = ztracy.Zone(@src());
        defer zone.End();

        const codec_ctx = self.codec_ctx.ptr();
        const video_stream = self.video_stream.ptr();

        const pkt = self._pkt.ptr();

        {
            const z = ztracy.ZoneN(@src(), "avcodec_send_frame");
            defer z.End();

            _ = try libav.err(c.avcodec_send_frame(codec_ctx, frame));
        }

        while (true) {
            const z_loop = ztracy.ZoneN(@src(), "write loop");
            defer z_loop.End();

            {
                const z = ztracy.ZoneN(@src(), "avcodec_receive_packet");
                defer z.End();

                const ret = c.avcodec_receive_packet(codec_ctx, pkt);
                if (ret == c.AVERROR(c.EAGAIN) or ret == c.AVERROR_EOF) break;
                _ = try libav.err(ret);
            }

            c.av_packet_rescale_ts(pkt, codec_ctx.time_base, video_stream.time_base);
            pkt.stream_index = video_stream.index;

            {
                const z = ztracy.ZoneN(@src(), "av_interleaved_write_frame");
                defer z.End();

                _ = try libav.err(c.av_interleaved_write_frame(self.fmt_ctx.ptr(), pkt));
            }
        }
    }

    pub fn deinit(self: *Encoder) void {
        self.writeVideoFrame(null) catch @panic("failed to flush encoder");
        _ = c.av_write_trailer(self.fmt_ctx.ptr());

        self._frame.deinit();
        if (self._hw) |*hw| hw.frame.deinit();
        self._pkt.deinit();

        self.codec_ctx.deinit();
        self.fmt_ctx.deinit();

        self.* = undefined;
    }
};
