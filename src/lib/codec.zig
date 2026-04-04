const std = @import("std");
const libav = @import("libav.zig");
const tracy = @import("tracy");
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

        mutex: TracyMutex,
        cond: std.Thread.Condition = .{},

        end_of_stream: std.atomic.Value(bool) = .init(false),
        should_quit: *const std.atomic.Value(bool),

        const log = std.log.scoped(.packet_queue);

        // INVARIANT: this is an SPSC Queue

        pub fn init(allocator: std.mem.Allocator, name: []const u8, should_quit: *const std.atomic.Value(bool)) Queue {
            return .{
                .mutex = .init(@src(), name),
                .list = LinearFifo.init(allocator),
                .should_quit = should_quit,
            };
        }

        pub fn deinit(self: *@This(), _: std.mem.Allocator) void {
            while (self.list.readItem()) |pkt| {
                var p: ?*c.AVPacket = pkt;
                c.av_packet_free(&p);
            }

            self.mutex.deinit();
            self.list.deinit();
        }

        pub fn push(self: *@This(), pkt: *c.AVPacket) !void {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

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

        pub fn pop(self: *@This()) !*c.AVPacket {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

            self.mutex.lock();
            defer self.mutex.unlock();

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "wait for packet", .color = .gray25 });
                defer z.end();

                while (self.list.readableLength() == 0) {
                    if (self.end_of_stream.load(.monotonic)) return error.end_of_stream;
                    if (self.should_quit.load(.monotonic)) return error.should_quit;

                    self.mutex.wait(&self.cond);
                }
            }

            return self.list.readItem() orelse unreachable;
        }

        /// Non-blocking pop - returns null immediately if queue is empty
        pub fn tryPop(self: *@This()) ?*c.AVPacket {
            const zone = tracy.Zone.begin(.{ .src = @src() });
            defer zone.end();

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
        tracy.setThreadName("packet read");

        var pkt = AvPacket.init();
        defer pkt.deinit();

        const audio_queue = &decode.queue.pkt.audio;
        const video_queue = &decode.queue.pkt.video;
        const fmt_ctx = &decode.fmt_ctx;

        while (!decode.should_quit.load(.monotonic)) {
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "read loop" });
            defer zone.end();

            switch (c.av_read_frame(fmt_ctx.ptr(), pkt.ptr())) {
                0...std.math.maxInt(c_int) => {
                    defer c.av_packet_unref(pkt.ptr());

                    const z = tracy.Zone.begin(.{ .src = @src(), .name = "read frame" });
                    defer z.end();

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

        sync: AudioPll = .{},

        /// Offset to align audio time with video stream time (set from first video PTS)
        /// This accounts for videos that don't start at PTS=0
        stream_start_offset: f64 = 0.0,

        is_muted: bool = true,

        stream: *c.SDL_AudioStream,

        const log = std.log.scoped(.audio);

        const AudioPll = struct {
            last_called: u64 = 0,
            last_returned: f64 = 0.0,

            const factor = 15.0; // smooth factor for drift correction
            const desync_threshold = 0.050; // s

            pub fn push(self: *@This(), hw_s: f64, offset_s: f64) f64 {
                const now_ns = c.SDL_GetTicksNS();
                const s = hw_s - offset_s;

                if (self.last_called == 0) { // init
                    self.last_called = now_ns;
                    self.last_returned = s;
                    return s;
                }

                const delta_time_s = @as(f64, @floatFromInt(now_ns - self.last_called)) / std.time.ns_per_s;
                self.last_called = now_ns;

                // Advance our clock by system time to maintain perfect linearity
                const sys_advanced = self.last_returned + delta_time_s;

                // Compare our linear expected time with the noisy/interpolated audio time
                const drift = s - sys_advanced;

                if (@abs(offset_s - 0.0) < std.math.floatEps(f64))
                    tracy.plot(.{ .name = "Audio Clock Drift (ms)", .value = .{ .f64 = drift * std.time.ms_per_s } });

                var smoothed: f64 = 0.0;
                if (@abs(drift) > desync_threshold) {
                    tracy.message(.{ .text = "snap audio clock to reality" });
                    smoothed = s; // snap to reality if drift is too large (e.g. seek)
                } else {
                    // Gently pull our linear clock towards the audio clock to correct over/under-shoots
                    // without ruining the frame-to-frame linearity
                    const alpha = 1.0 - std.math.exp(-factor * delta_time_s);
                    smoothed = sys_advanced + (drift * alpha);
                }

                const monotonic = @max(smoothed, self.last_returned);
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
            tracy.plotConfig(.{ .name = "Audio Clock Drift (ms)" });
            tracy.plotConfig(.{ .name = "Audio Samples Queued" });

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
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "AudioClock.seconds_passed" });
            defer zone.end();

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

                tracy.plot(.{ .name = "Audio Samples Queued", .value = .{ .i64 = @as(u32, @intFromFloat(queued)) / self.bytes_per_sample } });
                return self.sync.push(hw_secs, 0.0);
            } else {
                const duration_s: f64 = @as(f64, @floatFromInt(now_ns - self.last_hw_update_time_ns)) / std.time.ns_per_s;
                return self.sync.push(hw_secs, duration_s);
            }
        }
    };

    // A/V Sync Debug: Set to true to enable audio-side logging

    pub fn decode(decoder: *Decoder) !void {
        const log = std.log.scoped(.audio_decode);
        tracy.setThreadName("audio decode");

        const clock = &(decoder.audio_clock orelse return error.uninitialized_audio_clock);
        const codec_ctx = &decoder.audio_ctx;
        const pkt_queue = &decoder.queue.pkt.audio;

        const max_queue_sec = clock.hw_latency_secs * 1.1;
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

        var pending: ?*c.AVPacket = null;
        defer if (pending) |_| c.av_packet_free(&pending);

        while (!decoder.should_quit.load(.monotonic)) {
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "decode loop" });
            defer zone.end();

            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                const recv_z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_receive_frame" });
                defer recv_z.end();

                switch (c.avcodec_receive_frame(audio_ctx, src_frame)) {
                    0 => { // got a frame
                        defer c.av_frame_unref(src_frame);

                        const z = tracy.Zone.begin(.{ .src = @src(), .name = "process frame" });
                        defer z.end();

                        frame_count += 1;

                        {
                            const swr_z = tracy.Zone.begin(.{ .src = @src(), .name = "swr_convert_frame" });
                            defer swr_z.end();

                            _ = try libav.err(c.swr_convert_frame(swr, dst_frame, src_frame));
                        }

                        // Track stalls when audio buffer is full
                        {
                            const wait_z = tracy.Zone.begin(.{ .src = @src(), .name = "wait for audio buffer", .color = .gray25 });
                            defer wait_z.end();

                            while (true) {
                                if (c.SDL_GetAudioStreamQueued(clock.stream) < max_len) break;
                                std.Thread.sleep(5 * std.time.ns_per_ms);
                            }
                        }

                        {
                            const put_z = tracy.Zone.begin(.{ .src = @src(), .name = "SDL_PutAudioStreamData" });
                            defer put_z.end();

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

            var pkt: ?*c.AVPacket = blk: {
                if (pending) |pkt| {
                    defer pending = null;
                    break :blk pkt;
                }

                break :blk pkt_queue.pop() catch |e| switch (e) {
                    error.end_of_stream => {
                        const flush_ret = c.avcodec_send_packet(audio_ctx, null);
                        if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) _ = try libav.err(flush_ret);

                        continue;
                    },
                    error.should_quit => break,
                };
            };

            {
                const send_z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_send_packet" });
                defer send_z.end();

                const send_ret = c.avcodec_send_packet(audio_ctx, pkt);
                if (send_ret == c.AVERROR(c.EAGAIN)) {
                    pending = pkt;
                    continue;
                }

                c.av_packet_free(&pkt);
                if (send_ret == c.AVERROR_EOF) return;
                if (send_ret < 0) _ = try libav.err(send_ret);
            }
        }
    }
};

pub const video = struct {
    pub fn decode(decoder: *Decoder) !void {
        // const log = std.log.scoped(.video_decode);
        tracy.setThreadName("video decode");

        const frame_queue = &decoder.queue.frame;
        defer frame_queue.end_of_stream.store(true, .monotonic);

        const pkt_queue = &decoder.queue.pkt.video;
        const codec_ctx = &decoder.video_ctx;

        var src_frame: AvFrame = try .init();
        defer src_frame.deinit();

        const video_ctx = codec_ctx.*.inner.?;

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

        var pending: ?*c.AVPacket = null;
        defer if (pending) |_| c.av_packet_free(&pending);

        // TODO: using a fn ptr I think we can deduplicate this massive loop
        while (!decoder.should_quit.load(.monotonic)) {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "decode loop" });
            defer z.end();

            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                const recv_z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_receive_frame" });
                defer recv_z.end();

                switch (c.avcodec_receive_frame(video_ctx, src_frame.ptr())) {
                    0 => {
                        defer c.av_frame_unref(src_frame.ptr());

                        const dst_frame = frame_queue.acquire() catch |e| if (e != error.early_exit) return e else break :recv_loop;
                        defer frame_queue.commit(dst_frame);

                        _ = try libav.err(c.av_frame_copy_props(dst_frame, src_frame.ptr()));
                        try convert(decoder.video_ctx, src_frame.ptr(), dst_frame, sws);
                    },
                    c.AVERROR(c.EAGAIN) => break :recv_loop,
                    c.AVERROR_EOF => return,
                    else => |e| _ = try libav.err(e),
                }
            }

            var pkt: ?*c.AVPacket = blk: {
                if (pending) |pkt| {
                    pending = null;
                    break :blk pkt;
                }

                break :blk pkt_queue.pop() catch |e| switch (e) {
                    error.end_of_stream => {
                        const flush_ret = c.avcodec_send_packet(video_ctx, null);
                        if (flush_ret < 0 and flush_ret != c.AVERROR_EOF) _ = try libav.err(flush_ret);

                        continue;
                    },
                    error.should_quit => break,
                };
            };

            {
                const send_z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_send_packet" });
                defer send_z.end();

                const send_ret = c.avcodec_send_packet(video_ctx, pkt);
                if (send_ret == c.AVERROR(c.EAGAIN)) {
                    pending = pkt;
                    continue;
                }

                c.av_packet_free(&pkt);

                if (send_ret == c.AVERROR_EOF) return;
                if (send_ret < 0) _ = try libav.err(send_ret);
            }
        }
    }

    fn convert(codec_ctx: *const dec.AvCodecContext, src_frame: *c.AVFrame, dst_frame: *c.AVFrame, sws: *c.SwsContext) !void {
        @setRuntimeSafety(false);
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        if (codec_ctx.device) |_| {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "hwframe_transfer_data" });
            defer z.end();

            _ = try libav.err(c.av_hwframe_transfer_data(dst_frame, src_frame, 0));
        } else {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "sws_scale_frame" });
            defer z.end();

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

const TracyMutex = struct {
    inner: std.Thread.Mutex,
    _lock: *tracy.Lock,

    pub fn init(comptime src: std.builtin.SourceLocation, name: []const u8) TracyMutex {
        const _lock: *tracy.Lock = .init(.{ .src = src });
        _lock.customName(name);

        return .{
            .inner = .{},
            ._lock = _lock,
        };
    }

    pub fn deinit(self: @This()) void {
        self._lock.deinit();
    }

    pub fn lock(self: *@This()) void {
        _ = self._lock.beforeLock();
        defer self._lock.afterLock();

        self.inner.lock();
    }

    pub fn unlock(self: *@This()) void {
        self.inner.unlock();
        self._lock.afterUnlock();
    }

    pub fn wait(self: *@This(), cond: *std.Thread.Condition) void {
        // INVARIANT: this function is called when the lock is held
        self._lock.afterUnlock();
        defer self._lock.afterLock();

        cond.wait(&self.inner);
    }

    pub fn tryLock(self: *@This()) bool {
        const ret = self.inner.tryLock();
        self._lock.afterTryUnlock(ret); // NB: is actually afterTrylock

        return ret;
    }
};

pub const FrameQueue = struct {
    slot: Slot,
    read_idx: usize,
    write_idx: usize,

    end_of_stream: std.atomic.Value(bool) = .init(false),
    should_quit: *const std.atomic.Value(bool),

    mutex: TracyMutex,
    cond: std.Thread.Condition = .{},

    const Slot = struct { frame: []c.AVFrame, state: []State };
    const State = enum { empty, in_use, ready_to_reuse, writing };
    const Error = error{ ffmpeg_error, invalid_size, early_exit } || std.mem.Allocator.Error;

    // INVARIANT: This is an SPSC Queue

    // FIXME: replace with Resolution or Dimension or whatever
    const FrameOptions = struct { width: c_int, height: c_int };

    pub fn init(allocator: std.mem.Allocator, should_quit: *const std.atomic.Value(bool), count: usize, opt: FrameOptions) Error!FrameQueue {
        if (!std.math.isPowerOfTwo(count)) return error.invalid_size;

        const frames = try allocator.alloc(c.AVFrame, count);
        errdefer allocator.free(frames);

        const states = try allocator.alloc(State, count);
        errdefer allocator.free(states);

        for (frames, states) |*frame, *state| {
            frame.* = std.mem.zeroes(c.AVFrame);
            c.av_frame_unref(frame); // make the frames valid

            frame.width = opt.width;
            frame.height = opt.height;
            frame.format = c.AV_PIX_FMT_NV12;

            _ = try libav.err(c.av_frame_get_buffer(frame, 32));

            state.* = .empty;
        }

        return .{
            .mutex = .init(@src(), "FrameQueue"),
            .should_quit = should_quit,
            .slot = .{ .frame = frames, .state = states },
            .read_idx = 0,
            .write_idx = 0,
        };
    }

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        for (self.slot.frame) |*frame| c.av_frame_unref(frame);

        self.mutex.deinit();
        allocator.free(self.slot.frame);
        allocator.free(self.slot.state);
    }

    pub fn acquire(self: *@This()) Error!*c.AVFrame {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const idx = self.mask(self.write_idx);

        self.mutex.lock();
        defer self.mutex.unlock();

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "wait for available frame", .color = .gray25 });
            defer z.end();

            while (self.slot.state[idx] != .empty) {
                if (self.should_quit.load(.monotonic)) return error.early_exit;

                self.mutex.wait(&self.cond);
            }
        }

        self.slot.state[idx] = .writing;
        return &self.slot.frame[idx];
    }

    pub fn commit(self: *@This(), frame: *c.AVFrame) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const idx = self.mask(self.write_idx);

        self.mutex.lock();
        defer self.mutex.unlock();

        std.debug.assert(&self.slot.frame[idx] == frame);
        std.debug.assert(self.slot.state[idx] == .writing);

        self.slot.state[idx] = .in_use;
        self.write_idx += 1;
        self.cond.signal();
    }

    pub fn pop(self: *@This()) ?*c.AVFrame {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

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
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        self.mutex.lock();
        defer self.mutex.unlock();

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "frame search", .color = .gray25 });
            defer z.end();

            for (self.slot.frame, self.slot.state) |*frame, *state| {
                if (frame != used_frame) continue;
                std.debug.assert(state.* == .ready_to_reuse);

                state.* = .empty;
                return self.cond.signal();
            }
        }

        std.debug.panic("attempted to recycle a frame that was not in the queue", .{});
    }

    pub inline fn len(self: @This()) usize {
        return self.write_idx - self.read_idx;
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

        var frame_queue = try FrameQueue.init(allocator, should_quit, 0x20, .{
            .width = video_ctx.inner.?.width,
            .height = video_ctx.inner.?.height,
        });
        errdefer frame_queue.deinit(allocator);

        var video_queue = PacketQueue.init(allocator, "Video PacketQueue", should_quit);
        errdefer video_queue.deinit(allocator);

        var audio_queue = PacketQueue.init(allocator, "Audio PacketQueue", should_quit);
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

    sw_pix_fmt: c.AVPixelFormat,

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
                    c.AV_PIX_FMT_NV12,
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
            .sw_pix_fmt = sw_pix_fmt,
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

    pub fn encodeNv12Frame(self: *Encoder, y_buf: []const u8, uv_buf: []const u8, frame_pts: i64) !void {
        @setRuntimeSafety(false);

        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const sw_frame = self._frame.ptr();
        _ = try libav.err(c.av_frame_make_writable(sw_frame));

        const y_stride: usize = self.width;
        const uv_stride: usize = self.width; // RG is 2 bpp, width/2 pixels => width bytes

        // Hardware path
        if (self.sw_pix_fmt == c.AV_PIX_FMT_NV12) {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "hw memcpy" });
            defer z.end();

            sw_frame.pts = frame_pts;

            for (0..self.height) |h| {
                @memcpy(sw_frame.data[0][h * @as(usize, @intCast(sw_frame.linesize[0])) ..][0..y_stride], y_buf[h * y_stride ..][0..y_stride]);
            }

            for (0..self.height / 2) |h| {
                @memcpy(sw_frame.data[1][h * @as(usize, @intCast(sw_frame.linesize[1])) ..][0..uv_stride], uv_buf[h * uv_stride ..][0..uv_stride]);
            }
        } else {
            // Software path
            var src_frame: c.AVFrame = .{
                .format = c.AV_PIX_FMT_NV12,
                .width = @intCast(self.width),
                .height = @intCast(self.height),
            };

            src_frame.data[0] = @constCast(y_buf.ptr);
            src_frame.linesize[0] = @intCast(y_stride);

            src_frame.data[1] = @constCast(uv_buf.ptr);
            src_frame.linesize[1] = @intCast(uv_stride);

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "sws_scale_frame" });
                defer z.end();

                _ = try libav.err(c.sws_scale_frame(self.sws_ctx, sw_frame, &src_frame));
                sw_frame.pts = frame_pts;
            }
        }

        if (self._hw) |hw| {
            const hw_frame = hw.frame.ptr();

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "hw upload" });
                defer z.end();

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
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const UNKN_POS = -1;
        const audio_stream = self.audio_stream.ptr();

        c.av_packet_rescale_ts(pkt, in.time_base, audio_stream.time_base);
        pkt.pos = UNKN_POS;
        pkt.stream_index = audio_stream.index;

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "av_interleaved_write_frame" });
            defer z.end();

            _ = try libav.err(c.av_interleaved_write_frame(self.fmt_ctx.ptr(), pkt));
        }
    }

    fn writeVideoFrame(self: *Encoder, frame: ?*c.AVFrame) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const codec_ctx = self.codec_ctx.ptr();
        const video_stream = self.video_stream.ptr();

        const pkt = self._pkt.ptr();

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_send_frame" });
            defer z.end();

            _ = try libav.err(c.avcodec_send_frame(codec_ctx, frame));
        }

        while (true) {
            const z_loop = tracy.Zone.begin(.{ .src = @src(), .name = "write loop" });
            defer z_loop.end();

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_receive_packet" });
                defer z.end();

                const ret = c.avcodec_receive_packet(codec_ctx, pkt);
                if (ret == c.AVERROR(c.EAGAIN) or ret == c.AVERROR_EOF) break;
                _ = try libav.err(ret);
            }

            c.av_packet_rescale_ts(pkt, codec_ctx.time_base, video_stream.time_base);
            pkt.stream_index = video_stream.index;

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "av_interleaved_write_frame" });
                defer z.end();

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
