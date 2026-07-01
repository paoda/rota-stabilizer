const std = @import("std");
const libav = @import("libav.zig");
const tracy = @import("tracy");
const c = @import("../lib.zig").c;

const errify = @import("platform.zig").errify;

const LinearFifo = @import("fifo.zig").LinearFifo(*c.AVPacket, .dynamic);

const signal = @import("platform.zig").signal;

const enc = @import("libav.zig").enc;
const dec = @import("libav.zig").dec;
const sleep = @import("../lib.zig").sleep;

const AvPacket = @import("libav.zig").AvPacket;
const AvFrame = @import("libav.zig").AvFrame;
const Viewport = @import("../lib.zig").Viewport;
const Resolution = @import("../lib.zig").Resolution;

const errors = &@import("../lib.zig").errors;

// TODO: some universal thread sync primitive

pub const packet = struct {
    pub const Queue = struct { // FIXME: is there any point to rolling my own?
        list: LinearFifo,

        mutex: TracyMutex,
        cond: std.Thread.Condition = .{},

        end_of_stream: std.atomic.Value(bool) = .init(false),

        const log = std.log.scoped(.packet_queue);

        // INVARIANT: this is an SPSC Queue

        pub fn init(allocator: std.mem.Allocator, name: []const u8) Queue {
            return .{
                .mutex = .init(@src(), name),
                .list = LinearFifo.init(allocator),
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
                    if (signal.should_quit.load(.monotonic)) return error.should_quit;

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
        const log = std.log.scoped(.packet_read);
        defer log.debug("thread exit", .{});

        tracy.setThreadName("packet read");

        var pkt = AvPacket.init();
        defer pkt.deinit();

        const audio_queue = &decode.queue.pkt.audio;
        const video_queue = &decode.queue.pkt.video;
        const fmt_ctx = &decode.fmt_ctx;

        while (!signal.should_quit.load(.monotonic)) {
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

        stream: *c.SDL_AudioStream,

        volume: f32,

        // TODO(paoda): this needs to be an import from sdl.Error or something
        pub const InitError = error{sdl_error};

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

        pub fn init(ctx: *const dec.AvCodecContext, volume: f32) InitError!@This() {
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "audio.Clock.init" });
            defer zone.end();

            var desired: c.SDL_AudioSpec = std.mem.zeroes(c.SDL_AudioSpec);
            desired.freq = ctx.inner.?.sample_rate;
            desired.format = c.SDL_AUDIO_F32;
            desired.channels = ctx.inner.?.ch_layout.nb_channels;

            const stream = try errify(c.SDL_OpenAudioDeviceStream(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired, null, null));
            errdefer c.SDL_DestroyAudioStream(stream);

            var actual: c.SDL_AudioSpec = std.mem.zeroes(c.SDL_AudioSpec);
            var sample_frames: c_int = undefined;

            try errify(c.SDL_GetAudioDeviceFormat(c.SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &actual, &sample_frames));
            const offset = @as(f64, @floatFromInt(sample_frames)) / @as(f64, @floatFromInt(actual.freq));

            try errify(c.SDL_SetAudioStreamGain(stream, volume));

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
                .volume = volume,
            };
        }

        pub fn setVolume(self: *@This(), volume: f32) void {
            const next = @max(0.0, @min(1.0, volume));

            if (!c.SDL_SetAudioStreamGain(self.stream, next)) {
                return errors.add_set_volume_err(next);
            }

            self.volume = next;
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
        defer log.debug("thread exit", .{});

        tracy.setThreadName("audio decode");

        const clock = &(decoder.audio_clock orelse return error.uninitialized_audio_clock);
        const audio_ctx = decoder.audio_ctx.*.inner.?;
        const pkt_queue = &decoder.queue.pkt.audio;

        const max_queue_sec = clock.hw_latency_secs * 1.1;
        const bytes_per_sec: f64 = @floatFromInt(clock.sample_rate * clock.bytes_per_sample * clock.channels);
        const max_len: c_int = @intFromFloat(bytes_per_sec * max_queue_sec);

        log.info("rate: {} channels: {} bytes_per_sample: {}", .{ clock.sample_rate, clock.channels, clock.bytes_per_sample });
        log.info("queue: {d:.2}ms ({d} bytes)", .{ max_queue_sec * std.time.ms_per_s, @floor(max_queue_sec * bytes_per_sec) });

        var src_frame = try AvFrame.init();
        defer src_frame.deinit();

        var dst_frame = try AvFrame.init();
        defer dst_frame.deinit();

        var maybe_swr = c.swr_alloc();
        defer c.swr_free(&maybe_swr);

        const swr = maybe_swr orelse return error.out_of_memory;

        // debug
        var frame_count: u64 = 0;

        var pending: ?*c.AVPacket = null;
        defer if (pending) |_| c.av_packet_free(&pending);

        while (!signal.should_quit.load(.monotonic)) {
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "decode loop" });
            defer zone.end();

            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                const recv_z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_receive_frame" });
                defer recv_z.end();

                switch (c.avcodec_receive_frame(audio_ctx, src_frame.ptr())) {
                    0 => { // got a frame
                        defer c.av_frame_unref(src_frame.ptr());
                        defer c.av_frame_unref(dst_frame.ptr());

                        const z = tracy.Zone.begin(.{ .src = @src(), .name = "process frame" });
                        defer z.end();

                        frame_count += 1;

                        const source = src_frame.ptr();
                        const dest = dst_frame.ptr();

                        _ = try libav.err(c.av_channel_layout_copy(&dest.ch_layout, &audio_ctx.ch_layout));
                        dest.sample_rate = audio_ctx.sample_rate;
                        dest.format = c.AV_SAMPLE_FMT_FLT;

                        {
                            const swr_z = tracy.Zone.begin(.{ .src = @src(), .name = "swr_convert_frame" });
                            defer swr_z.end();

                            _ = try libav.err(c.swr_convert_frame(swr, dest, source));
                        }

                        // Track stalls when audio buffer is full
                        {
                            const wait_z = tracy.Zone.begin(.{ .src = @src(), .name = "wait for audio buffer", .color = .gray25 });
                            defer wait_z.end();

                            while (true) {
                                if (c.SDL_GetAudioStreamQueued(clock.stream) < max_len) break;
                                sleep(2 * std.time.ns_per_ms);
                            }
                        }

                        {
                            const put_z = tracy.Zone.begin(.{ .src = @src(), .name = "SDL_PutAudioStreamData" });
                            defer put_z.end();

                            const len = c.av_samples_get_buffer_size(
                                null,
                                dest.ch_layout.nb_channels,
                                dest.nb_samples,
                                dest.format,
                                0,
                            );
                            _ = c.SDL_PutAudioStreamData(clock.stream, dest.data[0], len);
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
                    pending = null;
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
    const FilterGraph = struct {
        graph: ?*c.AVFilterGraph,
        src: ?*c.AVFilterContext,
        sink: ?*c.AVFilterContext,

        const PullResult = enum { frame, again, eof };

        // FIXME: don't pass decoder in here?
        // FIXME: Hardware Acceleration

        fn init(decoder: *const Decoder) !@This() {
            var graph = c.avfilter_graph_alloc() orelse return error.out_of_memory;
            errdefer c.avfilter_graph_free(&graph);

            var src: ?*c.AVFilterContext = null;
            var sink: ?*c.AVFilterContext = null;

            const video_ctx = decoder.video_ctx.inner.?;
            const stream = decoder.stream(.video);

            const pixel_aspect: c.AVRational = blk: {
                if (stream.sample_aspect_ratio.den == 0) break :blk .{ .num = 1, .den = 1 };
                break :blk stream.sample_aspect_ratio;
            };

            const time_base: c.AVRational = blk: {
                if (stream.time_base.den == 0) break :blk video_ctx.time_base;
                break :blk stream.time_base;
            };

            var buf: [0x100]u8 = undefined;
            const args = try std.fmt.bufPrintZ(
                &buf,
                "video_size={}x{}:pix_fmt={}:time_base={}/{}:pixel_aspect={}/{}:colorspace={}:range={}",
                .{
                    video_ctx.width,
                    video_ctx.height,
                    c.AV_PIX_FMT_NV12,
                    time_base.num,
                    time_base.den,
                    pixel_aspect.num,
                    pixel_aspect.den,
                    video_ctx.colorspace,
                    video_ctx.color_range,
                },
            );

            const buffersrc = c.avfilter_get_by_name("buffer") orelse return error.ffmpeg_error;
            const buffersink = c.avfilter_get_by_name("buffersink") orelse return error.ffmpeg_error;

            _ = try libav.err(c.avfilter_graph_create_filter(&src, buffersrc, "video_in", args.ptr, null, graph));
            _ = try libav.err(c.avfilter_graph_create_filter(&sink, buffersink, "video_out", null, null, graph));

            switch (decoder.display_rotation) {
                0 => _ = try libav.err(c.avfilter_link(src.?, 0, sink.?, 0)),
                90, 270 => {
                    const arg = if (decoder.display_rotation == 90) "cclock" else "clock";
                    var transpose: ?*c.AVFilterContext = null;
                    const transpose_filter = c.avfilter_get_by_name("transpose") orelse return error.ffmpeg_error;

                    _ = try libav.err(c.avfilter_graph_create_filter(&transpose, transpose_filter, "display_rotate", arg, null, graph));
                    _ = try libav.err(c.avfilter_link(src.?, 0, transpose.?, 0));
                    _ = try libav.err(c.avfilter_link(transpose.?, 0, sink.?, 0));
                },
                180 => {
                    var hflip: ?*c.AVFilterContext = null;
                    var vflip: ?*c.AVFilterContext = null;

                    const hflip_filter = c.avfilter_get_by_name("hflip") orelse return error.ffmpeg_error;
                    const vflip_filter = c.avfilter_get_by_name("vflip") orelse return error.ffmpeg_error;

                    _ = try libav.err(c.avfilter_graph_create_filter(&hflip, hflip_filter, "display_hflip", null, null, graph));
                    _ = try libav.err(c.avfilter_graph_create_filter(&vflip, vflip_filter, "display_vflip", null, null, graph));
                    _ = try libav.err(c.avfilter_link(src.?, 0, hflip.?, 0));
                    _ = try libav.err(c.avfilter_link(hflip.?, 0, vflip.?, 0));
                    _ = try libav.err(c.avfilter_link(vflip.?, 0, sink.?, 0));
                },
                else => return error.unsupported_display_matrix,
            }

            _ = try libav.err(c.avfilter_graph_config(graph, null));

            return .{
                .src = src,
                .sink = sink,
                .graph = graph,
            };
        }

        fn deinit(self: *@This()) void {
            c.avfilter_graph_free(&self.graph);
            self.* = undefined;
        }

        fn push(self: *@This(), frame: ?*c.AVFrame) !void {
            const flags: c_int = if (frame == null) 0 else c.AV_BUFFERSRC_FLAG_KEEP_REF;
            _ = try libav.err(c.av_buffersrc_add_frame_flags(self.src.?, frame, flags));
        }

        fn pull(self: *@This(), frame: *c.AVFrame) !PullResult {
            switch (c.av_buffersink_get_frame(self.sink.?, frame)) {
                0 => return .frame,
                c.AVERROR(c.EAGAIN) => return .again,
                c.AVERROR_EOF => return .eof,
                else => |e| {
                    _ = try libav.err(e);
                    unreachable;
                },
            }
        }
    };

    pub fn decode(decoder: *Decoder) !void {
        const log = std.log.scoped(.video_decode);
        defer log.debug("thread exit", .{});

        tracy.setThreadName("video decode");

        // FIXME: idiomatic?
        const frame_queue = &decoder.queue.frame;
        const pkt_queue = &decoder.queue.pkt.video;
        const codec_ctx = &decoder.video_ctx;

        var src_frame: AvFrame = try .init();
        defer src_frame.deinit();

        var mid_frame: AvFrame = try .init();
        defer mid_frame.deinit();

        var dst_frame: AvFrame = try .init();
        defer dst_frame.deinit();

        const video_ctx = codec_ctx.*.inner.?;

        mid_frame.ptr().width = video_ctx.width;
        mid_frame.ptr().height = video_ctx.height;
        mid_frame.ptr().format = c.AV_PIX_FMT_NV12;
        _ = try libav.err(c.av_frame_get_buffer(mid_frame.ptr(), 32));

        var filter = try FilterGraph.init(decoder);
        defer filter.deinit();

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
        while (!signal.should_quit.load(.monotonic)) {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "decode loop" });
            defer z.end();

            // Try to receive a frame first if we have packets already in the decoder
            recv_loop: while (true) {
                const recv_z = tracy.Zone.begin(.{ .src = @src(), .name = "avcodec_receive_frame" });
                defer recv_z.end();

                switch (c.avcodec_receive_frame(video_ctx, src_frame.ptr())) {
                    0 => {
                        defer c.av_frame_unref(src_frame.ptr());

                        if (decoder.display_rotation == 0) {
                            // common path, skip the AvFilterGraph
                            const dst = frame_queue.acquire() catch |e| if (e != error.early_exit) return e else return;
                            defer frame_queue.commit(dst);

                            try convert(sws, .{ .inner = dst }, src_frame);
                        } else {
                            c.av_frame_unref(mid_frame.ptr());
                            c.av_frame_unref(dst_frame.ptr());

                            try convert(sws, mid_frame, src_frame);
                            try filter.push(mid_frame.ptr());
                            try drainFilter(&filter, frame_queue, dst_frame.ptr());
                        }
                    },
                    c.AVERROR(c.EAGAIN) => break :recv_loop,
                    c.AVERROR_EOF => {
                        try filter.push(null);
                        try drainFilter(&filter, frame_queue, dst_frame.ptr());
                        return frame_queue.end_of_stream.store(true, .monotonic);
                    },
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

                if (send_ret == c.AVERROR_EOF) return frame_queue.end_of_stream.store(true, .monotonic);
                if (send_ret < 0) _ = try libav.err(send_ret);
            }
        }
    }

    fn drainFilter(filter: *FilterGraph, frame_queue: *FrameQueue, frame: *c.AVFrame) !void {
        while (true) {
            c.av_frame_unref(frame);

            switch (try filter.pull(frame)) {
                .frame => {
                    const dst_frame = frame_queue.acquire() catch |e| if (e != error.early_exit) return e else return;
                    defer frame_queue.commit(dst_frame);

                    _ = try libav.err(c.av_frame_copy(dst_frame, frame));
                    _ = try libav.err(c.av_frame_copy_props(dst_frame, frame));
                },
                .again, .eof => return,
            }
        }
    }

    fn convert(sws: *c.SwsContext, dst_frame: AvFrame, src_frame: AvFrame) !void {
        @setRuntimeSafety(false);
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        if (src_frame.ptr().hw_frames_ctx != null) {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "hwframe_transfer_data" });
            defer z.end();

            _ = try libav.err(c.av_hwframe_transfer_data(dst_frame.ptr(), src_frame.ptr(), 0));
        } else {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "sws_scale_frame" });
            defer z.end();

            _ = try libav.err(c.sws_scale_frame(sws, dst_frame.ptr(), src_frame.ptr()));
        }

        // timing
        dst_frame.ptr().pts = src_frame.ptr().pts;

        dst_frame.ptr().colorspace = src_frame.ptr().colorspace;
        dst_frame.ptr().color_range = src_frame.ptr().color_range;
        dst_frame.ptr().color_primaries = src_frame.ptr().color_primaries;
        dst_frame.ptr().color_trc = src_frame.ptr().color_trc;
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

    mutex: TracyMutex,
    cond: std.Thread.Condition = .{},

    const Slot = struct { frame: []c.AVFrame, state: []State };
    const State = enum { empty, in_use, ready_to_reuse, writing };
    const Error = error{ invalid_size, early_exit };
    pub const InitError = std.mem.Allocator.Error || libav.Error;

    pub const capacity = 0x20;

    // INVARIANT: This is an SPSC Queue

    pub fn init(allocator: std.mem.Allocator, opt: Resolution) InitError!FrameQueue {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "FrameQueue.init" });
        defer zone.end();

        comptime std.debug.assert(std.math.isPowerOfTwo(capacity));

        const frames = try allocator.alloc(c.AVFrame, capacity);
        errdefer allocator.free(frames);
        errdefer for (frames) |*frame| c.av_frame_unref(frame); // imagine we get through only some av_frame_get_buffers

        const states = try allocator.alloc(State, capacity);
        errdefer allocator.free(states);

        for (frames, states) |*frame, *state| {
            frame.* = std.mem.zeroes(c.AVFrame);
            c.av_frame_unref(frame);

            frame.width = opt.width;
            frame.height = opt.height;
            frame.format = c.AV_PIX_FMT_NV12;

            _ = try libav.err(c.av_frame_get_buffer(frame, 32));

            state.* = .empty;
        }

        return .{
            .mutex = .init(@src(), "FrameQueue"),
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
                if (signal.should_quit.load(.monotonic)) return error.early_exit;
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
        self.cond.signal(); // wake up consumer in pop
    }

    pub fn pop(self: *@This()) ?*c.AVFrame {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const idx = self.mask(self.read_idx);

        self.mutex.lock();
        defer self.mutex.unlock();

        {
            const z = tracy.Zone.begin(.{ .src = @src(), .name = "wait for frame", .color = .gray25 });
            defer z.end();

            while (self.slot.state[idx] != .in_use) {
                if (self.end_of_stream.load(.monotonic)) return null;
                if (signal.should_quit.load(.monotonic)) return null;

                self.mutex.wait(&self.cond);
            }
        }

        self.slot.state[idx] = .ready_to_reuse;
        self.read_idx += 1;

        return &self.slot.frame[idx];
    }

    pub fn tryPop(self: *@This()) ?*c.AVFrame {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const idx = self.mask(self.read_idx);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.slot.state[idx] != .in_use) return null;

        self.slot.state[idx] = .ready_to_reuse;
        self.read_idx += 1;

        return &self.slot.frame[idx];
    }

    /// you *must* not call recycle() on this AvFrame
    pub fn peek(self: *@This()) ?*c.AVFrame {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const idx = self.mask(self.read_idx);

        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.slot.state[idx] != .in_use) return null;

        return &self.slot.frame[idx];
    }

    pub fn recycle(self: *@This(), used_frame: *c.AVFrame) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        // FIXME: can i do this using slices or something
        const diff = @intFromPtr(used_frame) - @intFromPtr(self.slot.frame.ptr);
        const idx = diff / @sizeOf(c.AVFrame);

        self.mutex.lock();
        defer self.mutex.unlock();

        std.debug.assert(idx < self.slot.frame.len);
        std.debug.assert(self.slot.state[idx] == .ready_to_reuse);

        self.slot.state[idx] = .empty;

        // Wake up the producer in acquire()
        self.cond.signal();
    }

    pub fn interrupt(self: *@This()) void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        self.mutex.lock();
        defer self.mutex.unlock();

        self.cond.broadcast();
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

    pub const InitError = error{
        ffmpeg_error,
        unsupported_display_matrix,
        unsupported_colour_depth,
    } || std.mem.Allocator.Error || FrameQueue.InitError || AudioClock.InitError || std.Thread.SpawnError;

    fmt_ctx: dec.AvFormatContext,

    video_ctx: *dec.AvCodecContext,
    audio_ctx: *dec.AvCodecContext,

    queue: Queues,

    audio_clock: ?AudioClock,

    // Important Information
    colour_space: c.AVColorSpace,
    resolution: Resolution,

    display_rotation: i16 = 0,

    pub const Queues = struct {
        frame: FrameQueue,
        pkt: struct { audio: PacketQueue, video: PacketQueue },

        fn deinit(self: *Queues, allocator: std.mem.Allocator) void {
            // wake up the mutexes so they can check for should_quit
            self.pkt.audio.interrupt();
            self.pkt.video.interrupt();
            self.frame.interrupt();

            self.pkt.audio.deinit(allocator);
            self.pkt.video.deinit(allocator);
            self.frame.deinit(allocator);
        }
    };

    pub const Handles = struct {
        pkt: std.Thread,
        audio: ?std.Thread,
        video: std.Thread,

        pub fn deinit(self: Handles) void {
            const zone = tracy.Zone.begin(.{ .src = @src(), .name = "Handles.deinit" });
            defer zone.end();

            self.pkt.join();
            self.video.join();

            if (self.audio) |a| a.join();
        }
    };

    const log = std.log.scoped(.decode);

    fn normalizeRotation(angle: f64) !i16 {
        const snapped = @as(i32, @intFromFloat(@round(angle / 90.0) * 90.0));
        if (@abs(angle - @as(f64, @floatFromInt(snapped))) > 1.0) return error.unsupported_display_matrix;

        var normalized = @mod(snapped, 360);
        if (normalized < 0) normalized += 360;

        return @intCast(normalized);
    }

    // TODO: DecoderOptions
    pub fn init(self: *Decoder, allocator: std.mem.Allocator, hw_device: c.AVHWDeviceType, volume: f32, path: []const u8, headless: bool) InitError!void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "Decoder.init" });
        defer zone.end();

        var fmt_ctx = try dec.AvFormatContext.init(path);
        errdefer fmt_ctx.deinit();

        const video_ctx = try allocator.create(dec.AvCodecContext);
        errdefer allocator.destroy(video_ctx);

        try video_ctx.init(.video, fmt_ctx, .{ .dev_type = hw_device });
        errdefer video_ctx.deinit();

        video_ctx.dump(fmt_ctx);

        const audio_ctx = try allocator.create(dec.AvCodecContext);
        errdefer allocator.destroy(audio_ctx);

        try audio_ctx.init(.audio, fmt_ctx, .{});
        errdefer audio_ctx.deinit();

        var display_rotation: i16 = 0;

        {
            const st: ?*c.AVStream = fmt_ctx.ptr().streams[@intCast(video_ctx.stream)];
            const data = st.?.codecpar.*.coded_side_data;
            const count: usize = @intCast(st.?.codecpar.*.nb_coded_side_data);

            // FIXME(paoda): be a little bit more intentional about how this looks in the terminal
            for (0..count) |i| {
                const ptr = &data[i];
                log.debug("{}: {s}", .{ i, c.av_packet_side_data_name(ptr.type) });

                switch (ptr.type) {
                    c.AV_PKT_DATA_DISPLAYMATRIX => {
                        const angle = c.av_display_rotation_get(@ptrCast(@alignCast(ptr.data)));
                        display_rotation = try normalizeRotation(angle);

                        log.debug("display matrix angle: {d:.2}", .{angle});
                        log.debug("display matrix snapped rotation: {}", .{display_rotation});
                    },
                    else => log.debug("found {s} side_data", .{c.av_packet_side_data_name(ptr.type)}),
                }
            }
        }

        const resolution: Resolution = blk: {
            const ctx = video_ctx.inner.?;

            if (display_rotation == 90 or display_rotation == 270)
                break :blk .{ .width = ctx.height, .height = ctx.width };

            break :blk .{ .width = ctx.width, .height = ctx.height };
        };

        // FIXME(paoda): this could be prettier
        log.debug(
            "display transform: angle={} pre={}x{} post={f}",
            .{
                display_rotation,
                video_ctx.inner.?.width,
                video_ctx.inner.?.height,
                resolution,
            },
        );

        if (resolution.width < resolution.height) {
            errors.add_portrait_err(path, resolution);
        }

        const sw_fmt = fmt_ctx.ptr().streams[@intCast(video_ctx.stream)].*.codecpar.*.format;
        switch (sw_fmt) {
            // TODO(paoda): support 10-bit
            // TODO(paoda): communicate this to the user
            c.AV_PIX_FMT_YUV420P10LE, c.AV_PIX_FMT_P010LE => return error.unsupported_colour_depth,
            else => {},
        }

        var frame_queue = try FrameQueue.init(allocator, .{
            .width = resolution.width,
            .height = resolution.height,
        });
        errdefer frame_queue.deinit(allocator);

        var video_queue = PacketQueue.init(allocator, "Video PacketQueue");
        errdefer video_queue.deinit(allocator);

        var audio_queue = PacketQueue.init(allocator, "Audio PacketQueue");
        errdefer audio_queue.deinit(allocator);

        const audio_clock: ?AudioClock = if (headless) null else try AudioClock.init(audio_ctx, volume);
        errdefer if (audio_clock) |clock| clock.deinit();

        self.* = .{
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
            .resolution = resolution,
            .display_rotation = display_rotation,
        };
    }

    pub fn deinit(self: *Decoder, allocator: std.mem.Allocator) void {
        const zone = tracy.Zone.begin(.{ .src = @src(), .name = "Decoder.deinit" });
        defer zone.end();

        if (self.audio_clock) |clock| clock.deinit();
        self.queue.deinit(allocator);

        self.audio_ctx.deinit();
        allocator.destroy(self.audio_ctx);

        self.video_ctx.deinit();
        allocator.destroy(self.video_ctx);

        self.fmt_ctx.deinit();
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

    pub fn duration(self: Decoder) f64 {
        const st = self.stream(.video);
        const end_pts: f64 = @floatFromInt(st.start_time + st.duration);

        return end_pts * c.av_q2d(st.time_base);
    }

    pub fn spawn(self: *Decoder, headless: bool) std.Thread.SpawnError!Handles {
        const pkt_handle = try std.Thread.spawn(.{}, packet.read, .{self});
        const video_handle = try std.Thread.spawn(.{}, video.decode, .{self});
        const audio_handle = if (headless) null else try std.Thread.spawn(.{}, audio.decode, .{self});

        return .{
            .pkt = pkt_handle,
            .video = video_handle,
            .audio = audio_handle,
        };
    }
};

pub const Encoder = struct {
    resolution: Resolution,

    codec_ctx: enc.AvCodecContext,
    fmt_ctx: enc.AvFormatContext,

    sws_ctx: ?*c.SwsContext,

    audio_stream: enc.AvStream,
    video_stream: enc.AvStream,

    _pkt: AvPacket,
    _frame: AvFrame, // points to NV12 Data from OpenGL

    // TODO: dont' always allocate this?
    _sw_frame: AvFrame, // for use in software encoding only

    _hw: ?struct { frame: AvFrame } = null,

    // guards against duplicate video DTS, which are supposed to be monotonic
    last_video_dts: ?i64 = null,

    const log = std.log.scoped(.encode);

    pub const Options = struct {
        encode_view: Viewport,
        bit_rate: c_int,

        decoder: *const Decoder,
    };

    pub fn init(self: *Encoder, opt: Options, device_type: ?c.AVHWDeviceType, path: []const u8) !void {
        const codec = c.AV_CODEC_ID_HEVC;

        if (device_type) |dev| {
            const dev_str = c.av_hwdevice_get_type_name(dev);

            if (self.initHardware(opt, dev, codec, path)) {
                return log.info("init {s} hevc encoder", .{dev_str});
            } else |_| {
                log.warn("failed to init {s} hevc encoder, trying h264", .{dev_str});
            }

            if (self.initHardware(opt, dev, c.AV_CODEC_ID_H264, path)) {
                return log.info("init {s} h264 encoder", .{dev_str});
            } else |e| {
                errors.add_encoding_fallback_err(dev_str, c.AV_CODEC_ID_H264, e);
            }
        }

        try self.initSoftware(opt, codec, path);
    }

    pub fn deinit(self: *Encoder) void {
        self.writeVideoFrame(null) catch @panic("failed to flush encoder");
        _ = c.av_write_trailer(self.fmt_ctx.ptr());

        self._frame.deinit();
        self._sw_frame.deinit();

        if (self._hw) |*hw| hw.frame.deinit();
        self._pkt.deinit();

        self.codec_ctx.deinit();
        self.fmt_ctx.deinit();

        self.* = undefined;
    }

    fn initShared(self: *Encoder, opt: Options, codec: enc.AvCodec, sw_pix_fmt: c.AVPixelFormat, path: []const u8) !void {
        var fmt_ctx = try enc.AvFormatContext.init(path);
        errdefer fmt_ctx.deinit();

        const encode_resolution = opt.encode_view.resolution();
        const width = encode_resolution.width;
        const height = encode_resolution.height;

        var codec_ctx = try enc.AvCodecContext.init(codec, fmt_ctx, .{
            .resolution = encode_resolution,
            .bit_rate = opt.bit_rate,
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

        self.* = .{
            .resolution = encode_resolution,
            .codec_ctx = codec_ctx,
            .fmt_ctx = fmt_ctx,

            .audio_stream = audio_stream,
            .video_stream = video_stream,

            ._pkt = try AvPacket.try_init(),
            ._frame = blk: {
                var frame = try AvFrame.init();
                errdefer frame.deinit();

                const ptr = frame.ptr();
                ptr.width = encode_resolution.width;
                ptr.height = encode_resolution.height;
                ptr.format = c.AV_PIX_FMT_NV12;

                ptr.linesize[0] = AvFrame.alignUp(encode_resolution.width, 32);
                ptr.linesize[1] = ptr.linesize[0];

                ptr.color_range = c.AVCOL_RANGE_MPEG;
                ptr.color_primaries = c.AVCOL_PRI_BT709;
                ptr.color_trc = c.AVCOL_TRC_BT709;
                ptr.colorspace = c.AVCOL_SPC_BT709;

                break :blk frame;
            },
            ._sw_frame = blk: {
                var frame = try AvFrame.init();
                errdefer frame.deinit();

                const ptr = frame.ptr();
                ptr.width = encode_resolution.width;
                ptr.height = encode_resolution.height;
                ptr.format = sw_pix_fmt;

                _ = try libav.err(c.av_frame_get_buffer(ptr, 32));
                break :blk frame;
            },
            ._hw = blk: {
                if (codec.hw == null) break :blk null;
                if (codec_ctx.ptr().hw_frames_ctx == null) break :blk null;
                const frame = try AvFrame.init();

                break :blk .{ .frame = frame };
            },
            .sws_ctx = blk: {
                const ptr: ?*c.SwsContext = c.sws_getContext(
                    width,
                    height,
                    c.AV_PIX_FMT_NV12,
                    width,
                    height,
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

    fn initHardware(self: *Encoder, opt: Options, device_type: c.AVHWDeviceType, codec_id: c.AVCodecID, path: []const u8) !void {
        const codec = enc.AvCodec.findHardware(device_type, codec_id) orelse return error.missing_encoder;
        try self.initShared(opt, codec, c.AV_PIX_FMT_NV12, path);
    }

    fn initSoftware(self: *Encoder, opt: Options, codec_id: c.AVCodecID, path: []const u8) !void {
        const codec = enc.AvCodec.findSoftware(codec_id);
        try self.initShared(opt, codec, codec.pix_fmt, path);
    }

    pub fn encodeNv12Frame(self: *Encoder, y_buf: []const u8, uv_buf: []const u8, frame_pts: i64) !void {
        const zone = tracy.Zone.begin(.{ .src = @src() });
        defer zone.end();

        const src_frame = self._frame.ptr();
        _ = c.av_frame_unref(self._frame.ptr());

        src_frame.width = self.resolution.width;
        src_frame.height = self.resolution.height;
        src_frame.format = c.AV_PIX_FMT_NV12;
        src_frame.pts = frame_pts;

        try self._frame.setYData(y_buf);
        try self._frame.setUvData(uv_buf);

        if (self._hw) |hw| {
            const hw_frame = hw.frame.ptr();
            c.av_frame_unref(hw_frame);

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "av_hwframe_get_buffer" });
                defer z.end();

                _ = try libav.err(c.av_hwframe_get_buffer(self.codec_ctx.ptr().hw_frames_ctx, hw_frame, 0));
            }
            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "av_hwframe_transfer_data" });
                defer z.end();

                _ = try libav.err(c.av_hwframe_transfer_data(hw_frame, src_frame, 0));
            }

            _ = try libav.err(c.av_frame_copy_props(hw_frame, src_frame));

            try self.writeVideoFrame(hw_frame);
        } else {
            const sw_frame = self._sw_frame.ptr();
            _ = try libav.err(c.av_frame_make_writable(sw_frame));

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "sws_scale_frame" });
                defer z.end();

                _ = try libav.err(c.sws_scale_frame(self.sws_ctx, sw_frame, src_frame));
                sw_frame.pts = src_frame.pts;
            }

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

            // some encoders can produce duplicate DTS when working with time_base and weird
            // framerates, we have to guard against this by ensuring that all DTS are monotonic and increasing
            if (self.last_video_dts) |dts| {
                if (pkt.dts <= dts) {
                    pkt.dts = dts + 1;
                    pkt.pts = @max(pkt.pts, pkt.dts);
                }
            }
            self.last_video_dts = pkt.dts;

            {
                const z = tracy.Zone.begin(.{ .src = @src(), .name = "av_interleaved_write_frame" });
                defer z.end();

                _ = try libav.err(c.av_interleaved_write_frame(self.fmt_ctx.ptr(), pkt));
            }
        }
    }
};
