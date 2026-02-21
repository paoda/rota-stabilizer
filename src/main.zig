const std = @import("std");
const builtin = @import("builtin");
const gl = @import("gl");
const zstbi = @import("zstbi");

const c = @import("lib.zig").c;
const video = @import("lib/codec.zig").video;
const audio = @import("lib/codec.zig").audio;
const packet = @import("lib/codec.zig").packet;
const platform = @import("lib/platform.zig");
const Encoder = @import("lib/codec.zig").Encoder;

const GpuResourceManager = @import("lib.zig").GpuResourceManager;
const BlurManager = @import("lib.zig").BlurManager;

const Ui = @import("lib/platform.zig").Ui;

const PacketQueue = @import("lib/codec.zig").packet.Queue;
const AudioClock = @import("lib/codec.zig").audio.Clock;
const FrameQueue = @import("lib/codec.zig").FrameQueue;
const DecodeContext = @import("lib/codec.zig").DecodeContext;

const dec = @import("lib/libav.zig").dec;
const AvFrame = @import("lib/libav.zig").AvFrame;
const AvPacket = @import("lib/libav.zig").AvPacket;

const Mat2 = @import("lib/math.zig").Mat2;
const Vec2 = @import("lib/math.zig").Vec2;
const mat2 = @import("lib/math.zig").mat2;
const vec2 = @import("lib/math.zig").vec2;

const sleep = @import("lib.zig").sleep;

const RGB24_BPP = @import("lib.zig").RGB24_BPP;

// ===== MODE SWITCH =====
// Set to true to render to file instead of realtime playback
const RENDER_MODE = true;

const Y_BPP = @import("lib.zig").Y_BPP;
const UV_BPP = @import("lib.zig").UV_BPP;

/// set to enable hardware decoding
const hw_device: ?c.AVHWDeviceType = switch (builtin.os.tag) {
    .linux => c.AV_HWDEVICE_TYPE_VAAPI,
    .windows => c.AV_HWDEVICE_TYPE_D3D11VA,
    // .macos => c.AV_HWDEVICE_TYPE_VIDEOTOOLBOX,
    else => null, // TODO: maybe use c.AV_HWDEVICE_TYPE_VULKAN on everything?
};

pub fn main() !void {
    const log = std.log.scoped(.ui);
    errdefer |err| if (err == error.sdl_error) log.err("SDL Error: {s}", .{c.SDL_GetError()});

    var gpa: std.heap.DebugAllocator(.{}) = .{ .backing_allocator = std.heap.c_allocator };
    defer std.debug.assert(gpa.deinit() == .ok);

    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();

    _ = args.skip(); // self
    const path = args.next() orelse return error.missing_file;

    // TODO: move this to Decoder struct

    var fmt_ctx = try dec.AvFormatContext.init(path);
    defer fmt_ctx.deinit();

    var video_ctx = try dec.AvCodecContext.init(allocator, .video, fmt_ctx, .{ .dev_type = hw_device });
    defer video_ctx.deinit(allocator);

    var audio_ctx = try dec.AvCodecContext.init(allocator, .audio, fmt_ctx, .{});
    defer audio_ctx.deinit(allocator);

    const vid_width: u32 = @intCast(video_ctx.inner.?.width);
    const vid_height: u32 = @intCast(video_ctx.inner.?.height);

    const ui = if (RENDER_MODE) try platform.createHeadless(1920, 1080) else try platform.createWindow(800, 800);
    defer ui.deinit();

    var audio_clock = try AudioClock.init(audio_ctx);
    defer audio_clock.deinit();

    // -- opengl --
    log.info("OpenGL device: {?s}", .{gl.GetString(gl.RENDERER)});
    log.info("OpenGL support (want 3.3): {?s}", .{gl.GetString(gl.VERSION)});

    gl.Enable(gl.BLEND);
    gl.BlendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.Disable(gl.FRAMEBUFFER_SRGB);
    _ = c.SDL_GL_SetSwapInterval(0);

    var res = try GpuResourceManager.init(allocator, vid_width, vid_height);
    defer res.deinit();

    const aspect = @as(f32, @floatFromInt(vid_width)) / @as(f32, @floatFromInt(vid_height));

    {
        const gcd = std.math.gcd(vid_width, vid_height);
        log.debug("Resolution: {}x{}", .{ vid_width, vid_height });
        log.debug("Aspect Ratio: {}:{} | {d:.5}", .{ vid_width / gcd, vid_height / gcd, aspect });
    }

    var stable_buffer = DoubleBuffer.init(res);
    const angle_calc = try AngleCalc.init(res, video_ctx.inner.?.colorspace);

    // -- opengl end --
    var queue = try FrameQueue.init(allocator, 0x80); // 1s at 120fps?
    defer queue.deinit(allocator);

    var should_quit: std.atomic.Value(bool) = .init(false);

    var video_pkt_queue = PacketQueue.init(allocator);
    defer video_pkt_queue.deinit(allocator);

    var audio_pkt_queue = PacketQueue.init(allocator);
    defer audio_pkt_queue.deinit(allocator);

    const read_opts: packet.ReadOptions = .{ .video_stream = video_ctx.stream, .audio_stream = audio_ctx.stream, .should_quit = &should_quit };

    const vid_decode: DecodeContext = .{ .codec_ctx = video_ctx, .fmt_ctx = fmt_ctx, .should_quit = &should_quit };
    const aud_decode: DecodeContext = .{ .codec_ctx = audio_ctx, .fmt_ctx = fmt_ctx, .should_quit = &should_quit };

    const pkt_handle = try std.Thread.spawn(.{}, packet.read, .{ &video_pkt_queue, &audio_pkt_queue, fmt_ctx, read_opts });
    defer pkt_handle.join();

    const video_decode = try std.Thread.spawn(.{}, video.decode, .{ &queue, &video_pkt_queue, vid_decode });
    defer video_decode.join();

    // In render mode, we consume audio packets directly instead of decoding for playback
    const audio_decode: ?std.Thread = if (!RENDER_MODE) try std.Thread.spawn(.{}, audio.decode, .{ &audio_clock, &audio_pkt_queue, aud_decode }) else null;
    defer if (audio_decode) |h| h.join();

    const w, const h = try ui.windowSize();
    var camera = Camera.init(vid_width, vid_height, w, h, angle_calc.colour_space);

    log.debug("colour space: {s}", .{c.av_color_space_name(video_ctx.inner.?.colorspace)});
    log.debug("colour range: {s}", .{c.av_color_range_name(video_ctx.inner.?.color_range)});
    log.debug("colour primaries: {s}", .{c.av_color_transfer_name(video_ctx.inner.?.color_trc)});

    var view = Viewport.init(w, h);
    const frame_rate = fmt_ctx.ptr().streams[@intCast(video_ctx.stream)].*.avg_frame_rate;
    const frame_period = 1.0 / c.av_q2d(frame_rate);

    // A/V Sync Debug State
    var frame_count: u64 = 0;

    if (RENDER_MODE) {
        // Initialize encoder
        var encoder = try Encoder.init(.{
            .width = @intCast(view.width),
            .height = @intCast(view.height),
            .input = .{
                .fmt_ctx = fmt_ctx,
                .audio_ctx = audio_ctx,
                .video_ctx = video_ctx,
            },
        });
        defer encoder.deinit();

        // Initialize encoder
        // var encoder = try Encoder.init(RENDER_OUTPUT_PATH, .{
        //     .width = @intCast(view.width),
        //     .height = @intCast(view.height),
        //     .fps = frame_rate,
        //     .input_fmt_ctx = fmt_ctx.ptr(),
        //     .input_video_stream = video_ctx.stream,
        //     .input_audio_stream = audio_ctx.stream,
        // });
        // defer encoder.deinit();

        const render_start_time = c.SDL_GetPerformanceCounter();
        const perf_freq: f64 = @floatFromInt(c.SDL_GetPerformanceFrequency());

        // Get estimated total frames for progress
        const video_stream = fmt_ctx.ptr().streams[@intCast(video_ctx.stream)];
        const audio_stream = fmt_ctx.ptr().streams[@intCast(audio_ctx.stream)];

        const estimated_frames: usize = blk: {
            // Try nb_frames first
            if (video_stream.*.nb_frames > 0) {
                break :blk @intCast(video_stream.*.nb_frames);
            }
            // Fall back to estimating from duration
            if (fmt_ctx.ptr().duration > 0) {
                const duration_secs = @as(f64, @floatFromInt(fmt_ctx.ptr().duration)) / @as(f64, c.AV_TIME_BASE);
                const fps = c.av_q2d(frame_rate);
                break :blk @intFromFloat(duration_secs * fps);
            }
            break :blk 0; // Unknown
        };

        // Initialize progress
        var progress_buffer: [256]u8 = undefined;
        const progress_node = std.Progress.start(.{
            .root_name = "Encoding",
            .estimated_total_items = estimated_frames,
            .draw_buffer = &progress_buffer,
        });
        defer progress_node.end();

        // Setup PBO double-buffering for async readback
        // This overlaps GPU→CPU transfer with encoding of the previous frame
        const rgb_size: isize = @intCast(@as(usize, @intCast(view.width)) * @as(usize, @intCast(view.height)) * RGB24_BPP);
        var readback_pbos: [2]c_uint = undefined;
        gl.GenBuffers(2, &readback_pbos);
        defer gl.DeleteBuffers(2, &readback_pbos);

        // Initialize both PBOs with the same size
        for (readback_pbos) |pbo| {
            gl.BindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
            gl.BufferData(gl.PIXEL_PACK_BUFFER, rgb_size, null, gl.STREAM_READ);
        }
        gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);

        var current_pbo: u1 = 0;
        var pending_frame_pts: ?i64 = null; // PTS of frame in the "previous" PBO waiting to be encoded

        while (!should_quit.load(.monotonic)) {
            // No event polling needed - window is hidden in render mode

            // Process any pending audio packets (remux to output)
            while (audio_pkt_queue.tryPop()) |audio_pkt| {
                defer {
                    var pkt: ?*c.AVPacket = audio_pkt;
                    c.av_packet_free(&pkt);
                }

                try encoder.writeAudioPacket(audio_stream, audio_pkt);
            }

            if (queue.pop()) |frame| {
                defer queue.recycle(frame);
                defer stable_buffer.swap();

                if (frame.format != c.AV_PIX_FMT_NV12) {
                    @branchHint(.cold);
                    const expected = c.av_get_pix_fmt_name(c.AV_PIX_FMT_NV12);
                    const actual = c.av_get_pix_fmt_name(frame.format);
                    log.err("unsupported pixel format: expected {s} got {s}", .{ expected, actual });
                    return error.ffmpeg_error;
                }

                frame_count += 1;

                // Upload Y plane to GPU
                {
                    gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, stable_buffer.pbo(.y));
                    defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

                    gl.BindTexture(gl.TEXTURE_2D, stable_buffer.tex(.y));
                    defer gl.BindTexture(gl.TEXTURE_2D, 0);

                    const pbo: ?[*]u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_UNPACK_BUFFER, gl.WRITE_ONLY));

                    if (pbo) |ptr| {
                        const bytes_per_line: usize = @intCast(frame.linesize[0]);
                        const height: usize = @intCast(frame.height);
                        const dst_stride = @as(usize, @intCast(frame.width * Y_BPP));

                        for (0..height) |y| {
                            const src_offset = y * bytes_per_line;
                            const dst_offset = y * dst_stride;
                            @memcpy(ptr[dst_offset..][0..dst_stride], frame.data[0][src_offset..][0..dst_stride]);
                        }
                        _ = gl.UnmapBuffer(gl.PIXEL_UNPACK_BUFFER);
                    }

                    gl.PixelStorei(gl.UNPACK_ALIGNMENT, 1);
                    gl.TexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, frame.width, frame.height, gl.RED, gl.UNSIGNED_BYTE, null);
                }

                // Upload UV plane to GPU
                {
                    gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, stable_buffer.pbo(.uv));
                    defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

                    gl.BindTexture(gl.TEXTURE_2D, stable_buffer.tex(.uv));
                    defer gl.BindTexture(gl.TEXTURE_2D, 0);

                    const pbo: ?[*]u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_UNPACK_BUFFER, gl.WRITE_ONLY));

                    if (pbo) |ptr| {
                        const bytes_per_line: usize = @intCast(frame.linesize[1]);
                        const height: usize = @intCast(@divTrunc(frame.height, 2));
                        const dst_stride = @as(usize, @intCast(@divTrunc(frame.width, 2) * UV_BPP));

                        for (0..height) |y| {
                            const src_offset = y * bytes_per_line;
                            const dst_offset = y * dst_stride;
                            @memcpy(ptr[dst_offset..][0..dst_stride], frame.data[1][src_offset..][0..dst_stride]);
                        }
                        _ = gl.UnmapBuffer(gl.PIXEL_UNPACK_BUFFER);
                    }

                    gl.PixelStorei(gl.UNPACK_ALIGNMENT, 2);
                    gl.TexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, @divTrunc(frame.width, 2), @divTrunc(frame.height, 2), gl.RG, gl.UNSIGNED_BYTE, null);
                }

                // Render the frame
                try render(&view, &stable_buffer, angle_calc, res, camera);

                // Start async readback into current PBO
                gl.BindBuffer(gl.PIXEL_PACK_BUFFER, readback_pbos[current_pbo]);
                gl.ReadPixels(0, 0, view.width, view.height, gl.RGB, gl.UNSIGNED_BYTE, null);
                gl.Flush(); // Ensure readback starts immediately (don't wait for glMapBuffer to trigger it)

                // If we have a pending frame in the other PBO, map and encode it
                if (pending_frame_pts) |pts| {
                    const prev_pbo = current_pbo +% 1;
                    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, readback_pbos[prev_pbo]);

                    const mapped: ?[*]const u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY));
                    if (mapped) |rgb_data| {
                        // try encoder.encodeRgbFrame(rgb_data[0..@intCast(rgb_size)], pts);
                        try encoder.encodeRgbFrame(rgb_data[0..@intCast(rgb_size)], pts);
                        _ = gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER);
                    }
                }
                gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);

                // Store current frame's PTS for next iteration, swap PBOs
                pending_frame_pts = frame.best_effort_timestamp;
                current_pbo +%= 1;

                // Update progress
                progress_node.setCompletedItems(frame_count);

                // No need to swap in headless render mode
            } else if (queue.end_of_stream.load(.monotonic) and queue.isEmpty()) {
                // Encode the last pending frame
                if (pending_frame_pts) |pts| {
                    const prev_pbo = current_pbo +% 1;
                    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, readback_pbos[prev_pbo]);

                    const mapped: ?[*]const u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_PACK_BUFFER, gl.READ_ONLY));
                    if (mapped) |rgb_data| {
                        // try encoder.encodeRgbFrame(rgb_data[0..@intCast(rgb_size)], pts);
                        try encoder.encodeRgbFrame(rgb_data[0..@intCast(rgb_size)], pts);
                        _ = gl.UnmapBuffer(gl.PIXEL_PACK_BUFFER);
                    }
                    gl.BindBuffer(gl.PIXEL_PACK_BUFFER, 0);
                }

                // End of video
                progress_node.setCompletedItems(frame_count);
                std.Progress.setStatus(.success);

                // Calculate and display final stats
                const elapsed: f64 = @as(f64, @floatFromInt(c.SDL_GetPerformanceCounter() - render_start_time)) / perf_freq;
                const video_duration = @as(f64, @floatFromInt(frame_count)) / c.av_q2d(frame_rate);
                const speed = video_duration / elapsed;
                log.info("Finished rendering {} frames in {d:.1}s ({d:.2}x realtime)", .{ frame_count, elapsed, speed });
                break;
            }
        }

        // Signal threads to quit (in case they're blocked waiting)
        should_quit.store(true, .monotonic);
        video_pkt_queue.quit();
        audio_pkt_queue.quit();
    } else {
        // ===== PLAYBACK MODE =====
        if (@import("builtin").mode == .Debug) {
            const drop_behind_ms = @max(0.025, frame_period * 2.0) * std.time.ms_per_s;
            const delay_ahead_ms = @max(0.012, frame_period * 1.5) * std.time.ms_per_s;
            log.info("Frame period: {d:.3}ms ({d:.2} fps)", .{ frame_period * std.time.ms_per_s, 1.0 / frame_period });
            log.info("Sync thresholds: drop_behind={d:.1}ms (~{d:.1} frames), delay_ahead={d:.1}ms (~{d:.1} frames), max_delay=500ms", .{
                drop_behind_ms,
                drop_behind_ms / (frame_period * std.time.ms_per_s),
                delay_ahead_ms,
                delay_ahead_ms / (frame_period * std.time.ms_per_s),
            });
            log.info("audio hw_latency: {d:.2}ms", .{audio_clock.hw_latency_secs * std.time.ms_per_s});
        }

        while (!should_quit.load(.monotonic)) {
            var event: c.SDL_Event = undefined;

            while (c.SDL_PollEvent(&event)) {
                switch (event.type) {
                    c.SDL_EVENT_QUIT => {
                        should_quit.store(true, .monotonic);
                        // Wake up threads blocked on queue pop
                        video_pkt_queue.quit();
                        audio_pkt_queue.quit();
                    },
                    c.SDL_EVENT_MOUSE_WHEEL => {
                        const wheel_delta = event.wheel.y * 0.1;
                        camera.adjustZoom(wheel_delta);
                        log.debug("Zoom: {d:.2}x", .{camera.zoom});
                    },
                    c.SDL_EVENT_WINDOW_RESIZED => {
                        view = Viewport.init(event.window.data1, event.window.data2);
                        camera.updateWindow(view.width, view.height);

                        log.debug("Window resized to {}x{}", .{ view.width, view.height });
                    },
                    c.SDL_EVENT_KEY_UP => switch (event.key.scancode) {
                        c.SDL_SCANCODE_M => {
                            try if (audio_clock.is_muted) audio_clock.unmute() else audio_clock.mute();
                        },
                        c.SDL_SCANCODE_L => {
                            // Debug: print world vs window viewport info
                            const world_bounds = camera.world_bounds;
                            log.debug("World bounds: {d:.1}x{d:.1}, Window: {}x{}", .{ world_bounds.x(), world_bounds.y(), camera.window_size[0], camera.window_size[1] });
                        },
                        c.SDL_SCANCODE_F11 => try ui.toggleFullscreen(),
                        else => {},
                    },
                    else => {},
                }
            }

            if (queue.pop()) |frame| {
                defer queue.recycle(frame);
                defer stable_buffer.swap();

                // Calculate PTS early - needed for initial sync
                const time_base: f64 = c.av_q2d(fmt_ctx.ptr().streams[@intCast(video_ctx.stream)].*.time_base);
                const pt_in_seconds = @as(f64, @floatFromInt(frame.best_effort_timestamp)) * time_base;

                // Track if this is the first frame (for delayed audio start)
                const is_first_frame = audio_clock.isPaused();
                const first_frame_pts: f64 = if (is_first_frame) pt_in_seconds else 0.0;

                if (frame.format != c.AV_PIX_FMT_NV12) {
                    @branchHint(.cold);

                    const expected = c.av_get_pix_fmt_name(c.AV_PIX_FMT_NV12);
                    const actual = c.av_get_pix_fmt_name(frame.format);
                    log.err("unsupported pixel format: expected {s} got {s}", .{ expected, actual });
                    return error.ffmpeg_error;
                }

                // FIXME: we currently assume colour space. We can't do that.

                // A/V sync thresholds - tuned for both high refresh (120fps) and standard (60fps) content
                // drop_behind: How far audio can get ahead before dropping video frames
                //   - 2 frames gives some tolerance without visible desync
                const drop_behind = @max(0.025, frame_period * 2.0);
                // delay_ahead: How far video can get ahead before we delay
                //   - 1.5 frames allows natural jitter without excessive delays
                const delay_ahead = @max(0.012, frame_period * 1.5);
                // max_delay: Cap on sleep time to handle PTS discontinuities
                //   - 500ms handles jumps without freezing UI too long
                const max_delay = 0.500;
                // desync_reset: Major desync threshold - skip frame entirely
                const desync_reset = 2.0;

                stable_buffer.set_display_time(pt_in_seconds);

                // Skip A/V sync on first frame - audio hasn't started yet
                if (is_first_frame) {
                    // Just render the first frame without sync logic
                    frame_count += 1;
                } else {
                    // A/V sync: compare the frame being DISPLAYED (previous frame from invert()) to audio clock
                    // Double-buffer pattern: we upload to current slot, but display from inverted slot
                    const audio_time = audio_clock.seconds_passed();
                    const frame_time = stable_buffer.invert().display_time(); // Frame actually being rendered
                    const diff = frame_time - audio_time;

                    // Increment frame counter for debug
                    frame_count += 1;

                    // skip frame
                    if (@abs(diff) > desync_reset) {
                        log.err("A/V desync: {d:.1}ms - skipping frame", .{diff * std.time.ms_per_s});
                        continue;
                    }

                    // drop frame
                    if (diff < -drop_behind) {
                        log.debug("DROP #{} | display: {d:.3}s upload: {d:.3}s audio: {d:.3}s diff: \x1B[36m{d:.1}ms\x1B[39m", .{
                            frame_count, frame_time, pt_in_seconds, audio_time, diff * std.time.ms_per_s,
                        });
                        continue;
                    }

                    // delay frame
                    if (diff > delay_ahead) {
                        const delay_ns = @min(diff * std.time.ns_per_s, max_delay * std.time.ns_per_s);
                        log.debug("DELAY #{} | display: {d:.3}s upload: {d:.3}s audio={d:.3}s diff: \x1B[33m{d:.1}ms\x1B[39m sleep: {d:.1}ms", .{
                            frame_count, frame_time, pt_in_seconds, audio_time, diff * std.time.ms_per_s, delay_ns / std.time.ns_per_ms,
                        });

                        sleep(@intFromFloat(delay_ns));
                    }
                }

                {
                    gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, stable_buffer.pbo(.y));
                    defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

                    gl.BindTexture(gl.TEXTURE_2D, stable_buffer.tex(.y));
                    defer gl.BindTexture(gl.TEXTURE_2D, 0);

                    const pbo: ?[*]u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_UNPACK_BUFFER, gl.WRITE_ONLY));

                    if (pbo) |ptr| {
                        // Copy line by line instead of a single large memcpy
                        const bytes_per_line: usize = @intCast(frame.linesize[0]);
                        const height: usize = @intCast(frame.height);

                        // Destination stride in the PBO may be different from the source
                        const dst_stride = @as(usize, @intCast(frame.width * Y_BPP));

                        for (0..height) |y| {
                            const src_offset = y * bytes_per_line;
                            const dst_offset = y * dst_stride;

                            @memcpy(ptr[dst_offset..][0..dst_stride], frame.data[0][src_offset..][0..dst_stride]);
                        }

                        _ = gl.UnmapBuffer(gl.PIXEL_UNPACK_BUFFER);
                    }

                    gl.PixelStorei(gl.UNPACK_ALIGNMENT, 1);

                    gl.TexSubImage2D(
                        gl.TEXTURE_2D,
                        0,
                        0,
                        0,
                        frame.width,
                        frame.height,
                        gl.RED,
                        gl.UNSIGNED_BYTE,
                        null,
                    );
                }

                {
                    gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, stable_buffer.pbo(.uv));
                    defer gl.BindBuffer(gl.PIXEL_UNPACK_BUFFER, 0);

                    gl.BindTexture(gl.TEXTURE_2D, stable_buffer.tex(.uv));
                    defer gl.BindTexture(gl.TEXTURE_2D, 0);

                    const pbo: ?[*]u8 = @ptrCast(gl.MapBuffer(gl.PIXEL_UNPACK_BUFFER, gl.WRITE_ONLY));

                    if (pbo) |ptr| {
                        // Copy line by line instead of a single large memcpy
                        const bytes_per_line: usize = @intCast(frame.linesize[1]);
                        const height: usize = @intCast(@divTrunc(frame.height, 2));

                        // Destination stride in the PBO may be different from the source
                        const dst_stride = @as(usize, @intCast(@divTrunc(frame.width, 2) * UV_BPP));

                        for (0..height) |y| {
                            const src_offset = y * bytes_per_line;
                            const dst_offset = y * dst_stride;

                            @memcpy(ptr[dst_offset..][0..dst_stride], frame.data[1][src_offset..][0..dst_stride]);
                        }

                        _ = gl.UnmapBuffer(gl.PIXEL_UNPACK_BUFFER);
                    }

                    gl.PixelStorei(gl.UNPACK_ALIGNMENT, 2);

                    gl.TexSubImage2D(
                        gl.TEXTURE_2D,
                        0,
                        0,
                        0,
                        @divTrunc(frame.width, 2),
                        @divTrunc(frame.height, 2),
                        gl.RG,
                        gl.UNSIGNED_BYTE,
                        null,
                    );
                }

                try render(
                    &view,
                    &stable_buffer,
                    angle_calc,
                    res,
                    camera,
                );

                // Start audio AFTER first frame is rendered (not before)
                // This ensures audio timing aligns with actual frame display
                if (is_first_frame) {
                    try audio_clock.start(first_frame_pts);
                }
            } else {
                // log.err("{}: video decode bottleneck", .{c.SDL_GetPerformanceCounter()}); // TODO: add adaptive sleeping here
                continue;
            }

            try ui.swap();
        }
    } // end of RENDER_MODE else block

    if (RENDER_MODE) {
        log.info("Render completed successfully", .{});
    }
}

const DoubleBuffer = struct {
    res: *const GpuResourceManager,

    display_times: [2]f64,
    current: u1,

    const Channel = enum { y, uv };

    const Inverted = struct {
        inner: DoubleBuffer,

        fn from(super: DoubleBuffer) Inverted {
            return .{
                .inner = .{
                    .res = super.res,
                    .display_times = super.display_times,
                    .current = super.current +% 1,
                },
            };
        }

        fn tex(self: @This()) Nv12Tex {
            const y = self.inner.res.tex.get(if (self.inner.current == 0) .y_front else .y_back);
            const uv = self.inner.res.tex.get(if (self.inner.current == 0) .uv_front else .uv_back);

            return .{ .y = y, .uv = uv };
        }

        fn display_time(self: @This()) f64 {
            return self.inner.display_times[self.inner.current];
        }
    };

    pub fn init(res: *const GpuResourceManager) DoubleBuffer {
        return .{
            .res = res,
            .display_times = .{ 0.0, 0.0 },
            .current = 0,
        };
    }

    fn tex(self: @This(), comptime ch: Channel) c_uint {
        return switch (ch) {
            .y => self.res.tex.get(if (self.current == 0) .y_front else .y_back),
            .uv => self.res.tex.get(if (self.current == 0) .uv_front else .uv_back),
        };
    }

    fn pbo(self: @This(), comptime ch: Channel) c_uint {
        return switch (ch) {
            .y => self.res.pbo.get(if (self.current == 0) .y_front else .y_back),
            .uv => self.res.pbo.get(if (self.current == 0) .uv_front else .uv_back),
        };
    }

    fn set_display_time(self: *@This(), in_seconds: f64) void {
        self.display_times[self.current] = in_seconds;
    }

    fn invert(self: @This()) Inverted {
        return Inverted.from(self);
    }

    fn swap(self: *@This()) void {
        self.current = self.current +% 1;
    }
};

const Id = enum(usize) {
    texture = 0,
    ring,
    circle,
    background,
    blur,
};

fn render(
    view: *Viewport,
    stable_buffer: *DoubleBuffer,
    angle_calc: AngleCalc,
    res: *const GpuResourceManager,
    camera: Camera,
) !void {
    gl.ClearColor(0, 0, 0, 0);
    gl.Clear(gl.COLOR_BUFFER_BIT);

    const tex = stable_buffer.invert().tex();

    angle_calc.execute(view, tex);

    {
        blur(res.blur(), res, view, tex, camera.colour_space, 4);
        const prog = res.prog.get(.bg);

        gl.UseProgram(prog);
        gl.BindVertexArray(res.vao.get(.tex));

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, res.blur().front.tex); // guaranteed to be the last modified texture

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.angle));

        const u_world_transform = camera.getBackgroundWorldTransform();
        const u_view_transform = Mat2.identity; // don't zoom in on background
        const u_clip_transform = camera.getViewClipTransform();

        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_angle"), 1);

        gl.Uniform1i(gl.GetUniformLocation(prog, "u_blur"), 0);

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    {
        const circle_prog = res.prog.get(.circle);
        const ring_prog = res.prog.get(.ring);

        const u_world_transform = camera.getUiWorldTransform();
        const u_view_transform = camera.getWorldViewTransform();
        const u_clip_transform = camera.getViewClipTransform();

        // Draw Transparent Puck
        gl.UseProgram(circle_prog);
        gl.BindVertexArray(res.vao.get(.circle));

        gl.UniformMatrix2fv(gl.GetUniformLocation(circle_prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(circle_prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(circle_prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});
        gl.DrawArrays(gl.TRIANGLE_FAN, 0, @intCast(res.meta.circle_len));

        // Draw Ring (matches ring in gameplay)
        gl.UseProgram(ring_prog);
        gl.BindVertexArray(res.vao.get(.ring));

        gl.UniformMatrix2fv(gl.GetUniformLocation(ring_prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(ring_prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(ring_prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});
        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, @intCast(res.meta.ring_len));
    }

    {
        const prog = res.prog.get(.tex);

        gl.UseProgram(prog);
        defer gl.UseProgram(0);

        gl.BindVertexArray(res.vao.get(.tex));
        defer gl.BindVertexArray(0);

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, tex.y);

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, tex.uv);

        gl.ActiveTexture(gl.TEXTURE2);
        gl.BindTexture(gl.TEXTURE_2D, res.tex.get(.angle));
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        const u_world_transform = camera.getVideoWorldTransform();
        const u_view_transform = camera.getWorldViewTransform();
        const u_clip_transform = camera.getViewClipTransform();

        const magic_aspect_ratio = 1.7763157895;

        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_world_transform"), 1, gl.FALSE, &.{u_world_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_view_transform"), 1, gl.FALSE, &.{u_view_transform.m});
        gl.UniformMatrix2fv(gl.GetUniformLocation(prog, "u_clip_transform"), 1, gl.FALSE, &.{u_clip_transform.m});

        gl.Uniform1i(gl.GetUniformLocation(prog, "u_y_tex"), 0);
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_uv_tex"), 1);
        gl.Uniform1i(gl.GetUniformLocation(prog, "u_angle"), 2);
        gl.Uniform1ui(gl.GetUniformLocation(prog, "u_colour_space"), camera.colour_space);
        gl.Uniform1f(gl.GetUniformLocation(prog, "u_ratio"), magic_aspect_ratio);

        gl.DrawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
}

fn blur(b: BlurManager, res: *const GpuResourceManager, view: *Viewport, src_tex: Nv12Tex, colour_space: c.AVColorSpace, comptime passes: u32) void {
    if (passes == 0) return;

    std.debug.assert(passes & 1 == 0);

    const width: c_int = @intCast(b.resolution.width);
    const height: c_int = @intCast(b.resolution.height);
    const program = res.prog.get(.blur);

    const cache: c_uint = blk: {
        var buf: [1]c_int = undefined;
        gl.GetIntegerv(gl.FRAMEBUFFER_BINDING, &buf);

        break :blk @intCast(buf[0]);
    };
    defer gl.BindFramebuffer(gl.FRAMEBUFFER, cache);

    view.set(width, height);
    defer view.restore();

    gl.BindVertexArray(res.vao.get(.blur));
    defer gl.BindVertexArray(0);

    gl.UseProgram(program);
    defer gl.UseProgram(0);

    gl.ActiveTexture(gl.TEXTURE1);
    gl.BindTexture(gl.TEXTURE_2D, src_tex.y);

    gl.ActiveTexture(gl.TEXTURE2);
    gl.BindTexture(gl.TEXTURE_2D, src_tex.uv);

    gl.Uniform2f(gl.GetUniformLocation(program, "u_resolution"), @floatFromInt(width), @floatFromInt(height));
    gl.Uniform1i(gl.GetUniformLocation(program, "u_screen"), 0);
    gl.Uniform1i(gl.GetUniformLocation(program, "u_y_tex"), 1);
    gl.Uniform1i(gl.GetUniformLocation(program, "u_uv_tex"), 2);
    gl.Uniform1ui(gl.GetUniformLocation(program, "u_colour_space"), colour_space);

    const horiz_loc = gl.GetUniformLocation(program, "u_horizontal");
    const use_nv12_loc = gl.GetUniformLocation(program, "u_use_nv12");

    gl.ActiveTexture(gl.TEXTURE0);

    for (0..passes) |i| {
        const current = b.current(i);
        const other = b.previous(i);

        gl.BindFramebuffer(gl.FRAMEBUFFER, current.fbo);
        gl.BindTexture(gl.TEXTURE_2D, other.tex);

        gl.Uniform1i(horiz_loc, @intFromBool(i % 2 == 0));
        gl.Uniform1i(use_nv12_loc, @intFromBool(i == 0));

        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }
}

const Viewport = struct {
    width: c_int,
    height: c_int,

    cached: ?[2]c_int = null,

    fn init(width: c_int, height: c_int) Viewport {
        gl.Viewport(0, 0, width, height);

        return .{ .width = width, .height = height };
    }

    fn set(self: *@This(), width: c_int, height: c_int) void {
        gl.Viewport(0, 0, width, height);
        self.cached = .{ self.width, self.height };

        self.width = width;
        self.height = height;
    }

    fn restore(self: *@This()) void {
        if (self.cached == null) @panic("viewport manager state corrupted");

        const view = self.cached.?;
        gl.Viewport(0, 0, view[0], view[1]);

        self.width = view[0];
        self.height = view[1];
        self.cached = null;
    }
};

const Nv12Tex = struct { y: c_uint, uv: c_uint };

const AngleCalc = struct {
    res: *const GpuResourceManager,
    colour_space: c.AVColorSpace,

    const log = std.log.scoped(.angle_calc);

    pub fn init(res: *const GpuResourceManager, colour_space: c.AVColorSpace) !AngleCalc {
        return .{ .res = res, .colour_space = colour_space };
    }

    pub fn execute(self: @This(), view: *Viewport, tex: Nv12Tex) void {
        const program = self.res.prog.get(.angle);

        view.set(1, 1);
        defer view.restore();

        gl.BindVertexArray(self.res.vao.get(.empty));
        defer gl.BindVertexArray(0);

        gl.BindFramebuffer(gl.FRAMEBUFFER, self.res.fbo.get(.angle));
        defer gl.BindFramebuffer(gl.FRAMEBUFFER, 0);

        gl.ActiveTexture(gl.TEXTURE0);
        gl.BindTexture(gl.TEXTURE_2D, tex.y);

        gl.ActiveTexture(gl.TEXTURE1);
        gl.BindTexture(gl.TEXTURE_2D, tex.uv);
        defer gl.BindTexture(gl.TEXTURE_2D, 0);

        gl.UseProgram(program);
        defer gl.UseProgram(0);

        gl.Uniform1i(gl.GetUniformLocation(program, "u_y_tex"), 0);
        gl.Uniform1i(gl.GetUniformLocation(program, "u_uv_tex"), 1);
        gl.Uniform1ui(gl.GetUniformLocation(program, "u_colour_space"), self.colour_space);

        gl.DrawArrays(gl.TRIANGLES, 0, 3);
    }
};

const Camera = struct {
    view_to_clip: Mat2,

    world_bounds: Vec2,
    window_size: [2]c_int,
    video_aspect: f32,

    scale: f32,
    inv_scale: f32,

    colour_space: c.AVColorSpace,

    zoom: f32 = 1.45, // TODO: revert to 1.0

    pub fn init(video_width: u32, video_height: u32, window_width: c_int, window_height: c_int, colour_space: c.AVColorSpace) Camera {
        const video_aspect = @as(f32, @floatFromInt(video_width)) / @as(f32, @floatFromInt(video_height));
        const window_aspect = @as(f32, @floatFromInt(window_width)) / @as(f32, @floatFromInt(window_height));

        const world_bounds = vec2(1.0, 1.0);
        const viewport_bounds = if (window_aspect > 1.0) vec2(window_aspect, 1.0) else vec2(1.0, 1.0 / window_aspect);
        const video_bounds = if (video_aspect > 1.0) vec2(1.0, 1.0 / video_aspect) else vec2(video_aspect, 1.0);

        const world_aspect = world_bounds.x() / world_bounds.y();
        const view_to_clip = calculateAspectCorrection(world_aspect, window_aspect);

        const scale = 1.0 / std.math.sqrt(video_bounds.x() * video_bounds.x() + video_bounds.y() * video_bounds.y());

        const viewport_diagonal = std.math.sqrt(viewport_bounds.x() * viewport_bounds.x() + viewport_bounds.y() * viewport_bounds.y());
        const inv_scale = viewport_diagonal / @min(video_bounds.x(), video_bounds.y());

        return .{
            .view_to_clip = view_to_clip,
            .world_bounds = world_bounds,
            .window_size = .{ window_width, window_height },
            .video_aspect = video_aspect,

            .colour_space = colour_space,

            .scale = scale,
            .inv_scale = inv_scale,
        };
    }

    pub fn updateWindow(self: *@This(), width: c_int, height: c_int) void {
        self.window_size = .{ width, height };

        const window_aspect = @as(f32, @floatFromInt(width)) / @as(f32, @floatFromInt(height));
        const world_aspect = self.world_bounds.x() / self.world_bounds.y();

        self.view_to_clip = calculateAspectCorrection(world_aspect, window_aspect);

        const viewport_bounds = if (window_aspect > 1.0) vec2(window_aspect, 1.0) else vec2(1.0, 1.0 / window_aspect);
        const video_bounds = if (self.video_aspect > 1.0) vec2(1.0, 1.0 / self.video_aspect) else vec2(self.video_aspect, 1.0);

        const viewport_diagonal = std.math.sqrt(viewport_bounds.x() * viewport_bounds.x() + viewport_bounds.y() * viewport_bounds.y());
        self.inv_scale = viewport_diagonal / @min(video_bounds.x(), video_bounds.y());
    }

    fn calculateAspectCorrection(world_aspect: f32, window_aspect: f32) Mat2 {
        if (window_aspect > world_aspect) {
            // Window wider than world - letterbox horizontally
            return Mat2.scaleXy(world_aspect / window_aspect, 1.0);
        } else {
            // Window taller than world - letterbox vertically
            return Mat2.scaleXy(1.0, window_aspect / world_aspect);
        }
    }

    pub fn getUiWorldTransform(self: @This()) Mat2 {
        return Mat2.scale(self.scale);
    }

    pub fn getVideoWorldTransform(self: @This()) Mat2 {
        const wide_scale = Mat2.scaleXy(1.0, 1.0 / self.video_aspect);
        const tall_scale = Mat2.scaleXy(self.video_aspect, 1.0);
        const aspect_transform = if (self.video_aspect > 1.0) wide_scale else tall_scale;

        const scale_transform = Mat2.scale(self.scale);
        return aspect_transform.mul(scale_transform);
    }

    pub fn getBackgroundWorldTransform(self: @This()) Mat2 {
        const wide_scale = Mat2.scaleXy(1.0, 1.0 / self.video_aspect);
        const tall_scale = Mat2.scaleXy(self.video_aspect, 1.0);
        const aspect_transform = if (self.video_aspect > 1.0) wide_scale else tall_scale;

        const scale_transform = Mat2.scale(self.inv_scale);
        return aspect_transform.mul(scale_transform);
    }

    pub fn getWorldViewTransform(self: @This()) Mat2 {
        return Mat2.scale(self.zoom);
    }

    pub fn getViewClipTransform(self: @This()) Mat2 {
        return self.view_to_clip;
    }

    pub fn adjustZoom(self: *@This(), delta: f32) void {
        self.setZoom(self.zoom + delta);
    }

    fn setZoom(self: *@This(), new_zoom: f32) void {
        std.log.info("zoom: {d:.2}", .{new_zoom});
        self.zoom = @max(1.0, @min(10.0, new_zoom));
    }
};
