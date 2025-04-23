//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

pub const c = @cImport({
    @cDefine("SDL_DISABLE_OLD_NAMES", {});
    @cDefine("SDL_MAIN_HANDLED", {});

    @cInclude("SDL3/SDL.h");
    @cInclude("SDL3/SDL_main.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libswscale/swscale.h");
    @cInclude("libswresample/swresample.h");
    @cInclude("libavutil/imgutils.h");
    @cInclude("libavutil/opt.h");

    @cInclude("stb_image_write.h");
});

pub inline fn libavError(value: c_int) error{ffmpeg_error}!c_int {
    if (value >= 0) return value;
    var buf: [0x100]u8 = undefined;
    const ret = c.av_strerror(value, &buf, buf.len);

    if (ret < 0) std.debug.panic("ffmpeg error handle failed: {}", .{value});

    std.debug.print("{s}\n", .{std.mem.sliceTo(&buf, 0)});
    return error.ffmpeg_error;
}

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

        _ = try libavError(c.av_frame_ref(frame, new_frame));
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

pub fn sleep(ns: u64) void {
    if (ns == 0) {
        @branchHint(.cold);
        return;
    }

    const freq = c.SDL_GetPerformanceFrequency();
    const start_counter = c.SDL_GetPerformanceCounter();

    // TODO: confirm I did the conversion right
    const duration: u64 = @intCast(@divTrunc(@as(i128, ns) * freq, std.time.ns_per_s));
    const target_counter: u64 = start_counter + duration;

    const threshold_ns = 15 * std.time.ns_per_ms; // Example threshold for coarse sleep

    // Perform coarse sleep if the total duration is significantly larger than the threshold
    if (ns > threshold_ns) {
        const ideal_ns = ns - threshold_ns;
        std.Thread.sleep(ideal_ns);
    }

    while (c.SDL_GetPerformanceCounter() < target_counter) std.atomic.spinLoopHint();

    // Optional: Debug print the actual elapsed time for verification
    // const end_counter = c.SDL_GetPerformanceCounter();
    // const actual_elapsed_ns: u64 = @intCast(@divTrunc(@as(i128, end_counter - start_counter) * std.time.ns_per_s, freq));

    // const log = std.log.scoped(.sleep);
    // log.debug("attempted {}ms late by {}ns", .{ ns / std.time.ns_per_ms, @as(i64, @intCast(actual_elapsed_ns)) - @as(i64, @intCast(ns)) });
}
