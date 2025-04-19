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
    buf: []c.AVFrame,
    read_idx: usize,
    write_idx: usize,

    mutex: std.Thread.Mutex = .{}, // TODO: switch to atomics

    const Error = error{ invalid_size, full, ffmpeg_error } || std.mem.Allocator.Error;

    pub fn init(allocator: std.mem.Allocator, count: usize) Error!FrameQueue {
        if (!std.math.isPowerOfTwo(count)) return error.invalid_size;

        const buf = try allocator.alloc(c.AVFrame, count);
        @memset(buf, std.mem.zeroes(c.AVFrame));

        for (buf) |*frame| c.av_frame_unref(frame);

        return .{ .buf = buf, .read_idx = 0, .write_idx = 0 };
    }

    pub fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.buf);
    }

    pub fn push(self: *@This(), to_be_copied: *const c.AVFrame) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.isFull()) return error.full;
        defer self.write_idx += 1;

        const ptr = &self.buf[self.mask(self.write_idx)];

        const valid_buffer =
            to_be_copied.width == ptr.width and
            to_be_copied.height == ptr.height and
            to_be_copied.format == ptr.format;

        if (!valid_buffer) {
            c.av_frame_unref(ptr);

            ptr.width = to_be_copied.width;
            ptr.height = to_be_copied.height;
            ptr.format = to_be_copied.format;

            _ = try libavError(c.av_frame_get_buffer(ptr, 32));
        }

        _ = try libavError(c.av_frame_copy(ptr, to_be_copied));
        _ = try libavError(c.av_frame_copy_props(ptr, to_be_copied));
    }

    pub fn pop(self: *@This(), dst_frame: *c.AVFrame) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.isEmpty()) return false;
        defer self.read_idx += 1;

        const src_frame = &self.buf[self.mask(self.read_idx)];

        const valid_buffer =
            src_frame.width == dst_frame.width and
            src_frame.height == dst_frame.height and
            src_frame.format == dst_frame.format;

        if (!valid_buffer) {
            c.av_frame_unref(dst_frame);

            dst_frame.width = src_frame.width;
            dst_frame.height = src_frame.height;
            dst_frame.format = src_frame.format;

            _ = try libavError(c.av_frame_get_buffer(dst_frame, 32));
        }

        _ = try libavError(c.av_frame_copy(dst_frame, src_frame));
        _ = try libavError(c.av_frame_copy_props(dst_frame, src_frame));

        return true;
    }

    inline fn len(self: @This()) usize {
        return self.write_idx - self.read_idx;
    }

    inline fn mask(self: @This(), idx: usize) usize {
        return idx & (self.buf.len - 1);
    }

    inline fn isEmpty(self: @This()) bool {
        return self.read_idx == self.write_idx;
    }

    inline fn isFull(self: @This()) bool {
        return self.len() == self.buf.len;
    }
};
