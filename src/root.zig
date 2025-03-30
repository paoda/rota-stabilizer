//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");
const testing = std.testing;

pub fn ThreadSafeRingBuffer(comptime T: type) type {
    return struct {
        inner: RingBuffer(T),
        mutex: std.Thread.Mutex,

        pub fn init(buf: []T) @This() {
            return .{ .inner = RingBuffer(T).init(buf), .mutex = .{} };
        }

        pub fn push(self: *@This(), value: T) !void {
            self.mutex.lock();
            defer self.mutex.unlock();

            try self.inner.push(value);
        }

        pub fn pop(self: *@This()) ?T {
            self.mutex.lock();
            defer self.mutex.unlock();

            return self.inner.pop();
        }

        pub fn len(self: *@This()) usize {
            self.mutex.lock();
            defer self.mutex.unlock();

            return self.inner.len();
        }
    };
}

pub fn RingBuffer(comptime T: type) type {
    return struct {
        buf: []T,
        read_idx: usize,
        write_idx: usize,

        pub fn init(buf: []T) @This() {
            std.debug.assert(std.math.isPowerOfTwo(buf.len));

            return .{ .buf = buf, .read_idx = 0, .write_idx = 0 };
        }

        pub fn push(self: *@This(), value: T) !void {
            if (self.isFull()) return error.out_of_memory;
            defer self.write_idx += 1;

            self.buf[self.mask(self.write_idx)] = value;
        }

        pub fn pop(self: *@This()) ?T {
            if (self.isEmpty()) return null;
            defer self.read_idx += 1;

            return self.buf[self.mask(self.read_idx)];
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

        pub fn len(self: @This()) usize {
            return self.write_idx - self.read_idx;
        }
    };
}
