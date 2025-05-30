//! By convention, root.zig is the root source file when making a library. If
//! you are making an executable, the convention is to delete this file and
//! start with main.zig instead.
const std = @import("std");

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
});

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

    const threshold_ns = 2 * std.time.ns_per_ms; // Example threshold for coarse sleep

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
