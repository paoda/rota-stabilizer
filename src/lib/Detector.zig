const Detector = @This();

const std = @import("std");
const tracy = @import("tracy");

const Resolution = @import("../lib.zig").Resolution;
const ContentRect = @import("../lib.zig").ContentRect;

const IVec2 = @import("math.zig").IVec2;
const ivec2 = @import("math.zig").ivec2;

const Component = struct {
    buf: []const u8,
    stride: usize,
};

// see src/shader/texture.frag (YUV -> RGB)
const y_offset = 16.0 / 255.0;
const y_scale = 255.0 / (235.0 - 16.0);
const uv_scale = 255.0 / (240.0 - 16.0);

pub const Config = struct {
    /// limited-range black is 16; +24 of headroom mirrors ffmpeg cropdetect's default limit
    luma_threshold: u8 = 40,
    /// scan every nth frame while searching
    search_interval: usize = 16,
    /// candidate must be unchanged for this many searches before validating
    stable_samples: usize = 8,
    /// number consecutive frames sampled during a validation burst
    burst_len: usize = 24,
    /// tolerable per-channel distance from 0.0/1.0 a marker may have
    sat_tolerance: f32 = 0.15,
    /// fraction of burst frames that must pass
    min_pass_rate: f32 = 0.9,
    /// max number of degrees between consecutive frames
    max_angle_step: f32 = 30.0,
    /// number of validation attempts before giving up
    max_failed_bursts: usize = 5,
    /// minimum size for a candidate ContentRect
    min_area_ratio: f32 = 0.25,
};

pub const Status = union(enum) {
    searching,
    validating,
    locked: ContentRect,
    failed,
};

config: Config,
res: Resolution,
colour_matrix: [9]f32, // column-major, same as Camera's

/// temporal max luma per row / column of the Y plane
row_max: []u8,
col_max: []u8,

frame_count: usize = 0,

candidate: ?ContentRect = null,
stable_count: usize = 0,

burst: Burst = .{},
failed_bursts: usize = 0,

status: Status = .searching,

enabled: bool = true,

const Burst = struct {
    target: ContentRect = undefined,
    fed: usize = 0,
    passed: usize = 0,
    last_angle: ?f32 = null,
};

const log = std.log.scoped(.content_detect);

pub fn init(allocator: std.mem.Allocator, res: Resolution, colour_matrix: [9]f32, config: Config) !Detector {
    const row_max = try allocator.alloc(u8, @intCast(res.height));
    errdefer allocator.free(row_max);

    const col_max = try allocator.alloc(u8, @intCast(res.width));
    errdefer allocator.free(col_max);

    @memset(row_max, 0);
    @memset(col_max, 0);

    return .{
        .config = config,
        .res = res,
        .colour_matrix = colour_matrix,
        .row_max = row_max,
        .col_max = col_max,
    };
}

pub fn deinit(self: *Detector, allocator: std.mem.Allocator) void {
    allocator.free(self.row_max);
    allocator.free(self.col_max);
}

pub fn reset(self: *Detector) void {
    @memset(self.row_max, 0);
    @memset(self.col_max, 0);

    self.frame_count = 0;
    self.candidate = null;
    self.stable_count = 0;
    self.burst = .{};
    self.failed_bursts = 0;
    self.status = .searching;
}

pub fn feed(self: *Detector, luma: Component, chroma: Component) void {
    defer self.frame_count += 1;

    switch (self.status) {
        .locked, .failed => {},
        .searching => self.search(luma),
        .validating => self.validate(luma, chroma),
    }
}

fn search(self: *Detector, luma: Component) void {
    if (self.frame_count % self.config.search_interval != 0) return;

    const zone = tracy.Zone.begin(.{ .src = @src(), .name = "Detector.search" });
    defer zone.end();

    self.accumulate(luma);
    const bounds = self.boundingBox() orelse return;

    // asking is this a native recording or functionally equivalent to one?
    if (bounds.eql(ContentRect.full(self.res))) {
        self.status = .{ .locked = bounds };
        return log.info("content already fills the frame. nothing to do", .{});
    }

    const area = @as(f32, @floatFromInt(bounds.frame.width)) * @as(f32, @floatFromInt(bounds.frame.height));
    const frame_area = @as(f32, @floatFromInt(self.res.width)) * @as(f32, @floatFromInt(self.res.height));
    if (area < self.config.min_area_ratio * frame_area) return;

    if (self.candidate) |prev| {
        self.stable_count = if (bounds.eql(prev)) self.stable_count + 1 else 0;
    }
    self.candidate = bounds;

    if (self.stable_count >= self.config.stable_samples) {
        log.debug("stable candidate {f} at ({}, {})", .{ bounds.frame, bounds.x, bounds.y });

        self.burst = .{ .target = bounds };
        self.status = .validating;
    }
}

fn validate(self: *Detector, luma: Component, chroma: Component) void {
    const zone = tracy.Zone.begin(.{ .src = @src(), .name = "Detector.validate" });
    defer zone.end();

    const target = self.burst.target;

    // the bounding box may still be growing. If this is the case, return to search
    if (self.frame_count % self.config.search_interval == 0) {
        self.accumulate(luma);

        if (self.boundingBox()) |bounds| {
            if (!bounds.eql(target)) {
                self.stable_count = 0;
                self.burst = .{};
                self.status = .searching;

                return log.debug("bounds grew mid-burst. return to search", .{});
            }
        }
    }

    const corners = self.sampleCorners(target, luma, chroma);
    var is_saturated = true;

    // confirm that the corners are either really close to 0 or really close to 1
    for (corners) |rgb| {
        for (rgb) |ch| {
            if (@min(ch, 1.0 - ch) > self.config.sat_tolerance) is_saturated = false;
        }
    }

    const angle = decodeAngle(corners);

    // is_smooth is true when the new angle is within max_angle_step in either direction
    const is_smooth = if (self.burst.last_angle) |prev| angleDelta(angle, prev) <= self.config.max_angle_step else true;
    self.burst.last_angle = angle;

    self.burst.fed += 1;
    if (is_saturated and is_smooth) self.burst.passed += 1;

    if (self.burst.fed < self.config.burst_len) return;
    // burst run is now complete

    const rate = @as(f32, @floatFromInt(self.burst.passed)) / @as(f32, @floatFromInt(self.burst.fed));

    if (rate >= self.config.min_pass_rate) {
        self.status = .{ .locked = target };
        return log.info("verified ({d:.0}%); locked {f} at ({}, {})", .{ rate * 100.0, target.frame, target.x, target.y });
    }

    self.failed_bursts += 1;
    log.debug("failed ({d:.0}%), {}/{} attempts", .{ rate * 100.0, self.failed_bursts, self.config.max_failed_bursts });

    self.stable_count = 0;
    self.burst = .{};
    self.status = if (self.failed_bursts >= self.config.max_failed_bursts) .failed else .searching;
}

fn accumulate(self: *Detector, luma: Component) void {
    const width: usize = @intCast(self.res.width);
    const height: usize = @intCast(self.res.height);

    for (0..height) |h| {
        const row = luma.buf[h * luma.stride ..][0..width];
        var max: u8 = self.row_max[h];

        // each luma value is gonna be a part of a row and column
        // so track and update them both
        for (row, self.col_max) |y, *col_max| {
            max = @max(max, y);
            col_max.* = @max(col_max.*, y);
        }

        self.row_max[h] = max;
    }
}

/// create bounding box **containing** everything that has ever exceeded the luma threshold
/// (including the values that determined the bound itself)
fn boundingBox(self: *const Detector) ?ContentRect {
    const threshold = self.config.luma_threshold;

    const top = findFirst(self.row_max, threshold) orelse return null;
    const bottom = findLast(self.row_max, threshold) orelse return null;
    const left = findFirst(self.col_max, threshold) orelse return null;
    const right = findLast(self.col_max, threshold) orelse return null;

    // divisible by two (grow outwards)
    const x = left & ~@as(usize, 1);
    const y = top & ~@as(usize, 1);

    // NB: these are likely the boundaries of the frame (unless portrait for some reason?)
    const right_ex = @min(@as(usize, @intCast(self.res.width)), (right + 2) & ~@as(usize, 1));
    const bottom_ex = @min(@as(usize, @intCast(self.res.height)), (bottom + 2) & ~@as(usize, 1));

    return .{
        .x = @intCast(x),
        .y = @intCast(y),
        .frame = .{ .width = @intCast(right_ex - x), .height = @intCast(bottom_ex - y) },
    };
}

fn findFirst(items: []const u8, threshold: u8) ?usize {
    for (items, 0..) |item, i| if (item >= threshold) return i;
    return null;
}

fn findLast(items: []const u8, threshold: u8) ?usize {
    std.debug.assert(items.len != 0);

    for (0..items.len) |i| {
        const idx = items.len - 1 - i;
        if (items[idx] >= threshold) return idx;
    }

    return null;
}

/// mean RGB channels of the four corners like in rotation.frag
/// order: top-left, top-right, bottom-left, bottom-right (bit weights high to low)
fn sampleCorners(self: *const Detector, rect: ContentRect, luma: Component, chroma: Component) [4][3]f32 {
    const ofs = 1;
    const size = 3;

    // need to at the very least be able to sample without going out of bounds
    std.debug.assert(rect.frame.width >= 2 * (ofs + size) and rect.frame.height >= 2 * (ofs + size));

    const x0: usize = @intCast(rect.x + ofs);
    const y0: usize = @intCast(rect.y + ofs);
    const x1: usize = @intCast(rect.x + rect.frame.width - ofs - size);
    const y1: usize = @intCast(rect.y + rect.frame.height - ofs - size);

    const top_left = ivec2(x0, y0);
    const top_right = ivec2(x1, y0);
    const btm_left = ivec2(x0, y1);
    const btm_right = ivec2(x1, y1);

    return .{
        self.sampleBox(size, top_left, luma, chroma),
        self.sampleBox(size, top_right, luma, chroma),
        self.sampleBox(size, btm_left, luma, chroma),
        self.sampleBox(size, btm_right, luma, chroma),
    };
}

// see sampleTexture in rotation.frag
fn sampleBox(self: *const Detector, comptime size: usize, pos: IVec2, luma: Component, chroma: Component) [3]f32 {
    var y_sum: f32 = 0.0;
    var u_sum: f32 = 0.0;
    var v_sum: f32 = 0.0;

    for (0..size) |dy| {
        for (0..size) |dx| {
            const px = pos.x() + dx;
            const py = pos.y() + dy;

            y_sum += @floatFromInt(luma.buf[py * luma.stride + px]);

            // NB: the UV plane is interleaved at half resolution
            const uv_idx = (py / 2) * chroma.stride + (px / 2) * 2;
            u_sum += @floatFromInt(chroma.buf[uv_idx]);
            v_sum += @floatFromInt(chroma.buf[uv_idx + 1]);
        }
    }

    const count: f32 = size * size;
    return decodeRgb(
        self.colour_matrix,
        y_sum / (count * 255.0), // div by 255.0 to normalize [0, 1]
        u_sum / (count * 255.0),
        v_sum / (count * 255.0),
    );
}

// see decode() in rotation.frag
fn decodeRgb(matrix: [9]f32, y_norm: f32, u_norm: f32, v_norm: f32) [3]f32 {
    const y_val = std.math.clamp((y_norm - y_offset) * y_scale, 0.0, 1.0);
    const u_val = std.math.clamp((u_norm - y_offset) * uv_scale, 0.0, 1.0) - 0.5;
    const v_val = std.math.clamp((v_norm - y_offset) * uv_scale, 0.0, 1.0) - 0.5;

    const m = matrix; // column-major: rgb = col0*y + col1*u + col2*v

    var rgb: [3]f32 = undefined;
    for (&rgb, 0..) |*ch, i| {
        ch.* = std.math.clamp(m[i] * y_val + m[3 + i] * u_val + m[6 + i] * v_val, 0.0, 1.0);
    }

    return rgb;
}

// see main() in rotation.frag
fn decodeAngle(corners: [4][3]f32) f32 {
    const threshold = 0.5;
    var value: u32 = 0;
    var weight: u32 = 2048;

    for (corners) |rgb| {
        for (rgb) |ch| {
            if (ch >= threshold) value += weight;
            weight /= 2;
        }
    }

    return 360.0 * @as(f32, @floatFromInt(value)) / 4096.0;
}

fn angleDelta(a: f32, b: f32) f32 {
    const diff = @abs(a - b);
    return @min(diff, 360.0 - diff);
}

// --- tests ---

const testing = std.testing;

// column-major bt709, identical to Camera.bt709 in main.zig
// zig fmt: off
const bt709 = [9]f32{
    1.0,      1.0,      1.0,
    0.0,     -0.1873,   1.8556,
    1.5748,  -0.4681,   0.0,
};
// zig fmt: on

const TestFrame = struct {
    y: []u8,
    uv: []u8,
    width: usize,
    height: usize,

    fn init(allocator: std.mem.Allocator, res: Resolution) !TestFrame {
        const width: usize = @intCast(res.width);
        const height: usize = @intCast(res.height);

        const y = try allocator.alloc(u8, width * height);
        errdefer allocator.free(y);

        const uv = try allocator.alloc(u8, width * ((height + 1) / 2)); // ceil: NV12 chroma has a row for every *two* luma rows
        errdefer allocator.free(uv);

        var self: TestFrame = .{ .y = y, .uv = uv, .width = width, .height = height };
        self.clear();

        return self;
    }

    fn deinit(self: TestFrame, allocator: std.mem.Allocator) void {
        allocator.free(self.y);
        allocator.free(self.uv);
    }

    /// limited-range black with neutral chroma
    fn clear(self: *TestFrame) void {
        @memset(self.y, 16);
        @memset(self.uv, 128);
    }

    fn setPixel(self: *TestFrame, x: usize, y_pos: usize, yuv: [3]u8) void {
        self.y[y_pos * self.width + x] = yuv[0];

        const uv_idx = (y_pos / 2) * self.width + (x / 2) * 2;
        self.uv[uv_idx] = yuv[1];
        self.uv[uv_idx + 1] = yuv[2];
    }

    /// mid gray: bright enough to trip the luma scan, fails marker saturation
    fn fillGray(self: *TestFrame, rect: ContentRect) void {
        const x: usize = @intCast(rect.x);
        const y_pos: usize = @intCast(rect.y);
        const w: usize = @intCast(rect.frame.width);
        const h: usize = @intCast(rect.frame.height);

        for (y_pos..y_pos + h) |r| {
            for (x..x + w) |col| {
                self.setPixel(col, r, .{ 128, 128, 128 });
            }
        }
    }

    /// 5x5 marker boxes at the corners of `rect`, encoding a 12-bit angle value
    fn setMarkers(self: *TestFrame, rect: ContentRect, value: u12) void {
        const box = 5;

        const x0: usize = @intCast(rect.x);
        const y0: usize = @intCast(rect.y);
        const x1: usize = @intCast(rect.x + rect.frame.width - box);
        const y1: usize = @intCast(rect.y + rect.frame.height - box);

        // top-left, top-right, bottom-left, bottom-right; bit weights high to low
        const positions: [4][2]usize = .{ .{ x0, y0 }, .{ x1, y0 }, .{ x0, y1 }, .{ x1, y1 } };

        for (positions, 0..) |pos, i| {
            const bits: u3 = @truncate(value >> @intCast(9 - i * 3));
            const yuv = rgbToYuv(.{
                @floatFromInt((bits >> 2) & 1),
                @floatFromInt((bits >> 1) & 1),
                @floatFromInt(bits & 1),
            });

            for (0..box) |dy| {
                for (0..box) |dx| {
                    self.setPixel(pos[0] + dx, pos[1] + dy, yuv);
                }
            }
        }
    }

    /// forward BT.709, limited range; the inverse of Detector.decodeRgb
    fn rgbToYuv(rgb: [3]f32) [3]u8 {
        const luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        const cb = (rgb[2] - luma) / 1.8556;
        const cr = (rgb[0] - luma) / 1.5748;

        return .{
            @intFromFloat(@round(16.0 + 219.0 * luma)),
            @intFromFloat(@round(128.0 + 224.0 * cb)),
            @intFromFloat(@round(128.0 + 224.0 * cr)),
        };
    }

    fn feedTo(self: *const TestFrame, detector: *Detector) void {
        detector.feed(
            .{ .buf = self.y, .stride = self.width },
            .{ .buf = self.uv, .stride = self.width },
        );
    }
};

const test_config: Detector.Config = .{
    .search_interval = 1,
    .stable_samples = 2,
    .burst_len = 4,
    .max_failed_bursts = 2,
};

test "full-frame content locks immediately without validation" {
    const res: Resolution = .{ .width = 64, .height = 36 };

    var detector = try Detector.init(testing.allocator, res, bt709, test_config);
    defer detector.deinit(testing.allocator);

    var frame = try TestFrame.init(testing.allocator, res);
    defer frame.deinit(testing.allocator);

    frame.fillGray(.full(res));
    frame.feedTo(&detector);

    try testing.expect(detector.status == .locked);
    try testing.expect(detector.status.locked.eql(.full(res)));
}

test "letterbox with visible markers locks with the exact device rect" {
    const res: Resolution = .{ .width = 640, .height = 360 };
    const device: ContentRect = .{ .x = 0, .y = 60, .frame = .{ .width = 640, .height = 240 } };

    var detector = try Detector.init(testing.allocator, res, bt709, test_config);
    defer detector.deinit(testing.allocator);

    var frame = try TestFrame.init(testing.allocator, res);
    defer frame.deinit(testing.allocator);

    frame.fillGray(device);
    frame.setMarkers(device, 0b111_111_111_111); // all white

    for (0..16) |_| frame.feedTo(&detector);

    try testing.expect(detector.status == .locked);
    try testing.expect(detector.status.locked.eql(device));
}

test "converges from play-area bbox to device rect once markers light up" {
    const res: Resolution = .{ .width = 640, .height = 360 };
    const device: ContentRect = .{ .x = 0, .y = 40, .frame = .{ .width = 640, .height = 280 } };
    const play_area: ContentRect = .{ .x = 80, .y = 80, .frame = .{ .width = 480, .height = 200 } };

    // plenty of headroom: the black-marker phase is *supposed* to fail bursts
    var config = test_config;
    config.max_failed_bursts = 100;

    var detector = try Detector.init(testing.allocator, res, bt709, config);
    defer detector.deinit(testing.allocator);

    var frame = try TestFrame.init(testing.allocator, res);
    defer frame.deinit(testing.allocator);

    // phase A: angle 0, markers black -> bbox is the play area, whose gray
    // corners fail saturation, so the detector must never lock onto it
    frame.fillGray(play_area);
    frame.setMarkers(device, 0);

    for (0..32) |_| frame.feedTo(&detector);
    try testing.expect(detector.status != .locked);

    // phase B: the device starts rotating smoothly; markers light up, the bbox
    // grows to the device rect, and validation passes
    var value: u12 = 2000;
    for (0..128) |_| {
        frame.setMarkers(device, value);
        frame.feedTo(&detector);

        value +%= 8; // ~0.7 degrees per frame

        if (detector.status == .locked) break;
    }

    try testing.expect(detector.status == .locked);
    try testing.expect(detector.status.locked.eql(device));
}

test "jumping angles are rejected and eventually fail" {
    const res: Resolution = .{ .width = 640, .height = 360 };
    const device: ContentRect = .{ .x = 0, .y = 60, .frame = .{ .width = 640, .height = 240 } };

    var detector = try Detector.init(testing.allocator, res, bt709, test_config);
    defer detector.deinit(testing.allocator);

    var frame = try TestFrame.init(testing.allocator, res);
    defer frame.deinit(testing.allocator);

    frame.fillGray(device);

    // saturated corners (red/black), but the decoded angle flips by 180 degrees
    // every frame -- physically impossible, so no lock
    var i: usize = 0;
    while (detector.status == .searching or detector.status == .validating) : (i += 1) {
        frame.setMarkers(device, if (i % 2 == 0) 0 else 0b100_000_000_000);
        frame.feedTo(&detector);

        if (i > 1000) break; // safety net
    }

    try testing.expect(detector.status == .failed);
}

test "bar noise below the luma threshold is ignored" {
    const res: Resolution = .{ .width = 640, .height = 360 };
    const device: ContentRect = .{ .x = 0, .y = 60, .frame = .{ .width = 640, .height = 240 } };

    var detector = try Detector.init(testing.allocator, res, bt709, test_config);
    defer detector.deinit(testing.allocator);

    var frame = try TestFrame.init(testing.allocator, res);
    defer frame.deinit(testing.allocator);

    frame.fillGray(device);
    frame.setMarkers(device, 0b111_111_111_111);

    // compression noise sprinkled through the bars, below the threshold
    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    for (0..200) |_| {
        const x = random.uintLessThan(usize, @intCast(res.width));
        const bar_y = random.uintLessThan(usize, 60);
        const top_or_bottom = if (random.boolean()) bar_y else 300 + bar_y;

        frame.y[top_or_bottom * @as(usize, @intCast(res.width)) + x] = 39; // luma_threshold - 1
    }

    for (0..16) |_| frame.feedTo(&detector);

    try testing.expect(detector.status == .locked);
    try testing.expect(detector.status.locked.eql(device));
}

test "reset returns to searching and re-converges" {
    const res: Resolution = .{ .width = 640, .height = 360 };
    const device: ContentRect = .{ .x = 0, .y = 60, .frame = .{ .width = 640, .height = 240 } };

    var detector = try Detector.init(testing.allocator, res, bt709, test_config);
    defer detector.deinit(testing.allocator);

    var frame = try TestFrame.init(testing.allocator, res);
    defer frame.deinit(testing.allocator);

    frame.fillGray(device);
    frame.setMarkers(device, 0b111_111_111_111);

    for (0..16) |_| frame.feedTo(&detector);
    try testing.expect(detector.status == .locked);

    detector.reset();
    try testing.expect(detector.status == .searching);

    for (0..16) |_| frame.feedTo(&detector);
    try testing.expect(detector.status == .locked);
    try testing.expect(detector.status.locked.eql(device));
}
