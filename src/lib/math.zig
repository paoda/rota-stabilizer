pub const Vec2 = struct { inner: [2]f32 };

pub const Mat2 = struct {
    inner: [4]f32,

    pub const identity: @This() = .{ .inner = .{ 1, 0, 0, 1 } };

    pub fn scale(s: f32) @This() {
        return .{ .inner = .{ s, 0, 0, s } };
    }

    pub fn scaleXy(sx: f32, sy: f32) @This() {
        return .{ .inner = .{ sx, 0, 0, sy } };
    }

    pub fn mul(left: @This(), right: @This()) @This() {
        const l = left.inner;
        const r = right.inner;

        return .{
            .inner = .{
                l[0] * r[0] + l[2] * r[1],
                l[1] * r[0] + l[3] * r[1],
                l[0] * r[2] + l[2] * r[3],
                l[1] * r[2] + l[3] * r[3],
            },
        };
    }
};

pub fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) Mat2 {
    return .{ .inner = .{ m00, m01, m10, m11 } };
}
