pub const Vec2 = struct {
    v: [2]f32,

    pub inline fn x(self: @This()) f32 {
        return self.v[0];
    }

    pub inline fn y(self: @This()) f32 {
        return self.v[1];
    }
};

pub fn vec2(x: f32, y: f32) Vec2 {
    return .{ .v = .{ x, y } };
}

pub const Mat2 = struct {
    m: [4]f32,

    pub const identity: @This() = mat2(1, 0, 0, 1);

    pub fn scale(s: f32) @This() {
        return mat2(s, 0, 0, s);
    }

    pub fn scaleXy(sx: f32, sy: f32) @This() {
        return mat2(sx, 0, 0, sy);
    }

    pub fn mul(left: @This(), right: @This()) @This() {
        const l = left.m;
        const r = right.m;

        return mat2(
            l[0] * r[0] + l[2] * r[1],
            l[1] * r[0] + l[3] * r[1],
            l[0] * r[2] + l[2] * r[3],
            l[1] * r[2] + l[3] * r[3],
        );
    }
};

pub fn mat2(m00: f32, m01: f32, m10: f32, m11: f32) Mat2 {
    return .{ .m = .{ m00, m01, m10, m11 } };
}
