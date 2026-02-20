#version 330 core
out vec4 frag_color;

in vec2 uv;

uniform sampler2D u_y_tex;
uniform sampler2D u_uv_tex;

uniform float u_ratio;
uniform uint u_colour_space;

const float border_radius = 20;
const float border = 0.0075;

const float Y_OFFSET = 16.0 / 255.0;
const float Y_SCALE = 255.0 / (235.0 - 16.0);
const float UV_SCALE = 255.0 / (240.0 - 16.0);

mat3 colourSpace() {
    const uint AVCOL_SPC_BT709 = 1u; // BT.709
    const uint AVCOL_SPC_BT470BG = 5u; // BT.601 (NTSC)
    const uint AVCOL_SPC_SMPTE170M = 6u; // BT.601 (PAL)

    const mat3 bt601 = mat3(
            1.0, 1.0, 1.0,
            0.0, -0.39465, 2.03211,
            1.13983, -0.58060, 0.0
        );

    const mat3 bt709 = mat3(
            1.0, 1.0, 1.0,
            0.0, -0.1873, 1.8556,
            1.5748, -0.4681, 0.0
        );

     switch (u_colour_space) {
        case AVCOL_SPC_BT709:
            return bt709;

        case AVCOL_SPC_BT470BG:
        case AVCOL_SPC_SMPTE170M :
            return bt601;
        default:
            return bt709; // FIXME: do i need to suport BT.2020?
    }
}

vec3 nv12ToRgb(float y_norm, vec2 uv_norm) {
    float y = (y_norm - Y_OFFSET) * Y_SCALE;
    vec2 uv = (uv_norm - Y_OFFSET) * UV_SCALE;

    y = clamp(y, 0.0, 1.0);
    uv = clamp(uv, 0.0, 1.0);

    vec3 yuv = vec3(y, uv.r - 0.5, uv.g - 0.5);

    // TODO: select colorspace based on AVFrame
    return clamp(colourSpace() * yuv, 0.0, 1.0);
}

// https://gamedev.stackexchange.com/questions/205467/add-a-rounded-border-to-a-texture-with-a-fragment-shader
float calcDistance(ivec2 resolution, vec2 uv) {
    vec2 positionInQuadrant = abs(uv * 2.0 - 1.0);
    vec2 extend = vec2(resolution) / 2.0;
    vec2 coords = positionInQuadrant * (extend + border_radius);
    vec2 delta = max(coords - extend, 0.);
    return length(delta);
}

void main() {
    ivec2 resolution = textureSize(u_y_tex, 0);
    float W = float(resolution.x);
    float H = float(resolution.y);

    float gameplay_height = W / u_ratio;
    float height_diff = H - gameplay_height;
    float threshold = (height_diff / 2) / H;
    float unit = 1.0 / H;

    if (uv.y < threshold || uv.y > (1 - threshold)) {
        discard;
    }

    vec2 content_uv = uv; // content uv

    if (height_diff > 0.0) {
        float gameplay_height_normalized = gameplay_height / H;

        if (gameplay_height_normalized > unit) { // If there is at least 1 line of pixels
            content_uv.y = (uv.y - threshold) / gameplay_height_normalized;
        } else {
            content_uv.y = 0.5; // FIXME: do we even care about what this is?
        }
    }

    float dist = calcDistance(resolution, content_uv);
    if (dist > border_radius) discard;

    if (content_uv.x >= (1.0 - border) || content_uv.x <= border || content_uv.y >= (1.0 - border * 2.0) || content_uv.y <= border * 2.0) {
        frag_color = vec4(vec3(1.0), 0.7); // TODO: make alpha channel runtime available?
        return;
    }

    vec3 rgb = nv12ToRgb(texture(u_y_tex, uv).r, texture(u_uv_tex, uv).rg);
    frag_color = vec4(rgb, 1.0);
}
