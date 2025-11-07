#version 450
layout(local_size_x = 16, local_size_y = 16) in;

// storage image we will write to
layout(binding = 0, rgba32f) uniform writeonly image2D destImage;
// push constants for resolution / time / seed
layout(push_constant) uniform Push {
    ivec2 size;
    float time;
} pushConsts;

// Simple hash-based noise
uint hash(uvec2 v) {
    v = (v * 1664525u + 1013904223u);
    v.x += v.y * 1664525u;
    return v.x ^ v.y;
}
float rnd(ivec2 p) {
    return float(hash(uvec2(p))) / 4294967295.0;
}
float noise(vec2 uv) {
    ivec2 i = ivec2(floor(uv));
    vec2 f = fract(uv);
    float a = rnd(i);
    float b = rnd(i + ivec2(1,0));
    float c = rnd(i + ivec2(0,1));
    float d = rnd(i + ivec2(1,1));
    vec2 u = f*f*(3.0-2.0*f);
    return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}

void main() {
    ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (gid.x >= pushConsts.size.x || gid.y >= pushConsts.size.y) return;

    vec2 uv = vec2(gid) / vec2(pushConsts.size);
    // move & scale uv with time for animation if wanted
    float t = pushConsts.time * 0.1;
    float n = 0.0;
    float amp = 1.0;
    float freq = 1.0;
    // simple fbm
    for (int i = 0; i < 5; ++i) {
        n += amp * noise((uv * freq + vec2(t)));
        freq *= 2.0;
        amp *= 0.5;
    }
    // color mapping
    vec3 col = vec3(n);
    // add a radial vignette
    vec2 c = uv - 0.5;
    float r = length(c);
    col *= smoothstep(0.8, 0.3, r);

    imageStore(destImage, gid, vec4(col, 1.0));
}