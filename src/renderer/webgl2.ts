/**
 * WebGL2 painting backend — a peer of `createCanvas2DRenderer`.
 *
 * It mirrors canvas2d's incremental accumulation on the GPU. Two R8
 * framebuffer-attached count textures at image resolution hold, per pixel, how
 * many selected masks cover it (`bright`) and how many selected masks' dilated
 * edges cover it (`outline`). A selection toggle uploads only the toggled
 * mask's coverage into one reused scratch texture and draws it into those
 * targets with additive (select) or reverse-subtractive (deselect) blending —
 * so a toggle costs one mask's worth of work, not the whole selection's, and
 * peak VRAM is a constant number of textures regardless of how many masks are
 * selected.
 *
 * The added/removed sets come from `resolveAppliedDelta`, the same function
 * canvas2d uses, so the two backends agree on delta semantics by construction
 * rather than by two implementations happening to match.
 *
 * WHAT THIS DOES AND DOES NOT SAVE. It removes the per-rebuild
 * `countsToImageData()` -> `createImageBitmap()` round trip canvas2d performs
 * on the main thread. It does NOT remove the per-mask PNG -> ImageData ->
 * Uint8Array coverage decode in `SegmentViewer.tsx`; that is what produces
 * `Scene.masks[].coverage` and it is unchanged by this file.
 *
 * The scratch texture is uploaded at the MASK's own size (`mask.width` /
 * `mask.height` when present, else the image size) and sampled at image
 * resolution, so the sampler performs the upsample. Nothing emits smaller
 * masks yet; this is the mechanism, not a realized saving.
 *
 * COORDINATE CONVENTION — the one thing that will bite an editor. Nothing is
 * uploaded flipped (`UNPACK_FLIP_Y_WEBGL` stays 0), so for every texture here
 * "texture row 0" is "image row 0". In an FBO pass that means
 * `int(gl_FragCoord.y)` IS the image row index, which is what lets the dash
 * stipple and the edge test use the same integer coordinates canvas2d's CPU
 * loops use. The single flip happens in the composite pass's vertex shader,
 * which maps v = 0 to the TOP of the viewport.
 */
import { resolveAppliedDelta } from '../core';
import type { Renderer, Scene } from './types';

/** canvas2d's DIM_COLOR is `rgba(0, 0, 0, 0.55)`. */
const DIM_ALPHA = 0.55;
/** canvas2d's OUTLINE_COLOR / DRAFT_COLOR, `#f97316` = rgb(249, 115, 22). */
const OUTLINE_R = 249 / 255;
const OUTLINE_G = 115 / 255;
const OUTLINE_B = 22 / 255;
/** Must stay equal to canvas2d's OUTLINE_RADIUS. */
const OUTLINE_RADIUS = 3;
/** Triangles per draft vertex dot. */
const DOT_SEGMENTS = 16;
/**
 * One count in an R8 UNORM target. Additive blending at 1/255 increments a
 * UNORM8 texel by exactly one ULP, so counts are EXACT up to 255 concurrently
 * selected masks. Past 255 a texel saturates and stays lit until the selection
 * drops back under the ceiling. Raising the ceiling means moving the count
 * targets to R16F, which needs EXT_color_buffer_float and is therefore not
 * core WebGL2 — see makeTarget().
 */
const COUNT_UNIT = 1 / 255;

const CONTEXT_ATTRS: WebGLContextAttributes = {
  alpha: false,
  antialias: false,
  depth: false,
  stencil: false,
};

type Gl = WebGL2RenderingContext;
interface Target {
  tex: WebGLTexture;
  fbo: WebGLFramebuffer;
}
interface MaskRef {
  coverage: Uint8Array;
  width: number;
  height: number;
}

/** Attribute-less full-screen triangle; see the coordinate-convention note. */
const VERT_QUAD = `#version 300 es
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = vec2(p.x, 1.0 - p.y);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * Coverage byte 1 becomes 1/255 in an R8 texture, NOT 1.0 — every coverage
 * test in this file compares against 0.5/255.
 */
const FRAG_MASK = `#version 300 es
precision highp float;
uniform sampler2D uMask;
uniform vec2 uImageSize;
uniform float uAmount;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
void main() {
  float c = texture(uMask, gl_FragCoord.xy / uImageSize).r > COV_EPS ? 1.0 : 0.0;
  fragColor = vec4(c * uAmount, 0.0, 0.0, 1.0);
}
`;

/**
 * Same rule as buildEdgeCoverage: a pixel is an edge when it is covered and at
 * least one of its 4 orthogonal neighbours is uncovered OR outside the image,
 * so a mask touching the border still gets an outline along that border.
 */
const FRAG_EDGE = `#version 300 es
precision highp float;
uniform sampler2D uMask;
uniform vec2 uImageSize;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
float cov(vec2 p) {
  return texture(uMask, p / uImageSize).r > COV_EPS ? 1.0 : 0.0;
}
void main() {
  vec2 p = gl_FragCoord.xy;
  if (cov(p) == 0.0) { fragColor = vec4(0.0); return; }
  float l = p.x >= 1.0 ? cov(p - vec2(1.0, 0.0)) : 0.0;
  float r = p.x <= uImageSize.x - 1.0 ? cov(p + vec2(1.0, 0.0)) : 0.0;
  float d = p.y >= 1.0 ? cov(p - vec2(0.0, 1.0)) : 0.0;
  float u = p.y <= uImageSize.y - 1.0 ? cov(p + vec2(0.0, 1.0)) : 0.0;
  fragColor = vec4((l * r * d * u) < 1.0 ? 1.0 : 0.0, 0.0, 0.0, 1.0);
}
`;

/**
 * Half of a radius-N Chebyshev (8-connected square) dilation. It is separable,
 * so two 7-tap passes replace one 49-tap window — RADIUS is injected from
 * OUTLINE_RADIUS so the shader cannot drift from canvas2d's constant. The dash
 * stipple and the count increment ride on the second (vertical) pass.
 */
const FRAG_DILATE = `#version 300 es
precision highp float;
uniform sampler2D uSrc;
uniform vec2 uImageSize;
uniform vec2 uStep;
uniform float uAmount;
uniform bool uDashed;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
const int RADIUS = ${OUTLINE_RADIUS};
void main() {
  ivec2 pi = ivec2(gl_FragCoord.xy);
  if (uDashed && ((pi.x + pi.y) & 1) == 1) { fragColor = vec4(0.0); return; }
  vec2 p = gl_FragCoord.xy;
  float m = 0.0;
  for (int i = -RADIUS; i <= RADIUS; i++) {
    vec2 q = p + uStep * float(i);
    if (q.x < 0.0 || q.y < 0.0 || q.x > uImageSize.x || q.y > uImageSize.y) continue;
    m = max(m, texture(uSrc, q / uImageSize).r > COV_EPS ? 1.0 : 0.0);
  }
  fragColor = vec4(m * uAmount, 0.0, 0.0, 1.0);
}
`;

/**
 * One pass, no per-mask work. Reproduces canvas2d's paint(): base image, then
 * a dim layer filled everywhere and punched out where the ACTIVE bright count
 * (hover when hovering, else selection) is non-zero, then the selection
 * outline and the hover outline as opaque source-over blits.
 */
const FRAG_COMPOSITE = `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uBase;
uniform sampler2D uBright;
uniform sampler2D uOutline;
uniform sampler2D uHoverOutline;
uniform float uDimAlpha;
uniform vec3 uOutlineColor;
out vec4 fragColor;
const float COV_EPS = 0.5 / 255.0;
void main() {
  vec3 col = texture(uBase, vUv).rgb;
  float bright = texture(uBright, vUv).r > COV_EPS ? 1.0 : 0.0;
  col = mix(col, vec3(0.0), uDimAlpha * (1.0 - bright));
  float outline = max(
    texture(uOutline, vUv).r > COV_EPS ? 1.0 : 0.0,
    texture(uHoverOutline, vUv).r > COV_EPS ? 1.0 : 0.0);
  col = mix(col, uOutlineColor, outline);
  fragColor = vec4(col, 1.0);
}
`;

const VERT_DRAFT = `#version 300 es
in vec2 aPos;
uniform vec2 uImageSize;
void main() {
  gl_Position = vec4(
    aPos.x / uImageSize.x * 2.0 - 1.0,
    1.0 - aPos.y / uImageSize.y * 2.0,
    0.0, 1.0);
}
`;

const FRAG_DRAFT = `#version 300 es
precision highp float;
uniform vec3 uColor;
out vec4 fragColor;
void main() { fragColor = vec4(uColor, 1.0); }
`;

/**
 * Triangulate the draft polygon in IMAGE space: one quad per segment plus a
 * disc per vertex. canvas2d strokes the same path with `lineJoin: 'round'` and
 * `lineCap: 'round'`; the vertex discs (radius lineWidth * 1.8, which is wider
 * than the stroke) are what stand in for those round joins and caps, and they
 * are drawn by canvas2d too. The path is closed at 3+ points, matching
 * canvas2d's `if (draftPoints.length >= 3) ctx.closePath()`.
 *
 * Exported for the unit test in this file's sibling suite? No — kept private.
 * The geometry is verified through pixels in tests/browser, not by shape.
 */
function buildDraftGeometry(
  points: readonly { x: number; y: number }[],
  imageWidth: number,
  imageHeight: number,
): Float32Array {
  const lineWidth = Math.max(2, Math.min(imageWidth, imageHeight) / 300);
  const half = lineWidth / 2;
  const dot = lineWidth * 1.8;
  const n = points.length;
  const segments = n >= 3 ? n : n - 1;
  const out: number[] = [];
  for (let i = 0; i < segments; i += 1) {
    const a = points[i];
    const b = points[(i + 1) % n];
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    const len = Math.hypot(dx, dy);
    if (len === 0) continue;
    dx /= len;
    dy /= len;
    const nx = -dy * half;
    const ny = dx * half;
    out.push(a.x + nx, a.y + ny, b.x + nx, b.y + ny, b.x - nx, b.y - ny);
    out.push(a.x + nx, a.y + ny, b.x - nx, b.y - ny, a.x - nx, a.y - ny);
  }
  for (const p of points) {
    for (let k = 0; k < DOT_SEGMENTS; k += 1) {
      const t0 = ((k / DOT_SEGMENTS) * Math.PI) * 2;
      const t1 = (((k + 1) / DOT_SEGMENTS) * Math.PI) * 2;
      out.push(
        p.x, p.y,
        p.x + Math.cos(t0) * dot, p.y + Math.sin(t0) * dot,
        p.x + Math.cos(t1) * dot, p.y + Math.sin(t1) * dot,
      );
    }
  }
  return new Float32Array(out);
}

function compileShader(gl: Gl, type: number, source: string): WebGLShader | null {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, source);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.warn('note-scanner: WebGL2 shader compile failed:', gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

function linkProgram(gl: Gl, vertSrc: string, fragSrc: string): WebGLProgram | null {
  const vs = compileShader(gl, gl.VERTEX_SHADER, vertSrc);
  const fs = vs ? compileShader(gl, gl.FRAGMENT_SHADER, fragSrc) : null;
  if (!vs || !fs) {
    if (vs) gl.deleteShader(vs);
    if (fs) gl.deleteShader(fs);
    return null;
  }
  const prog = gl.createProgram();
  if (!prog) {
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    return null;
  }
  gl.attachShader(prog, vs);
  gl.attachShader(prog, fs);
  // Only the draft program has an attribute; pinning it to 0 lets the draft
  // VAO be configured once at build time instead of queried per frame.
  gl.bindAttribLocation(prog, 0, 'aPos');
  gl.linkProgram(prog);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.warn('note-scanner: WebGL2 program link failed:', gl.getProgramInfoLog(prog));
    gl.deleteProgram(prog);
    return null;
  }
  return prog;
}

/**
 * `filter` is NOT a free knob — which textures may be LINEAR is decided by what
 * is stored in them.
 *
 * NEAREST is mandatory for every COVERAGE texture: the scratch upload, the
 * count/edge targets and the hover targets. The mask -> image-resolution step
 * is the F4 upsample, and a LINEAR filter there yields FRACTIONAL coverage,
 * which corrupts the exact one-ULP count arithmetic the R8 targets rely on (a
 * half-lit texel is not "half a mask"; it is a wrong count).
 *
 * The base PHOTO has no such constraint: it is colour, not a count, and it is
 * magnified by `paint()` sizing the drawing buffer to `imageWidth * dpr` while
 * the texture stays `imageWidth` wide. NEAREST there is hard pixel replication
 * where canvas2d's `drawImage` interpolates, worst at fractional dpr
 * (1.25 / 1.5). So `baseTex` is the one LINEAR texture here. At dpr 1 every
 * sample lands on a texel centre and LINEAR returns exactly the NEAREST value,
 * which is why the dpr-1 AC3 differential is unmoved by this.
 */
function makeTexture(gl: Gl, filter: number = gl.NEAREST): WebGLTexture | null {
  const tex = gl.createTexture();
  if (!tex) return null;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function makeTarget(gl: Gl, width: number, height: number): Target | null {
  const tex = makeTexture(gl);
  if (!tex) return null;
  // R8, not R16F: R8 is color-renderable in core WebGL2, R16F is not without
  // EXT_color_buffer_float. See COUNT_UNIT for the 255-mask ceiling this buys.
  gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R8, width, height);
  const fbo = gl.createFramebuffer();
  if (!fbo) {
    gl.deleteTexture(tex);
    return null;
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteFramebuffer(fbo);
    gl.deleteTexture(tex);
    return null;
  }
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  return { tex, fbo };
}

/**
 * Every (vertex, fragment) pair `createWebGL2Renderer` links, in one list so
 * the probe and `buildPrograms()` cannot drift apart.
 */
const PROGRAM_SOURCES: readonly (readonly [string, string])[] = [
  [VERT_QUAD, FRAG_MASK],
  [VERT_QUAD, FRAG_EDGE],
  [VERT_QUAD, FRAG_DILATE],
  [VERT_QUAD, FRAG_COMPOSITE],
  [VERT_DRAFT, FRAG_DRAFT],
];

/**
 * Does this browser actually give us a usable WebGL2 renderer, right now?
 *
 * A browser can expose `WebGL2RenderingContext` and still refuse a context
 * (blocklisted driver, lost GPU process), and it can hand back a context that
 * then fails to compile, link, or produce a complete R8 framebuffer. So the
 * probe does all four on a 1x1 throwaway canvas and drops the context again.
 * `createDefaultRenderer()` is the only caller.
 *
 * It links ALL FIVE programs, not just the composite one. This is the only
 * failure check that changes the factory's answer — a failure inside an
 * already-selected live instance just makes `paint()` a no-op (see the note on
 * `failed`) — so a driver that compiles the composite shader but rejects, say,
 * the dilation loop must be caught HERE or the viewer paints nothing at all.
 */
export function probeWebGL2Support(): boolean {
  if (typeof document === 'undefined') return false;
  try {
    const probe = document.createElement('canvas');
    probe.width = 1;
    probe.height = 1;
    const gl = probe.getContext('webgl2', CONTEXT_ATTRS) as Gl | null;
    if (!gl) return false;
    const progs: (WebGLProgram | null)[] = [];
    for (const [vert, frag] of PROGRAM_SOURCES) progs.push(linkProgram(gl, vert, frag));
    const linked = progs.every(Boolean);
    const target = linked ? makeTarget(gl, 1, 1) : null;
    for (const p of progs) if (p) gl.deleteProgram(p);
    if (target) {
      gl.deleteFramebuffer(target.fbo);
      gl.deleteTexture(target.tex);
    }
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return Boolean(linked && target);
  } catch {
    // jsdom throws outright from getContext when the `canvas` package is
    // absent, which is exactly the "no WebGL2 here" answer we want.
    return false;
  }
}

export function createWebGL2Renderer(): Renderer {
  let canvas: HTMLCanvasElement | null = null;
  let gl: Gl | null = null;
  let failed = false;
  let contextLost = false;

  let progMask: WebGLProgram | null = null;
  let progEdge: WebGLProgram | null = null;
  let progDilate: WebGLProgram | null = null;
  let progComposite: WebGLProgram | null = null;
  let progDraft: WebGLProgram | null = null;
  let quadVao: WebGLVertexArrayObject | null = null;
  let draftVao: WebGLVertexArrayObject | null = null;
  let draftBuf: WebGLBuffer | null = null;
  const uniforms = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>();

  let texWidth = 0;
  let texHeight = 0;
  let bright: Target | null = null;
  let outline: Target | null = null;
  let edgeA: Target | null = null;
  let edgeB: Target | null = null;
  let hoverBright: Target | null = null;
  let hoverOutline: Target | null = null;
  let scratchTex: WebGLTexture | null = null;
  let baseTex: WebGLTexture | null = null;
  let baseSource: CanvasImageSource | null = null;

  let prevSelected = new Set<string>();
  let hoverKey = '';
  /**
   * The coverage each currently-applied id was added WITH. It holds references
   * to arrays SegmentViewer already owns — no copies, no GPU memory — and
   * exists so a deselect can subtract exactly what was added even after the id
   * has dropped out of `scene.masks`. This is the direct analogue of canvas2d's
   * covIdxCache/edgeIdxCache: cache first, `scene.masks` on a miss.
   */
  const appliedMasks = new Map<string, MaskRef>();

  function uni(prog: WebGLProgram, name: string): WebGLUniformLocation | null {
    let m = uniforms.get(prog);
    if (!m) {
      m = new Map();
      uniforms.set(prog, m);
    }
    if (!m.has(name)) m.set(name, gl!.getUniformLocation(prog, name));
    return m.get(name) ?? null;
  }

  function bindTex(unit: number, tex: WebGLTexture | null, prog: WebGLProgram, name: string): void {
    const g = gl!;
    g.activeTexture(g.TEXTURE0 + unit);
    g.bindTexture(g.TEXTURE_2D, tex);
    g.uniform1i(uni(prog, name), unit);
  }

  function buildPrograms(): boolean {
    const g = gl!;
    // Positional, from the same list the probe links, so the probe cannot end
    // up validating a different set of shaders than the renderer builds.
    [progMask, progEdge, progDilate, progComposite, progDraft] = PROGRAM_SOURCES.map(
      ([vert, frag]) => linkProgram(g, vert, frag),
    );
    quadVao = g.createVertexArray();
    draftVao = g.createVertexArray();
    draftBuf = g.createBuffer();
    if (!progMask || !progEdge || !progDilate || !progComposite || !progDraft) return false;
    if (!quadVao || !draftVao || !draftBuf) return false;
    g.bindVertexArray(draftVao);
    g.bindBuffer(g.ARRAY_BUFFER, draftBuf);
    g.enableVertexAttribArray(0);
    g.vertexAttribPointer(0, 2, g.FLOAT, false, 0, 0);
    g.bindVertexArray(null);
    return true;
  }

  function destroyTargets(): void {
    const g = gl;
    if (!g) return;
    for (const t of [bright, outline, edgeA, edgeB, hoverBright, hoverOutline]) {
      if (!t) continue;
      g.deleteFramebuffer(t.fbo);
      g.deleteTexture(t.tex);
    }
    bright = outline = edgeA = edgeB = hoverBright = hoverOutline = null;
  }

  /**
   * `edgeA` / `edgeB` are the dilation ping-pong pair and are live only inside
   * `runOutlinePasses`, yet they are allocated here and held for the renderer's
   * lifetime — two full-resolution R8 targets, ~24 MB on a 12 MP image. That is
   * a deliberate residency-vs-allocation-churn trade: a selection toggle runs
   * these passes once per added or removed mask, and allocating a pair of
   * image-sized targets per toggle would put that churn on the frame that has
   * to stay responsive.
   */
  function ensureTargets(width: number, height: number): boolean {
    if (bright && texWidth === width && texHeight === height) return true;
    destroyTargets();
    const g = gl!;
    bright = makeTarget(g, width, height);
    outline = makeTarget(g, width, height);
    edgeA = makeTarget(g, width, height);
    edgeB = makeTarget(g, width, height);
    hoverBright = makeTarget(g, width, height);
    hoverOutline = makeTarget(g, width, height);
    texWidth = width;
    texHeight = height;
    // Fresh, zeroed count buffers mean nothing has been accumulated yet.
    prevSelected = new Set();
    appliedMasks.clear();
    hoverKey = '';
    if (!bright || !outline || !edgeA || !edgeB || !hoverBright || !hoverOutline) {
      destroyTargets();
      failed = true;
      return false;
    }
    return true;
  }

  function ensureBase(scene: Scene): void {
    const g = gl!;
    if (baseTex && baseSource === scene.base) return;
    // LINEAR, unlike every other texture here — see makeTexture(). The base is
    // colour magnified to `imageWidth * dpr`, not coverage feeding a count.
    if (!baseTex) baseTex = makeTexture(g, g.LINEAR);
    if (!baseTex) {
      failed = true;
      return;
    }
    g.bindTexture(g.TEXTURE_2D, baseTex);
    // Never flipped — see the coordinate-convention note at the top.
    g.pixelStorei(g.UNPACK_FLIP_Y_WEBGL, 0);
    try {
      // `Scene.base` is a CanvasImageSource, which is exactly TexImageSource
      // plus SVGImageElement — the one member WebGL cannot upload. Narrowing
      // rather than widening keeps the cast honest: an SVGImageElement would
      // make texImage2D throw, and the catch below handles it like any other
      // upload failure. SegmentViewer only ever supplies an HTMLImageElement.
      g.texImage2D(
        g.TEXTURE_2D,
        0,
        g.RGBA,
        g.RGBA,
        g.UNSIGNED_BYTE,
        scene.base as Exclude<CanvasImageSource, SVGImageElement>,
      );
    } catch (err) {
      // KNOWN EDGE: a cross-origin base image served without CORS headers
      // makes texImage2D throw SecurityError, where canvas2d's drawImage would
      // merely taint the canvas and keep painting. SegmentViewer sets
      // `crossOrigin` on its loader, so this is reachable only for a host that
      // serves the image without `Access-Control-Allow-Origin`. Refusing to
      // paint (with this warning) beats painting a black board silently.
      console.warn('note-scanner: WebGL2 renderer cannot upload the base image:', err);
      failed = true;
      return;
    }
    baseSource = scene.base;
  }

  function maskRef(scene: Scene, id: string): MaskRef | null {
    const cached = appliedMasks.get(id);
    if (cached) return cached;
    const mask = scene.masks.get(id);
    if (!mask) return null;
    const width = mask.width ?? scene.imageWidth;
    const height = mask.height ?? scene.imageHeight;
    // A coverage array that disagrees with its declared dimensions would
    // upload garbage rather than a wrong-but-plausible mask, so refuse it and
    // leave the counts untouched.
    if (mask.coverage.length !== width * height) return null;
    return { coverage: mask.coverage, width, height };
  }

  function uploadScratch(ref: MaskRef): void {
    const g = gl!;
    if (!scratchTex) scratchTex = makeTexture(g);
    g.bindTexture(g.TEXTURE_2D, scratchTex);
    g.pixelStorei(g.UNPACK_ALIGNMENT, 1);
    g.texImage2D(
      g.TEXTURE_2D, 0, g.R8, ref.width, ref.height, 0, g.RED, g.UNSIGNED_BYTE, ref.coverage,
    );
  }

  type Blend = 'add' | 'sub' | 'replace';

  function runPass(
    prog: WebGLProgram,
    target: Target,
    blend: Blend,
    setUniforms: (prog: WebGLProgram) => void,
  ): void {
    const g = gl!;
    g.bindFramebuffer(g.FRAMEBUFFER, target.fbo);
    g.viewport(0, 0, texWidth, texHeight);
    if (blend === 'replace') {
      g.disable(g.BLEND);
    } else {
      g.enable(g.BLEND);
      g.blendFunc(g.ONE, g.ONE);
      // FUNC_REVERSE_SUBTRACT is dst - src, which is the deselect direction.
      g.blendEquation(blend === 'add' ? g.FUNC_ADD : g.FUNC_REVERSE_SUBTRACT);
    }
    g.useProgram(prog);
    g.bindVertexArray(quadVao);
    setUniforms(prog);
    g.drawArrays(g.TRIANGLES, 0, 3);
    g.disable(g.BLEND);
  }

  function clearTarget(target: Target): void {
    const g = gl!;
    g.bindFramebuffer(g.FRAMEBUFFER, target.fbo);
    g.viewport(0, 0, texWidth, texHeight);
    g.disable(g.BLEND);
    g.clearColor(0, 0, 0, 0);
    g.clear(g.COLOR_BUFFER_BIT);
  }

  /** scratch -> edgeA (edge) -> edgeB (horizontal half) -> target (vertical half). */
  function runOutlinePasses(target: Target, blend: Blend, dashed: boolean): void {
    const g = gl!;
    runPass(progEdge!, edgeA!, 'replace', (prog) => {
      bindTex(0, scratchTex, prog, 'uMask');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
    });
    runPass(progDilate!, edgeB!, 'replace', (prog) => {
      bindTex(0, edgeA!.tex, prog, 'uSrc');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform2f(uni(prog, 'uStep'), 1, 0);
      g.uniform1f(uni(prog, 'uAmount'), 1);
      g.uniform1i(uni(prog, 'uDashed'), 0);
    });
    runPass(progDilate!, target, blend, (prog) => {
      bindTex(0, edgeB!.tex, prog, 'uSrc');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform2f(uni(prog, 'uStep'), 0, 1);
      g.uniform1f(uni(prog, 'uAmount'), COUNT_UNIT);
      g.uniform1i(uni(prog, 'uDashed'), dashed ? 1 : 0);
    });
  }

  function applyMask(scene: Scene, id: string, blend: 'add' | 'sub'): void {
    const g = gl!;
    const ref = maskRef(scene, id);
    if (!ref) {
      if (blend === 'sub') appliedMasks.delete(id);
      return;
    }
    uploadScratch(ref);
    runPass(progMask!, bright!, blend, (prog) => {
      bindTex(0, scratchTex, prog, 'uMask');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform1f(uni(prog, 'uAmount'), COUNT_UNIT);
    });
    runOutlinePasses(outline!, blend, false);
    if (blend === 'add') appliedMasks.set(id, ref);
    else appliedMasks.delete(id);
  }

  function applyDelta(scene: Scene): void {
    const { added, removed, applied } = resolveAppliedDelta(
      prevSelected,
      scene.selectedIds,
      (id) => scene.masks.has(id),
    );
    for (const id of added) applyMask(scene, id, 'add');
    for (const id of removed) applyMask(scene, id, 'sub');
    prevSelected = applied;
  }

  /** Hover is one mask that REPLACES the selection's bright window, so it
   *  never touches the count targets. Same `hoverKey` guard canvas2d uses. */
  function rebuildHover(scene: Scene): void {
    const g = gl!;
    const key = scene.hoveredId ?? '';
    if (key === hoverKey) return;
    hoverKey = key;
    const ref = scene.hoveredId ? maskRef(scene, scene.hoveredId) : null;
    if (!ref) {
      clearTarget(hoverBright!);
      clearTarget(hoverOutline!);
      return;
    }
    uploadScratch(ref);
    runPass(progMask!, hoverBright!, 'replace', (prog) => {
      bindTex(0, scratchTex, prog, 'uMask');
      g.uniform2f(uni(prog, 'uImageSize'), texWidth, texHeight);
      g.uniform1f(uni(prog, 'uAmount'), COUNT_UNIT);
    });
    runOutlinePasses(hoverOutline!, 'replace', true);
  }

  function drawDraft(scene: Scene): void {
    if (scene.draftPoints.length === 0) return;
    const g = gl!;
    const verts = buildDraftGeometry(scene.draftPoints, scene.imageWidth, scene.imageHeight);
    if (verts.length === 0) return;
    g.useProgram(progDraft!);
    g.bindVertexArray(draftVao);
    g.bindBuffer(g.ARRAY_BUFFER, draftBuf);
    g.bufferData(g.ARRAY_BUFFER, verts, g.DYNAMIC_DRAW);
    g.uniform2f(uni(progDraft!, 'uImageSize'), scene.imageWidth, scene.imageHeight);
    g.uniform3f(uni(progDraft!, 'uColor'), OUTLINE_R, OUTLINE_G, OUTLINE_B);
    g.drawArrays(g.TRIANGLES, 0, verts.length / 2);
  }

  function composite(scene: Scene): void {
    const g = gl!;
    g.bindFramebuffer(g.FRAMEBUFFER, null);
    g.viewport(0, 0, canvas!.width, canvas!.height);
    g.disable(g.BLEND);
    g.clearColor(0, 0, 0, 1);
    g.clear(g.COLOR_BUFFER_BIT);
    g.useProgram(progComposite!);
    g.bindVertexArray(quadVao);
    bindTex(0, baseTex, progComposite!, 'uBase');
    bindTex(1, scene.hoveredId ? hoverBright!.tex : bright!.tex, progComposite!, 'uBright');
    bindTex(2, outline!.tex, progComposite!, 'uOutline');
    bindTex(3, hoverOutline!.tex, progComposite!, 'uHoverOutline');
    g.uniform1f(uni(progComposite!, 'uDimAlpha'), DIM_ALPHA);
    g.uniform3f(uni(progComposite!, 'uOutlineColor'), OUTLINE_R, OUTLINE_G, OUTLINE_B);
    g.drawArrays(g.TRIANGLES, 0, 3);
    drawDraft(scene);
  }

  function paint(scene: Scene): void {
    if (!canvas || !gl || failed || contextLost) return;
    const { imageWidth: w, imageHeight: h, devicePixelRatio: dpr } = scene;
    // Assigning canvas.width/height reallocates and clears the drawing buffer
    // even when the value is unchanged, and SegmentViewer paints on every
    // hover. Skipping the no-op assignment is safe because composite() clears
    // the default framebuffer itself before every frame.
    const nextW = Math.floor(w * dpr);
    const nextH = Math.floor(h * dpr);
    if (canvas.width !== nextW) canvas.width = nextW;
    if (canvas.height !== nextH) canvas.height = nextH;
    if (!ensureTargets(w, h)) return;
    ensureBase(scene);
    if (failed) return;
    applyDelta(scene);
    rebuildHover(scene);
    composite(scene);
  }

  const onContextLost = (event: Event): void => {
    // Without preventDefault the context is never restorable.
    event.preventDefault();
    contextLost = true;
  };

  const onContextRestored = (): void => {
    contextLost = false;
    if (!gl) return;
    // Every GL object created before the loss is gone. Rebuild from nothing and
    // re-accumulate the selection on the next draw.
    uniforms.clear();
    progMask = progEdge = progDilate = progComposite = progDraft = null;
    quadVao = draftVao = null;
    draftBuf = null;
    bright = outline = edgeA = edgeB = hoverBright = hoverOutline = null;
    scratchTex = null;
    baseTex = null;
    baseSource = null;
    texWidth = texHeight = 0;
    prevSelected = new Set();
    appliedMasks.clear();
    hoverKey = '';
    failed = !buildPrograms();
  };

  /**
   * Full GL teardown for whichever canvas `canvas` currently points at.
   * Shared by `dispose()` and by `init()`'s different-canvas path: `Renderer`
   * is public API and nothing stops a caller from calling `init()` again with
   * a second canvas on the same instance. Without this, the first canvas's
   * context, every program/texture/VAO/buffer built for it, and its
   * context-loss listeners would leak — nothing else ever tears them down.
   */
  function teardownGl(): void {
    const g = gl;
    if (canvas) {
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);
    }
    if (g) {
      destroyTargets();
      if (scratchTex) g.deleteTexture(scratchTex);
      if (baseTex) g.deleteTexture(baseTex);
      for (const p of [progMask, progEdge, progDilate, progComposite, progDraft]) {
        if (p) g.deleteProgram(p);
      }
      if (quadVao) g.deleteVertexArray(quadVao);
      if (draftVao) g.deleteVertexArray(draftVao);
      if (draftBuf) g.deleteBuffer(draftBuf);
    }
    uniforms.clear();
    progMask = progEdge = progDilate = progComposite = progDraft = null;
    quadVao = draftVao = null;
    draftBuf = null;
    scratchTex = null;
    baseTex = null;
    baseSource = null;
    texWidth = texHeight = 0;
    prevSelected = new Set();
    appliedMasks.clear();
    hoverKey = '';
    failed = false;
    contextLost = false;
    gl = null;
  }

  return {
    init(next) {
      // SegmentViewer calls init() before every draw, so the common case (same
      // canvas, already bound) must be cheap and must not re-register the
      // context-loss listeners.
      if (canvas === next && (gl || failed)) return;
      if (canvas && canvas !== next && (gl || failed)) teardownGl();
      canvas = next;
      failed = false;
      contextLost = false;
      gl = next.getContext('webgl2', CONTEXT_ATTRS) as Gl | null;
      if (!gl) {
        failed = true;
        return;
      }
      next.addEventListener('webglcontextlost', onContextLost);
      next.addEventListener('webglcontextrestored', onContextRestored);
      failed = !buildPrograms();
    },
    draw(scene) {
      paint(scene);
    },
    resize(scene) {
      paint(scene);
    },
    dispose() {
      teardownGl();
      canvas = null;
    },
    /**
     * Drop the retained coverage reference for these ids.
     *
     * Per the Renderer interface's contract this must NOT touch the count
     * targets or `prevSelected`: subtracting here as well as in
     * `resolveAppliedDelta`'s `removed` branch would double-count and drive the
     * R8 counts below zero, where they clamp at 0 and permanently un-brighten
     * pixels other selected masks still cover.
     *
     * No GPU resource is per-id in this renderer — the scratch texture is
     * re-uploaded per delta and nothing per-mask is retained (which is why
     * AC5's texture bound holds) — so this drops only the CPU-side coverage
     * reference, exactly mirroring canvas2d evicting its index caches.
     */
    evict(ids) {
      for (const id of ids) appliedMasks.delete(id);
    },
  };
}
