// stochastic-mountain.js — photographic, generative SoCal terrain for praxagent.ai
// Keeps the original StochasticMountain(canvasId) API while replacing the
// line-walk canvas with a single adaptive WebGL2 pass.

(function (global) {
  'use strict';

  var scriptUrl = document.currentScript && document.currentScript.src
    ? document.currentScript.src
    : document.baseURI;
  var assetBase = new URL('.', scriptUrl);
  var desktopImageUrl = new URL('assets/joshua-tree-bg.webp', assetBase).href;
  var mobileImageUrl = new URL('assets/joshua-tree-bg-mobile.webp', assetBase).href;
  var groundMaskUrl = new URL('assets/joshua-tree-ground-mask.png', assetBase).href;

  var QUAD_VERTEX_SHADER = `#version 300 es
layout(location = 0) in vec2 a_position;
out vec2 v_screenUv;

void main() {
  v_screenUv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

  function landscapeFragmentShader(lowPower) {
    return `#version 300 es
#define LOW_POWER ${lowPower ? 1 : 0}
precision highp float;

uniform sampler2D u_image;
uniform sampler2D u_groundMask;
uniform vec2 u_resolution;
uniform vec2 u_imageSize;
uniform float u_time;
uniform float u_motionAmount;
uniform float u_darkMode;

in vec2 v_screenUv;
out vec4 outColor;

vec2 coverUv(vec2 screenUv) {
  float screenAspect = u_resolution.x / u_resolution.y;
  float imageAspect = u_imageSize.x / u_imageSize.y;
  vec2 crop = vec2(1.0);

  if (screenAspect > imageAspect) {
    crop.y = imageAspect / screenAspect;
  } else {
    crop.x = screenAspect / imageAspect;
  }

  float centerTop = 0.5 + 0.12 * (1.0 - crop.y);
  vec2 centerBottom = vec2(0.5, 1.0 - centerTop);
  return (screenUv - 0.5) * crop + centerBottom;
}

float hash21(vec2 point) {
  point = fract(point * vec2(123.34, 456.21));
  point += dot(point, point + 45.32);
  return fract(point.x * point.y);
}

float noise21(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  local = local * local * (3.0 - 2.0 * local);

  return mix(
    mix(hash21(cell), hash21(cell + vec2(1.0, 0.0)), local.x),
    mix(hash21(cell + vec2(0.0, 1.0)), hash21(cell + vec2(1.0)), local.x),
    local.y
  );
}

float fbm(vec2 point) {
  float value = 0.0;
  float amplitude = 0.5;
  mat2 rotation = mat2(0.80, -0.60, 0.60, 0.80);

  for (int index = 0; index < 4; index++) {
#if LOW_POWER
    if (index == 3) break;
#endif
    value += amplitude * noise21(point);
    point = rotation * point * 2.03 + 17.1;
    amplitude *= 0.5;
  }

  return value;
}

float luminance(vec3 color) {
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

float cloudWeight(vec3 color) {
  float light = smoothstep(0.48, 0.70, luminance(color));
  float neutral = 1.0 - smoothstep(0.035, 0.155, color.b - color.r);
  return light * (0.34 + neutral * 0.66);
}

float cloudAt(vec2 imageUv) {
  return cloudWeight(texture(u_image, clamp(imageUv, 0.001, 0.999)).rgb);
}

float turingField(vec2 point, float time) {
  vec2 warp = vec2(
    fbm(point * 4.7 + vec2(time * 0.025, -time * 0.018)),
    fbm(point * 5.3 + vec2(-time * 0.021, time * 0.016) + 8.4)
  );
  vec2 field = point * vec2(19.0, 23.0) + (warp - 0.5) * 4.2;
  float horizontal = sin(
    field.x + 1.7 * sin(field.y * 0.74 - time * 0.38)
  );
  float vertical = sin(
    field.y * 1.12 + 1.45 * sin(field.x * 0.63 + time * 0.31)
  );
  return clamp(0.5 + 0.25 * (horizontal + vertical), 0.0, 1.0);
}

vec3 webCell(vec2 point) {
  vec2 cell = floor(point);
  vec2 local = fract(point);
  float nearest = 1e9;
  float secondNearest = 1e9;
  vec2 nearestVector = vec2(0.0);
  vec2 secondVector = vec2(0.0);

  for (int y = -1; y <= 1; y++) {
    for (int x = -1; x <= 1; x++) {
      vec2 neighbor = vec2(float(x), float(y));
      vec2 id = cell + neighbor;
      vec2 jitter = 0.18 + 0.64 * vec2(
        hash21(id + vec2(17.17, 53.71)),
        hash21(id + vec2(91.73, 27.19))
      );
      vec2 delta = neighbor + jitter - local;
      float distanceSquared = dot(delta, delta);

      if (distanceSquared < nearest) {
        secondNearest = nearest;
        secondVector = nearestVector;
        nearest = distanceSquared;
        nearestVector = delta;
      } else if (distanceSquared < secondNearest) {
        secondNearest = distanceSquared;
        secondVector = delta;
      }
    }
  }

  float separation = max(length(secondVector - nearestVector), 0.001);
  float edgeDistance = (secondNearest - nearest) / (2.0 * separation);
  vec2 edgeNormal = normalize(nearestVector - secondVector);
  return vec3(max(edgeDistance, 0.0), edgeNormal);
}

void main() {
  vec2 imageUv = coverUv(v_screenUv);
  vec2 topUv = vec2(imageUv.x, 1.0 - imageUv.y);
  vec2 imagePixel = 1.0 / u_imageSize;
  vec2 maskPixel = 1.0 / vec2(textureSize(u_groundMask, 0));
  vec3 source = texture(u_image, imageUv).rgb;
  float ground = clamp(texture(u_groundMask, imageUv).r, 0.0, 1.0);
  float skyMask = 1.0 - ground;
  vec3 sky = source;

  // The branch is coherent across broad sky/ground regions. It avoids running
  // the cloud pass underneath opaque terrain while preserving feathered edges.
  if (ground < 0.999) {
    float cloudEnvelope = cloudWeight(source);
    cloudEnvelope = max(cloudEnvelope, cloudAt(imageUv + imagePixel * vec2(10.0, 0.0)));
    cloudEnvelope = max(cloudEnvelope, cloudAt(imageUv - imagePixel * vec2(10.0, 0.0)));
    cloudEnvelope = max(cloudEnvelope, cloudAt(imageUv + imagePixel * vec2(0.0, 8.0)));
    cloudEnvelope = max(cloudEnvelope, cloudAt(imageUv - imagePixel * vec2(0.0, 8.0)));
#if !LOW_POWER
    cloudEnvelope = max(cloudEnvelope, cloudAt(imageUv + imagePixel * vec2(7.0, 6.0)));
    cloudEnvelope = max(cloudEnvelope, cloudAt(imageUv - imagePixel * vec2(7.0, 6.0)));
#endif
    cloudEnvelope = smoothstep(0.08, 0.78, cloudEnvelope);

    vec2 cloudPoint = topUv * vec2(3.5, 5.9);
    float cloudNoiseA = fbm(
      cloudPoint + vec2(u_time * 0.105, -u_time * 0.070)
    );
    float cloudNoiseB = fbm(
      cloudPoint.yx * 1.13 + vec2(-u_time * 0.086, u_time * 0.058) + 12.4
    );
    float lateralBreath = sin(u_time * 0.245 + topUv.y * 1.8);
    vec2 cloudFlowTop = vec2(
      (cloudNoiseA - 0.5) * 0.0080
        + sin(topUv.y * 15.0 + u_time * 0.32) * 0.0055
        + lateralBreath * 0.0040,
      (cloudNoiseB - 0.5) * 0.0060
        + cos(topUv.x * 12.0 - u_time * 0.27) * 0.0032
    );
    cloudFlowTop *= u_motionAmount;

    vec2 cloudFlowUv = vec2(cloudFlowTop.x, -cloudFlowTop.y);
    vec2 cloudTangentTop = normalize(vec2(1.0, 0.22 + (cloudNoiseB - 0.5) * 0.72));
    vec2 cloudTangentUv = vec2(cloudTangentTop.x, -cloudTangentTop.y);
    float silkRadius = mix(2.2, 6.5, cloudEnvelope);
    vec2 silkStep = cloudTangentUv * silkRadius / u_imageSize;
    vec2 silkCenter = clamp(imageUv + cloudFlowUv, 0.001, 0.999);

#if LOW_POWER
    vec3 silk = texture(u_image, silkCenter).rgb * 0.56;
    silk += texture(u_image, clamp(silkCenter + silkStep, 0.001, 0.999)).rgb * 0.22;
    silk += texture(u_image, clamp(silkCenter - silkStep, 0.001, 0.999)).rgb * 0.22;
    vec3 movingClouds = silk;
#else
    vec3 silk = texture(u_image, silkCenter).rgb * 0.34;
    silk += texture(u_image, clamp(silkCenter + silkStep, 0.001, 0.999)).rgb * 0.22;
    silk += texture(u_image, clamp(silkCenter - silkStep, 0.001, 0.999)).rgb * 0.22;
    silk += texture(u_image, clamp(silkCenter + silkStep * 2.0, 0.001, 0.999)).rgb * 0.11;
    silk += texture(u_image, clamp(silkCenter - silkStep * 2.0, 0.001, 0.999)).rgb * 0.11;

    vec2 shadowFlow = vec2(-cloudFlowTop.x * 0.46, cloudFlowTop.y * 0.36);
    vec3 shadowVeil = texture(
      u_image,
      clamp(imageUv + shadowFlow + silkStep * 0.55, 0.001, 0.999)
    ).rgb;
    vec3 movingClouds = mix(silk, shadowVeil, 0.17);
#endif

    float movedCloud = cloudWeight(movingClouds);
    float movingEnvelope = max(cloudEnvelope, movedCloud);
    float skyMotion = (0.32 + movingEnvelope * 0.68) * skyMask * u_motionAmount;
    sky = mix(source, movingClouds, skyMotion);
  }

  vec3 terrain = source;
  float groundMotion = 0.0;

  // The terrain pass runs only where the real photo mask says there is ground.
  if (ground > 0.001) {
    // Slower, shallower movement keeps the ground alive without making the
    // high-contrast rock detail feel visually unstable.
    float groundTime = u_time * 0.62;
    float pattern = turingField(topUv, groundTime);
    float travelingWave = 0.5 + 0.5 * sin(
      (topUv.x * 0.72 + topUv.y * 1.34) * 28.0
        - groundTime * 1.35 + pattern * 4.0
    );
    vec2 baseFlowTop = vec2(
      sin(pattern * 6.2831853 + groundTime * 0.33),
      cos(pattern * 5.41 - groundTime * 0.29)
    );
    baseFlowTop *= 0.0004 + 0.00125 * travelingWave;

    vec2 webPoint = topUv * vec2(12.0, 16.0);
    webPoint += vec2(
      0.22 * sin(webPoint.y * 0.57 + 0.35 * sin(webPoint.x * 0.31)),
      0.18 * sin(webPoint.x * 0.49 - 0.30 * sin(webPoint.y * 0.37))
    );

    vec3 webData = webCell(webPoint);
    float edgeDistance = webData.x;
    vec2 edgeNormalTop = webData.yz;
    float webPhase = dot(webPoint, vec2(0.72, 0.41))
      - groundTime * 1.05 + pattern * 3.0;
    float webSigned = sin(webPhase);

    float antialiasWidth = clamp(
      12.0 / min(u_resolution.x, u_resolution.y),
      0.0015,
      0.040
    );
    float webHalo = 1.0 - smoothstep(
      0.050 + antialiasWidth,
      0.220 + antialiasWidth,
      edgeDistance
    );
    float primaryThread = 1.0 - smoothstep(
      max(0.0, 0.018 - antialiasWidth),
      0.062 + antialiasWidth,
      edgeDistance
    );
    float echoCenterA = 0.095 + 0.016 * sin(webPhase * 0.72);
    float echoThreadA = 1.0 - smoothstep(
      0.014 + antialiasWidth,
      0.046 + antialiasWidth,
      abs(edgeDistance - echoCenterA)
    );
#if !LOW_POWER
    float echoCenterB = 0.175 + 0.020 * sin(webPhase * 0.54 + 1.7);
    float echoThreadB = 1.0 - smoothstep(
      0.012 + antialiasWidth,
      0.040 + antialiasWidth,
      abs(edgeDistance - echoCenterB)
    );
#endif

    vec2 refractionTop = baseFlowTop * (0.78 + 0.30 * webHalo);
    refractionTop += edgeNormalTop * (0.00032 * webHalo * webSigned);
    vec2 refractionUv = vec2(refractionTop.x, -refractionTop.y);
    vec2 terrainCenter = clamp(imageUv + refractionUv, 0.001, 0.999);
    float displacedGround = texture(u_groundMask, terrainCenter).r;
    groundMotion = smoothstep(
      0.10,
      0.68,
      min(ground, displacedGround)
    ) * u_motionAmount;

    terrain = texture(u_image, terrainCenter).rgb;
    float maskUp8 = texture(
      u_groundMask,
      clamp(imageUv + vec2(0.0, maskPixel.y * 8.0), 0.001, 0.999)
    ).r;
    float maskUp20 = texture(
      u_groundMask,
      clamp(imageUv + vec2(0.0, maskPixel.y * 20.0), 0.001, 0.999)
    ).r;
#if !LOW_POWER
    float maskUp38 = texture(
      u_groundMask,
      clamp(imageUv + vec2(0.0, maskPixel.y * 38.0), 0.001, 0.999)
    ).r;
#endif
    float ridgeThreadA = clamp(ground - maskUp8, 0.0, 1.0);
    float ridgeThreadB = clamp(maskUp8 - maskUp20, 0.0, 1.0);
#if !LOW_POWER
    float ridgeThreadC = clamp(maskUp20 - maskUp38, 0.0, 1.0);
#endif
    float ridgePhase = topUv.x * 31.0 - groundTime * 1.18 + pattern * 2.6;

    float webRelief = primaryThread * webSigned * 0.015;
    webRelief += echoThreadA * sin(webPhase * 1.13 + 1.2) * 0.010;
#if !LOW_POWER
    webRelief += echoThreadB * sin(webPhase * 0.91 + 3.0) * 0.006;
#endif
    float ridgeRelief = ridgeThreadA * sin(ridgePhase) * 0.022;
    ridgeRelief += ridgeThreadB * sin(ridgePhase * 0.91 + 1.8) * 0.014;
#if !LOW_POWER
    ridgeRelief += ridgeThreadC * sin(ridgePhase * 0.77 + 3.4) * 0.009;
#endif
    float organicRelief = (pattern - 0.5) * 0.020;
    terrain *= 1.0
      + (webRelief + ridgeRelief + organicRelief) * groundMotion * 0.48;
  }

  vec3 baseScene = mix(sky, source, ground);
  vec3 color = mix(baseScene, terrain, groundMotion);
  float vignette = 1.0 - 0.12 * smoothstep(0.20, 0.86, length(v_screenUv - 0.5));
  color *= vignette;
  color = mix(color, color * vec3(0.70, 0.76, 0.84), 0.42 * u_darkMode);
  outColor = vec4(color, 1.0);
}
`;
  }

  function compileShader(gl, type, source) {
    var shader = gl.createShader(type);
    if (!shader) throw new Error('Unable to create a WebGL shader.');

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      var message = gl.getShaderInfoLog(shader) || 'Unknown shader error.';
      gl.deleteShader(shader);
      throw new Error(message);
    }

    return shader;
  }

  function createProgram(gl, vertexSource, fragmentSource) {
    var vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    var fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    var program = gl.createProgram();

    if (!program) throw new Error('Unable to create a WebGL program.');

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      var message = gl.getProgramInfoLog(program) || 'Unknown program error.';
      gl.deleteProgram(program);
      throw new Error(message);
    }

    return program;
  }

  function createTexture(gl, image, textureUnit) {
    var texture = gl.createTexture();
    if (!texture) throw new Error('Unable to create a WebGL texture.');

    gl.activeTexture(textureUnit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      image
    );
    return texture;
  }

  function loadImage(url, priority) {
    return new Promise(function (resolve, reject) {
      var image = new Image();
      image.decoding = 'async';
      if ('fetchPriority' in image) image.fetchPriority = priority || 'auto';
      image.onload = function () { resolve(image); };
      image.onerror = function () { reject(new Error('Unable to load ' + url)); };
      image.src = url;

      if (image.complete && image.naturalWidth) resolve(image);
    });
  }

  global.StochasticMountain = function (canvasId) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return null;

    if (canvas.__stochasticMountain) {
      canvas.__stochasticMountain.destroy();
    }

    var gl = null;
    try {
      gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        premultipliedAlpha: false,
        preserveDrawingBuffer: false,
        powerPreference: 'default'
      });
    } catch (error) {
      gl = null;
    }

    var destroyed = false;
    var contextLost = false;
    var intersecting = true;
    var compact = canvas.getBoundingClientRect().width < 720;
    var animationFrame = 0;
    var simulationTime = 0;
    var lastDraw = 0;
    var previousFrame = performance.now();
    var resources = null;
    var images = null;
    var resizeObserver = null;
    var intersectionObserver = null;
    var themeObserver = null;
    var initGeneration = 0;
    var reducedMotion = global.matchMedia('(prefers-reduced-motion: reduce)');

    function isDarkMode() {
      return document.documentElement.getAttribute('data-theme') === 'dark';
    }

    function deleteResources() {
      if (!resources || !gl || contextLost) {
        resources = null;
        return;
      }

      gl.deleteProgram(resources.program);
      gl.deleteTexture(resources.imageTexture);
      gl.deleteTexture(resources.maskTexture);
      gl.deleteBuffer(resources.quadBuffer);
      gl.deleteVertexArray(resources.quadVao);
      resources = null;
    }

    function createPipeline(isCompact) {
      var program = createProgram(
        gl,
        QUAD_VERTEX_SHADER,
        landscapeFragmentShader(isCompact)
      );

      return {
        program: program,
        image: gl.getUniformLocation(program, 'u_image'),
        groundMask: gl.getUniformLocation(program, 'u_groundMask'),
        resolution: gl.getUniformLocation(program, 'u_resolution'),
        imageSize: gl.getUniformLocation(program, 'u_imageSize'),
        time: gl.getUniformLocation(program, 'u_time'),
        motion: gl.getUniformLocation(program, 'u_motionAmount'),
        darkMode: gl.getUniformLocation(program, 'u_darkMode')
      };
    }

    function measureCanvas() {
      var bounds = canvas.getBoundingClientRect();
      var nextCompact = bounds.width < 720;
      var qualityChanged = nextCompact !== compact;
      compact = nextCompact;

      if (!bounds.width || !bounds.height) {
        return { valid: false, qualityChanged: qualityChanged };
      }

      var dprCap = compact ? 1 : 1.25;
      var pixelBudget = compact ? 750000 : 1600000;
      var budgetDpr = Math.sqrt(pixelBudget / (bounds.width * bounds.height));
      var dpr = Math.max(0.35, Math.min(
        global.devicePixelRatio || 1,
        dprCap,
        budgetDpr
      ));
      var width = Math.max(1, Math.round(bounds.width * dpr));
      var height = Math.max(1, Math.round(bounds.height * dpr));

      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }

      return { valid: true, qualityChanged: qualityChanged };
    }

    function replacePipeline() {
      if (!resources || !gl || contextLost) return;
      var nextPipeline = createPipeline(compact);
      gl.deleteProgram(resources.program);
      resources.program = nextPipeline.program;
      resources.image = nextPipeline.image;
      resources.groundMask = nextPipeline.groundMask;
      resources.resolution = nextPipeline.resolution;
      resources.imageSize = nextPipeline.imageSize;
      resources.time = nextPipeline.time;
      resources.motion = nextPipeline.motion;
      resources.darkMode = nextPipeline.darkMode;
    }

    function draw() {
      if (!gl || !resources || destroyed || contextLost) return;

      var measurement = measureCanvas();
      if (!measurement.valid) return;
      if (measurement.qualityChanged) replacePipeline();

      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.35, 0.45, 0.54, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(resources.program);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, resources.imageTexture);
      gl.uniform1i(resources.image, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, resources.maskTexture);
      gl.uniform1i(resources.groundMask, 1);

      gl.uniform2f(resources.resolution, canvas.width, canvas.height);
      gl.uniform2f(
        resources.imageSize,
        images.photo.naturalWidth,
        images.photo.naturalHeight
      );
      gl.uniform1f(resources.time, simulationTime);
      gl.uniform1f(resources.motion, reducedMotion.matches ? 0 : 1);
      gl.uniform1f(resources.darkMode, isDarkMode() ? 1 : 0);
      gl.bindVertexArray(resources.quadVao);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);

      canvas.classList.add('is-ready');
    }

    function shouldAnimate() {
      return Boolean(
        resources &&
        !destroyed &&
        !contextLost &&
        intersecting &&
        !document.hidden &&
        !reducedMotion.matches
      );
    }

    function stopLoop() {
      if (animationFrame) global.cancelAnimationFrame(animationFrame);
      animationFrame = 0;
    }

    function requestNextFrame() {
      if (!animationFrame && shouldAnimate()) {
        animationFrame = global.requestAnimationFrame(frame);
      }
    }

    function frame(now) {
      animationFrame = 0;
      if (!shouldAnimate()) return;

      var interval = 1000 / (compact ? 30 : 45);
      if (!lastDraw) lastDraw = now - interval;
      var sinceDraw = now - lastDraw;

      if (sinceDraw >= interval * 0.9) {
        simulationTime += Math.min((now - previousFrame) / 1000, 0.05);
        previousFrame = now;
        lastDraw = now - (sinceDraw % interval);
        draw();
      }

      requestNextFrame();
    }

    function syncLoop(drawStill) {
      if (destroyed) return;

      if (!shouldAnimate()) {
        stopLoop();
        if (drawStill && resources && !contextLost && !document.hidden) draw();
        return;
      }

      previousFrame = performance.now();
      lastDraw = 0;
      requestNextFrame();
    }

    function setupResources() {
      if (!gl || !images || destroyed || contextLost) return;

      deleteResources();
      measureCanvas();

      var pipeline = createPipeline(compact);
      var imageTexture = createTexture(gl, images.photo, gl.TEXTURE0);
      var maskTexture = createTexture(gl, images.mask, gl.TEXTURE1);
      var quadBuffer = gl.createBuffer();
      var quadVao = gl.createVertexArray();

      if (!quadBuffer || !quadVao) {
        throw new Error('Unable to allocate the WebGL scene.');
      }

      gl.bindVertexArray(quadVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
        gl.STATIC_DRAW
      );
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.bindVertexArray(null);

      resources = {
        program: pipeline.program,
        image: pipeline.image,
        groundMask: pipeline.groundMask,
        resolution: pipeline.resolution,
        imageSize: pipeline.imageSize,
        time: pipeline.time,
        motion: pipeline.motion,
        darkMode: pipeline.darkMode,
        imageTexture: imageTexture,
        maskTexture: maskTexture,
        quadBuffer: quadBuffer,
        quadVao: quadVao
      };

      simulationTime = 0;
      draw();
      syncLoop(false);
    }

    function initialize() {
      var generation = ++initGeneration;
      var useMobileImage = global.matchMedia('(max-width: 719px)').matches;

      Promise.all([
        loadImage(useMobileImage ? mobileImageUrl : desktopImageUrl, 'high'),
        loadImage(groundMaskUrl, 'high')
      ]).then(function (loaded) {
        if (destroyed || contextLost || generation !== initGeneration) return;
        images = { photo: loaded[0], mask: loaded[1] };
        setupResources();
      }).catch(function (error) {
        canvas.classList.remove('is-ready');
        console.error('The generative landscape could not start.', error);
      });
    }

    function onResize() {
      if (!resources || destroyed || contextLost) return;
      if (intersecting && !document.hidden) draw();
      syncLoop(false);
    }

    function onIntersection(entries) {
      intersecting = Boolean(entries[0] && entries[0].isIntersecting);
      if (!intersecting) stopLoop();
      else syncLoop(true);
    }

    function onVisibilityChange() {
      if (document.hidden) stopLoop();
      else syncLoop(true);
    }

    function onReducedMotionChange() {
      simulationTime = 0;
      syncLoop(true);
    }

    function onThemeChange() {
      if (resources && intersecting && !document.hidden) draw();
    }

    function onContextLost(event) {
      event.preventDefault();
      contextLost = true;
      ++initGeneration;
      stopLoop();
      resources = null;
      canvas.classList.remove('is-ready');
    }

    function onContextRestored() {
      if (destroyed) return;
      contextLost = false;
      resources = null;
      initialize();
    }

    function destroy() {
      if (destroyed) return;
      destroyed = true;
      ++initGeneration;
      stopLoop();
      resizeObserver && resizeObserver.disconnect();
      intersectionObserver && intersectionObserver.disconnect();
      themeObserver && themeObserver.disconnect();
      document.removeEventListener('visibilitychange', onVisibilityChange);
      canvas.removeEventListener('webglcontextlost', onContextLost);
      canvas.removeEventListener('webglcontextrestored', onContextRestored);

      if (reducedMotion.removeEventListener) {
        reducedMotion.removeEventListener('change', onReducedMotionChange);
      } else if (reducedMotion.removeListener) {
        reducedMotion.removeListener(onReducedMotionChange);
      }

      deleteResources();
      canvas.classList.remove('is-ready');
      delete canvas.dataset.landscapeFallback;
      if (canvas.__stochasticMountain === controller) {
        delete canvas.__stochasticMountain;
      }
    }

    var controller = {
      destroy: destroy,
      redraw: function () { if (!destroyed) draw(); }
    };
    canvas.__stochasticMountain = controller;

    if (!gl) return controller;

    // Browsers isolate every file:// URL, so local image files cannot legally
    // become WebGL textures. Keep the photographic CSS fallback and avoid a
    // noisy SecurityError; the animated version must be viewed over HTTP(S).
    if (global.location.protocol === 'file:') {
      canvas.dataset.landscapeFallback = 'file-protocol';
      console.info(
        'The animated landscape is using its still fallback. Open this page through a local web server to enable WebGL motion.'
      );
      return controller;
    }

    resizeObserver = new ResizeObserver(onResize);
    resizeObserver.observe(canvas.parentElement || canvas);

    if ('IntersectionObserver' in global) {
      intersectionObserver = new IntersectionObserver(onIntersection, {
        threshold: 0.01
      });
      intersectionObserver.observe(canvas.parentElement || canvas);
    }

    themeObserver = new MutationObserver(onThemeChange);
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });

    document.addEventListener('visibilitychange', onVisibilityChange);
    canvas.addEventListener('webglcontextlost', onContextLost);
    canvas.addEventListener('webglcontextrestored', onContextRestored);

    if (reducedMotion.addEventListener) {
      reducedMotion.addEventListener('change', onReducedMotionChange);
    } else if (reducedMotion.addListener) {
      reducedMotion.addListener(onReducedMotionChange);
    }

    initialize();
    return controller;
  };
})(window);
