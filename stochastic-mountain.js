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
uniform float u_shuffleAmount;

in vec2 v_screenUv;
out vec4 outColor;

vec2 coverUvFor(vec2 screenUv, vec2 imageSize) {
  float screenAspect = u_resolution.x / u_resolution.y;
  float imageAspect = imageSize.x / imageSize.y;
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

vec2 coverUv(vec2 screenUv) {
  return coverUvFor(screenUv, u_imageSize);
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

  for (int index = 0; index < 5; index++) {
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

void main() {
  vec2 imageUv = coverUv(v_screenUv);
  vec2 topUv = vec2(imageUv.x, 1.0 - imageUv.y);
  vec2 imagePixel = 1.0 / u_imageSize;
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

  // Organic refraction keeps the original psychedelic terrain movement, but
  // deliberately omits the Voronoi web, fracture lines, rims, and glints.
  if (ground > 0.001) {
    float groundTime = u_time * 0.62;
    float pattern = turingField(topUv, groundTime);
    float travelingWave = 0.5 + 0.5 * sin(
      (topUv.x * 0.72 + topUv.y * 1.34) * 28.0
        - groundTime * 1.35 + pattern * 4.0
    );
    // Preserve the existing mid-ground motion and add stronger parallax-like
    // displacement as the terrain approaches the bottom of the frame.
    float foregroundDepth = smoothstep(0.48, 1.0, topUv.y);
    float depthGain = 1.0 + foregroundDepth * 2.4;

    vec2 organicFlowTop = vec2(
      sin(pattern * 6.2831853 + groundTime * 0.33),
      cos(pattern * 5.41 - groundTime * 0.29)
    );
    organicFlowTop *= (0.0007 + 0.0022 * travelingWave) * depthGain;
    organicFlowTop += vec2(
      sin(topUv.y * 18.0 - groundTime * 0.44),
      cos(topUv.x * 15.0 + groundTime * 0.37)
    ) * 0.00045 * depthGain;

    vec2 refractionUv = vec2(organicFlowTop.x, -organicFlowTop.y);
    vec2 terrainCenter = clamp(imageUv + refractionUv, 0.001, 0.999);
    float displacedGround = texture(u_groundMask, terrainCenter).r;
    groundMotion = smoothstep(
      0.10,
      0.68,
      min(ground, displacedGround)
    ) * u_motionAmount;

    terrain = texture(u_image, terrainCenter).rgb;
    float breathingRelief = (pattern - 0.5) * 0.018;
    breathingRelief += sin(
      travelingWave * 6.2831853 + groundTime * 0.35
    ) * 0.006;
    terrain *= 1.0
      + breathingRelief * groundMotion * (1.0 + foregroundDepth * 0.75);

  }

  vec3 baseScene = mix(sky, source, ground);
  vec3 color = mix(baseScene, terrain, groundMotion);

  // Tiny cells migrate to deterministic random destinations. A curved offset
  // gives each cell an individual ballistic path while the JS timeline applies
  // damped spring easing to the shared travel amount.
  float shufflePosition = clamp(u_shuffleAmount, -0.12, 1.12);
  float shuffleProgress = clamp(shufflePosition, 0.0, 1.0);
#if LOW_POWER
  vec2 shuffleGrid = max(floor(u_resolution / 2.0), vec2(1.0));
#else
  vec2 shuffleGrid = max(floor(u_resolution / 3.0), vec2(1.0));
#endif
  vec2 cell = floor(v_screenUv * shuffleGrid);
  vec2 localPixel = fract(v_screenUv * shuffleGrid);
  vec2 randomCell = floor(vec2(
    hash21(cell + vec2(17.2, 63.8)),
    hash21(cell + vec2(91.7, 24.5))
  ) * shuffleGrid);
  vec2 randomDestination = (randomCell + localPixel) / shuffleGrid;
  vec2 curveDirection = vec2(
    hash21(cell + vec2(7.1, 83.4)),
    hash21(cell + vec2(61.8, 13.2))
  ) - 0.5;
  float arc = sin(shuffleProgress * 3.14159265);
  vec2 sampleScreen = mix(
    v_screenUv,
    randomDestination,
    shufflePosition
  ) + curveDirection * arc * 0.075;
  vec3 shuffledPixels = texture(
    u_image,
    clamp(coverUvFor(sampleScreen, u_imageSize), 0.001, 0.999)
  ).rgb;
  float shuffleEngagement = smoothstep(0.0, 0.22, shuffleProgress);
  color = mix(color, shuffledPixels, shuffleEngagement);

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
    var initGeneration = 0;
    var reducedMotion = global.matchMedia('(prefers-reduced-motion: reduce)');

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
        shuffleAmount: gl.getUniformLocation(program, 'u_shuffleAmount')
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
      resources.shuffleAmount = nextPipeline.shuffleAmount;
    }

    function driftAndSnap01(value) {
      var clamped = Math.max(0, Math.min(1, value));
      var driftEnd = 0.82;

      if (clamped < driftEnd) {
        var driftProgress = clamped / driftEnd;
        return 0.58 * Math.pow(driftProgress, 1.7);
      }

      var snapProgress = (clamped - driftEnd) / (1 - driftEnd);
      return 0.58 + 0.42 * Math.pow(snapProgress, 2.7);
    }

    function currentShuffleAmount() {
      if (reducedMotion.matches) return 0;

      var originalHold = 8;
      var shuffledHold = 2.4;
      var transition = 6;
      var phase = simulationTime % (
        originalHold + shuffledHold + transition * 2
      );

      if (phase < originalHold) return 0;
      if (phase < originalHold + transition) {
        return driftAndSnap01((phase - originalHold) / transition);
      }
      if (phase < originalHold + transition + shuffledHold) return 1;
      return 1 - driftAndSnap01(
        (phase - originalHold - transition - shuffledHold) / transition
      );
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
      var shuffleAmount = currentShuffleAmount();
      gl.uniform1f(resources.shuffleAmount, shuffleAmount);
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
        shuffleAmount: pipeline.shuffleAmount,
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
        images = {
          photo: loaded[0],
          mask: loaded[1]
        };
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
