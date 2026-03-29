// stochastic-mountain.js — Animated mountain for praxagent.ai
// Walk lines continuously drift and reshape organically.

(function() {
  'use strict';

  function mulberry32(a) {
    return function() {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var v = Math.imul(a ^ a >>> 15, 1 | a);
      v = v + Math.imul(v ^ v >>> 7, 61 | v) ^ v;
      return ((v ^ v >>> 14) >>> 0) / 4294967296;
    };
  }

  var themes = {
    dark: {
      bgTop: '#0f1117', bgBot: '#060810', fade: '15,17,23',
      fR: function(d) { return 2 + d * 14 | 0; },
      fG: function(d) { return 5 + d * 25 | 0; },
      fB: function(d) { return 16 + d * 50 | 0; },
      fA: function(d) { return 0.025 + d * 0.05; },
      sR: function(d) { return 30 + (d * 55 | 0); },
      sG: function(d) { return 65 + (d * 80 | 0); },
      sB: function(d) { return 140 + (d * 70 | 0); },
      sA: function(d) { return 0.02 + d * 0.07; },
      hR: function(h) { return h > 0.7 ? 120 : 55; },
      hG: function(h) { return h > 0.7 ? 185 : 125; },
      hB: function()  { return 250; },
      hA: function(h, r) { return h > 0.7 ? 0.22 + r * 0.18 : 0.10 + r * 0.14; }
    },
    light: {
      bgTop: '#f8fafc', bgBot: '#edf0f5', fade: '248,250,252',
      fR: function(d) { return 190 - d * 40 | 0; },
      fG: function(d) { return 205 - d * 35 | 0; },
      fB: function(d) { return 225 - d * 30 | 0; },
      fA: function(d) { return 0.025 + d * 0.05; },
      sR: function(d) { return 25 + (d * 42 | 0); },
      sG: function(d) { return 55 + (d * 60 | 0); },
      sB: function(d) { return 120 + (d * 50 | 0); },
      sA: function(d) { return 0.04 + d * 0.10; },
      hR: function(h) { return h > 0.7 ? 30 : 16; },
      hG: function(h) { return h > 0.7 ? 80 : 50; },
      hB: function(h) { return h > 0.7 ? 175 : 155; },
      hA: function(h, r) { return h > 0.7 ? 0.30 + r * 0.20 : 0.16 + r * 0.18; }
    }
  };

  // Multi-frequency drift — punchy, organic reshaping
  function drift(time, x, ph, sp, fr, amp) {
    return (
      Math.sin(time * sp[0] + x * fr[0] + ph[0]) * 0.40 +
      Math.sin(time * sp[1] + x * fr[1] + ph[1]) * 0.35 +
      Math.sin(time * sp[2] + x * fr[2] + ph[2]) * 0.25
    ) * amp;
  }

  // Ridgeline drift
  function ridgeDrift(time, x) {
    return (
      Math.sin(time * 0.50 + x * 0.005) * 2.5 +
      Math.sin(time * 0.35 + x * 0.014 + 1.5) * 1.8 +
      Math.sin(time * 0.70 + x * 0.003 + 3.2) * 1.2
    );
  }

  window.StochasticMountain = function(canvasId) {
    var canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    var ctx = canvas.getContext('2d');

    var W = 0, H = 0, DPR = 1;
    var theme = detectTheme();
    var destroyed = false;
    var animId = null;
    var RIDGE_N = 400;
    var ridge = [];
    var walks = [];
    var heroes = [];
    var peaks = [];
    var particles = [];
    var rand;

    function detectTheme() {
      return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    }

    function buildRidge() {
      rand = mulberry32(42);
      var pts = new Array(RIDGE_N + 1);
      // Mountain occupies the lower portion — sharp peak around 44% down
      var anchors = [
        [0.00, 0.92], [0.06, 0.86], [0.14, 0.78], [0.22, 0.82],
        [0.30, 0.72], [0.38, 0.74], [0.44, 0.62], [0.48, 0.50],
        [0.50, 0.44], [0.52, 0.50], [0.56, 0.62], [0.62, 0.68],
        [0.70, 0.74], [0.78, 0.80], [0.86, 0.86], [0.93, 0.91],
        [1.00, 0.94]
      ];

      for (var a = 0; a < anchors.length; a++) {
        pts[Math.floor(anchors[a][0] * RIDGE_N)] = H * anchors[a][1];
      }
      for (var a = 0; a < anchors.length - 1; a++) {
        var si = Math.floor(anchors[a][0] * RIDGE_N);
        var ei = Math.floor(anchors[a + 1][0] * RIDGE_N);
        var sy = pts[si], ey = pts[ei];
        for (var i = si; i <= ei; i++) {
          var f = (i - si) / (ei - si);
          pts[i] = sy + (ey - sy) * f * f * (3 - 2 * f);
        }
      }
      for (var i = 1; i < RIDGE_N; i++) {
        var xn = i / RIDGE_N;
        var peakProx = Math.exp(-Math.pow((xn - 0.50) / 0.22, 2));
        pts[i] += (rand() - 0.5) * (5 + peakProx * 14);
      }
      var s = new Array(RIDGE_N + 1);
      s[0] = pts[0]; s[RIDGE_N] = pts[RIDGE_N];
      for (var i = 1; i < RIDGE_N; i++) {
        s[i] = pts[i - 1] * 0.2 + pts[i] * 0.6 + pts[i + 1] * 0.2;
      }
      ridge = s;
    }

    function rAt(x) {
      var idx = Math.max(0, Math.min(RIDGE_N, (x / W) * RIDGE_N));
      var i0 = Math.floor(idx);
      var i1 = Math.min(i0 + 1, RIDGE_N);
      var frac = idx - i0;
      return ridge[i0] * (1 - frac) + ridge[i1] * frac;
    }

    function makeDriftParams(rng) {
      return {
        ph: [rng() * 6.28, rng() * 6.28, rng() * 6.28],
        sp: [0.50 + rng() * 0.40, 0.90 + rng() * 0.50, 1.40 + rng() * 0.70],
        fr: [0.003 + rng() * 0.005, 0.008 + rng() * 0.010, 0.015 + rng() * 0.015]
      };
    }

    function buildWalks() {
      walks = [];
      var mobile = W < 600;
      var n = mobile ? 70 : 150;
      var steps = mobile ? 140 : 250;

      for (var w = 0; w < n; w++) {
        var pts = [], y = 0;
        var depth = rand() * rand() * 0.50;
        var vol = 0.2 + rand() * rand() * 3.5;
        var amp = 0.1 + rand() * 0.9;

        for (var i = 0; i <= steps; i++) {
          var x = (i / steps) * W;
          var rY = rAt(x);
          var baseline = rY + (H - rY) * depth;
          y += (rand() - 0.5) * vol;
          y *= 0.988;
          var fy = baseline + y * amp * 10;
          var ceil = depth < 0.1 ? rY - 18 - rand() * 12 : rY - 3;
          if (fy < ceil) fy = ceil + rand() * 4;
          pts.push({ x: x, y: fy });
        }

        var dp = makeDriftParams(rand);
        var driftAmp = (1 - depth) * (4.0 + rand() * 7.0);

        walks.push({
          p: pts, d: depth, v: vol,
          dp: dp, driftAmp: driftAmp,
          brPh1: rand() * 6.28, brPh2: rand() * 6.28
        });
      }
      walks.sort(function(a, b) { return b.d - a.d; });
    }

    function buildHeroes() {
      heroes = [];
      var mobile = W < 600;
      var n = mobile ? 10 : 22;
      var steps = mobile ? 140 : 250;

      for (var ri = 0; ri < n; ri++) {
        var rr = mulberry32(7000 + ri);
        var pts = [], y = 0;
        var off = (rr() - 0.5) * 22;
        var vol = 0.5 + rr() * 2;
        var above = rr() > 0.5;

        for (var i = 0; i <= steps; i++) {
          var x = (i / steps) * W;
          var rY = rAt(x);
          y += (rr() - 0.5) * vol;
          y *= 0.982;
          var fy = rY + off + y * 8;
          if (above && fy < rY - 30) fy = rY - 30 + rr() * 5;
          if (!above && fy < rY - 5) fy = rY - 5 + rr() * 3;
          pts.push({ x: x, y: fy });
        }

        var dp = makeDriftParams(rr);
        heroes.push({
          p: pts, hue: rr(), br: rr(), above: above,
          dp: dp, driftAmp: 7 + rr() * 10,
          brPh1: rr() * 6.28, brPh2: rr() * 6.28
        });
      }
    }

    // Find sharp peaks on the ridgeline (local y-minima = highest screen points)
    function findPeaks() {
      peaks = [];
      var win = 15; // scan window
      for (var i = win; i < RIDGE_N - win; i++) {
        var isMin = true;
        for (var j = 1; j <= win; j++) {
          if (ridge[i] >= ridge[i - j] || ridge[i] >= ridge[i + j]) {
            isMin = false; break;
          }
        }
        if (isMin) {
          // Sharpness = how much lower this peak is vs its neighbors
          var avgNeighbor = (ridge[i - win] + ridge[i + win]) / 2;
          var sharpness = avgNeighbor - ridge[i];
          peaks.push({
            idx: i,
            x: (i / RIDGE_N) * W,
            y: ridge[i],
            sharpness: sharpness
          });
        }
      }
      // Sort sharpest first
      peaks.sort(function(a, b) { return b.sharpness - a.sharpness; });
      // Keep top 6 at most
      if (peaks.length > 6) peaks.length = 6;
    }

    function rebuild() {
      buildRidge();
      findPeaks();
      buildWalks();
      buildHeroes();
      particles = [];
    }

    function resize() {
      var rect = canvas.parentElement.getBoundingClientRect();
      DPR = Math.min(window.devicePixelRatio || 1, 2);
      W = rect.width;
      H = rect.height;
      if (!W || !H) return;
      canvas.width = W * DPR;
      canvas.height = H * DPR;
      canvas.style.width = W + 'px';
      canvas.style.height = H + 'px';
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      rebuild();
    }

    function draw(ts) {
      if (destroyed) return;
      var time = (ts || 0) * 0.001;
      if (!W || !H) { animId = requestAnimationFrame(draw); return; }

      var t = themes[theme];
      ctx.clearRect(0, 0, W, H);

      // Background
      var bg = ctx.createLinearGradient(0, 0, 0, H);
      bg.addColorStop(0, t.bgTop);
      bg.addColorStop(0.5, t.bgTop);
      bg.addColorStop(1, t.bgBot);
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      // Mountain mass fill (drifts with ridge)
      ctx.beginPath();
      ctx.moveTo(0, rAt(0) + ridgeDrift(time, 0));
      for (var i = 1; i <= RIDGE_N; i++) {
        var rx = (i / RIDGE_N) * W;
        ctx.lineTo(rx, ridge[i] + ridgeDrift(time, rx));
      }
      ctx.lineTo(W, H + 2);
      ctx.lineTo(0, H + 2);
      ctx.closePath();
      var mg = ctx.createLinearGradient(0, H * 0.4, 0, H);
      mg.addColorStop(0, 'rgba(' + t.fR(0.9) + ',' + t.fG(0.9) + ',' + t.fB(0.9) + ',0.20)');
      mg.addColorStop(0.4, 'rgba(' + t.fR(0.5) + ',' + t.fG(0.5) + ',' + t.fB(0.5) + ',0.12)');
      mg.addColorStop(1, 'rgba(' + t.fR(0.1) + ',' + t.fG(0.1) + ',' + t.fB(0.1) + ',0.03)');
      ctx.fillStyle = mg;
      ctx.fill();

      // Walks — visible drifting
      var breathe = 1 + 0.08 * Math.sin(time * 0.5);

      for (var w = 0; w < walks.length; w++) {
        var wk = walks[w];
        var pts = wk.p;
        var d = 1 - wk.d;
        var bMod = 0.55 + 0.45 * (
          Math.sin(time * 0.45 + wk.brPh1) * 0.55 +
          Math.sin(time * 0.70 + wk.brPh2) * 0.45
        );

        if (w % 5 === 0) {
          ctx.beginPath();
          for (var i = 0; i < pts.length; i++) {
            var yo = drift(time, pts[i].x, wk.dp.ph, wk.dp.sp, wk.dp.fr, wk.driftAmp);
            if (i === 0) ctx.moveTo(pts[i].x, pts[i].y + yo);
            else ctx.lineTo(pts[i].x, pts[i].y + yo);
          }
          ctx.lineTo(W + 2, H + 2);
          ctx.lineTo(-2, H + 2);
          ctx.closePath();
          ctx.fillStyle = 'rgba(' + t.fR(d) + ',' + t.fG(d) + ',' + t.fB(d) + ',' + t.fA(d * 0.4) + ')';
          ctx.fill();
        }

        ctx.beginPath();
        for (var i = 0; i < pts.length; i++) {
          var yo = drift(time, pts[i].x, wk.dp.ph, wk.dp.sp, wk.dp.fr, wk.driftAmp);
          if (i === 0) ctx.moveTo(pts[i].x, pts[i].y + yo);
          else ctx.lineTo(pts[i].x, pts[i].y + yo);
        }
        var sm = wk.v > 2 ? 1.0 : 0.65;
        var alpha = t.sA(d * sm) * bMod * breathe;
        ctx.strokeStyle = 'rgba(' + t.sR(d) + ',' + t.sG(d) + ',' + t.sB(d) + ',' + alpha + ')';
        ctx.lineWidth = wk.v > 2 ? (0.4 + d * 0.7) : (0.25 + d * 0.45);
        ctx.stroke();
      }

      // Hero walks — strongest drift
      for (var ri = 0; ri < heroes.length; ri++) {
        var h = heroes[ri];
        var hb = 0.50 + 0.50 * (
          Math.sin(time * 0.55 + h.brPh1) * 0.50 +
          Math.sin(time * 0.85 + h.brPh2) * 0.50
        );

        ctx.beginPath();
        for (var i = 0; i < h.p.length; i++) {
          var yo = drift(time, h.p[i].x, h.dp.ph, h.dp.sp, h.dp.fr, h.driftAmp);
          if (i === 0) ctx.moveTo(h.p[i].x, h.p[i].y + yo);
          else ctx.lineTo(h.p[i].x, h.p[i].y + yo);
        }
        var ab = h.above ? 1.4 : 1.0;
        var br = t.hA(h.hue, h.br) * ab * hb;
        ctx.strokeStyle = 'rgba(' + t.hR(h.hue) + ',' + t.hG(h.hue) + ',' + t.hB(h.hue) + ',' + br + ')';
        ctx.lineWidth = h.above ? (0.9 + h.br * 1.3) : (0.6 + h.br * 0.9);
        ctx.stroke();
      }

      // Ridgeline
      ctx.beginPath();
      ctx.moveTo(0, rAt(0) + ridgeDrift(time, 0));
      for (var i = 1; i <= RIDGE_N; i++) {
        var rx = (i / RIDGE_N) * W;
        ctx.lineTo(rx, ridge[i] + ridgeDrift(time, rx));
      }
      ctx.strokeStyle = 'rgba(' + t.sR(1) + ',' + t.sG(1) + ',' + t.sB(1) + ',0.20)';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Ridgeline glow
      ctx.beginPath();
      ctx.moveTo(0, rAt(0) + ridgeDrift(time, 0));
      for (var i = 1; i <= RIDGE_N; i++) {
        var rx = (i / RIDGE_N) * W;
        ctx.lineTo(rx, ridge[i] + ridgeDrift(time, rx));
      }
      ctx.strokeStyle = 'rgba(' + t.sR(1) + ',' + t.sG(1) + ',' + t.sB(1) + ',0.06)';
      ctx.lineWidth = 5;
      ctx.stroke();

      // ── Glitter particles from sharp peaks ───────────────
      // Spawn
      if (peaks.length > 0 && particles.length < 25) {
        // Sharper peaks spawn more often
        for (var pi = 0; pi < peaks.length; pi++) {
          var spawnChance = 0.012 + peaks[pi].sharpness * 0.0008;
          if (Math.random() < spawnChance) {
            var pk = peaks[pi];
            var rd = ridgeDrift(time, pk.x);
            particles.push({
              x: pk.x + (Math.random() - 0.5) * 8,
              y: pk.y + rd - 2,
              vx: (Math.random() - 0.5) * 0.6,
              vy: -(0.3 + Math.random() * 0.7),
              life: 1,
              decay: 0.004 + Math.random() * 0.006,
              size: 1 + Math.random() * 2,
              twinkle: Math.random() * 6.28,
              twinkleSpd: 3 + Math.random() * 4,
              hue: Math.random()
            });
          }
        }
      }

      // Update & draw
      for (var i = particles.length - 1; i >= 0; i--) {
        var p = particles[i];
        p.x += p.vx;
        p.vy -= 0.003; // gentle acceleration upward
        p.y += p.vy;
        p.vx *= 0.998;
        p.life -= p.decay;

        if (p.life <= 0) { particles.splice(i, 1); continue; }

        var twk = 0.5 + 0.5 * Math.sin(time * p.twinkleSpd + p.twinkle);
        var alpha = p.life * twk;
        var sz = p.size * (0.5 + 0.5 * p.life);

        // Pick color: white core, tinted glow
        var cr, cg, cb;
        if (theme === 'dark') {
          cr = p.hue > 0.5 ? 160 : 100; cg = 200; cb = 255;
        } else {
          cr = p.hue > 0.5 ? 40 : 8; cg = 90; cb = 180;
        }

        // Glow
        var gl = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, sz * 4);
        gl.addColorStop(0, 'rgba(' + cr + ',' + cg + ',' + cb + ',' + (0.3 * alpha) + ')');
        gl.addColorStop(1, 'rgba(' + cr + ',' + cg + ',' + cb + ',0)');
        ctx.fillStyle = gl;
        ctx.beginPath();
        ctx.arc(p.x, p.y, sz * 4, 0, Math.PI * 2);
        ctx.fill();

        // 4-pointed star core
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(time * 0.5 + p.twinkle);
        ctx.strokeStyle = 'rgba(255,255,255,' + (alpha * 0.9) + ')';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(-sz, 0); ctx.lineTo(sz, 0);
        ctx.moveTo(0, -sz); ctx.lineTo(0, sz);
        ctx.stroke();
        // Bright center dot
        ctx.fillStyle = 'rgba(255,255,255,' + alpha + ')';
        ctx.beginPath();
        ctx.arc(0, 0, sz * 0.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }

      // Top fade — keeps the upper area clean for content
      var tf = ctx.createLinearGradient(0, 0, 0, H * 0.52);
      tf.addColorStop(0, 'rgba(' + t.fade + ',1)');
      tf.addColorStop(0.65, 'rgba(' + t.fade + ',1)');
      tf.addColorStop(1, 'rgba(' + t.fade + ',0)');
      ctx.fillStyle = tf;
      ctx.fillRect(0, 0, W, H * 0.52);

      // Bottom fade
      var bf = ctx.createLinearGradient(0, H * 0.90, 0, H);
      bf.addColorStop(0, 'rgba(' + t.fade + ',0)');
      bf.addColorStop(1, 'rgba(' + t.fade + ',0.65)');
      ctx.fillStyle = bf;
      ctx.fillRect(0, H * 0.90, W, H * 0.10);

      animId = requestAnimationFrame(draw);
    }

    // Init
    resize();

    var ro = new ResizeObserver(function() { if (!destroyed) resize(); });
    ro.observe(canvas.parentElement);

    var mo = new MutationObserver(function() {
      var nt = detectTheme();
      if (nt !== theme) theme = nt;
    });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

    function onThemeChange(e) { if (e.detail) theme = e.detail.theme; }
    window.addEventListener('themeChanged', onThemeChange);

    animId = requestAnimationFrame(draw);

    return {
      destroy: function() {
        destroyed = true;
        if (animId) cancelAnimationFrame(animId);
        ro.disconnect();
        mo.disconnect();
        window.removeEventListener('themeChanged', onThemeChange);
      }
    };
  };
})();
