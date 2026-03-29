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

  // Multi-frequency drift — visible, organic reshaping
  function drift(time, x, ph, sp, fr, amp) {
    return (
      Math.sin(time * sp[0] + x * fr[0] + ph[0]) * 0.45 +
      Math.sin(time * sp[1] + x * fr[1] + ph[1]) * 0.35 +
      Math.sin(time * sp[2] + x * fr[2] + ph[2]) * 0.20
    ) * amp;
  }

  // Ridgeline drift
  function ridgeDrift(time, x) {
    return (
      Math.sin(time * 0.25 + x * 0.004) * 1.5 +
      Math.sin(time * 0.18 + x * 0.011 + 1.5) * 1.0 +
      Math.sin(time * 0.35 + x * 0.002 + 3.2) * 0.8
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
    var rand;

    function detectTheme() {
      return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
    }

    function buildRidge() {
      rand = mulberry32(42);
      var pts = new Array(RIDGE_N + 1);
      // Mountain occupies the lower portion — peak around 55% down
      var anchors = [
        [0.00, 0.92], [0.06, 0.86], [0.14, 0.78], [0.22, 0.82],
        [0.30, 0.72], [0.38, 0.74], [0.45, 0.60], [0.50, 0.55],
        [0.55, 0.58], [0.62, 0.66], [0.70, 0.72], [0.78, 0.78],
        [0.86, 0.84], [0.93, 0.90], [1.00, 0.94]
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
        // Fast enough to see clearly within seconds
        sp: [0.30 + rng() * 0.25, 0.55 + rng() * 0.35, 0.90 + rng() * 0.50],
        fr: [0.002 + rng() * 0.004, 0.006 + rng() * 0.008, 0.012 + rng() * 0.012]
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
        // Visible drift: surface walks move more
        var driftAmp = (1 - depth) * (2.5 + rand() * 5.0);

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
          dp: dp, driftAmp: 5 + rr() * 8,
          brPh1: rr() * 6.28, brPh2: rr() * 6.28
        });
      }
    }

    function rebuild() {
      buildRidge();
      buildWalks();
      buildHeroes();
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
      var breathe = 1 + 0.06 * Math.sin(time * 0.3);

      for (var w = 0; w < walks.length; w++) {
        var wk = walks[w];
        var pts = wk.p;
        var d = 1 - wk.d;
        var bMod = 0.65 + 0.35 * (
          Math.sin(time * 0.25 + wk.brPh1) * 0.6 +
          Math.sin(time * 0.38 + wk.brPh2) * 0.4
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
        var hb = 0.6 + 0.4 * (
          Math.sin(time * 0.3 + h.brPh1) * 0.55 +
          Math.sin(time * 0.45 + h.brPh2) * 0.45
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
