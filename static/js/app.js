/* Deepfake Detection - scanner UI logic (app.js) */
(function () {
  "use strict";

  var scanner = document.getElementById("scanner");
  var scanState = document.getElementById("scan-state");
  var dz = document.getElementById("dropzone");
  var fileInput = document.getElementById("file-input");
  var stage = document.getElementById("stage");
  var stageImg = document.getElementById("stage-img");
  var stageVid = document.getElementById("stage-vid");
  var replaceBtn = document.getElementById("replace-btn");
  var metaName = document.getElementById("meta-name");
  var metaSize = document.getElementById("meta-size");
  var detectBtn = document.getElementById("detect-btn");
  var errorZone = document.getElementById("error-zone");
  var tabsRow = document.getElementById("tabs-row");
  var stabEls = Array.prototype.slice.call(document.querySelectorAll(".stab"));
  var verdictView = document.getElementById("verdict-view");
  var loader = document.getElementById("loader");
  var loaderMsg = document.getElementById("loader-msg");
  var loaderStep = document.getElementById("loader-step");

  var state = { file: null, sessionId: null, urls: {}, activeTab: null };

  /* ================================================ document chrome: nav, reveal, top */
  var topbar = document.getElementById("topbar");
  var toTop = document.getElementById("to-top");
  var navLinks = Array.prototype.slice.call(document.querySelectorAll(".nav-link"));

  function onScroll() {
    topbar.classList.toggle("scrolled", window.scrollY > 10);
    toTop.classList.toggle("hidden", window.scrollY < 320);
    var pos = window.scrollY + 130;
    var current = document.getElementById("analyze");
    navLinks.forEach(function (l) {
      var sec = document.getElementById(l.getAttribute("href").slice(1));
      if (sec && sec.offsetTop <= pos && sec.offsetTop + sec.offsetHeight > pos) current = sec;
    });
    navLinks.forEach(function (l) {
      l.classList.toggle("active", l.getAttribute("href") === "#" + current.id);
    });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("load", onScroll);
  toTop.addEventListener("click", function () { window.scrollTo({ top: 0, behavior: "smooth" }); });

  var revealObs = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) { en.target.classList.add("visible"); revealObs.unobserve(en.target); }
    });
  }, { threshold: 0.12 });
  Array.prototype.forEach.call(document.querySelectorAll(".reveal"), function (el) { revealObs.observe(el); });

  /* animated stat counters */
  var countObs = new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (!en.isIntersecting) return;
      countObs.unobserve(en.target);
      var el = en.target, to = parseFloat(el.dataset.to), suffix = el.dataset.suffix || "";
      var start = null, dur = 1200;
      function tick(ts) {
        if (!start) start = ts;
        var p = Math.min((ts - start) / dur, 1);
        var eased = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(to * eased) + (p < 1 ? "" : suffix);
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }, { threshold: 0.4 });
  Array.prototype.forEach.call(document.querySelectorAll(".count"), function (el) { countObs.observe(el); });

  /* ================================================ readiness */
  function checkReady() {
    fetch("/api/status")
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (!s.ready) {
          if (s.error) {
            var p = document.createElement("p");
            p.className = "alert error";
            p.textContent = "Model load failed: " + s.error;
            dz.parentNode.insertBefore(p, dz.nextSibling);
            detectBtn.disabled = true;
          } else { setTimeout(checkReady, 2000); }
        }
      })
      .catch(function () { setTimeout(checkReady, 3000); });
  }
  checkReady();

  /* ================================================ upload UI */
  function clickInput() { fileInput.click(); }
  dz.addEventListener("click", clickInput);
  dz.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); clickInput(); }
  });
  replaceBtn.addEventListener("click", function (e) { e.stopPropagation(); clickInput(); });

  var ctaBtn = document.getElementById("cta-btn");
  var heroAnalyze = document.getElementById("hero-analyze");
  [ctaBtn, heroAnalyze].forEach(function (b) {
    b.addEventListener("click", function () {
      document.getElementById("analyze").scrollIntoView({ behavior: "smooth" });
      setTimeout(function () { clickInput(); }, 500);
    });
  });

  fileInput.addEventListener("change", function () { pickFile(fileInput.files[0]); });
  ["dragover", "dragenter"].forEach(function (ev) {
    dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("dragover"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("dragover"); });
  });
  dz.addEventListener("drop", function (e) {
    var files = e.dataTransfer.files;
    if (files.length) pickFile(files[0]);
  });

  function humanSize(b) {
    return b >= 1048576 ? (b / 1048576).toFixed(2) + " MB" : (b / 1024).toFixed(1) + " KB";
  }

  function clientCheck(file) {
    var name = file.name.toLowerCase();
    if (!/\.(jpe?g|png|mp4|avi)$/.test(name)) {
      showError("Unsupported format. Please use JPEG, PNG, MP4 or AVI."); return false;
    }
    var isImg = /\.(jpe?g|png)$/.test(name);
    var max = isImg ? window.CONFIG.maxImageMb * 1048576 : window.CONFIG.maxVideoMb * 1048576;
    if (file.size > max) {
      showError(isImg ? "File too large. Maximum: 10 MB for images."
                       : "File too large. Maximum: 100 MB for videos.");
      return false;
    }
    return true;
  }

  function setScanState(text, busy) {
    scanState.textContent = text;
    scanState.style.color = busy ? "var(--accent)" : "";
  }

  function pickFile(file) {
    hideError();
    if (!file) return;
    if (!clientCheck(file)) return;

    state.file = file;
    state.urls = {};
    var isImg = /\.(jpe?g|png)$/i.test(file.name);
    stageImg.hidden = !isImg;
    stageVid.hidden = isImg;
    if (isImg) { stageImg.src = URL.createObjectURL(file); }
    else { stageVid.src = URL.createObjectURL(file); }
    metaName.textContent = file.name;
    metaSize.textContent = humanSize(file.size);
    dz.classList.add("hidden");
    stage.classList.remove("hidden");
    setScanState("Ready to scan", false);
    detectBtn.removeAttribute("disabled");
  }

  function showError(msg) {
    errorZone.textContent = msg;
    errorZone.classList.remove("hidden");
    setScanState("Failed", true);
  }
  function hideError() { errorZone.classList.add("hidden"); }

  /* ================================================ scan tabs */
  var TAB_SEQ = ["source", "face", "heatmap", "verdict"];
  function setActiveTab(name, showView) {
    state.activeTab = name;
    stabEls.forEach(function (b) {
      var t = b.dataset.tab;
      b.classList.toggle("active", t === name);
    });
    if (showView === false) return;
    var isVerdict = name === "verdict";
    verdictView.classList.toggle("hidden", !isVerdict);
    if (!isVerdict && state.urls[name]) {
      var isImg = /\.(jpe?g|png)$/i.test(state.urls[name]);
      stageImg.hidden = !isImg;
      stageVid.hidden = isImg;
      (isImg ? stageImg : stageVid).src = state.urls[name];
      if (name === "source" && state.file) stageVid.pause();
      stage.classList.remove("hidden");
    }
  }
  stabEls.forEach(function (b) {
    b.addEventListener("click", function () { setActiveTab(b.dataset.tab); });
  });

  /* ================================================ detect */
  detectBtn.addEventListener("click", function () {
    if (!state.file || detectBtn.disabled) return;
    detectBtn.disabled = true;
    hideError();
    scanner.classList.add("scanning");
    setScanState("Scanning…", true);
    tabsRow.classList.remove("hidden");
    tabsRow.scrollIntoView({ behavior: "smooth", block: "nearest" });
    showLoader("Analyzing media…", "Uploading");
    setActiveTab("source", false);

    var fd = new FormData();
    fd.append("file", state.file);

    fetch("/api/upload", { method: "POST", body: fd })
      .then(function (r) { return r.json().then(function (d) { if (!r.ok) throw d; return d; }); })
      .then(function (up) {
        state.sessionId = up.session_id;
        return runDetect(up);
      })
      .catch(function (err) {
        if (err && err.error) showError(err.error.message);
        else showError("An internal error occurred. Please try again.");
      })
      .finally(function () {
        hideLoader();
        scanner.classList.remove("scanning");
        detectBtn.disabled = false;
      });
  });

  var STEPS = [
    ["Detecting faces…", "MTCNN face localization", "face"],
    ["Running deep learning models…", "CNN + ViT + LSTM ensemble", "heatmap"],
    ["Generating Grad-CAM proof…", "Explainability heatmap", "verdict"]
  ];
  function showLoader(text, step) {
    loaderMsg.textContent = text;
    loaderStep.textContent = step || "";
    loader.classList.remove("hidden");
  }
  function hideLoader() { loader.classList.add("hidden"); }

  function runDetect(up) {
    var i = 0;
    var timer = setInterval(function () {
      if (i >= STEPS.length) return;
      var s = STEPS[i];
      loaderMsg.textContent = s[0];
      loaderStep.textContent = s[1];
      setActiveTab(s[2], false);
      i += 1;
    }, 2600);

    return fetch("/api/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId })
    }).then(function (r) {
      clearInterval(timer);
      return r.json().then(function (d) { if (!r.ok) throw d; return d; });
    }).then(function (resp) {
      renderResult(resp.result);
    });
  }

  /* ================================================ render */
  function renderResult(res) {
    var verdict = res.verdict;
    var conf = res.confidence;
    var isFake = verdict === "FAKE";

    state.urls = {
      source: res.media_url,
      face: res.face_url,
      heatmap: res.heatmap_url
    };

    var badge = document.getElementById("verdict-badge");
    badge.textContent = verdict;
    badge.className = "badge " + verdict + " show";
    var ring = document.getElementById("verdict-ring");
    ring.className = "verdict-ring pop-" + (isFake ? "fake" : "real");

    var fill = document.getElementById("conf-fill");
    fill.className = "conf-fill " + (isFake ? "fake" : "real");
    fill.style.width = "0%";
    requestAnimationFrame(function () {
      setTimeout(function () { fill.style.width = conf + "%"; }, 60);
    });
    animateCounter(conf);

    document.getElementById("verdict-note").innerHTML = isFake
      ? "This media shows strong signs of <b>AI manipulation</b>. Please verify before trusting it."
      : "No significant signs of manipulation found. This media appears <b>authentic</b>.";

    var vm = document.getElementById("video-meta");
    if (res.kind === "video") {
      vm.classList.remove("hidden");
      vm.innerHTML =
        "Frames analyzed: <b>" + res.faces_analyzed + "</b>" +
        (res.most_manipulated_frame ? " &middot; Most suspicious frame: <b>#" + res.most_manipulated_frame + "</b>" : "") +
        (res.video_fps ? " &middot; Extraction rate: <b>" + res.video_fps + " fps</b>" : "");
    } else {
      vm.classList.add("hidden");
    }
    document.getElementById("run-meta").textContent =
      "Analyzed the detected face only &middot; frames: " + (res.faces_analyzed || 1);

    setScanState("Scan complete", false);
    setActiveTab("verdict");
    var vv = verdictView.getBoundingClientRect();
    if (vv.top < 0) {
      window.scrollTo({ top: window.scrollY + vv.top - 90, behavior: "smooth" });
    }
  }

  function animateCounter(target) {
    var el = document.getElementById("conf-text");
    var start = null, dur = 1100;
    function tick(ts) {
      if (!start) start = ts;
      var p = Math.min((ts - start) / dur, 1);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = (target * eased).toFixed(1) + "%";
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ================================================ again */
  document.getElementById("again-btn").addEventListener("click", function () {
    verdictView.classList.add("hidden");
    stage.classList.add("hidden");
    dz.classList.remove("hidden");
    tabsRow.classList.add("hidden");
    detectBtn.disabled = true;
    state = { file: null, sessionId: null, urls: {}, activeTab: null };
    fileInput.value = "";
    setScanState("Waiting for a file", false);
    document.getElementById("analyze").scrollIntoView({ behavior: "smooth", block: "start" });
  });
})();