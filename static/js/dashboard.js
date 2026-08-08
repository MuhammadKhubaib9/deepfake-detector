/* DeepFake Detection System - metrics dashboard (dashboard.js, UC-07) */
(function () {
  "use strict";

  var state = { models: {}, order: [], active: null };

  function $(id) { return document.getElementById(id); }

  fetch("/api/metrics")
    .then(function (r) { return r.json().then(function (d) { if (!r.ok) throw d; return d; }); })
    .then(function (doc) {
      state.srs = doc.srs_targets || {};
      state.note = doc.note || "";
      state.models = doc.models || {};
      state.order = Object.keys(state.models);
      if (!state.order.length) throw { error: { message: "No model metrics found." } };
      state.active = state.order[0];
      buildTabs();
      render();
      $("dash-note").textContent = state.note;
    })
    .catch(function (err) {
      var el = $("metrics-error");
      el.textContent = (err && err.error && err.error.message) || "Failed to load metrics. Run: python scripts/generate_metrics.py";
      el.classList.remove("hidden");
    });

  function buildTabs() {
    var tabs = $("model-tabs");
    tabs.innerHTML = "";
    state.order.forEach(function (key) {
      var b = document.createElement("button");
      b.className = "tab" + (key === state.active ? " active" : "");
      b.textContent = state.models[key].model_name;
      b.addEventListener("click", function () {
        state.active = key;
        buildTabs();
        render();
      });
      tabs.appendChild(b);
    });
  }

  function render() {
    var m = state.models[state.active];
    var met = m.metrics || {};
    var t = state.srs || {};
    var grid = $("stat-grid");
    grid.innerHTML = "";

    [["Accuracy", "accuracy", t.accuracy],
     ["Precision", "precision", null],
     ["Recall", "recall", null],
     ["F1-Score", "f1", t.f1],
     ["ROC-AUC", "roc_auc", t.roc_auc]].forEach(function (row) {
      var label = row[0], key = row[1], target = row[2];
      var card = document.createElement("div");
      card.className = "stat";
      var v = met[key];
      card.innerHTML =
        '<div class="k">' + label + "</div>" +
        '<div class="v">' + (v != null ? (v * 100).toFixed(1) + "%" : "—") + "</div>";
      if (target != null && v != null) {
        var ok = v >= target;
        card.innerHTML += '<div class="' + (ok ? "pass" : "fail") + '">' +
          (ok ? "✓ ≥ SRS target" : "✗ below " + (target * 100).toFixed(0) + "%") + "</div>";
      }
      grid.appendChild(card);
    });

    var showImg = function (id, url) {
      var el = $(id);
      if (url) { el.src = url; el.parentNode.classList.remove("hidden"); }
      else { el.parentNode.classList.add("hidden"); }
    };
    showImg("cm-img", m.confusion_url);
    showImg("roc-img", m.roc_url);

    $("dash-note").textContent =
      state.note + " Dataset: " + (m.dataset || "—");
  }
})();