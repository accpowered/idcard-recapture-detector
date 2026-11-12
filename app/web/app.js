/* 居民身份证翻拍识别工具 — 前端逻辑 */
(function () {
  "use strict";

  var CLASS_TEXT = { original: "原件实拍", screen_recapture: "屏幕翻拍", print_recapture: "打印翻拍" };
  var RISK_TEXT = { low: "低风险·原件", medium: "中风险·建议复核", high: "高风险·疑似翻拍" };
  var MODE_TEXT = { screen_only: "仅屏幕翻拍", screen_and_print: "屏幕+打印翻拍" };
  var GATE_TEXT = {
    below_min_f1: "macro-F1 低于质量门",
    missing_class_recall: "存在必备类别召回为 0",
    current_changed: "训练期间当前版本被切换",
    mode_changed: "训练期间识别模式被切换",
    auto_activate_disabled: "已关闭自动启用"
  };
  var VIEWS = ["analyze", "batch", "history", "review", "models", "settings"];
  var state = {
    settings: null, models: null, dataset: null,
    selectedVersion: localStorage.getItem("modelVersion") || "auto",
    batchFiles: [], batchRes: null,
    trainingJobId: null, trainingRunning: false,
    maskFields: true, lastResult: null, datasetDefaultsApplied: false
  };

  function $(id) { return document.getElementById(id); }
  function esc(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function show(el) { if (el) el.classList.remove("hidden"); }
  function hide(el) { if (el) el.classList.add("hidden"); }
  function text(el, value) { if (el) el.textContent = value == null ? "" : String(value); }

  function globalError(message) {
    var el = $("globalError");
    if (!el) return;
    if (!message) { hide(el); el.textContent = ""; return; }
    el.textContent = message;
    show(el);
  }

  async function api(path, opts) {
    var res = await fetch(path, opts || {});
    var body = null;
    try { body = await res.json(); }
    catch (e) { throw new Error("服务返回异常（HTTP " + res.status + "）"); }
    if (!res.ok || (body && body.code != null && body.code !== 0)) {
      var err = new Error((body && (body.error || body.message)) || ("HTTP " + res.status));
      err.payload = body;
      err.status = res.status;
      throw err;
    }
    return body;
  }
  function jsonOpts(body) {
    return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  }
  function errorText(err) {
    var out = err && err.message ? err.message : String(err);
    if (err && err.payload && err.payload.issues && err.payload.issues.length) {
      out += "：" + err.payload.issues.map(function (i) { return i.message; }).join("；");
    }
    return out;
  }

  // ------------------------------------------------------------------ 视图路由
  function currentView() {
    var view = (location.hash || "").replace("#", "");
    return VIEWS.indexOf(view) >= 0 ? view : "analyze";
  }
  function switchView(view) {
    if (VIEWS.indexOf(view) < 0) view = "analyze";
    document.querySelectorAll(".view").forEach(function (section) { section.classList.remove("active"); });
    var target = $("view-" + view);
    if (target) target.classList.add("active");
    document.querySelectorAll(".nav-btn").forEach(function (btn) {
      btn.classList.toggle("active", btn.dataset.view === view);
    });
    globalError("");
    if (view === "history") loadHistory();
    if (view === "review") loadReview();
    if (view === "models") loadModelsView();
    if (view === "settings") loadSettingsView();
  }
  function gotoView(view) {
    if (location.hash !== "#" + view) location.hash = "#" + view;
    switchView(view);
  }
  function bindNav() {
    document.querySelectorAll(".nav-btn").forEach(function (btn) {
      btn.addEventListener("click", function () { gotoView(btn.dataset.view); });
    });
    window.addEventListener("hashchange", function () { switchView(currentView()); });
  }

  // ------------------------------------------------------------------ 版本选择
  function fillVersionSelects() {
    var models = state.models || {};
    var current = models.current || {};
    var versions = models.versions || [];
    [$("modelVersionSelect"), $("batchModelVersionSelect")].forEach(function (select) {
      if (!select) return;
      var html = '<option value="auto">跟随当前版本（' + esc(current.version || "—") + '）</option>';
      versions.forEach(function (v) {
        var label = v.version + " · " + (MODE_TEXT[v.mode] || v.mode) +
          (v.active ? "（当前）" : "") + (v.compatible ? "" : "（不兼容）");
        html += '<option value="' + esc(v.version) + '"' + (v.compatible ? "" : " disabled") + ">" + esc(label) + "</option>";
      });
      select.innerHTML = html;
    });
    syncVersionSelects();
  }
  function syncVersionSelects() {
    var value = state.selectedVersion || "auto";
    var versions = (state.models && state.models.versions) || [];
    var ok = value === "auto" || versions.some(function (v) { return v.version === value && v.compatible; });
    if (!ok) {
      value = "auto";
      state.selectedVersion = "auto";
      localStorage.setItem("modelVersion", "auto");
    }
    [$("modelVersionSelect"), $("batchModelVersionSelect")].forEach(function (select) {
      if (select) select.value = value;
    });
  }
  function setSelectedVersion(value) {
    state.selectedVersion = value || "auto";
    localStorage.setItem("modelVersion", state.selectedVersion);
    syncVersionSelects();
  }

  // ------------------------------------------------------------------ 识别
  function bindAnalyze() {
    var dropzone = $("dropzone");
    var fileInput = $("fileInput");
    $("pickBtn").addEventListener("click", function (e) { e.stopPropagation(); fileInput.click(); });
    dropzone.addEventListener("click", function () { fileInput.click(); });
    dropzone.addEventListener("dragover", function (e) { e.preventDefault(); dropzone.classList.add("dragover"); });
    dropzone.addEventListener("dragleave", function () { dropzone.classList.remove("dragover"); });
    dropzone.addEventListener("drop", function (e) {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      if (e.dataTransfer.files.length) analyzeFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", function () {
      if (fileInput.files.length) analyzeFile(fileInput.files[0]);
    });
    $("modelVersionSelect").addEventListener("change", function () { setSelectedVersion(this.value); });
    $("resetBtn").addEventListener("click", resetAnalyze);
  }

  async function analyzeFile(file) {
    hide($("analyzeError"));
    var fd = new FormData();
    fd.append("file", file);
    fd.append("model_version", state.selectedVersion || "auto");
    try {
      var res = await api("/api/analyze", { method: "POST", body: fd });
      state.lastResult = res.result;
      renderAnalyze(res.result);
    } catch (err) {
      var el = $("analyzeError");
      el.textContent = errorText(err);
      show(el);
    }
  }

  function resetAnalyze() {
    hide($("analyzeResult"));
    $("dropzone").style.display = "";
    $("fileInput").value = "";
    hide($("analyzeError"));
    state.lastResult = null;
  }

  function renderAnalyze(result) {
    show($("analyzeResult"));
    $("dropzone").style.display = "none";
    $("annotatedImg").src = result.annotated_b64 || "";
    var rc = result.recapture || {};
    var badge = $("riskBadge");
    badge.textContent = RISK_TEXT[rc.risk] || rc.risk || "—";
    badge.className = "risk-badge risk-" + (rc.risk || "low");
    text($("classLabel"), CLASS_TEXT[rc.class] || rc.class || "—");
    text($("scoreVal"), "score " + (typeof rc.score === "number" ? rc.score.toFixed(2) : "0.00"));
    text($("modelUsed"), "模型 " + (rc.model_version || "—") + " · " + (MODE_TEXT[rc.mode] || rc.mode || "—"));

    var probsEl = $("probs");
    var html = "";
    var classes = rc.classes || Object.keys(rc.probs || {});
    classes.forEach(function (key) {
      var value = (rc.probs || {})[key] || 0;
      html += '<div class="prob-row"><span class="name">' + esc(CLASS_TEXT[key] || key) + "</span>" +
        '<div class="prob-bar"><div class="prob-fill" style="width:' + (value * 100).toFixed(0) + '%"></div></div>' +
        '<span class="val">' + value.toFixed(3) + "</span></div>";
    });
    probsEl.innerHTML = html;

    var evidenceEl = $("evidenceList");
    var evidenceHtml = "";
    (rc.evidences || []).forEach(function (item) {
      evidenceHtml += '<li><span class="e-name">' + esc(item.name) + "</span>" +
        '<div class="e-bar"><div class="e-fill" style="width:' + ((item.score || 0) * 100).toFixed(0) + '%"></div></div>' +
        '<span class="e-detail">' + esc(item.detail || "") + " (" + (item.score || 0).toFixed(2) + ")</span></li>";
    });
    evidenceEl.innerHTML = evidenceHtml;

    // 未安装 OCR 时不展示「关键字段」卡片，避免整块空表格。
    var hasOcr = !!(result.ocr && result.ocr.available);
    var fieldCard = $("fieldCard");
    if (fieldCard) fieldCard.classList.toggle("hidden", !hasOcr);
    if (hasOcr) {
      renderFields(result.fields || {});
      text($("ocrStatus"), "OCR：已启用（PP-OCR）");
    }
  }

  function renderFields(fields) {
    var rows = [
      ["姓名", fields.name], ["性别", fields.gender + (fields.gender_consistent === false ? " ⚠" : "")],
      ["民族", fields.ethnicity], ["出生", fields.birth + (fields.birth_consistent === false ? " ⚠" : "")],
      ["住址", fields.address], ["公民身份号码", fields.id_number + (fields.id_checksum_valid === false ? " ⚠" : "")],
      ["签发机关", fields.authority], ["有效期限", fields.valid_period]
    ];
    var table = $("fieldTable");
    var html = "";
    rows.forEach(function (row) {
      var value = row[1];
      html += "<tr><td>" + esc(row[0]) + "</td><td>" + (value ? esc(value) : "<span class='muted'>—</span>") + "</td></tr>";
    });
    table.innerHTML = html;
  }

  // ------------------------------------------------------------------ 批量
  function bindBatch() {
    var input = $("batchInput");
    $("batchPickBtn").addEventListener("click", function () { input.click(); });
    input.addEventListener("change", function () {
      state.batchFiles = Array.prototype.slice.call(input.files);
      $("batchStartBtn").disabled = state.batchFiles.length === 0;
      text($("batchStatus"), "已选择 " + state.batchFiles.length + " 张");
    });
    $("batchStartBtn").addEventListener("click", startBatch);
    $("exportCsvBtn").addEventListener("click", exportCSV);
    $("exportJsonBtn").addEventListener("click", exportJSON);
    $("batchModelVersionSelect").addEventListener("change", function () { setSelectedVersion(this.value); });
  }

  async function startBatch() {
    var fd = new FormData();
    state.batchFiles.forEach(function (file) { fd.append("files", file); });
    fd.append("model_version", state.selectedVersion || "auto");
    $("batchStartBtn").disabled = true;
    try {
      var res = await api("/api/batch", { method: "POST", body: fd });
      pollBatch(res.task_id);
    } catch (err) {
      text($("batchStatus"), "提交失败：" + errorText(err));
      $("batchStartBtn").disabled = false;
    }
  }

  async function pollBatch(tid) {
    try {
      var res = await api("/api/batch/" + tid);
      var bar = $("batchProgress");
      bar.style.width = (res.total ? (res.done / res.total) * 100 : 0) + "%";
      var tbody = document.querySelector("#batchTable tbody");
      var html = "";
      (res.results || []).forEach(function (item) {
        var rc = (item.result && item.result.recapture) || {};
        html += "<tr><td>" + esc(item.file_name) + "</td><td>" + esc(CLASS_TEXT[rc.class] || "—") + "</td>" +
          "<td>" + esc(rc.risk || "—") + "</td><td>" + (rc.score != null ? rc.score.toFixed(2) : "—") + "</td>" +
          "<td>" + esc(rc.model_version || "—") + "</td><td>" + (item.error ? "失败" : "完成") + "</td></tr>";
      });
      tbody.innerHTML = html;
      if (res.status === "not_found") {
        text($("batchStatus"), "批量任务不存在或已过期，请重新提交");
        $("batchStartBtn").disabled = false;
        return;
      }
      if (res.status === "done") {
        text($("batchStatus"), "完成 " + res.done + "/" + res.total +
          ((res.errors || []).length ? "，失败 " + res.errors.length + " 张" : ""));
        $("exportCsvBtn").disabled = false;
        $("exportJsonBtn").disabled = false;
        state.batchRes = res;
      } else if (res.status === "running") {
        text($("batchStatus"), "处理中 " + res.done + "/" + res.total);
        setTimeout(function () { pollBatch(tid); }, 600);
      }
    } catch (err) {
      text($("batchStatus"), "轮询失败：" + errorText(err));
    }
  }

  function exportCSV() {
    var res = state.batchRes;
    if (!res) return;
    var csv = "文件,类别,风险,分数,模型\n";
    (res.results || []).forEach(function (item) {
      var rc = (item.result && item.result.recapture) || {};
      csv += '"' + item.file_name + '",' + (rc.class || "") + "," + (rc.risk || "") + "," +
        (rc.score != null ? rc.score : "") + "," + (rc.model_version || "") + "\n";
    });
    download(csv, "batch_result.csv", "text/csv");
  }
  function exportJSON() {
    if (!state.batchRes) return;
    download(JSON.stringify(state.batchRes, null, 2), "batch_result.json", "application/json");
  }
  function download(content, name, type) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([content], { type: type }));
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ------------------------------------------------------------------ 历史
  async function loadHistory() {
    try {
      var res = await api("/api/history");
      var tbody = document.querySelector("#historyTable tbody");
      var html = "";
      (res.items || []).forEach(function (item) {
        var summary = item.summary || {};
        html += "<tr><td>" + new Date(item.created_at * 1000).toLocaleString() + "</td>" +
          "<td>" + esc(item.file_name) + "</td><td>" + esc(CLASS_TEXT[summary.class] || "—") + "</td>" +
          "<td>" + esc(summary.risk || "—") + "</td><td>" + esc(summary.model_version || "—") + "</td>" +
          "<td>" + esc(summary.name || "—") + "</td><td>" + esc(summary.id_number || "—") + "</td></tr>";
      });
      tbody.innerHTML = html || '<tr><td colspan="7" class="muted">暂无历史记录</td></tr>';
    } catch (err) {
      globalError("历史加载失败：" + errorText(err));
    }
  }

  // ------------------------------------------------------------------ 复核
  async function loadReview() {
    try {
      var res = await api("/api/feedback?status=pending");
      var el = $("reviewList");
      var items = res.items || [];
      if (!items.length) {
        el.innerHTML = '<p class="muted">暂无待复核样本。</p>';
        return;
      }
      var html = "";
      items.forEach(function (item) {
        html += '<div class="review-item" data-id="' + esc(item.id) + '">' +
          '<div class="grow">' + esc(item.file_name) + '<br/><span class="muted">标记类别：' +
          esc(CLASS_TEXT[item.true_class] || item.true_class) + "</span></div>" +
          '<select class="review-select"><option value="original">原件</option>' +
          '<option value="screen_recapture">屏幕翻拍</option><option value="print_recapture">打印翻拍</option></select>' +
          '<button class="btn primary" data-action="confirm">确认并入训练集</button>' +
          '<button class="btn" data-action="reject">拒绝</button></div>';
      });
      el.innerHTML = html;
      el.querySelectorAll(".review-item").forEach(function (row) {
        row.querySelector('[data-action="confirm"]').addEventListener("click", function () { confirmReview(row); });
        row.querySelector('[data-action="reject"]').addEventListener("click", function () { rejectReview(row); });
      });
    } catch (err) {
      globalError("复核队列加载失败：" + errorText(err));
    }
  }
  async function confirmReview(row) {
    try {
      await api("/api/feedback/" + row.getAttribute("data-id") + "/confirm",
        jsonOpts({ confirmed_class: row.querySelector(".review-select").value }));
      loadReview();
    } catch (err) { globalError("确认失败：" + errorText(err)); }
  }
  async function rejectReview(row) {
    try {
      await api("/api/feedback/" + row.getAttribute("data-id") + "/reject", { method: "POST" });
      loadReview();
    } catch (err) { globalError("拒绝失败：" + errorText(err)); }
  }

  // ------------------------------------------------------------------ 模型与训练
  function bindModels() {
    $("uploadBtn").addEventListener("click", uploadSamples);
    $("trainStartBtn").addEventListener("click", startTrain);
    $("versionsTable").addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-version]");
      if (btn) activateVersion(btn.getAttribute("data-version"));
    });
    $("datasetSummary").addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-delete-name]");
      if (btn) deleteSample(btn.getAttribute("data-delete-class"), btn.getAttribute("data-delete-name"));
    });
    $("trainResult").addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-activate]");
      if (btn) activateVersion(btn.getAttribute("data-activate"));
    });
  }

  async function loadModelsView() {
    globalError("");
    try {
      await loadModels();
      await loadDataset();
    } catch (err) {
      globalError("模型页加载失败：" + errorText(err));
    }
  }

  async function loadModels() {
    state.models = await api("/api/models");
    renderRuntimeBadge();
    fillVersionSelects();
    renderModelCurrent();
    renderVersions();
  }

  function renderRuntimeBadge() {
    var el = $("runtimeBadge");
    if (!el) return;
    var model = (state.models && state.models.current) || {};
    var mode = (state.models && state.models.modes && state.models.modes.current) ||
      (state.settings && state.settings.editable && state.settings.editable.recapture.mode) || "";
    el.textContent = "模型 " + (model.version || "—") + " · " + (MODE_TEXT[mode] || mode || "—");
    el.classList.toggle("warn", model.compatible === false);
  }

  function renderModelCurrent() {
    var current = (state.models && state.models.current) || {};
    text($("modelCurrent"), "当前版本：" + (current.version || "—") + " · " +
      (MODE_TEXT[current.mode] || current.mode || "—") + " · 类别：" +
      ((current.classes || []).map(function (c) { return CLASS_TEXT[c] || c; }).join(" / ") || "—"));
    var warning = $("modelWarning");
    var messages = [];
    if (current.compatible === false) {
      messages.push("当前模型不兼容当前识别模式，识别请求会被拦截；请启用兼容版本或切换识别模式。");
    }
    ((state.models && state.models.warnings) || []).forEach(function (w) { messages.push(w); });
    if (messages.length) {
      warning.innerHTML = messages.map(esc).join("<br/>");
      show(warning);
    } else {
      hide(warning);
    }
  }

  function renderVersions() {
    var versions = (state.models && state.models.versions) || [];
    var tbody = $("versionsTable").querySelector("tbody");
    var html = "";
    versions.forEach(function (v) {
      var metrics = v.metrics || {};
      var status;
      if (v.active) status = '<span class="status-badge ok">当前</span>';
      else if (!v.valid) status = '<span class="status-badge error">损坏</span>';
      else if (!v.compatible) status = '<span class="status-badge warn">不兼容</span>';
      else status = '<span class="status-badge">候选</span>';
      var action = "";
      if (!v.active && v.valid && v.compatible) {
        action = '<button class="btn small" data-version="' + esc(v.version) + '">启用</button>';
      } else if (!v.active && v.reason) {
        action = '<span class="muted">' + esc(v.reason) + "</span>";
      }
      html += "<tr><td>" + esc(v.version) + "</td><td>" + esc(MODE_TEXT[v.mode] || v.mode || "—") + "</td>" +
        "<td>" + esc((v.classes || []).map(function (c) { return CLASS_TEXT[c] || c; }).join(" / ")) + "</td>" +
        "<td>" + (v.created_at ? new Date(v.created_at * 1000).toLocaleString() : "—") + "</td>" +
        "<td>" + fmt(metrics.accuracy) + "</td><td>" + fmt(metrics.macro_f1) + "</td>" +
        "<td>" + status + "</td><td>" + action + "</td></tr>";
    });
    tbody.innerHTML = html || '<tr><td colspan="8" class="muted">暂无模型版本</td></tr>';
  }

  async function activateVersion(version) {
    if (!version) return;
    if (!window.confirm("切换当前模型版本为 " + version + " ？")) return;
    try {
      await api("/api/models/activate", jsonOpts({ version: version }));
      await loadModels();
      applyUiSettings();
    } catch (err) {
      globalError("切换版本失败：" + errorText(err));
    }
  }

  async function loadDataset() {
    var report = await api("/api/train/dataset");
    state.dataset = report;
    renderDataset(report);
  }

  function renderDataset(report) {
    text($("trainModeTag"), MODE_TEXT[report.mode] || report.mode || "—");
    var summary = $("datasetSummary");
    var statusMap = { missing: ["error", "缺样本"], low: ["warn", "偏少"], ok: ["ok", "可用"], good: ["ok", "充足"] };
    var html = "";
    var optional = report.optional_classes || [];
    (report.classes || []).forEach(function (cls) {
      var st = statusMap[cls.status] || ["", cls.status];
      if (cls.status === "missing" && optional.indexOf(cls.name) >= 0) {
        st = ["warn", "建议补充"];
      }
      html += '<div class="dataset-card"><div class="dc-head"><b>' + esc(cls.label) + "</b>" +
        '<span class="status-badge ' + st[0] + '">' + esc(st[1]) + "</span></div>" +
        '<div class="dc-count">' + cls.count + ' <span class="muted">张</span></div>' +
        '<div class="muted">最低 ' + cls.min_required + " · 建议 " + cls.recommended + "</div>";
      var recent = cls.recent_files || [];
      if (recent.length) {
        html += '<div class="dc-files"><span class="muted">最近：</span>';
        recent.slice(0, 4).forEach(function (name) {
          html += '<span class="file-chip">' + esc(name) +
            '<button class="chip-x" title="删除该样本" data-delete-class="' + esc(cls.name) +
            '" data-delete-name="' + esc(name) + '">×</button></span>';
        });
        html += "</div>";
      }
      html += "</div>";
    });
    summary.innerHTML = html;
    $("datasetGuidance").innerHTML = (report.guidance || []).map(function (g) {
      return "<li>" + esc(g) + "</li>";
    }).join("");
    $("datasetIssues").innerHTML = (report.issues || []).map(function (issue) {
      return '<div class="banner ' + (issue.level === "error" ? "error" : "warn") + '">' + esc(issue.message) + "</div>";
    }).join("");

    var uploadClass = $("uploadClass");
    var previous = uploadClass.value;
    uploadClass.innerHTML = (report.classes || []).map(function (cls) {
      return '<option value="' + esc(cls.name) + '">' + esc(cls.label) + "</option>";
    }).join("");
    if (previous) uploadClass.value = previous;

    if (!state.datasetDefaultsApplied) {
      var training = ((state.settings && state.settings.effective) || {}).training || {};
      if (training.epochs) $("trainEpochs").value = training.epochs;
      if (training.lr) $("trainLr").value = training.lr;
      if (training.target_fpr) $("trainTargetFpr").value = training.target_fpr;
      state.datasetDefaultsApplied = true;
    }

    $("trainStartBtn").disabled = !report.ready || state.trainingRunning;
    text($("trainHint"), report.ready
      ? (state.trainingRunning ? "训练进行中…" : "数据集体检通过，可开始训练")
      : "数据集体检未通过，请先补齐素材");
  }

  async function uploadSamples() {
    var classSelect = $("uploadClass");
    var input = $("uploadFiles");
    if (!input.files.length) { text($("uploadStatus"), "请先选择图片"); return; }
    var fd = new FormData();
    fd.append("class", classSelect.value);
    for (var i = 0; i < input.files.length; i++) fd.append("files", input.files[i]);
    text($("uploadStatus"), "上传中…");
    try {
      var res = await api("/api/train/samples", { method: "POST", body: fd });
      var skipped = res.skipped || [];
      text($("uploadStatus"), "已上传 " + res.saved.length + " 张，跳过 " + skipped.length + " 张" +
        (skipped.length ? "（" + skipped.map(function (s) { return s.reason; }).join("；") + "）" : ""));
      input.value = "";
      renderDataset(res.dataset);
    } catch (err) {
      text($("uploadStatus"), "上传失败：" + errorText(err));
    }
  }

  async function deleteSample(className, name) {
    if (!window.confirm("删除样本 " + className + "/" + name + " ？")) return;
    try {
      var res = await api("/api/train/samples/delete", jsonOpts({ "class": className, name: name }));
      text($("uploadStatus"), "已删除 " + name);
      renderDataset(res.dataset);
    } catch (err) {
      text($("uploadStatus"), "删除失败：" + errorText(err));
    }
  }

  async function startTrain() {
    var body = {
      epochs: parseInt($("trainEpochs").value, 10) || 200,
      lr: parseFloat($("trainLr").value) || 0.5,
      limit_per_class: parseInt($("trainLimit").value, 10) || 0,
      target_fpr: parseFloat($("trainTargetFpr").value) || 0.05,
      init_from_current: $("trainInitFrom").checked,
      auto_activate: true
    };
    var label = $("trainLabel").value.trim();
    if (label) body.label = label;
    $("trainStartBtn").disabled = true;
    $("trainResult").innerHTML = "";
    text($("trainHint"), "提交训练任务…");
    try {
      var res = await api("/api/train", jsonOpts(body));
      state.trainingJobId = res.job_id;
      state.trainingRunning = true;
      hide($("trainLogs"));
      pollTrain(res.job_id);
    } catch (err) {
      state.trainingRunning = false;
      $("trainStartBtn").disabled = false;
      text($("trainHint"), "提交失败：" + errorText(err));
    }
  }

  async function pollTrain(jobId) {
    try {
      var st = await api("/api/train/" + jobId);
      state.trainingRunning = st.status === "running";
      renderTrainStatus(st);
      if (state.trainingRunning) {
        setTimeout(function () { pollTrain(jobId); }, 1000);
      } else {
        state.trainingJobId = null;
        await loadModels();
        await loadDataset();
      }
    } catch (err) {
      state.trainingRunning = false;
      text($("trainHint"), "训练状态获取失败：" + errorText(err));
    }
  }

  function renderTrainStatus(st) {
    $("trainProgress").style.width = (st.progress || 0) + "%";
    var logs = $("trainLogs");
    var lines = st.logs || [];
    if (lines.length) {
      show(logs);
      logs.textContent = lines.join("\n");
      logs.scrollTop = logs.scrollHeight;
    }
    if (st.status === "running") {
      text($("trainHint"), "训练中：" + stageText(st.stage) + " " + (st.progress || 0) + "%");
      $("trainStartBtn").disabled = true;
    } else if (st.status === "failed") {
      text($("trainHint"), "训练失败：" + (st.error || "未知错误"));
      var issues = (st.issues || []).map(function (issue) {
        return '<div class="banner error">' + esc(issue.message) + "</div>";
      }).join("");
      $("trainResult").innerHTML = '<div class="banner error">训练失败：' + esc(st.error || "未知错误") + "</div>" + issues;
      $("trainStartBtn").disabled = false;
    } else if (st.status === "done") {
      renderTrainResult(st.result || {});
      $("trainStartBtn").disabled = false;
    }
  }

  function stageText(stage) {
    var map = { dataset: "数据集体检", extract: "特征提取", train: "训练模型", evaluate: "评估与阈值校准", promote: "生成版本" };
    return map[stage] || stage || "";
  }

  function renderTrainResult(result) {
    var metrics = result.metrics || {};
    var html = "";
    if (result.promoted) {
      html += '<div class="banner success">训练完成，已自动升级当前版本为 <b>' + esc(result.version) + "</b>" +
        (result.previous_version ? "，上一版本 " + esc(result.previous_version) + " 已保留" : "") + "。</div>";
    } else {
      html += '<div class="banner warn">训练完成但未自动启用（' +
        esc(GATE_TEXT[result.gate] || result.gate || "未知原因") + "），版本 <b>" +
        esc(result.version) + "</b> 已保存为候选。</div>";
    }
    html += '<table class="batch-table"><thead><tr><th>版本</th><th>Acc</th><th>macro-F1</th>' +
      "<th>AUC</th><th>EER</th><th>高风险阈值</th></tr></thead><tbody><tr>" +
      "<td>" + esc(result.version) + "</td><td>" + fmt(metrics.accuracy) + "</td>" +
      "<td>" + fmt(metrics.macro_f1) + "</td><td>" + fmt(metrics.auc_recapture) + "</td>" +
      "<td>" + fmt(metrics.eer) + "</td><td>" + fmt((result.thresholds || {}).high) + "</td></tr></tbody></table>";
    var buttons = "";
    if (result.promoted && result.previous_version) {
      buttons += '<button class="btn" data-activate="' + esc(result.previous_version) + '">回退到 ' +
        esc(result.previous_version) + "</button>";
    }
    if (!result.promoted) {
      buttons += '<button class="btn" data-activate="' + esc(result.version) + '">仍然启用此版本</button>';
    }
    if (buttons) html += '<div class="toolbar">' + buttons + "</div>";
    $("trainResult").innerHTML = html;
  }

  function fmt(value) {
    return (value == null || isNaN(value)) ? "—" : Number(value).toFixed(3);
  }

  // ------------------------------------------------------------------ 设置
  function bindSettings() {
    $("saveSettingsBtn").addEventListener("click", saveSettings);
    $("showTrainingToggle").addEventListener("change", function () {
      if (!state.settings) return;
      state.settings.editable = state.settings.editable || {};
      state.settings.editable.ui = state.settings.editable.ui || {};
      state.settings.editable.ui.show_training = this.checked;
      applyUiSettings();
    });
  }

  async function loadSettingsView() {
    try {
      var settings = await api("/api/settings");
      state.settings = settings;
      renderSettingsForm(settings);
      applyUiSettings();
      var health = await api("/api/health");
      $("healthDump").textContent = JSON.stringify(health, null, 2);
      $("settingsEffective").textContent = JSON.stringify(settings.effective, null, 2);
    } catch (err) {
      globalError("设置页加载失败：" + errorText(err));
    }
  }

  function renderSettingsForm(settings) {
    var editable = settings.editable || {};
    var ui = editable.ui || {};
    var mode = (editable.recapture || {}).mode || "screen_and_print";
    var training = editable.training || {};
    $("showTrainingToggle").checked = ui.show_training !== false;
    document.querySelectorAll('input[name="recaptureMode"]').forEach(function (radio) {
      radio.checked = radio.value === mode;
    });
    $("keepVersions").value = training.keep_versions || 10;
    $("targetFpr").value = training.target_fpr || 0.05;
    $("settingsWarnings").innerHTML = (settings.warnings || []).map(function (warning) {
      return '<div class="banner warn">' + esc(warning) + "</div>";
    }).join("");
    text($("settingsStatus"), "");
  }

  async function saveSettings() {
    var selected = document.querySelector('input[name="recaptureMode"]:checked');
    var patch = {
      ui: { show_training: $("showTrainingToggle").checked },
      recapture: { mode: selected ? selected.value : "screen_and_print" },
      training: {
        keep_versions: parseInt($("keepVersions").value, 10) || 10,
        target_fpr: parseFloat($("targetFpr").value) || 0.05
      }
    };
    text($("settingsStatus"), "保存中…");
    try {
      var res = await api("/api/settings", jsonOpts(patch));
      state.settings = await api("/api/settings");
      state.datasetDefaultsApplied = false;
      await loadModels();
      renderSettingsForm(state.settings);
      applyUiSettings();
      var messages = [];
      if (res.switched_to) messages.push("已自动切换到兼容版本 " + res.switched_to);
      if (res.warnings && res.warnings.length) messages = messages.concat(res.warnings);
      text($("settingsStatus"), messages.length ? messages.join("；") : "已保存，立即生效");
    } catch (err) {
      text($("settingsStatus"), "保存失败：" + errorText(err));
    }
  }

  function applyUiSettings() {
    var editable = (state.settings && state.settings.editable) || {};
    var ui = editable.ui || {};
    var showTraining = ui.show_training !== false;
    var card = $("trainCard");
    if (card) card.classList.toggle("hidden", !showTraining);
    var mode = (editable.recapture || {}).mode || "screen_and_print";
    text($("modeTag"), MODE_TEXT[mode] || mode);
    text($("trainModeTag"), MODE_TEXT[mode] || mode);
  }

  // ------------------------------------------------------------------ 启动
  async function bootstrap() {
    bindNav();
    bindAnalyze();
    bindBatch();
    bindModels();
    bindSettings();
    try {
      var results = await Promise.all([api("/api/settings"), api("/api/models")]);
      state.settings = results[0];
      state.models = results[1];
      applyUiSettings();
      renderRuntimeBadge();
      fillVersionSelects();
    } catch (err) {
      globalError("初始化失败：" + errorText(err));
    }
    switchView(currentView());
  }

  bootstrap();
})();
