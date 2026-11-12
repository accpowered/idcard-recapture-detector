/* 居民身份证翻拍识别工具 — 前端逻辑 */
(function () {
  "use strict";

  const RISK_TEXT = { low: "低风险·原件", medium: "中风险·建议复核", high: "高风险·疑似翻拍" };
  const CLASS_TEXT = { original: "原件实拍", screen_recapture: "屏幕翻拍", print_recapture: "打印翻拍" };
  let maskFields = true;
  let lastAnalyzeResult = null;

  // ---- 视图切换 ----
  const navBtns = document.querySelectorAll(".nav-btn");
  navBtns.forEach((b) => b.addEventListener("click", () => switchView(b.dataset.view)));
  function switchView(v) {
    document.querySelectorAll(".view").forEach((s) => s.classList.remove("active"));
    document.getElementById("view-" + v).classList.add("active");
    navBtns.forEach((b) => b.classList.toggle("active", b.dataset.view === v));
    if (v === "history") loadHistory();
    if (v === "review") loadReview();
    if (v === "models") loadModels();
    if (v === "settings") loadSettings();
  }

  // ---- 识别 ----
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  document.getElementById("pickBtn").addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("dragover"); });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    e.preventDefault(); dropzone.classList.remove("dragover");
    if (e.dataTransfer.files.length) analyzeFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", () => { if (fileInput.files.length) analyzeFile(fileInput.files[0]); });
  document.getElementById("toggleMask").addEventListener("click", () => {
    maskFields = !maskFields;
    if (lastAnalyzeResult) renderFields(lastAnalyzeResult.fields, lastAnalyzeResult.recapture);
  });

  async function analyzeFile(file) {
    hide(document.getElementById("analyzeError"));
    const fd = new FormData();
    fd.append("file", file);
    try {
      const res = await fetch("/api/analyze", { method: "POST", body: fd }).then((r) => r.json());
      if (res.code !== 0) { showError(res.error || "识别失败"); return; }
      lastAnalyzeResult = res.result;
      renderAnalyze(res.result);
    } catch (e) { showError("请求失败：" + e.message); }
  }

  function showError(msg) { const el = document.getElementById("analyzeError"); el.textContent = msg; el.classList.remove("hidden"); }

  function renderAnalyze(r) {
    document.getElementById("analyzeResult").classList.remove("hidden");
    document.getElementById("dropzone").style.display = "none";
    document.getElementById("annotatedImg").src = r.annotated_b64 || "";
    const rc = r.recapture;
    const badge = document.getElementById("riskBadge");
    badge.textContent = RISK_TEXT[rc.risk] || rc.risk;
    badge.className = "risk-badge risk-" + rc.risk;
    document.getElementById("classLabel").textContent = CLASS_TEXT[rc.class] || rc.class;
    document.getElementById("scoreVal").textContent = "score " + rc.score.toFixed(2);
    const probsEl = document.getElementById("probs");
    probsEl.innerHTML = "";
    [["original", "原件"], ["screen_recapture", "屏幕"], ["print_recapture", "打印"]].forEach(([k, label]) => {
      const v = rc.probs[k] || 0;
      probsEl.innerHTML += '<div class="prob-row"><span class="name">' + label + '</span>' +
        '<div class="prob-bar"><div class="prob-fill" style="width:' + (v * 100).toFixed(0) + '%"></div></div>' +
        '<span class="val">' + v.toFixed(3) + '</span></div>';
    });
    const ev = document.getElementById("evidenceList");
    ev.innerHTML = "";
    (rc.evidences || []).forEach((e) => {
      ev.innerHTML += '<li><span class="e-name">' + e.name + '</span>' +
        '<div class="e-bar"><div class="e-fill" style="width:' + (e.score * 100).toFixed(0) + '%"></div></div>' +
        '<span class="e-detail">' + (e.detail || "") + ' (' + e.score.toFixed(2) + ')</span></li>';
    });
    renderFields(r.fields, rc);
    const ocr = document.getElementById("ocrStatus");
    ocr.textContent = r.ocr && r.ocr.available
      ? "OCR：已启用（PP-OCR）" : "OCR：未启用（安装 rapidocr 后可提取字段）";
  }

  function renderFields(fields, rc) {
    const rows = [
      ["姓名", fields.name], ["性别", fields.gender + (fields.gender_consistent === false ? " ⚠" : "")],
      ["民族", fields.ethnicity], ["出生", fields.birth + (fields.birth_consistent === false ? " ⚠" : "")],
      ["住址", fields.address], ["公民身份号码", fields.id_number + (fields.id_checksum_valid === false ? " ⚠" : "")],
      ["签发机关", fields.authority], ["有效期限", fields.valid_period],
    ];
    const t = document.getElementById("fieldTable");
    t.innerHTML = "";
    rows.forEach(([k, v]) => {
      t.innerHTML += "<tr><td>" + k + "</td><td>" + (v || "<span class='muted'>—</span>") + "</td></tr>";
    });
  }

  // ---- 批量 ----
  const batchInput = document.getElementById("batchInput");
  let batchFiles = [];
  document.getElementById("batchPickBtn").addEventListener("click", () => batchInput.click());
  batchInput.addEventListener("change", () => {
    batchFiles = Array.from(batchInput.files);
    document.getElementById("batchStartBtn").disabled = batchFiles.length === 0;
    document.getElementById("batchStatus").textContent = "已选择 " + batchFiles.length + " 张";
  });
  document.getElementById("batchStartBtn").addEventListener("click", startBatch);

  async function startBatch() {
    const fd = new FormData();
    batchFiles.forEach((f) => fd.append("files", f));
    const res = await fetch("/api/batch", { method: "POST", body: fd }).then((r) => r.json());
    const tid = res.task_id;
    document.getElementById("batchStartBtn").disabled = true;
    pollBatch(tid);
  }

  async function pollBatch(tid) {
    const res = await fetch("/api/batch/" + tid).then((r) => r.json());
    const bar = document.getElementById("batchProgress");
    bar.style.width = (res.total ? (res.done / res.total) * 100 : 0) + "%";
    const tbody = document.querySelector("#batchTable tbody");
    tbody.innerHTML = "";
    (res.results || []).forEach((r) => {
      const rc = (r.result && r.result.recapture) || {};
      tbody.innerHTML += "<tr><td>" + r.file_name + "</td><td>" + (CLASS_TEXT[rc.class] || "—") + "</td>" +
        "<td>" + (rc.risk || "—") + "</td><td>" + (rc.score != null ? rc.score.toFixed(2) : "—") + "</td>" +
        "<td>" + (r.error ? "失败" : "完成") + "</td></tr>";
    });
    if (res.status === "done") {
      document.getElementById("batchStatus").textContent = "完成 " + res.done + "/" + res.total;
      document.getElementById("exportCsvBtn").disabled = false;
      document.getElementById("exportJsonBtn").disabled = false;
      window._batchRes = res;
    } else if (res.status === "running") {
      document.getElementById("batchStatus").textContent = "处理中 " + res.done + "/" + res.total;
      setTimeout(() => pollBatch(tid), 600);
    }
  }

  function exportCSV() {
    const res = window._batchRes; if (!res) return;
    let csv = "文件,类别,风险,分数\n";
    (res.results || []).forEach((r) => {
      const rc = (r.result && r.result.recapture) || {};
      csv += '"' + r.file_name + '",' + (rc.class || "") + "," + (rc.risk || "") + "," + (rc.score != null ? rc.score : "") + "\n";
    });
    download(csv, "batch_result.csv", "text/csv");
  }
  function exportJSON() {
    download(JSON.stringify(window._batchRes, null, 2), "batch_result.json", "application/json");
  }
  document.getElementById("exportCsvBtn").addEventListener("click", exportCSV);
  document.getElementById("exportJsonBtn").addEventListener("click", exportJSON);

  function download(content, name, type) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([content], { type: type }));
    a.download = name; a.click(); URL.revokeObjectURL(a.href);
  }

  // ---- 历史 ----
  async function loadHistory() {
    const res = await fetch("/api/history").then((r) => r.json());
    const tbody = document.querySelector("#historyTable tbody");
    tbody.innerHTML = "";
    (res.items || []).forEach((it) => {
      tbody.innerHTML += "<tr><td>" + new Date(it.created_at * 1000).toLocaleString() + "</td>" +
        "<td>" + it.file_name + "</td><td>" + (CLASS_TEXT[it.summary.class] || "—") + "</td>" +
        "<td>" + (it.summary.risk || "—") + "</td><td>" + (it.summary.name || "—") + "</td>" +
        "<td>" + (it.summary.id_number || "—") + "</td><td></td></tr>";
    });
  }

  // ---- 复核 ----
  async function loadReview() {
    const res = await fetch("/api/feedback").then((r) => r.json());
    const el = document.getElementById("reviewList");
    el.innerHTML = "";
    if (!(res.items || []).length) { el.innerHTML = '<p class="muted">暂无待复核样本。</p>'; return; }
    res.items.forEach((it) => {
      el.innerHTML += '<div class="review-item" id="rv-' + it.id + '">' +
        '<div class="grow">' + it.file_name + '<br/><span class="muted">标记类别：' + (CLASS_TEXT[it.true_class] || it.true_class) + '</span></div>' +
        '<select id="sel-' + it.id + '"><option value="original">原件</option><option value="screen_recapture">屏幕翻拍</option><option value="print_recapture">打印翻拍</option></select>' +
        '<button class="btn primary" onclick="window.confirmReview(\'' + it.id + '\')">确认并入训练集</button>' +
        '<button class="btn" onclick="window.rejectReview(\'' + it.id + '\')">拒绝</button></div>';
    });
  }
  window.confirmReview = async function (id) {
    const cls = document.getElementById("sel-" + id).value;
    await fetch("/api/feedback/" + id + "/confirm", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmed_class: cls }) });
    loadReview();
  };
  window.rejectReview = async function (id) {
    await fetch("/api/feedback/" + id + "/reject", { method: "POST" });
    loadReview();
  };

  // ---- 模型 ----
  async function loadModels() {
    const res = await fetch("/api/models").then((r) => r.json());
    const el = document.getElementById("modelInfo");
    el.innerHTML = "<p>当前版本：<b>" + (res.current.version || "未知") + "</b></p>" +
      "<p class='muted'>历史版本：" + (res.versions.join(", ") || "无") + "</p>" +
      (res.current.metrics ? "<pre class='pre'>" + res.current.metrics + "</pre>" : "");
  }
  document.getElementById("trainForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const body = {
      data_dir: document.getElementById("trainData").value,
      epochs: parseInt(document.getElementById("trainEpochs").value, 10),
      out: document.getElementById("trainOut").value,
    };
    const res = await fetch("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }).then((r) => r.json());
    pollTrain(res.job_id);
  });
  async function pollTrain(jobId) {
    const res = await fetch("/api/train/" + jobId).then((r) => r.json());
    const el = document.getElementById("trainStatus");
    if (res.status === "done") el.textContent = "训练完成 → " + (res.weights_path || "");
    else if (res.status === "failed") el.textContent = "训练失败：" + res.error;
    else { el.textContent = "训练中…"; setTimeout(() => pollTrain(jobId), 1500); }
  }

  // ---- 设置 ----
  async function loadSettings() {
    const res = await fetch("/api/health").then((r) => r.json());
    document.getElementById("settingsDump").textContent = JSON.stringify(res, null, 2);
  }

  function hide(el) { el.classList.add("hidden"); }

  // 识别结果：隐藏拖拽区 / 换一张
  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      document.getElementById("analyzeResult").classList.add("hidden");
      document.getElementById("dropzone").classList.remove("hidden");
      document.getElementById("dropzone").style.display = "";
      fileInput.value = "";
    });
  }

})();