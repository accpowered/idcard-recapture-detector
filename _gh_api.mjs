import https from "node:https";
const TOKEN = process.env.DSH_CM_GITHUB_TOKEN;
if (!TOKEN) { console.error("NO_TOKEN"); process.exit(1); }
const OWNER = "accpowered";
const REPO = process.env.REPO_NAME || "idcard-recapture-detector";
const DESC = "居民身份证翻拍识别工具：卡证定位 + 翻拍判定（原件/屏幕/打印）+ 中文 OCR 字段校验，纯 CPU Web 应用，支持再训练闭环。";

function api(method, path, body) {
  return new Promise((resolve, reject) => {
    const data = body ? JSON.stringify(body) : null;
    const req = https.request({
      host: "api.github.com", path, method,
      headers: {
        "Authorization": "Bearer " + TOKEN,
        "Accept": "application/vnd.github+json",
        "User-Agent": "dsh-publisher",
        "X-GitHub-Api-Version": "2022-11-28",
        ...(data ? { "Content-Length": Buffer.byteLength(data) } : {}),
      },
    }, (res) => {
      let chunks = "";
      res.on("data", (d) => { chunks += d; });
      res.on("end", () => {
        let json = null;
        try { json = JSON.parse(chunks); } catch { /* ignore */ }
        resolve({ status: res.statusCode, json, text: chunks });
      });
    });
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

const user = await api("GET", "/user");
console.log("USER", user.status, user.json && user.json.login);
if (user.status !== 200 || !user.json || user.json.login !== OWNER) {
  console.log("IDENTITY_MISMATCH", JSON.stringify({ status: user.status, login: user.json && user.json.login }));
  process.exit(1);
}

let repo = await api("POST", "/user/repos", { name: REPO, description: DESC, private: false, auto_init: false });
if (repo.status === 201) {
  console.log("CREATED", repo.json.full_name);
} else if (repo.status === 422) {
  repo = await api("GET", "/repos/" + OWNER + "/" + REPO);
  console.log("EXISTS", repo.status, repo.json && repo.json.full_name);
} else {
  console.log("CREATE_FAILED", repo.status, (repo.text || "").slice(0, 300));
  process.exit(1);
}
if (repo.status !== 200 && repo.status !== 201) { console.log("REPO_ERROR", repo.status); process.exit(1); }
console.log("REPO_INFO " + JSON.stringify({
  full_name: repo.json.full_name,
  html_url: repo.json.html_url,
  private: repo.json.private,
  default_branch: repo.json.default_branch,
  ssh_url: repo.json.ssh_url,
}));
process.exit(0);
