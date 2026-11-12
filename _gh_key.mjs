import fs from "node:fs";
import https from "node:https";
const TOKEN = process.env.DSH_CM_GITHUB_TOKEN;
if (!TOKEN) { console.error("NO_TOKEN"); process.exit(1); }
const OWNER = "accpowered";
const REPO = process.env.REPO_NAME || "idcard-recapture-detector";

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
        resolve({ status: res.statusCode, json });
      });
    });
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

const mode = process.argv[2];
if (mode === "add") {
  const pub = fs.readFileSync(process.argv[3], "utf8").trim();
  const r = await api("POST", "/repos/" + OWNER + "/" + REPO + "/keys", {
    title: "dsh-temp-publish-" + Date.now(), key: pub, read_only: false,
  });
  if (r.status !== 201) { console.log("ADD_FAILED", r.status, JSON.stringify(r.json)); process.exit(1); }
  console.log("KEY_ID " + r.json.id);
} else if (mode === "delete") {
  const id = process.argv[3];
  const r = await api("DELETE", "/repos/" + OWNER + "/" + REPO + "/keys/" + id);
  console.log("DELETE_KEY_STATUS " + r.status);
} else if (mode === "list") {
  const r = await api("GET", "/repos/" + OWNER + "/" + REPO + "/keys");
  console.log("KEYS " + JSON.stringify((r.json || []).map((k) => ({ id: k.id, title: k.title }))));
}
process.exit(0);
