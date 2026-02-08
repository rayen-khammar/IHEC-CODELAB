import { getAlerts } from "./api.js";
import { $, setLoading, showToast } from "./ui.js";
import { POLL_ALERTS_MS } from "./config.js";

function tone(sev) {
  if (sev === "HIGH") return "bad";
  if (sev === "MEDIUM") return "warn";
  return "";
}

function render(listEl, items) {
  listEl.innerHTML = items.map(a => `
    <div class="card" style="margin-top: 10px; padding: 12px; background: rgba(255,255,255,0.03); box-shadow:none;">
      <div class="row">
        <div style="font-weight: 700;">
          <a class="link" href="./stock.html?symbol=${encodeURIComponent(a.symbol)}">${a.symbol}</a> — ${a.title}
        </div>
        <span class="badge ${tone(a.severity)}">${a.severity}</span>
      </div>
      <div class="mini" style="margin-top: 6px;">
        ${new Date(a.ts).toLocaleString()} • ${a.type}
      </div>
    </div>
  `).join("") || `<div class="sub">No alerts found.</div>`;
}

async function load() {
  const root = $("#alerts-root");
  setLoading(root, true);

  try {
    const data = await getAlerts();
    $("#alerts-updated").textContent = new Date(data.last_updated).toLocaleString();

    const f = $("#filter").value;
    const items = (data.items || []).filter(a => f === "ALL" ? true : a.type === f);

    render($("#alerts-list"), items);
  } catch (e) {
    showToast("Failed to load alerts. Check API.", "error");
  } finally {
    setLoading(root, false);
  }
}

$("#btn-refresh").addEventListener("click", load);
$("#filter").addEventListener("change", load);

load();
setInterval(load, POLL_ALERTS_MS);
