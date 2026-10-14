// Plain JS chat client: no framework, no CDN.
const chatEl = document.getElementById("chat");
const form = document.getElementById("composer");
const input = document.getElementById("q");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const fileInput = document.getElementById("file");

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = "msg " + who;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
  return div;
}

function addCitations(botDiv, citations) {
  if (!citations || !citations.length) return;
  const box = document.createElement("div");
  box.className = "citations";
  citations.forEach((c) => {
    const el = document.createElement("div");
    el.className = "cite";
    const pageBit = c.page ? " (page " + c.page + ")" : "";
    const title = document.createElement("div");
    title.textContent = "[" + c.ref + "] " + c.source + pageBit + "  -  score " + c.score + " (click to expand)";
    const excerpt = document.createElement("div");
    excerpt.className = "excerpt";
    excerpt.textContent = c.excerpt;
    el.appendChild(title);
    el.appendChild(excerpt);
    el.addEventListener("click", () => el.classList.toggle("open"));
    box.appendChild(el);
  });
  botDiv.appendChild(box);
  chatEl.scrollTop = chatEl.scrollHeight;
}

async function refreshStatus() {
  try {
    const res = await fetch("/health");
    const h = await res.json();
    statusEl.textContent = h.documents + " docs, " + h.chunks + " chunks indexed";
  } catch (e) {
    statusEl.textContent = "offline";
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  sendBtn.disabled = true;
  addMessage(question, "user");
  const typing = addMessage("thinking...", "bot typing");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error("HTTP " + res.status);
    const data = await res.json();
    typing.classList.remove("typing");
    typing.textContent = data.answer;
    addCitations(typing, data.citations);
  } catch (err) {
    typing.textContent = "Error: " + err.message;
  } finally {
    sendBtn.disabled = false;
    chatEl.scrollTop = chatEl.scrollHeight;
    refreshStatus();
  }
});

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  addMessage("Ingesting " + file.name + "...", "bot typing");
  try {
    const res = await fetch("/ingest", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "upload failed");
    addMessage(
      "Ingested " + (data.source || file.name) + ": " + data.chunks_added + " chunks added.",
      "bot"
    );
  } catch (err) {
    addMessage("Ingest failed: " + err.message, "bot");
  }
  fileInput.value = "";
  refreshStatus();
});

addMessage(
  "Hi! Ask me anything about the indexed documents. Try: \"What is the return policy?\" or \"How do I reset my password?\"",
  "bot"
);
refreshStatus();
