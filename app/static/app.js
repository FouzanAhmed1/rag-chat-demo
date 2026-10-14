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

function getSessionId() {
  let id = null;
  try {
    id = localStorage.getItem("rag_session_id");
  } catch (e) {
    /* storage unavailable, fall through */
  }
  if (!id) {
    id =
      (window.crypto && crypto.randomUUID
        ? crypto.randomUUID()
        : "sess-" + Date.now().toString(36) + Math.random().toString(36).slice(2));
    try {
      localStorage.setItem("rag_session_id", id);
    } catch (e) {
      /* storage unavailable */
    }
  }
  return id;
}

// Stream an answer from POST /chat/stream, rendering tokens as they arrive.
// Throws if the stream is unusable so the caller can fall back to /chat.
async function askStream(question, botDiv) {
  const res = await fetch("/chat/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getSessionId(),
    },
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) {
    throw new Error("stream unavailable (HTTP " + res.status + ")");
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let gotDone = false;
  let citations = null;
  botDiv.classList.remove("typing");
  botDiv.textContent = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split("\n\n");
    buf = parts.pop();
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") {
        gotDone = true;
        break;
      }
      let evt;
      try {
        evt = JSON.parse(payload);
      } catch (e) {
        throw new Error("could not parse stream event");
      }
      if (evt.error) throw new Error(evt.error);
      if (evt.token) botDiv.textContent += evt.token;
      if (evt.citations) citations = evt.citations;
      chatEl.scrollTop = chatEl.scrollHeight;
    }
    if (gotDone) break;
  }
  if (!gotDone) throw new Error("stream ended without [DONE]");
  return citations;
}

async function askOnce(question, botDiv) {
  const res = await fetch("/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Session-Id": getSessionId(),
    },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) throw new Error("HTTP " + res.status);
  const data = await res.json();
  botDiv.classList.remove("typing");
  botDiv.textContent = data.answer;
  return data.citations;
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
    let citations;
    try {
      citations = await askStream(question, typing);
    } catch (streamErr) {
      // Graceful fallback: the plain non-streaming endpoint.
      citations = await askOnce(question, typing);
    }
    addCitations(typing, citations);
  } catch (err) {
    typing.classList.remove("typing");
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
