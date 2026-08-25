// Client lecteur Lutheria : /ws/listen -> affichage avec patch partial/final.
// partial : texte malgache provisoire ; final (même id) : remplace par le français.
"use strict";

const list = document.getElementById("lines");
const statusEl = document.getElementById("status");
const lines = new Map(); // id -> <li>

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = cls || "";
}

function nearBottom() {
  return window.innerHeight + window.scrollY >= document.body.offsetHeight - 120;
}

function scrollToBottom(force) {
  if (force || nearBottom()) window.scrollTo({ top: document.body.scrollHeight });
}

function render(msg) {
  let li = lines.get(msg.id);
  if (!li) {
    li = document.createElement("li");
    const stick = nearBottom();
    list.appendChild(li);
    lines.set(msg.id, li);
    scrollToBottom(stick);
  }

  if (msg.state === "partial") {
    li.innerHTML = "";
    const mg = document.createElement("span");
    mg.className = "mg";
    mg.textContent = msg.text_mg + " …";
    li.appendChild(mg);
  } else {
    li.innerHTML = "";
    const fr = document.createElement("div");
    fr.className = "fr";
    fr.textContent = msg.text_fr || "(traduction indisponible)";
    const mg = document.createElement("div");
    mg.className = "mg";
    mg.textContent = msg.text_mg || "";
    li.appendChild(fr);
    li.appendChild(mg);
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/listen`);

  ws.onopen = () => setStatus("Connecté", "ok");
  ws.onclose = () => {
    setStatus("Déconnecté — reconnexion dans 2 s", "err");
    setTimeout(connect, 2000); // reconnexion automatique
  };
  ws.onerror = () => ws.close();
  ws.onmessage = (e) => {
    try {
      render(JSON.parse(e.data));
    } catch {
      /* message illisible : ignoré */
    }
  };
}

connect();
