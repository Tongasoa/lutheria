// Client micro Lutheria : getUserMedia -> PCM int16 16 kHz -> WebSocket binaire.
// Protocole : frames binaires (PCM mono), ack = trame vide du serveur.
"use strict";

const TARGET_SR = 16000;
const SEND_CHUNK_SAMPLES = 1600; // 100 ms @16 kHz

const els = {
  start: document.getElementById("start"),
  stop: document.getElementById("stop"),
  status: document.getElementById("status"),
  token: document.getElementById("token"),
  log: document.getElementById("log"),
};

let ctx = null;
let node = null;
let stream = null;
let ws = null;

// --- rééchantillonnage linéaire (qualité suffisante pour ASR, cf. ADR 0001) ---
class Resampler {
  constructor(srcRate) {
    this.ratio = srcRate / TARGET_SR; // échantillons d'entrée par échantillon de sortie
    this.acc = 0;
    this.count = 0;
    this.pos = 0; // position fractionnaire dans le bloc courant (interpolation)
  }

  process(input) {
    const out = [];
    if (this.ratio > 1) {
      // décimation avec moyenne sur les `step` échantillons absorbés par sortie
      const step = Math.floor(this.ratio);
      for (let i = 0; i < input.length; i++) {
        this.acc += input[i];
        if (++this.count === step) {
          out.push(this.acc / step);
          this.acc = 0;
          this.count = 0;
        }
      }
    } else {
      // sur-échantillonnage (rare) : interpolation linéaire avec report de phase
      let pos = this.pos;
      while (pos < input.length) {
        const i0 = Math.floor(pos);
        const t = pos - i0;
        const s0 = input[i0];
        const s1 = i0 + 1 < input.length ? input[i0 + 1] : s0;
        out.push(s0 + (s1 - s0) * t);
        pos += this.ratio;
      }
      this.pos = pos - input.length;
    }
    return new Int16Array(out.map(toInt16));
  }
}

function toInt16(f) {
  const s = Math.max(-1, Math.min(1, f));
  return s < 0 ? s * 0x8000 : s * 0x7fff;
}

// --- file d'émission : on n'envoie que par paquets de ~100 ms ---
class Sender {
  constructor(ws) {
    this.ws = ws;
    this.buffer = new Int16Array(SEND_CHUNK_SAMPLES * 2); // marge
    this.len = 0;
  }

  push(samples) {
    let offset = 0;
    while (offset < samples.length) {
      const space = this.buffer.length - this.len;
      const take = Math.min(space, samples.length - offset);
      this.buffer.set(samples.subarray(offset, offset + take), this.len);
      this.len += take;
      offset += take;
      if (this.len === this.buffer.length) this.flush();
    }
  }

  flush() {
    if (this.ws.readyState === WebSocket.OPEN && this.len > 0) {
      this.ws.send(this.buffer.slice(0, this.len).buffer);
    }
    this.len = 0;
  }
}

let sender = null;

function setStatus(text, cls) {
  els.status.textContent = text;
  els.status.className = cls || "";
}

function log(msg) {
  const line = document.createElement("div");
  line.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
  els.log.prepend(line);
}

async function start() {
  const token = els.token.value.trim();
  if (!token) return setStatus("Token requis", "err");

  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws/mic?token=${encodeURIComponent(token)}`);
  ws.binaryType = "arraybuffer";
  ws.onopen = () => setStatus("Connecté", "ok");
  ws.onclose = (e) => { setStatus(`Déconnecté (${e.code})`, "err"); stopCapture(); };
  ws.onerror = () => setStatus("Erreur WebSocket", "err");
  ws.onmessage = () => {}; // acks : rien à faire

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (e) {
    return setStatus("Accès micro refusé", "err");
  }

  ctx = new AudioContext();
  await ctx.audioWorklet.addModule("pcm-worklet.js");
  const resampler = new Resampler(ctx.sampleRate);
  sender = new Sender(ws);

  node = new AudioWorkletNode(ctx, "pcm-capture");
  node.port.onmessage = (e) => sender.push(resampler.process(e.data.samples));
  ctx.createMediaStreamSource(stream).connect(node); // pas de connexion à destination (larsen)

  els.start.disabled = true;
  els.stop.disabled = false;
  setStatus("Micro actif", "ok");
  log("Capture démarrée (" + ctx.sampleRate + " Hz -> 16 kHz)");
}

function stopCapture() {
  if (sender) sender.flush();
  if (node) { node.port.onmessage = null; node.disconnect(); node = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (ctx) { ctx.close(); ctx = null; }
  els.start.disabled = false;
  els.stop.disabled = true;
}

els.start.addEventListener("click", start);
els.stop.addEventListener("click", () => { if (ws) ws.close(1000); stopCapture(); });
