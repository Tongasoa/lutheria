// AudioWorklet : capture float32 -> envoi au thread principal pour conversion PCM 16 kHz.
// Aucun traitement lourd ici : on ne fait que transmettre les blocs AudioBuffer.
class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this._seq = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    // mono : premier canal ; copie défensive (le buffer est réutilisé par WebAudio)
    this.port.postMessage({ seq: this._seq++, samples: new Float32Array(input[0]) });
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
