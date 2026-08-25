#!/usr/bin/env python
"""Client de fumée Lutheria : envoie un fichier audio au serveur et affiche les traductions.

Usage :
    python scripts/smoke_client.py --server wss://MON_IP --token TOKEN --file audio.wav
    python scripts/smoke_client.py --server ws://127.0.0.1:8000 --token TOKEN --file audio.wav

Simule exactement le client web : PCM 16 kHz par trames binaires de 100 ms,
écoute des messages partial/final. Utile pour valider un déploiement EC2 sans
navigateur (certificat auto-signé accepté via ssl no-verify avec --insecure).
"""

import argparse
import asyncio
import contextlib
import json
import wave


def load_pcm_16k(path: str) -> bytes:
    """Lit un WAV mono/piste-moyennée, rééchantillonne grossièrement vers 16 kHz."""
    with wave.open(path, "rb") as w:
        sr, n, ch, width = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        if width != 2:
            raise SystemExit("Le WAV doit être PCM 16 bits")
        raw = w.readframes(n)
    samples = raw
    if ch > 1:
        import array

        a = array.array("h", raw)
        samples = array.array(
            "h", [sum(a[i : i + ch]) // ch for i in range(0, len(a), ch)]
        )
        raw = samples.tobytes()
    if sr != 16000:
        import numpy as np

        a = np.frombuffer(raw, dtype=np.int16)
        target = int(len(a) * 16000 / sr)
        a = np.interp(
            np.linspace(0, len(a) - 1, target), np.arange(len(a)), a
        ).astype(np.int16)
        raw = a.tobytes()
    return raw


async def run(server: str, token: str, pcm: bytes, insecure: bool) -> None:
    import ssl

    import websockets

    ctx = None
    if server.startswith("wss") and insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    url_mic = f"{server}/ws/mic?token={token}"
    url_listen = f"{server}/ws/listen"

    async with websockets.connect(url_listen, ssl=ctx) as listener:
        async with websockets.connect(url_mic, ssl=ctx, max_size=2**20) as mic:
            listen_task = asyncio.create_task(drain(listener))
            chunk = 3200  # 100 ms
            for i in range(0, len(pcm), chunk):
                await mic.send(pcm[i : i + chunk])
                await mic.recv()  # ack
                await asyncio.sleep(0.09)  # cadence quasi temps réel
            print("-- fin d'émission, écoute 10 s --")
            await asyncio.sleep(10)
            listen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await listen_task


async def drain(listener) -> None:
    try:
        while True:
            msg = json.loads(await listener.recv())
            state = msg.get("state", "?")
            icon = "🟡" if state == "partial" else "🟢"
            line = (
                msg.get("text_mg", "")
                if state == "partial"
                else f"{msg.get('text_fr', '')}   [{msg.get('text_mg', '')}]"
            )
            print(f"{icon} #{msg.get('id')} {line}")
    except Exception:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True, help="ex: wss://1.2.3.4 ou ws://localhost:8000")
    p.add_argument("--token", required=True)
    p.add_argument("--file", required=True, help="fichier WAV (PCM 16 bits)")
    p.add_argument("--insecure", action="store_true", help="accepter certificat auto-signé")
    args = p.parse_args()

    pcm = load_pcm_16k(args.file)
    print(f"{len(pcm)/32000:.1f}s d'audio chargé depuis {args.file}")
    asyncio.run(run(args.server, args.token, pcm, args.insecure))


if __name__ == "__main__":
    main()
