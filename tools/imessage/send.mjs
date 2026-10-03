// Send one iMessage through Photon. Called by backend/caregiver.py.
//   node send.mjs <phone-or-email>   (message text on stdin)
// Env (either form):
//   Spectrum (Photon cloud project):  PHOTON_IMESSAGE_ADDRESS = project ID (a UUID)
//                                     PHOTON_IMESSAGE_TOKEN   = project secret key
//   Advanced iMessage server:         PHOTON_IMESSAGE_ADDRESS = "https://..." or "host:port"
//                                     PHOTON_IMESSAGE_TOKEN   = bearer token
// Prints one JSON line: {"ok": true, "id": "..."} or {"ok": false, "error": "..."}
const to = process.argv[2];
const address = (process.env.PHOTON_IMESSAGE_ADDRESS || "").trim();
const token = (process.env.PHOTON_IMESSAGE_TOKEN || "").trim();
const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function done(result) {
  process.stdout.write(JSON.stringify(result) + "\n");
  process.exit(result.ok ? 0 : 1);
}

let text = "";
for await (const chunk of process.stdin) text += chunk;
text = text.trim();

if (!to || !text) done({ ok: false, error: "usage: node send.mjs <recipient> < message.txt" });
if (!address || !token) done({ ok: false, error: "PHOTON_IMESSAGE_ADDRESS and PHOTON_IMESSAGE_TOKEN must be set" });

async function viaSpectrum() {
  const { Spectrum } = await import("spectrum-ts");
  const { imessage } = await import("spectrum-ts/providers/imessage");
  const app = await Spectrum({ projectId: address, projectSecret: token, providers: [imessage.config()] });
  try {
    const space = await imessage(app).space.create(to);
    const sent = await space.send(text);
    return { ok: true, id: sent?.id ?? null, via: "spectrum" };
  } finally {
    await app.stop();
  }
}

async function viaServer() {
  let im;
  try {
    if (/^https?:\/\//.test(address)) {
      const { createHttpClient } = await import("@photon-ai/advanced-imessage");
      im = createHttpClient({ address, token });
    } else {
      const { createClient } = await import("@photon-ai/advanced-imessage/grpc");
      im = createClient({ address, token });
    }
    const sent = await im.messages.sendText(`any;-;${to}`, text);
    return { ok: true, id: sent?.guid ?? null, via: "server" };
  } finally {
    try { await im?.close(); } catch {}
  }
}

try {
  done(await (UUID.test(address) ? viaSpectrum() : viaServer()));
} catch (err) {
  let error = String(err?.message || err).slice(0, 300);
  if (/target not allowed/i.test(error)) {
    // Photon's free plan can't start conversations: the recipient must message the project's number first.
    error = "Photon won't message this number yet: text your Photon iMessage number from this phone once, then retry";
  }
  done({ ok: false, error });
}
