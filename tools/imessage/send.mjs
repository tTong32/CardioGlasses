// Send one iMessage through Photon. Called by backend/caregiver.py.
//   node send.mjs <phone-or-email>   (message text on stdin)
// Env: PHOTON_IMESSAGE_ADDRESS  "https://..." for the HTTP API, or "host:port" for gRPC
//      PHOTON_IMESSAGE_TOKEN    bearer token from Photon
// Prints one JSON line: {"ok": true, "guid": "..."} or {"ok": false, "error": "..."}
import { createHttpClient } from "@photon-ai/advanced-imessage";

const to = process.argv[2];
const address = process.env.PHOTON_IMESSAGE_ADDRESS;
const token = process.env.PHOTON_IMESSAGE_TOKEN;

function done(result) {
  process.stdout.write(JSON.stringify(result) + "\n");
  process.exit(result.ok ? 0 : 1);
}

let text = "";
for await (const chunk of process.stdin) text += chunk;
text = text.trim();

if (!to || !text) done({ ok: false, error: "usage: node send.mjs <recipient> < message.txt" });
if (!address || !token) done({ ok: false, error: "PHOTON_IMESSAGE_ADDRESS and PHOTON_IMESSAGE_TOKEN must be set" });

let im;
try {
  if (/^https?:\/\//.test(address)) {
    im = createHttpClient({ address, token });
  } else {
    const { createClient } = await import("@photon-ai/advanced-imessage/grpc");
    im = createClient({ address, token });
  }
  const sent = await im.messages.sendText(`any;-;${to}`, text);
  await im.close();
  done({ ok: true, guid: sent?.guid ?? null });
} catch (err) {
  try { await im?.close(); } catch {}
  done({ ok: false, error: String(err?.message || err) });
}
