const express = require("express");
const axios = require("axios");
const fs = require("fs").promises; // ⚡ Changed to promises for async non-blocking file I/O
const fsSync = require("fs");      // Used strictly for initial directory creation check
const path = require("path");
const crypto = require("crypto");
const cors = require("cors");
const https = require("https");    // ⚡ Added for TCP Connection Pooling
const http = require("http");      // ⚡ Added for ultra-lean internal pings

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const PORT = process.env.PORT || 10000;
const API_KEY = process.env.GLM_API_KEY;
const MASTER_PROMPT = process.env.MASTER_PROMPT || "";

const GLM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions";

// ⚡ Optimized Axios Instance with Persistent TCP Sockets
const axiosInstance = axios.create({
  timeout: 600000,
  httpsAgent: new https.Agent({
    keepAlive: true,        // Reuses the same network socket for subsequent messages
    maxSockets: 100,        // Max concurrent open sockets
    keepAliveMsecs: 1000    // Keeps the channel hot
  })
});

const SESS_DIR = path.join(__dirname, "sessions");
if (!fsSync.existsSync(SESS_DIR)) fsSync.mkdirSync(SESS_DIR);

const activeStreams = new Map();

function sessionFile(id) {
  return path.join(SESS_DIR, `${id}.json`);
}

// ⚡ Refactored to handle asynchronous non-blocking disk reads
async function loadSession(id) {
  const f = sessionFile(id);
  try {
    const data = await fs.readFile(f, "utf8");
    return JSON.parse(data);
  } catch {
    return { structured_memory: null, messages: [] };
  }
}

// ⚡ Refactored to handle asynchronous non-blocking disk writes
async function saveSession(id, data) {
  try {
    await fs.writeFile(sessionFile(id), JSON.stringify(data));
  } catch (err) {
    console.error("Session save error:", err);
  }
}

function getConversationId(body) {
  if (body.conversation_id) return body.conversation_id;
  const base = body.messages?.[0]?.content || "default";
  return crypto.createHash("sha256").update(base).digest("hex");
}

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body;
  const convoId = getConversationId(body);

  try {
    if (activeStreams.has(convoId)) {
      try { activeStreams.get(convoId).end(); } catch {}
    }
    activeStreams.set(convoId, res);

    // ⚡ Await the async file reader
    const session = await loadSession(convoId);

    session.messages = body.messages || [];
    // ⚡ Await the async file writer
    await saveSession(convoId, session);

    const finalMessages = [];

    if (MASTER_PROMPT) {
      finalMessages.push({ role: "system", content: MASTER_PROMPT });
    }

    if (session.structured_memory) {
      finalMessages.push({
        role: "system",
        content: "LONG-TERM MEMORY:\n" + session.structured_memory
      });
    }

    finalMessages.push(...session.messages);

    const upstream = await axiosInstance.post(
      GLM_ENDPOINT,
      {
        ...body,
        model: process.env.MODEL_NAME || "z-ai/glm5",
        messages: finalMessages,
        stream: true,
        chat_template_kwargs: {"reasoning_effort":"high"}// Preserved exactly as requested// Preserved exactly as requested
         // Preserved exactly as requested
      },
      {
        headers: {
          Authorization: `Bearer ${API_KEY}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream"
        },
        responseType: "stream"
      }
    );

    res.setHeader("Content-Type", "text/event-stream");
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");

    let buffer = "";

    upstream.data.on("data", chunk => {
      buffer += chunk.toString();

      // ⚡ High-performance stream scanning (Prevents fragmentation chunk drops)
      let lineIndex;
      while ((lineIndex = buffer.indexOf("\n")) !== -1) {
        const line = buffer.substring(0, lineIndex).trim();
        buffer = buffer.substring(lineIndex + 1);

        if (!line.startsWith("data:")) continue;

        // Fast slice execution instead of allocating new strings via string replacements
        const data = line.slice(5).trim();

        if (data === "[DONE]") {
          res.write("data: [DONE]\n\n");
          res.end();
          return;
        }

        try {
          const parsed = JSON.parse(data);
          const delta = parsed?.choices?.[0]?.delta;
          if (!delta) continue;

          const out = {
            id: parsed.id || "chatcmpl-" + Date.now(),
            object: "chat.completion.chunk",
            created: parsed.created || Math.floor(Date.now() / 1000),
            model: process.env.MODEL_NAME || "z-ai/glm5",
            choices: [
              {
                index: 0,
                delta: delta,
                finish_reason: parsed.choices?.[0]?.finish_reason || null
              }
            ]
          };

          res.write(`data: ${JSON.stringify(out)}\n\n`);
        } catch {}
      }
    });

    upstream.data.on("end", () => {
      res.end();
      activeStreams.delete(convoId);
    });

    upstream.data.on("error", err => {
      console.error(err);
      res.end();
      activeStreams.delete(convoId);
    });

  } catch (err) {
    console.error(err);
    res.status(500).json({ error: "proxy failure" });
    activeStreams.delete(convoId);
  }
});

app.get("/ping", (req, res) => {
  res.send("alive");
});

app.listen(PORT, () => {
  console.log("LLM Proxy running optimally");
});

// ⚡ Ultra-lean native keep-alive loop (Removes Axios wrapper allocations)
setInterval(() => {
  http.get(`http://localhost:${PORT}/ping`, (res) => {
    res.resume(); // Consume response memory immediately
  }).on("error", () => {});
}, 240000);
