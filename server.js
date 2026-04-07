const express = require("express");
const axios = require("axios");
const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const PORT = process.env.PORT || 10000;
const API_KEY = process.env.GLM_API_KEY;
const MASTER_PROMPT = process.env.MASTER_PROMPT || "";

const GLM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions";

const axiosInstance = axios.create({
  timeout: 600000
});

const SESS_DIR = path.join(__dirname, "sessions");
if (!fs.existsSync(SESS_DIR)) fs.mkdirSync(SESS_DIR);

const activeStreams = new Map();

function sessionFile(id) {
  return path.join(SESS_DIR, `${id}.json`);
}

function loadSession(id) {
  const f = sessionFile(id);
  if (fs.existsSync(f)) return JSON.parse(fs.readFileSync(f));
  return { structured_memory: null, messages: [] };
}

function saveSession(id, data) {
  fs.writeFileSync(sessionFile(id), JSON.stringify(data));
}

function getConversationId(body) {
  if (body.conversation_id) return body.conversation_id;
  const base = body.messages?.[0]?.content || "default";
  return crypto.createHash("sha256").update(base).digest("hex");
}

app.post("/v1/chat/completions", async (req, res) => {
  try {
    const body = req.body;
    const convoId = getConversationId(body);

    if (activeStreams.has(convoId)) {
      try { activeStreams.get(convoId).end(); } catch {}
    }
    activeStreams.set(convoId, res);

    const session = loadSession(convoId);

    // 🔥 LIMIT CONTEXT (prevents drift in Chub)
    session.messages = body.messages;
    saveSession(convoId, session);

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
        max_tokens: 4096
        reasoning_budget: 16384,
    chat_template_kwargs: {"enable_thinking":true}
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

      const events = buffer.split("\n\n");
      buffer = events.pop();

      for (const event of events) {
        if (!event.startsWith("data:")) continue;

        const data = event.replace("data:", "").trim();

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
            id: "chatcmpl-" + Date.now(),
            object: "chat.completion.chunk",
            created: Math.floor(Date.now() / 1000),
            model: process.env.MODEL_NAME || "z-ai/glm5",
            choices: [
              {
                index: 0,
                delta: delta,
                finish_reason: null
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
  }
});

app.get("/ping", (req, res) => {
  res.send("alive");
});

app.listen(PORT, () => {
  console.log("LLM Proxy running");
});

// keep render alive
setInterval(async () => {
  try {
    await axios.get(`http://localhost:${PORT}/ping`);
  } catch {}
}, 240000);
