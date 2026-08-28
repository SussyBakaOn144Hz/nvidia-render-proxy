const express = require("express");
const axios = require("axios");
const fs = require("fs").promises;
const fsSync = require("fs");
const path = require("path");
const crypto = require("crypto");
const cors = require("cors");
const https = require("https");
const http = require("http");

const app = express();
app.use(cors());
app.use(express.json({ limit: "20mb" }));

const PORT = process.env.PORT || 10000;
const API_KEY = process.env.GLM_API_KEY;
const MASTER_PROMPT = process.env.MASTER_PROMPT || "";
const MODEL_NAME = process.env.MODEL_NAME || "moonshotai/kimi-k3";

const NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions";

const axiosInstance = axios.create({
  timeout: 600000,
  httpsAgent: new https.Agent({
    keepAlive: true,
    maxSockets: 100,
    keepAliveMsecs: 1000
  })
});

const SESS_DIR = path.join(__dirname, "sessions");
if (!fsSync.existsSync(SESS_DIR)) fsSync.mkdirSync(SESS_DIR);

const activeStreams = new Map();

function sessionFile(id) {
  return path.join(SESS_DIR, `${id}.json`);
}

async function loadSession(id) {
  const f = sessionFile(id);
  try {
    const data = await fs.readFile(f, "utf8");
    return JSON.parse(data);
  } catch {
    return { structured_memory: null, messages: [] };
  }
}

async function saveSession(id, data) {
  try {
    await fs.writeFile(sessionFile(id), JSON.stringify(data));
  } catch (err) {
    console.error("Session save error:", err);
  }
}

function getConversationId(body) {
  if (body.conversation_id) return body.conversation_id;
  const firstMsg = body.messages?.[0]?.content || "default";
  const lastUserMsg = body.messages?.filter(m => m.role === "user").pop()?.content || "";
  return crypto.createHash("sha256").update(firstMsg + lastUserMsg.slice(0, 50)).digest("hex");
}

app.post("/v1/chat/completions", async (req, res) => {
  const body = req.body;
  const convoId = getConversationId(body);

  try {
    if (activeStreams.has(convoId)) {
      try { activeStreams.get(convoId).end(); } catch {}
    }
    activeStreams.set(convoId, res);

    const session = await loadSession(convoId);
    session.messages = body.messages || [];
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
      NVIDIA_ENDPOINT,
      {
        ...body,
        model: MODEL_NAME,
        messages: finalMessages,
        stream: true,
        max_tokens: body.max_tokens || 8192,
        reasoning_effort: "high"
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
    let isThinking = false;

    upstream.data.on("data", chunk => {
      buffer += chunk.toString();

      let lineIndex;
      while ((lineIndex = buffer.indexOf("\n")) !== -1) {
        const line = buffer.substring(0, lineIndex).trim();
        buffer = buffer.substring(lineIndex + 1);

        if (!line.startsWith("data:")) continue;

        const data = line.slice(5).trim();

        if (data === "[DONE]") {
          if (isThinking) {
            const closeTag = {
              id: "chatcmpl-" + Date.now(),
              object: "chat.completion.chunk",
              created: Math.floor(Date.now() / 1000),
              model: MODEL_NAME,
              choices: [{ index: 0, delta: { content: "\n</think>\n\n" }, finish_reason: null }]
            };
            res.write(`data: ${JSON.stringify(closeTag)}\n\n`);
          }
          res.write("data: [DONE]\n\n");
          res.end();
          return;
        }

        try {
          const parsed = JSON.parse(data);
          const delta = parsed?.choices?.[0]?.delta;
          if (!delta) continue;

          let mappedContent = "";

          // Maps reasoning stream tokens to <think> ... </think> tags
          if (delta.reasoning_content) {
            if (!isThinking) {
              isThinking = true;
              mappedContent = "<think>\n" + delta.reasoning_content;
            } else {
              mappedContent = delta.reasoning_content;
            }
          } else if (delta.content) {
            if (isThinking) {
              isThinking = false;
              mappedContent = "\n</think>\n\n" + delta.content;
            } else {
              mappedContent = delta.content;
            }
          }

          const out = {
            id: parsed.id || "chatcmpl-" + Date.now(),
            object: "chat.completion.chunk",
            created: parsed.created || Math.floor(Date.now() / 1000),
            model: MODEL_NAME,
            choices: [
              {
                index: 0,
                delta: { content: mappedContent },
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
      console.error("Upstream stream error:", err);
      res.end();
      activeStreams.delete(convoId);
    });

  } catch (err) {
    if (err.response?.data?.on) {
      let errData = "";
      err.response.data.on("data", c => (errData += c.toString()));
      err.response.data.on("end", () => {
        console.error(`Upstream Error [HTTP ${err.response.status}]:`, errData);
      });
    } else {
      console.error("Proxy Error:", err.message);
    }

    if (!res.headersSent) {
      res.status(err.response?.status || 500).json({
        error: {
          message: err.message || "Proxy failure",
          status: err.response?.status
        }
      });
    } else {
      res.end();
    }
    activeStreams.delete(convoId);
  }
});

app.get("/ping", (req, res) => {
  res.send("alive");
});

app.listen(PORT, () => {
  console.log(`LLM Proxy running on port ${PORT}`);
});

setInterval(() => {
  http.get(`http://localhost:${PORT}/ping`, (res) => {
    res.resume();
  }).on("error", () => {});
}, 240000);
