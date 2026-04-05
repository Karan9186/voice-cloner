const express = require("express");
const axios = require("axios");
const multer = require("multer");
const FormData = require("form-data");

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 15 * 1024 * 1024,
    files: 1,
  },
});

const PYTHON_API_URL = process.env.PYTHON_API_URL || "http://127.0.0.1:5001";
const PORT = Number(process.env.PORT || 3000);

app.use(express.json({ limit: "1mb" }));

app.post("/clone", upload.single("audio"), async (req, res) => {
  if (!req.body.text || !req.body.text.trim()) {
    return res.status(400).json({ status: "error", message: "Field 'text' is required." });
  }

  if (!req.file) {
    return res.status(400).json({ status: "error", message: "Field 'audio' is required." });
  }

  const form = new FormData();
  form.append("text", req.body.text.trim());
  form.append("audio", req.file.buffer, {
    filename: req.file.originalname || "reference.wav",
    contentType: req.file.mimetype || "audio/wav",
  });

  for (const fieldName of ["language", "temperature", "speed", "length_penalty", "repetition_penalty", "trim_silence"]) {
    if (req.body[fieldName] !== undefined) {
      form.append(fieldName, String(req.body[fieldName]));
    }
  }

  try {
    const upstream = await axios.post(`${PYTHON_API_URL}/clone`, form, {
      headers: form.getHeaders(),
      responseType: "stream",
      maxBodyLength: Infinity,
      maxContentLength: Infinity,
      timeout: 0,
      validateStatus: () => true,
    });

    res.status(upstream.status);
    for (const [header, value] of Object.entries(upstream.headers)) {
      if (header.toLowerCase() === "transfer-encoding") {
        continue;
      }
      res.setHeader(header, value);
    }

    upstream.data.on("error", (streamError) => {
      console.error("Python stream error:", streamError.message);
      if (!res.headersSent) {
        res.status(502).json({ status: "error", message: "Upstream audio stream failed." });
      } else {
        res.destroy(streamError);
      }
    });

    req.on("aborted", () => upstream.data.destroy());
    upstream.data.pipe(res);
  } catch (error) {
    console.error("Proxy clone error:", error.message);
    if (!res.headersSent) {
      res.status(502).json({ status: "error", message: "Failed to reach Python TTS service." });
    }
  }
});

app.all("/health", async (_req, res) => {
  try {
    const response = await axios.post(`${PYTHON_API_URL}/health`, {}, { timeout: 5000 });
    res.status(response.status).json(response.data);
  } catch (error) {
    console.error("Proxy health error:", error.message);
    res.status(502).json({ status: "error", message: "Python health check failed." });
  }
});

app.listen(PORT, () => {
  console.log(`Node streaming proxy running on port ${PORT}`);
});
