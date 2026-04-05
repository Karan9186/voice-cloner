const fs = require("fs");
const path = require("path");
const axios = require("axios");
const FormData = require("form-data");

async function main() {
  const text = process.argv[2] || "This is a streaming XTTS clone test from the Node proxy.";
  const audioPath = process.argv[3];
  const outputPath = process.argv[4] || path.join(process.cwd(), "test-output.wav");
  const targetUrl = process.env.NODE_PROXY_URL || "http://127.0.0.1:3000/generate";

  if (!audioPath) {
    throw new Error('Usage: node test-client.js "text" path/to/reference.wav [output.wav]');
  }

  const form = new FormData();
  form.append("text", text);
  form.append("audio", fs.createReadStream(audioPath));

  const response = await axios.post(targetUrl, form, {
    headers: form.getHeaders(),
    responseType: "stream",
    timeout: 0,
  });

  await new Promise((resolve, reject) => {
    const output = fs.createWriteStream(outputPath);
    response.data.pipe(output);
    response.data.on("error", reject);
    output.on("finish", resolve);
    output.on("error", reject);
  });

  console.log(`Saved streamed audio to ${outputPath}`);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
