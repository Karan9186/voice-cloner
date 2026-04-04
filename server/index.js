const express = require("express");
const axios = require("axios");
const fs = require("fs");
const path = require("path");

const app = express();
app.use(express.json());

app.post("/voice", async (req, res) => {
  try {
    const { text } = req.body;

    const response = await axios({
      url: "http://localhost:5001/generate",
      method: "POST",
      data: { text },
      responseType: "stream"
    });

    const filePath = path.join(__dirname, "../outputs/output.wav");
    const writer = fs.createWriteStream(filePath);

    response.data.pipe(writer);

    writer.on("finish", () => {
      res.download(filePath);
    });

  } catch (err) {
    console.error(err);
    res.status(500).send("Error generating voice");
  }
});

app.listen(3000, () => {
  console.log("Node server running on port 3000");
});