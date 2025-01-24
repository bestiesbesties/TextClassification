const fs = require("fs");
const path = require("path");
const child_process = require("child_process");

const express = require("express"); // voor routing
const multer = require("multer"); // voor storage
const { stdout } = require("process");

const uploadDir = path.join(__dirname, "uploads")

if (!fs.existsSync(uploadDir)) {
    console.log("Created uploads folder in: ", __dirname)
    fs.mkdirSync(uploadDir, { recursive: true})
}

// MULTER
const storage = multer.diskStorage({
    destination : (req, file, callback) => {
        callback(null, uploadDir);
    },
    filename : (req, file, callback) => {
        callback(null, "upload-" + file.originalname);
    }
});
const upload = multer({storage});

const app = express();

app.use("/app", express.static(path.join(__dirname, 'app')));

app.get("/", (request, response) => {
    fs.readFile("app/public/home.html", "utf8", (err, data) => {

        if (err) {
            response.status(500).send("Error reading html")
        }
        response.send(data); 
    });
});

app.post("/upload_run", upload.array("files"), (request, response) => {
    console.log("Uploading file");
    console.log("Client request:", request)
    const uploadedFilepath = request.files[0].path
    const embeddingModelName = request.body.embedding_model_name

    const useFaiss = "False"
    console.log("Executing procces on arguments:", uploadedFilepath, embeddingModelName, useFaiss)
    const python = child_process.exec("python3 main.py "+ uploadedFilepath + " " + embeddingModelName + " " + useFaiss, (error, stdout, stderr) => {
        if (error) {
            console.log("error: ", error)
            return response.status(500)
        }
        if (stderr) {
            console.log("stderr: ", stderr)
            return response.status(500)
        }
        
        console.log("stdout: ", stdout)
        const cleanedJson = JSON.parse(stdout.trim())
        console.log("cleanedJson: ", cleanedJson)
        
        response.status(200).json(cleanedJson)
    });
});

const PORT = 4500;
app.listen(PORT, () => {
    console.log("Server running on http://localhost:" + PORT)
});