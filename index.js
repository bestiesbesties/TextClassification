const fs = require("fs");
const path = require("path");
const child_process = require("child_process");
const express = require("express"); // Routing
const multer = require("multer"); // Storage
const WebSocket = require('ws'); // Realtime coms
const crypto = require('crypto')

const PORTEXPRESS = 4500;
const PORTWS = 4501;

//*WEBSOCKET
const sendWebSocketServerMessageToAll = (type, message, activeWebSocketSessions) => {
    console.log("TOALL: ", message);
    activeWebSocketSessions.forEach((socket, sessionID) => {
        const body = {
            "type" : type,
            "message" : message
        };
        socket.send(JSON.stringify(body));
    });
};

const webSocketServer = new WebSocket.Server({ port: PORTWS });
const activeWebSocketSessions = new Map();

webSocketServer.on("connection", (socket, request) => {

    console.log(socket)
    const sessionID = crypto.randomUUID()
    activeWebSocketSessions.set(sessionID, socket)
    sendWebSocketServerMessageToAll("terminal", `SERVER: Started WebSocket Session on ID: ${sessionID}`, activeWebSocketSessions)

    socket.on("close", (socket, reason) => {
        sendWebSocketServerMessageToAll("terminal", `SERVER: Closing session ${sessionID}`, activeWebSocketSessions)
        activeWebSocketSessions.delete(sessionID)
    })

})

//Server folder structure
const uploadDir = "./uploads"
if (!fs.existsSync(uploadDir)) {
    console.log("Created uploads folder in: ", __dirname)
    fs.mkdirSync(uploadDir, { recursive: true})
}

//*MULTER
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

app.use("/app", express.static("./app"));

app.get("/", (request, response) => {
    fs.readFile("app/public/home.html", "utf8", (err, data) => {

        if (err) {
            response.status(500).send("Error reading html")
        }
        response.send(data); 
    });
});

app.post("/upload_run", upload.array("files"), (request, response) => {

    sendWebSocketServerMessageToAll("terminal", `SERVER: Storing recieved request.files:  ${JSON.stringify(request.files)}`, activeWebSocketSessions)
    const uploadedFilepath = request.files[0].path
    const embeddingModelName = request.body.embedding_model_name

    const useFaiss = "True"

    sendWebSocketServerMessageToAll("terminal", `SERVER: Executing Python process on arguments: ${uploadedFilepath} ${embeddingModelName} ${useFaiss}`, activeWebSocketSessions)
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
        console.log(`cleanedJson: ${JSON.stringify(cleanedJson)}`)
        
        fs.unlink(request.files[0].path, (error) => {
            if (error) {
                sendWebSocketServerMessageToAll("terminal", "SERVER: Could not delete upload", activeWebSocketSessions);
            } else {
                sendWebSocketServerMessageToAll("terminal", "SERVER: Deleted upload on disk", activeWebSocketSessions);
            }
        });       

        response.status(200).json(cleanedJson)
    });

});

app.listen(PORTEXPRESS, () => {
    console.log("Server running on http://localhost:" + PORTEXPRESS)
});

