// import { sendToTerminal } from "./app/public/scripts/terminalengine.js";
// import fs from 'fs';
// import express from 'express'; // Routing
// import multer from 'multer'; // Storage
// import WebSocket from "ws"; // Realtime coms
// import path from 'path';
// import { exec, spawn } from 'child_process';
// import { stdout } from 'process';

const fs = require("fs");
const path = require("path");
const child_process = require("child_process");
const express = require("express"); // Routing
const multer = require("multer"); // Storage
const WebSocket = require('ws'); // Realtime coms
const crypto = require('crypto')

//*WEBSOCKET
const sendWebSocketServerMessageToAll = (type, message, activeWebSocketSessions) => {
    activeWebSocketSessions.forEach((socket, sessionID) => {
        console.log(message);
        const body = {
            "type" : type,
            "message" : message
        };
        socket.send(JSON.stringify(body)    );
        // socket.send(JSON.stringify(body));
    });
};

const webSocketServer = new WebSocket.Server({ port: 4501 });
const activeWebSocketSessions = new Map();

webSocketServer.on("connection", (socket, request) => {

    const sessionID = crypto.randomUUID()
    activeWebSocketSessions.set(sessionID, socket)
    console.log("TEMPLog", activeWebSocketSessions)
    sendWebSocketServerMessageToAll("terminal", `SERVER: Started WebSocket Session on ID: ${sessionID}`, activeWebSocketSessions)

    socket.on("close", (socket, reason) => {
        console.log(`SERVER: Session ${sessionID}`)
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

    sendWebSocketServerMessageToAll("terminal", "SERVER: Uploading file.", activeWebSocketSessions)
    const uploadedFilepath = request.files[0].path
    const embeddingModelName = request.body.embedding_model_name

    const useFaiss = "False"

    sendWebSocketServerMessageToAll("terminal", `SERVER: Executing process on arguments: ${uploadedFilepath} ${embeddingModelName} ${useFaiss}`, activeWebSocketSessions)
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

