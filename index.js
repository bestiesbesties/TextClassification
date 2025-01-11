const fs = require("fs")
const path = require("path");
const express = require("express");

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

    let PORT = 4500
    app.listen(PORT, () => {
        console.log("Server running on http://localhost:" + PORT)
    });