// event.preventDefault() voorkomt dat welke browser iets standaards doet.
// FromData() kan simpele gegevens, maar ook complexe bestanden inpakken
// new maakt een nieuwe instance, ipv een constructor functie als functie aan te roepen

import { sendToTerminal } from "./terminalengine.js";

const dropzone = document.getElementById("dropzone");

dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover")
});

dropzone.addEventListener("dragenter", (event) => {
    event.preventDefault();
});

dropzone.addEventListener("drop", async (event) => {
    sendToTerminal("CLIENT: Handeling drop event");
    event.preventDefault(); // voorkomt dat de browser het bestand opent ipv opslaat.
    event.stopPropagation(); // voorkomt dat andere eventlisterers getriggerd worden
    dropzone.classList.remove("dragover");

    document.getElementById("spinner").style.display = "block";
    document.getElementById("body").classList.add("darken");

    sendToTerminal("CLIENT: Placing data in form structure");
    const embeddingModelName = document.getElementsByClassName("model-button active")[0].textContent

    const files = event.dataTransfer.files;
    sendToTerminal(`CLIENT: Amount of files recieved in drop: ${files.length}`)

    console.log("event", event)
    console.log("event.dataTransfer.files:", event.dataTransfer.files)
    const formData = new FormData();
    formData.append("files", files[0]);
    formData.append("embedding_model_name", embeddingModelName);
    try {
        sendToTerminal("CLIENT: Posting to server");
        const response = await fetch("/upload_run", {
            method: "POST",
            body: formData
        });
        
        if (response.ok) {
            const result = await response.json();
            sendToTerminal(`CLIENT: Recieved: ", ${JSON.stringify(result)}`);
            document.getElementById("economy-value").innerText = result.economy + "%";
            document.getElementById("health-value").innerText = result.health + "%";
            document.getElementById("tech-value").innerText = result.tech + "%";
            document.getElementById("agriculture-value").innerText = result.agriculture + "%";

            document.getElementById("construction-value").innerText = result.construction + "%";
            document.getElementById("education-value").innerText = result.education + "%";
            document.getElementById("legal-value").innerText = result.legal + "%";
            document.getElementById("retail-value").innerText = result.retail + "%";
        } else {
            sendToTerminal("CLIENT: Error on serverside");
        }

    } catch (error) {
        document.getElementById("spinner").style.display = "block";
        document.getElementById("body").classList.add("darken")
        sendToTerminal(`CLIENT: An error occured: ${error}`)
    }
    document.getElementById("spinner").style.display = "none";
    document.getElementById("body").classList.remove("darken")
});