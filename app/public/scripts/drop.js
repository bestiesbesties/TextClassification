// event.preventDefault() voorkomt dat welke browser iets standaards doet.
// FromData() kan simpele gegevens, maar ook complexe bestanden inpakken
// new maakt een nieuwe instance, ipv een constructor functie als functie aan te roepen

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

dropzone.addEventListener("drop", (event) => {
    console.log("Handeling drop event")
    event.preventDefault(); // voorkomt dat de browser het bestand opent ipv opslaat.
    dropzone.classList.remove("dragover");

    console.log("Handeling drop event")
    const files = event.dataTransfer.files;
    const formData = new FormData();
    formData.append("files", files[0]);

    console.log("Posting to server")
    fetch("/upload_run", {
        method: "POST",
        body: formData
    }).then(response => {
        if (response.ok) {
            console.log("Bestand succesvol verstuurd naar de server");
            alert("Bestand succesvol verstuurd naar de server");
        } else {
            console.log("Bestand onsuccesvol verstuurd naar de server");
            alert("Bestand succesvol verstuurd naar de server");
        }
    }).catch((error) => {
        console.error("Fout: ", error);
    });
})