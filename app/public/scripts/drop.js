// event.preventDefault() voorkomt dat welke browser iets standaards doet.
// FromData() kan simpele gegevens, maar ook complexe bestanden inpakken
// new maakt een nieuwe instance, ipv een constructor functie als functie aan te roepen

const dropzone = document.getElementById("dropzone");
console.log("hello world");

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
    event.preventDefault(); // voorkomt dat de browser het bestand opent ipv opslaat.
    dropzone.classList.remove("dragover");
    const files = event.dataTransfer.files;
    console.log(files);
    const formData = new FormData();

    for (let file of files) {
        formData.append("files", file);
    }

    fetch("/upload", {
        method: "POST",
        body: formData
    }).then(response => {
        if (response.ok) {
            alert("Bestand succesvol verstuurd naar de server");
        } else {
            alert("Bestand onsuccesvol verstuurd naar de server");
        }
    }).catch((error) => {
        console.error("Fout: ", error);
    });
})