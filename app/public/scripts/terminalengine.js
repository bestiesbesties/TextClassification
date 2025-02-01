export function sendToTerminal(message) {
    console.log(message);
    const activeTerminal = document.getElementById("activeTerminal");

    const row = document.createElement("div");
    row.className = "realtime-text";
    row.id = crypto.randomUUID();
    row.textContent = ">>> "+ message;

    activeTerminal.appendChild(row);
    activeTerminal.scrollTop = activeTerminal.scrollHeight;
}