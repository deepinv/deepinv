// Unfold the user guide tab in navigation when loading the index.html
window.addEventListener('load', (event) => {
    if (!document.URL.includes("index.html")) {
        return
    }
    var menu = document.querySelector(".wy-menu ul li:first-child")
    if (!menu.classList.contains("current")) {
        menu.classList.add("current")
    }
});