window.addEventListener('load', (event) => {
    if (!document.URL.includes("index.html")) {
        return
    }
    var menu = document.querySelector(".wy-menu ul li:first-child")
    if (!menu.classList.contains("current")) {
        menu.classList.add("current")
    }
});