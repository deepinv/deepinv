/* Renders the community showcases from the build-time data injected by
   _static/community/data/<kind>.data.js, which set:

     window.DI_COMMUNITY.scholar        = {updated, source_url, scholar: [...]}
     window.DI_COMMUNITY.contributors   = {updated, source_url, contributors: [...]}

   Reading a <script>-loaded global (rather than fetching JSON at page load)
   means the showcases also work when the built HTML is opened directly from
   disk (file://), where fetch() is blocked by the browser. This script runs on
   both index.html and community.html and only fills whichever containers are
   present on the page.
   Data is produced at build time by docs/source/_ext/community_data.py. */
(function () {
    "use strict";

    var SCHOLAR_INITIAL_ROWS = 10;

    function el(tag, cls, text) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text != null) e.textContent = text;
        return e;
    }

    function hideShowcase(node) {
        var s = node.closest(".di-showcase");
        if (s) s.style.display = "none";
    }

    /* ------------------------------------------------ Powering SotA research */

    function renderScholar(container, data) {
        var papers = (data && data.scholar) || [];
        container.textContent = "";
        if (!papers.length) {
            hideShowcase(container);
            return;
        }

        papers.forEach(function (p) {
            var item = el("div", "di-scholar-item");
            if (p.authors) item.appendChild(el("div", "di-scholar-authors", p.authors));

            var titleWrap = el("div");
            if (p.url) {
                var a = el("a", "di-scholar-title", p.title);
                a.href = p.url;
                a.target = "_blank";
                a.rel = "noopener noreferrer";
                titleWrap.appendChild(a);
            } else {
                titleWrap.appendChild(el("span", "di-scholar-title", p.title));
            }
            item.appendChild(titleWrap);

            if (p.venue || p.year) {
                var meta = el("div", "di-scholar-meta");
                if (p.venue) meta.appendChild(el("span", "di-scholar-venue", p.venue));
                if (p.venue && p.year) meta.appendChild(document.createTextNode(" · "));
                if (p.year) meta.appendChild(el("span", "di-scholar-year", p.year));
                item.appendChild(meta);
            }
            container.appendChild(item);
        });

        // Cap the visible height at the first N rows; the rest are reachable by
        // scrolling inside the list. A small peek hints there is more below.
        var rows = container.querySelectorAll(".di-scholar-item");
        if (rows.length > SCHOLAR_INITIAL_ROWS) {
            requestAnimationFrame(function () {
                var cutoff = rows[SCHOLAR_INITIAL_ROWS];
                if (cutoff) container.style.maxHeight = cutoff.offsetTop + 26 + "px";
            });
        } else {
            container.style.maxHeight = "none";
        }
    }

    /* ------------------------------------------------------------ Contributors */

    function renderContributors(container, data) {
        var people = (data && data.contributors) || [];
        container.textContent = "";
        if (!people.length) {
            hideShowcase(container);
            return;
        }

        people.forEach(function (c) {
            var card = el("a", "di-contrib");
            card.href = c.html_url || "https://github.com/" + c.login;
            card.target = "_blank";
            card.rel = "noopener noreferrer";
            if (c.location) card.title = c.location;

            var img = el("img", "di-contrib-avatar");
            img.loading = "lazy";
            img.alt = c.name || c.login;
            var av = c.avatar_url || "";
            img.src = av + (av.indexOf("?") === -1 ? "?" : "&") + "s=128";
            card.appendChild(img);

            card.appendChild(el("span", "di-contrib-name", c.name || c.login));
            if (c.location) card.appendChild(el("span", "di-contrib-loc", c.location));
            container.appendChild(card);
        });
    }

    function init() {
        var di = window.DI_COMMUNITY || {};

        var scholar = document.querySelector("[data-di-scholar]");
        if (scholar) {
            if (di.scholar) renderScholar(scholar, di.scholar);
            else hideShowcase(scholar);
        }

        var contrib = document.querySelector("[data-di-contributors]");
        if (contrib) {
            if (di.contributors) renderContributors(contrib, di.contributors);
            else hideShowcase(contrib);
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
