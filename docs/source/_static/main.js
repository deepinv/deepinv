// Minimal dependency-free sortable tables for the benchmark pages.
// Click a header cell to sort; toggling direction on repeated clicks.
// Cells formatted like "26.68 ± 1.47" sort by their leading number, and
// non-numeric cells (e.g. "nan ± nan") always sink to the bottom.
document.addEventListener("DOMContentLoaded", function () {
    function text(row, i) {
        var cell = row.children[i];
        return cell ? cell.textContent.trim() : "";
    }

    function number(value) {
        var match = value.replace(/,/g, "").match(/-?\d+(\.\d+)?/);
        return match ? parseFloat(match[0]) : NaN;
    }

    function comparer(i, asc) {
        return function (a, b) {
            var v1 = text(a, i),
                v2 = text(b, i);
            var n1 = number(v1),
                n2 = number(v2);
            // Keep NaN cells at the bottom regardless of sort direction.
            if (isNaN(n1) !== isNaN(n2)) return isNaN(n1) ? 1 : -1;
            var result =
                !isNaN(n1) && !isNaN(n2) ? n1 - n2 : v1.localeCompare(v2);
            return asc ? result : -result;
        };
    }

    document.querySelectorAll("table.sortable-table").forEach(function (table) {
        var headers = table.querySelectorAll("thead th");
        var tbody = table.querySelector("tbody");
        if (!tbody) return;

        headers.forEach(function (th, i) {
            th.addEventListener("click", function () {
                var asc = th.dataset.sorted !== "asc";
                var rows = Array.prototype.slice.call(tbody.rows);
                rows.sort(comparer(i, asc));
                rows.forEach(function (row) {
                    tbody.appendChild(row);
                });
                headers.forEach(function (h) {
                    delete h.dataset.sorted;
                });
                th.dataset.sorted = asc ? "asc" : "desc";
            });
        });
    });
});
