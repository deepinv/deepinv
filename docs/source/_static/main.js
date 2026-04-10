// Custom DataTables type for "mean ± std" formatted cells.
// Detects strings like "26.68 ± 1.47" and sorts numerically by the mean value.

// Sphinx wraps list-table cells in <p> tags, so DataTables receives HTML like
// "<p>26.68 ± 1.47</p>". We strip tags and decode common HTML entities before
// matching, so that the type detection and sort order work on the plain text.
function _stripHtml(str) {
    return str
        .replace(/<[^>]+>/g, '')          // remove all HTML tags
        .replace(/&plusmn;/gi, '\u00B1')  // &plusmn; -> ±
        .replace(/&amp;/gi,   '&')
        .replace(/&lt;/gi,    '<')
        .replace(/&gt;/gi,    '>')
        .replace(/&nbsp;/gi,  ' ')
        .trim();
}

$.fn.dataTable.ext.type.detect.unshift(function (data) {
    if (typeof data === 'string') {
        var text = _stripHtml(data);
        if (/^[-+]?\d+(\.\d+)?\s*\u00B1\s*\d+(\.\d+)?$/.test(text)) {
            return 'mean-std';
        }
    }
    return null;
});

$.fn.dataTable.ext.type.order['mean-std-pre'] = function (data) {
    if (typeof data === 'string') {
        var text = _stripHtml(data);
        var match = text.match(/^([-+]?\d+(?:\.\d+)?)/);
        if (match) {
            return parseFloat(match[1]);
        }
    }
    return 0;
};

$(document).ready(function () {
    if ($('.sortable-table').length > 0) {
        $('.sortable-table').each(function () {
            var psnrColIndex = -1;
            $(this).find('thead th').each(function (i) {
                if ($(this).text().trim().toUpperCase().indexOf('PSNR') !== -1) {
                    psnrColIndex = i;
                }
            });

            $(this).DataTable({
                "paging": false,    // Often better for benchmark lists
                "info": false,      // Hides the "Showing 1 of X entries"
                "searching": true,  // Adds a search box
                "ordering": true,   // Explicitly enable ordering
                // Sort by PSNR descending if present, otherwise no default sort
                "order": psnrColIndex >= 0 ? [[psnrColIndex, 'desc']] : [],
            });
        });
    }
});