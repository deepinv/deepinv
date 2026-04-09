// Custom DataTables type for "mean ± std" formatted cells.
// Detects strings like "26.68 ± 1.47" and sorts numerically by the mean value.
$.fn.dataTable.ext.type.detect.unshift(function (data) {
    if (typeof data === 'string' && /^[-+]?\d+(\.\d+)?\s*\u00B1\s*\d+(\.\d+)?$/.test(data.trim())) {
        return 'mean-std';
    }
    return null;
});

$.fn.dataTable.ext.type.order['mean-std-pre'] = function (data) {
    if (typeof data === 'string') {
        var match = data.match(/^([-+]?\d+(?:\.\d+)?)/);
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