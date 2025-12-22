$(document).ready(function () {
    // Check if the table exists and isn't already initialized
    if ($('.sortable-table').length > 0) {
        $('.sortable-table').DataTable({
            "paging": false,    // Often better for benchmark lists
            "info": false,      // Hides the "Showing 1 of X entries"
            "searching": true,  // Adds a search box
            "ordering": true,   // Explicitly enable ordering
            "order": [],        // Don't apply a default sort on load
        });
    }
});