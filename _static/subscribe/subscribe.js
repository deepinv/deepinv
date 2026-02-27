function validateEmail() {
    const email = document.getElementById("emailInput").value.trim();
    const button = document.getElementById("subscribeBtn");

    if (email.length > 0 && email.includes("@")) {
        button.disabled = false;
        button.classList.add("enabled");
    } else {
        button.disabled = true;
        button.classList.remove("enabled");
    }
}

async function submitAndRedirect() {
    const email = document.getElementById("emailInput").value.trim();
    const encodedEmail = encodeURIComponent(email);

    try {
        const sheetFormURL = "https://docs.google.com/forms/d/e/1FAIpQLScAmkOxqMA8Kr_xSxVkglr6snJYk7z_F_sjhUFz3hxxPF1qeg/formResponse";
        const entryID = "entry.372491860";

        await fetch(sheetFormURL, {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: `${entryID}=${encodedEmail}`
        });
    } catch (err) {
        console.warn(err);
    }

    const surveyForm = "https://docs.google.com/forms/d/e/1FAIpQLScYdfkqB4fp_cEElILQuRjk8FZ3mPjJlYccNTVYIjgJ_F2e6g/viewform?usp=pp_url&entry.512719014=";
    window.open(surveyForm + encodedEmail, "_blank");
}


