document
  .getElementById("telegram-form")
  .addEventListener("submit", async function (event) {
    event.preventDefault();

    const username = document.getElementById("telegram-username").value;

    try {
      const response = await fetch(
        `http://127.0.0.1:8000/check_shower/${username}`
      );
      const data = await response.json();

      if (response.ok) {
        // Store the result in Chrome storage
        chrome.storage.sync.set(
          { username: username, hasShoweredToday: data.has_showered_today },
          () => {
            console.log(
              "Stored in Chrome storage:",
              username,
              data.has_showered_today
            );
          }
        );

        // Display the result
        if (data.has_showered_today === true) {
          document.getElementById(
            "output"
          ).innerText = `User ${username} has showered today. Refresh this page if it is currently blocked.`;
        } else {
          document.getElementById(
            "output"
          ).innerText = `User ${username} has NOT showered today. Coding pages will be blocked.`;
        }

        document.getElementById("remove-button").style.display = "block";
      } else {
        document.getElementById("output").innerText = `Error: ${data.detail}`;
      }
    } catch (err) {
      console.error("Error:", err);
      document.getElementById("output").innerText = `Error: ${err.message}`;
    }
  });

// Handle removal of username
document.getElementById("remove-button").addEventListener("click", function () {
  chrome.storage.sync.remove("username", () => {
    console.log("Username removed from Chrome storage");
    document.getElementById("output").innerText = "Username removed.";
    document.getElementById("remove-button").style.display = "none";
    chrome.runtime.sendMessage({ action: "stopBlocking" });
  });
});
