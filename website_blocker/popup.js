document.addEventListener("DOMContentLoaded", () => {
  const siteInput = document.getElementById("site-input");
  const addSiteButton = document.getElementById("add-site");
  const blockedSitesList = document.getElementById("blocked-sites");

  function updateUI(sites) {
    blockedSitesList.innerHTML = "";
    sites.forEach((site) => {
      const li = document.createElement("li");
      li.textContent = site;
      const removeButton = document.createElement("button");
      removeButton.textContent = "Remove";
      removeButton.onclick = () => removeSite(site);
      li.appendChild(removeButton);
      blockedSitesList.appendChild(li);
    });
  }

  function removeSite(site) {
    chrome.storage.sync.get({ blockedSites: [] }, function (data) {
      const updatedSites = data.blockedSites.filter((s) => s !== site);
      chrome.storage.sync.set({ blockedSites: updatedSites });
    });
  }

  addSiteButton.addEventListener("click", () => {
    const site = siteInput.value.trim();
    if (site) {
      chrome.storage.sync.get({ blockedSites: [] }, function (data) {
        if (!data.blockedSites.includes(site)) {
          const updatedSites = [...data.blockedSites, site];
          chrome.storage.sync.set({ blockedSites: updatedSites });
        }
      });
      siteInput.value = "";
    }
  });

  chrome.storage.sync.get({ blockedSites: [] }, function (data) {
    updateUI(data.blockedSites);
  });
});
