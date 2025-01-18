chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ blockedSites: [] });
});

chrome.storage.onChanged.addListener((changes, namespace) => {
  if (changes.blockedSites) {
    const sites = changes.blockedSites.newValue || [];
    const rules = sites.map((site, index) => ({
      id: index + 1,
      priority: 1,
      action: { type: "block" },
      condition: { urlFilter: site, resourceTypes: ["main_frame"] },
    }));

    chrome.declarativeNetRequest.updateDynamicRules({
      removeRuleIds: Array.from({ length: 1000 }, (_, i) => i + 1), // Clear all existing rules
      addRules: rules,
    });
  }
});
