const BLOCK_SITES = ["leetcode.com", "github.com"];

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ blockedSites: BLOCK_SITES });
});

// Listen for changes in Chrome storage to update the block status
chrome.storage.sync.get(["hasShoweredToday", "username"], (data) => {
  if (data.hasShoweredToday === false) {
    updateBlockRules(BLOCK_SITES);
  } else {
    stopBlockingSites();
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "stopBlocking") {
    stopBlockingSites();
  }
});

// Update the blocking rules dynamically based on shower status
function updateBlockRules(sites) {
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

// Stop blocking websites by removing the dynamic rules
function stopBlockingSites() {
  chrome.declarativeNetRequest.updateDynamicRules({
    removeRuleIds: Array.from({ length: 1000 }, (_, i) => i + 1), // Clear all blocking rules
    addRules: [],
  });
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    chrome.storage.sync.get(["hasShoweredToday", "username"], (data) => {
      if (data.username) {
        console.log(`Checking shower status for: ${data.username}`);
        if (data.hasShoweredToday === false) {
          // Block sites
          updateBlockRules(BLOCK_SITES);
        } else {
          // Stop blocking sites
          stopBlockingSites();
        }
      } else {
        // If no username, stop blocking
        stopBlockingSites();
      }
    });
  }
});
