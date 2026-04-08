// ============================================================
// THEME TOGGLE
// ============================================================
(function () {
  const STORAGE_KEY = "ep-theme";
  const html = document.documentElement;

  function getStored() {
    return localStorage.getItem(STORAGE_KEY);
  }

  function applyTheme(theme) {
    if (theme === "dark") {
      html.classList.add("dark");
    } else {
      html.classList.remove("dark");
    }
  }

  function toggleTheme() {
    const current = html.classList.contains("dark") ? "dark" : "light";
    const next = current === "dark" ? "light" : "dark";
    localStorage.setItem(STORAGE_KEY, next);
    applyTheme(next);
    updateToggleLabel();
  }

  function updateToggleLabel() {
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    const isDark = html.classList.contains("dark");
    btn.textContent = isDark ? "[light]" : "[dark]";
    btn.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
  }

  // Apply on load (default: light; respect stored preference)
  const stored = getStored();
  if (stored) {
    applyTheme(stored);
  }
  // else: default is light (no class added)

  document.addEventListener("DOMContentLoaded", function () {
    updateToggleLabel();
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.addEventListener("click", toggleTheme);
  });
})();

// ============================================================
// MOBILE NAV HAMBURGER
// ============================================================
document.addEventListener("DOMContentLoaded", function () {
  const hamburger = document.getElementById("nav-hamburger");
  const navLinks = document.getElementById("nav-links");
  if (hamburger && navLinks) {
    hamburger.addEventListener("click", function () {
      const open = navLinks.classList.toggle("open");
      hamburger.setAttribute("aria-expanded", open);
    });
  }
});

// ============================================================
// ACTIVE NAV LINK
// ============================================================
document.addEventListener("DOMContentLoaded", function () {
  const path = window.location.pathname;
  document.querySelectorAll(".nav-links a").forEach(function (link) {
    const href = link.getAttribute("href");
    if (href === "/" && path === "/") {
      link.classList.add("active");
    } else if (href !== "/" && path.startsWith(href)) {
      link.classList.add("active");
    }
  });
});

// ============================================================
// PUBLICATION CARD VIDEO HOVER
// ============================================================
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".pub-card").forEach(function (card) {
    const video = card.querySelector("video");
    if (!video) return;

    card.addEventListener("mouseenter", function () {
      video.play().catch(() => {});
    });
    card.addEventListener("mouseleave", function () {
      video.pause();
      video.currentTime = 0;
    });
  });
});

// ============================================================
// BLOG TAG FILTER
// ============================================================
document.addEventListener("DOMContentLoaded", function () {
  const filterBtns = document.querySelectorAll("[data-tag-filter]");
  const blogItems = document.querySelectorAll("[data-tags]");
  if (!filterBtns.length) return;

  let activeTag = null;

  filterBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const tag = btn.dataset.tagFilter;

      if (activeTag === tag) {
        // deselect
        activeTag = null;
        filterBtns.forEach((b) => b.classList.remove("active"));
        blogItems.forEach((item) => item.classList.remove("hidden"));
        return;
      }

      activeTag = tag;
      filterBtns.forEach((b) =>
        b.classList.toggle("active", b.dataset.tagFilter === tag)
      );

      if (tag === "all") {
        activeTag = null;
        blogItems.forEach((item) => item.classList.remove("hidden"));
        filterBtns.forEach((b) => b.classList.remove("active"));
        return;
      }

      blogItems.forEach(function (item) {
        const tags = (item.dataset.tags || "").split(",");
        item.classList.toggle("hidden", !tags.includes(tag));
      });
    });
  });
});

// ============================================================
// BACKLOG TAG FILTER
// ============================================================
document.addEventListener("DOMContentLoaded", function () {
  const filterBtns = document.querySelectorAll("[data-backlog-filter]");
  const cards = document.querySelectorAll("[data-backlog-tags]");
  if (!filterBtns.length) return;

  let activeTag = null;

  filterBtns.forEach(function (btn) {
    btn.addEventListener("click", function () {
      const tag = btn.dataset.backlogFilter;

      if (activeTag === tag || tag === "all") {
        activeTag = null;
        filterBtns.forEach((b) => b.classList.remove("active"));
        cards.forEach((c) => c.classList.remove("hidden"));
        return;
      }

      activeTag = tag;
      filterBtns.forEach((b) =>
        b.classList.toggle("active", b.dataset.backlogFilter === tag)
      );

      cards.forEach(function (card) {
        const tags = (card.dataset.backlogTags || "").split(",");
        card.classList.toggle("hidden", !tags.includes(tag));
      });
    });
  });
});
