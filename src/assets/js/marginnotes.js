/**
 * Moves .marginnote elements out of the post body and into the sidebar column,
 * positioned at the same vertical offset as where they appeared in the text.
 * Only runs on wide viewports (mirrors the CSS 860px breakpoint).
 */
(function () {
  function run() {
    if (window.innerWidth <= 860) return;

    var notes = Array.prototype.slice.call(
      document.querySelectorAll('.post-body .marginnote')
    );
    if (!notes.length) return;

    var layout = document.querySelector('.post-layout');
    if (!layout) return;

    var layoutTop = layout.getBoundingClientRect().top;

    // Measure all vertical offsets BEFORE any DOM changes.
    var tops = notes.map(function (note) {
      return Math.max(0, note.getBoundingClientRect().top - layoutTop);
    });

    // Ensure the sidebar column exists.
    var sidebar = document.querySelector('.post-sidenotes');
    if (!sidebar) {
      sidebar = document.createElement('aside');
      sidebar.className = 'post-sidenotes';
      sidebar.setAttribute('aria-label', 'Margin notes');
      // Place explicitly in grid column 2, row 1.
      sidebar.style.gridColumn = '2';
      sidebar.style.gridRow = '1';
      layout.appendChild(sidebar);
    }
    sidebar.style.position = 'relative';

    // Detach each note from the post body and re-attach in the sidebar.
    notes.forEach(function (note, i) {
      note.parentNode.removeChild(note);
      note.style.position = 'absolute';
      note.style.top = tops[i] + 'px';
      note.style.width = '100%';
      note.style.float = 'none';
      note.style.marginLeft = '0';
      sidebar.appendChild(note);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
  } else {
    run();
  }
})();
