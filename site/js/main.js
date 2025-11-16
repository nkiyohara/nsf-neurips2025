// Expand / Collapse all
(function () {
  const controls = document.querySelectorAll('[data-toggle-details]');
  if (!controls.length) return;

  const setAll = open => {
    document.querySelectorAll('#results details').forEach(d => { d.open = open; });
  };

  controls.forEach(btn => {
    btn.addEventListener('click', () => {
      const shouldOpen = (btn.getAttribute('data-toggle-details') || '').toLowerCase() === 'open';
      setAll(shouldOpen);
    }, { passive: true });
  });
})();

// Deep-link support: open detail if hash matches
(function () {
  const id = location.hash.replace('#', '');
  if (!id) return;
  const d = document.getElementById(id);
  if (d && d.tagName.toLowerCase() === 'details') { d.open = true; d.scrollIntoView({ block: 'start' }); }
})();

// Keep video posters in sync with the active color theme so the first frame shows before playback.
(function () {
  const videos = Array.from(document.querySelectorAll('video[data-poster-light]'));
  if (!videos.length) return;

  const prefersDarkQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  const pickPoster = (video, theme) => {
    if (theme === 'dark') {
      return video.dataset.posterDark || video.dataset.posterLight || '';
    }
    return video.dataset.posterLight || video.dataset.posterDark || '';
  };
  const applyPosters = () => {
    const explicitTheme = document.documentElement.getAttribute('data-theme');
    const prefersDark = prefersDarkQuery ? prefersDarkQuery.matches : false;
    const theme = explicitTheme === 'dark' || (!explicitTheme && prefersDark) ? 'dark' : 'light';
    videos.forEach(video => {
      const poster = pickPoster(video, theme);
      if (poster) video.setAttribute('poster', poster);
    });
  };

  applyPosters();

  if (prefersDarkQuery) {
    const handler = () => applyPosters();
    if (typeof prefersDarkQuery.addEventListener === 'function') {
      prefersDarkQuery.addEventListener('change', handler);
    } else if (typeof prefersDarkQuery.addListener === 'function') {
      prefersDarkQuery.addListener(handler);
    }
  }

  if ('MutationObserver' in window) {
    const observer = new MutationObserver(applyPosters);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  }
})();

// Sync image grid caption heights
(function () {
  const grids = document.querySelectorAll('.image-grid');
  if (!grids.length) return;

  const sync = () => {
    grids.forEach(grid => {
      const rows = new Map();
      grid.querySelectorAll('figure').forEach(fig => {
        const caption = fig.querySelector('figcaption');
        if (!caption) return;
        caption.style.minHeight = '';
        const key = fig.offsetTop;
        if (!rows.has(key)) rows.set(key, []);
        rows.get(key).push(caption);
      });
      rows.forEach(captions => {
        const maxHeight = Math.max(...captions.map(cap => cap.offsetHeight));
        captions.forEach(cap => cap.style.minHeight = `${maxHeight}px`);
      });
    });
  };

  window.addEventListener('load', sync, { passive: true });
  window.addEventListener('resize', sync, { passive: true });
  grids.forEach(grid => grid.querySelectorAll('img').forEach(img => {
    if (!img.complete) img.addEventListener('load', sync, { passive: true, once: true });
  }));
  sync();
})();

// Copy to clipboard functionality
(function () {
  const buttons = document.querySelectorAll('[data-copy-target]');
  buttons.forEach(btn => {
    const selector = btn.getAttribute('data-copy-target');
    if (!selector) return;
    const target = document.querySelector(selector);
    if (!target) return;

    const originalLabel = (btn.getAttribute('aria-label') || 'Copy').trim();
    const originalTitle = btn.getAttribute('title') || originalLabel;
    btn.setAttribute('aria-label', originalLabel);
    btn.setAttribute('title', originalTitle);
    const feedbackEl = btn.querySelector('.copy-feedback');

    let resetTimer = null;
    const showFeedback = (message, isError = false) => {
      if (resetTimer) window.clearTimeout(resetTimer);
      if (message) {
        btn.setAttribute('aria-label', message);
        btn.setAttribute('title', message);
        if (feedbackEl) {
          feedbackEl.textContent = message;
          feedbackEl.classList.toggle('is-error', Boolean(isError));
          feedbackEl.classList.add('is-visible');
        }
        resetTimer = window.setTimeout(() => {
          btn.setAttribute('aria-label', originalLabel);
          btn.setAttribute('title', originalTitle);
          if (feedbackEl) {
            feedbackEl.textContent = '';
            feedbackEl.classList.remove('is-visible');
            feedbackEl.classList.remove('is-error');
          }
        }, 1800);
      } else {
        btn.setAttribute('aria-label', originalLabel);
        btn.setAttribute('title', originalTitle);
        if (feedbackEl) {
          feedbackEl.textContent = '';
          feedbackEl.classList.remove('is-visible');
          feedbackEl.classList.remove('is-error');
        }
      }
    };

    btn.addEventListener('click', async () => {
      const text = target.textContent;
      if (!text) return;

      let copied = false;
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        try {
          await navigator.clipboard.writeText(text);
          copied = true;
        } catch (_err) {
          copied = false;
        }
      }

      if (!copied) {
        try {
          const textarea = document.createElement('textarea');
          textarea.value = text;
          textarea.setAttribute('readonly', '');
          textarea.style.position = 'fixed';
          textarea.style.opacity = '0';
          document.body.appendChild(textarea);
          textarea.focus();
          textarea.select();
          textarea.setSelectionRange(0, textarea.value.length);
          copied = document.execCommand ? document.execCommand('copy') : false;
          document.body.removeChild(textarea);
        } catch (_err) {
          copied = false;
        }
      }

      if (copied) {
        showFeedback('Copied!');
      } else {
        showFeedback('Copy failed', true);
      }
    });
  });
})();

// Image lightbox for figures
(function () {
  const figures = document.querySelectorAll('.image-grid figure, figure.figure-one');
  const lightbox = document.getElementById('figure-lightbox');
  if (!figures.length || !lightbox) return;

  const lightboxImg = lightbox.querySelector('img');
  const lightboxCaption = lightbox.querySelector('figcaption');
  const dismissControls = lightbox.querySelectorAll('[data-lightbox-dismiss]');

  let activeTrigger = null;

  const closeLightbox = () => {
    if (!lightbox.classList.contains('is-active')) return;
    lightbox.classList.remove('is-active');
    document.body.classList.remove('lightbox-open');
    lightbox.setAttribute('aria-hidden', 'true');
    if (lightboxImg) {
      lightboxImg.removeAttribute('src');
      lightboxImg.removeAttribute('alt');
      lightboxImg.setAttribute('aria-hidden', 'true');
      lightboxImg.setAttribute('role', 'presentation');
    }
    if (lightboxCaption) {
      lightboxCaption.textContent = '';
      lightboxCaption.style.display = '';
    }
    if (activeTrigger) {
      activeTrigger.focus({ preventScroll: true });
      activeTrigger = null;
    }
  };

  const openLightbox = fig => {
    const img = fig.querySelector('img');
    if (!img || !lightboxImg) return;
    const caption = fig.querySelector('figcaption');
    const captionText = caption ? caption.textContent.trim() : '';
    const src = img.currentSrc || img.src;
    if (src) lightboxImg.src = src;
    lightboxImg.alt = img.alt || captionText || 'Expanded figure';
    lightboxImg.removeAttribute('aria-hidden');
    lightboxImg.removeAttribute('role');

    if (lightboxCaption) {
      if (captionText) {
        lightboxCaption.textContent = captionText;
        lightboxCaption.style.display = '';
      } else {
        lightboxCaption.textContent = '';
        lightboxCaption.style.display = 'none';
      }
    }

    activeTrigger = fig;
    lightbox.classList.add('is-active');
    lightbox.setAttribute('aria-hidden', 'false');
    document.body.classList.add('lightbox-open');
    const closeButton = lightbox.querySelector('.lightbox__close');
    if (closeButton) closeButton.focus({ preventScroll: true });
  };

  figures.forEach(fig => {
    fig.classList.add('is-lightbox-trigger');
    if (!fig.hasAttribute('tabindex')) fig.setAttribute('tabindex', '0');
    if (!fig.hasAttribute('role')) fig.setAttribute('role', 'button');
    const labelSource = fig.querySelector('figcaption') || fig.querySelector('img');
    if (labelSource) {
      const labelText = labelSource.getAttribute('alt') || labelSource.textContent;
      if (labelText) fig.setAttribute('aria-label', `Expand figure: ${labelText}`);
    } else {
      fig.setAttribute('aria-label', 'Expand figure');
    }

    fig.addEventListener('click', event => {
      event.preventDefault();
      openLightbox(fig);
    });

    fig.addEventListener('keydown', event => {
      if (event.defaultPrevented) return;
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        openLightbox(fig);
      }
    });
  });

  dismissControls.forEach(control => {
    control.addEventListener('click', event => {
      event.preventDefault();
      closeLightbox();
    });
  });

  lightbox.addEventListener('click', event => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape') {
      closeLightbox();
    }
  });
})();
