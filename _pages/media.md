---
layout: page
title: Movies & Books
permalink: /media/
---

<div class="media-tabs" role="tablist">
  <button type="button" class="sticker tactile media-tab" data-tab="movies" role="tab" aria-selected="true">Movies</button>
  <button type="button" class="sticker tactile media-tab" data-tab="books" role="tab" aria-selected="false">Books</button>
</div>

<div class="media-panel" data-panel="movies">
  <p class="page-subtitle">Recently watched, synced from <a href="https://letterboxd.com/damiavr/" target="_blank" rel="noopener">Letterboxd</a>.</p>
  <div class="media-grid">
    {% for movie in site.data.movies %}
    <a href="{{ movie.link }}" target="_blank" rel="noopener" class="media-card">
      {% if movie.poster %}
      <img src="{{ movie.poster }}" alt="" class="media-card__cover" loading="lazy" />
      {% endif %}
      <div class="media-card__body">
        <h3 class="media-card__title">{{ movie.title }}{% if movie.year %} <span class="media-card__year">({{ movie.year }})</span>{% endif %}</h3>
        {% if movie.rating %}{% include star-rating.html rating=movie.rating %}{% endif %}
        {% if movie.watched_date %}<p class="media-card__meta">Watched {{ movie.watched_date | date: "%-d %B %Y" }}</p>{% endif %}
      </div>
    </a>
    {% endfor %}
  </div>
</div>

<div class="media-panel" data-panel="books" hidden>
  <div class="media-grid">
    {% for book in site.data.books %}
    <div class="media-card">
      {% if book.cover %}
      <img src="{{ book.cover }}" alt="" class="media-card__cover" loading="lazy" />
      {% endif %}
      <div class="media-card__body">
        <h3 class="media-card__title">{{ book.title }}</h3>
        {% if book.author %}<p class="media-card__year">{{ book.author }}</p>{% endif %}
        {% if book.rating %}{% include star-rating.html rating=book.rating %}{% endif %}
        {% if book.date_read %}<p class="media-card__meta">Read {{ book.date_read | date: "%-d %B %Y" }}</p>{% endif %}
        {% if book.comment %}<p class="media-card__comment">{{ book.comment }}</p>{% endif %}
      </div>
    </div>
    {% endfor %}
  </div>
</div>

<script>
  (function () {
    var tabs = document.querySelectorAll('.media-tab');
    var panels = document.querySelectorAll('.media-panel');

    function activate(name) {
      tabs.forEach(function (tab) {
        var active = tab.dataset.tab === name;
        tab.classList.toggle('sticker--tomato', active);
        tab.setAttribute('aria-selected', active);
      });
      panels.forEach(function (panel) {
        panel.hidden = panel.dataset.panel !== name;
      });
    }

    tabs.forEach(function (tab) {
      tab.addEventListener('click', function () {
        activate(tab.dataset.tab);
        history.replaceState(null, '', '#' + tab.dataset.tab);
      });
    });

    var initial = location.hash.replace('#', '');
    activate(initial === 'books' ? 'books' : 'movies');
  })();
</script>
