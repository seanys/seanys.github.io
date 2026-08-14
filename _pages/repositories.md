---
layout: page
permalink: /repositories/
title: Repositories
nav: true
nav_order: 4
_styles: |
  .post-header {
    display: none;
  }
---

{% if site.data.repositories.github_users %}

<div class="repositories repository-profile-row d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-stretch">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.liquid username=user %}
    {% include repository/repo_stats.liquid username=user %}
  {% endfor %}
</div>
{% endif %}

---

{% if site.data.repositories.github_repos %}

## GitHub Repositories

<div class="repositories repository-list-row d-flex flex-wrap flex-md-row flex-column justify-content-between align-items-center">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.liquid repository=repo %}
  {% endfor %}
</div>
{% endif %}
