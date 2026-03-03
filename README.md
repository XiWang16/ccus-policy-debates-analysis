A site that scrapes and republishes information on Canada's House of Commons.

License
=======

Code is released under the AGPLv3 (see below). However, any site you create
using this code cannot use the openparliament.ca name or logo, except as
acknowledgement.

Copyright (C) Michael Mulley (michaelmulley.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

Usage
============

This is the source code for a specific site and isn't adapted for reuse or
other purposes. But if it's useful to you, that's great!

The application runs on a recent version of Python (3.12 as of this writing).
Your best bet to get an environment running is probably Docker. A sample
docker-compose.yml is provided in the config-examples directory; copy it into the
same directory as this file and run `docker compose up`.

The app uses the Django framework, which can work with a variety of databases
(including sqlite if you don't want to run a database server), but it's only been
tested with PostgreSQL. You can download a dump of all our Parliamentary data to
load into Postgres from <https://openparliament.ca/data-download/>.

CCUS analysis module
====================

In addition to the core openparliament.ca site, this repository now includes
a research-focused analysis module under `parliament/ccus_analysis/`, added
by Xi Wang to study debates around carbon capture, utilization, and storage
(CCUS) policy in the House of Commons.

The `ccus_analysis` package currently provides:

* A multi-step pipeline (`CCUSAnalysisPipeline`) that:
  - resolves a manually curated list of CCUS-relevant bills against the
    OpenParliament API,
  - fetches Hansard speeches for each matched bill and session,
  - identifies CCUS-relevant speeches and paragraphs using keyword- and
    embedding-based matching, and
  - extracts political actors and classifies their stance and arguments using
    an opinion classifier.
* Structured outputs in `parliament/ccus_analysis/output/`:
  - JSON dumps (`ccus_*.json`) with bills, speeches, actor-level opinions,
    jurisdictions, and a high-level summary; and
  - CSV tables (`ccus_*.csv`) that can be used directly for statistical
    analysis and visualization.
* Downstream statistical scripts, including:
  - `argument_frames_by_party.py`, which loads argument-level data and runs
    a chi-squared test of argument frames by party, writing
    `argument_frames_by_party.csv`; and
  - `speech_level_stance.py`, which propagates actor-level opinions down to
    individual speeches and tests party × stance distributions, writing
    `speech_level_stances.csv`.
* A visualization step (`step5_vis.py`) that materializes interactive HTML
  dashboards (e.g. stance breakdowns, actor and bill overviews) in
  `parliament/ccus_analysis/output/vis/`.

The CCUS module is designed so that future work can:

* swap in different keyword providers or policy domains (e.g. other climate
  or energy topics) while reusing the same pipeline structure;
* extend the analysis with additional LLM-based tasks (for example, finer-
  grained argument typologies or rhetorical strategies); and
* integrate new visualizations or export formats without changing the core
  site or API.
