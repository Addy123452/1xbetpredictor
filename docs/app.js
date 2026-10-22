/* Static client-side predictor: reproduces the trained multinomial logistic
 * regression in JavaScript (standardize -> logits -> softmax). */
(function () {
  'use strict';

  var DATA = null;

  function el(id) { return document.getElementById(id); }

  function esc(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function fillSelect(select, teams, preferred) {
    var chosen = teams.indexOf(preferred) >= 0 ? preferred : teams[0];
    select.innerHTML = teams.map(function (team) {
      return '<option value="' + esc(team) + '"' +
        (team === chosen ? ' selected' : '') + '>' + esc(team) + '</option>';
    }).join('');
  }

  function buildRow(home, away, neutral) {
    var hs = DATA.teams[home];
    var as = DATA.teams[away];
    return {
      home_elo: hs.elo,
      away_elo: as.elo,
      elo_diff: hs.elo - as.elo,
      home_form_ppg: hs.form_ppg,
      away_form_ppg: as.form_ppg,
      form_diff: hs.form_ppg - as.form_ppg,
      home_form_gd: hs.form_gd,
      away_form_gd: as.form_gd,
      gd_diff: hs.form_gd - as.form_gd,
      home_home_ppg: hs.home_ppg,
      away_away_ppg: as.away_ppg,
      opp_strength_home: as.elo,
      opp_strength_away: hs.elo,
      neutral: neutral ? 1 : 0,
    };
  }

  function predict(home, away, neutral) {
    var m = DATA.model;
    var row = buildRow(home, away, neutral);
    var scaled = m.feature_columns.map(function (column, i) {
      return (row[column] - m.scaler_mean[i]) / m.scaler_scale[i];
    });

    var logits = m.coef.map(function (weights, k) {
      var total = m.intercept[k];
      for (var i = 0; i < weights.length; i += 1) {
        total += weights[i] * scaled[i];
      }
      return total;
    });

    var max = Math.max.apply(null, logits);
    var exps = logits.map(function (value) { return Math.exp(value - max); });
    var sum = exps.reduce(function (a, b) { return a + b; }, 0);

    var probs = {};
    m.classes.forEach(function (label, i) { probs[label] = exps[i] / sum; });
    return probs;
  }

  function pct(value) { return (value * 100).toFixed(1) + '%'; }

  function render() {
    var home = el('homeTeam').value;
    var away = el('awayTeam').value;
    var neutral = el('neutralVenue').checked;
    var result = el('result');

    if (!home || !away) { return; }
    if (home === away) {
      result.innerHTML = '<p class="warn">Pick two different teams.</p>';
      return;
    }

    var p = predict(home, away, neutral);
    var winner = 'H';
    if (p.D > p[winner]) { winner = 'D'; }
    if (p.A > p[winner]) { winner = 'A'; }

    function row(key, label) {
      return '<div class="prob-row' + (key === winner ? ' top' : '') + '">' +
        '<span class="prob-label">' + label + '</span>' +
        '<span class="prob-value">' + pct(p[key]) + '</span>' +
        '<span class="bar"><span class="fill" style="width:' + pct(p[key]) + '"></span></span>' +
        '</div>';
    }

    var hs = DATA.teams[home];
    var as = DATA.teams[away];
    var winnerName = winner === 'H' ? home : (winner === 'A' ? away : 'Draw');

    result.innerHTML =
      '<div class="match-title">' + esc(home) + ' <span>vs</span> ' + esc(away) +
        (neutral ? ' <em>(neutral)</em>' : '') + '</div>' +
      '<div class="probs">' + row('H', 'Home win') + row('D', 'Draw') + row('A', 'Away win') + '</div>' +
      '<p class="most">Most likely: <b>' + esc(winnerName) + '</b> · confidence ' + pct(p[winner]) + '</p>' +
      '<div class="facts">' +
        '<span>Elo ' + hs.elo.toFixed(0) + ' vs ' + as.elo.toFixed(0) + '</span>' +
        '<span>Form ' + hs.form_ppg.toFixed(2) + ' vs ' + as.form_ppg.toFixed(2) + '</span>' +
        '<span>Elo gap ' + (hs.elo - as.elo).toFixed(0) + '</span>' +
      '</div>';
  }

  function init() {
    fetch('predictor_data.json')
      .then(function (res) {
        if (!res.ok) { throw new Error('HTTP ' + res.status); }
        return res.json();
      })
      .then(function (data) {
        DATA = data;
        var teams = Object.keys(data.teams).sort(function (a, b) { return a.localeCompare(b); });
        fillSelect(el('homeTeam'), teams, 'Brazil');
        fillSelect(el('awayTeam'), teams, 'Argentina');
        el('predictBtn').addEventListener('click', render);
        el('homeTeam').addEventListener('change', render);
        el('awayTeam').addEventListener('change', render);
        el('neutralVenue').addEventListener('change', render);
        render();
      })
      .catch(function (err) {
        el('result').innerHTML = '<p class="warn">Could not load predictor data (' +
          esc(err.message) + '). Run <code>python -m src.training.export_web</code> first.</p>';
      });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
