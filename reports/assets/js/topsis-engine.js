/* Module 2: Candidate Filtering & MCDA-TOPSIS Engine (Single Responsibility: Domain Math & CSV) */
(function (global) {
  'use strict';

  const DEFAULT_CRITERIA = Object.freeze({
    maxDv: 6.5,
    maxDur: 450,
    maxOcc: 3,
    minObs: 50,
    minDiam: 25,
    grpFilter: 'ALL',
    w1: 0.40,
    w2: 0.25,
    w3: 0.20,
    w4: 0.15
  });

  function filterCandidates(records, criteria) {
    return records.filter(function (d) {
      if (d.dv > criteria.maxDv) return false;
      if (d.dur > criteria.maxDur) return false;
      if (d.occ > criteria.maxOcc) return false;
      if (d.obs < criteria.minObs) return false;
      if (d.diam < criteria.minDiam) return false;
      if (criteria.grpFilter === 'ALL') return true;
      if (criteria.grpFilter === 'KNOWN') return d.has_spec === 1;
      return String(d.grp || '').indexOf(criteria.grpFilter) === 0;
    });
  }

  function rankCandidatesTOPSIS(records, criteria) {
    const cand = filterCandidates(records, criteria);
    if (cand.length === 0) {
      return [];
    }

    let maxLogObs = 1.0;
    let maxLogArc = 1.0;
    for (let i = 0; i < cand.length; i += 1) {
      const lo = Math.log1p(cand[i].obs);
      const la = Math.log1p(cand[i].arc);
      if (lo > maxLogObs) maxLogObs = lo;
      if (la > maxLogArc) maxLogArc = la;
    }

    const feats = cand.map(function (d) {
      const boundedOcc = Math.min(9, Math.max(0, d.occ));
      const infoIdx = 100.0 * (
        0.35 * (Math.log1p(d.obs) / maxLogObs) +
        0.25 * (Math.log1p(d.arc) / maxLogArc) +
        0.25 * ((9.0 - boundedOcc) / 9.0) +
        0.10 * d.has_spec +
        0.05 * d.diam_m_real
      );
      const c1 = 1.0 / (d.dv * Math.sqrt(Math.max(d.dur, 1.0)));
      const c2 = infoIdx;
      const c3 = Math.log1p(d.diam);
      const c4 = Math.log1p(d.traj);
      return Object.assign({}, d, { infoIdx: infoIdx, c1: c1, c2: c2, c3: c3, c4: c4 });
    });

    function vectorNorm(key) {
      let sumSq = 0.0;
      for (let i = 0; i < feats.length; i += 1) {
        sumSq += feats[i][key] * feats[i][key];
      }
      return Math.sqrt(sumSq) || 1.0;
    }

    const n1 = vectorNorm('c1');
    const n2 = vectorNorm('c2');
    const n3 = vectorNorm('c3');
    const n4 = vectorNorm('c4');

    const wSum = (criteria.w1 + criteria.w2 + criteria.w3 + criteria.w4) || 1.0;
    const nw1 = criteria.w1 / wSum;
    const nw2 = criteria.w2 / wSum;
    const nw3 = criteria.w3 / wSum;
    const nw4 = criteria.w4 / wSum;

    let ib1 = -Infinity, iw1 = Infinity;
    let ib2 = -Infinity, iw2 = Infinity;
    let ib3 = -Infinity, iw3 = Infinity;
    let ib4 = -Infinity, iw4 = Infinity;

    for (let i = 0; i < feats.length; i += 1) {
      const r = feats[i];
      r.v1 = (r.c1 / n1) * nw1;
      r.v2 = (r.c2 / n2) * nw2;
      r.v3 = (r.c3 / n3) * nw3;
      r.v4 = (r.c4 / n4) * nw4;

      if (r.v1 > ib1) ib1 = r.v1;
      if (r.v1 < iw1) iw1 = r.v1;
      if (r.v2 > ib2) ib2 = r.v2;
      if (r.v2 < iw2) iw2 = r.v2;
      if (r.v3 > ib3) ib3 = r.v3;
      if (r.v3 < iw3) iw3 = r.v3;
      if (r.v4 > ib4) ib4 = r.v4;
      if (r.v4 < iw4) iw4 = r.v4;
    }

    for (let i = 0; i < feats.length; i += 1) {
      const r = feats[i];
      const db = Math.hypot(r.v1 - ib1, r.v2 - ib2, r.v3 - ib3, r.v4 - ib4);
      const dw = Math.hypot(r.v1 - iw1, r.v2 - iw2, r.v3 - iw3, r.v4 - iw4);
      r.topsis = (db + dw) > 0 ? (100.0 * dw / (db + dw)) : 50.0;
    }

    feats.sort(function (a, b) {
      return (b.topsis - a.topsis) || (a.dv - b.dv);
    });

    return feats;
  }

  function buildQualifiedCSV(ranked) {
    const header = [
      'rank', 'spkid', 'designacao', 'asteroide_oficial', 'espectro',
      'delta_v_kms', 'duracao_dias', 'diametro_ref_m', 'observacoes', 'occ', 'score_topsis'
    ];
    const rows = ranked.map(function (r, idx) {
      const safePdes = '"' + String(r.pdes).replace(/"/g, '""') + '"';
      const safeName = '"' + String(r.name).replace(/"/g, '""') + '"';
      const safeSpec = '"' + String(r.spec).replace(/"/g, '""') + '"';
      return [
        idx + 1, r.spkid, safePdes, safeName, safeSpec,
        r.dv, r.dur, r.diam, r.obs, r.occ, r.topsis.toFixed(2)
      ].join(',');
    });
    return '\uFEFF' + [header.join(',')].concat(rows).join('\n');
  }

  global.NewSpaceTopsisEngine = {
    DEFAULT_CRITERIA: DEFAULT_CRITERIA,
    filterCandidates: filterCandidates,
    rankCandidatesTOPSIS: rankCandidatesTOPSIS,
    buildQualifiedCSV: buildQualifiedCSV
  };
})(typeof window !== 'undefined' ? window : globalThis);
