'use strict';
const $ = s => document.querySelector(s);
let batch = null;
let currentPage = 1;
const PAGE_SIZE = 50;

function el(tag, text, cls) { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; if (cls) node.className = cls; return node; }
function status(text, error = false) { const node = $('#status'); node.textContent = text; node.hidden = !text; node.classList.toggle('error', error); node.setAttribute('role', error ? 'alert' : 'status'); }
function route(path) {
  const page = ({'/':'home','/analyze':'analyze','/bulk':'bulk','/dashboard':'dashboard','/about':'about'})[path] || 'home';
  document.querySelectorAll('[data-page]').forEach(node => { node.hidden = node.dataset.page !== page; });
  document.querySelectorAll('nav a').forEach(node => { node.classList.toggle('active', node.pathname === path); if (node.pathname === path) node.setAttribute('aria-current','page'); else node.removeAttribute('aria-current'); });
  if (page === 'about') loadModel();
  window.scrollTo(0, 0);
}
document.addEventListener('click', event => {
  const link = event.target.closest('a');
  if (link && link.origin === location.origin && ['/','/analyze','/bulk','/dashboard','/about'].includes(link.pathname) && !event.ctrlKey && !event.metaKey) {
    event.preventDefault(); history.pushState({}, '', link.pathname); route(link.pathname);
  }
});
window.addEventListener('popstate', () => route(location.pathname));
$('#review').addEventListener('input', () => { $('#count').textContent = `${$('#review').value.length.toLocaleString()} / 5,000`; });

async function submit(form, url, options) {
  const button = form.querySelector('[type=submit]'); const original = button.textContent;
  button.disabled = true; button.textContent = 'Analyzing…'; form.setAttribute('aria-busy','true'); status('Analyzing the review history…');
  try {
    const response = await fetch(url, {...options, signal: AbortSignal.timeout(180000)});
    let data; try { data = await response.json(); } catch { throw new Error('The server is starting or unavailable. Please retry shortly.'); }
    if (!response.ok) throw new Error(data.error || 'Could not analyze the reviews.');
    batch = data; currentPage = 1; renderResults(); history.pushState({}, '', '/dashboard'); route('/dashboard'); status('');
  } catch (error) {
    status(error.name === 'TimeoutError' ? 'Analysis timed out. Try a smaller CSV.' : error.message, true);
  } finally { button.disabled = false; button.textContent = original; form.removeAttribute('aria-busy'); }
}

$('#single-form').addEventListener('submit', event => {
  event.preventDefault();
  submit(event.currentTarget, '/api/analyze', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({review:$('#review').value, product:$('#product').value})});
});
$('#bulk-form').addEventListener('submit', event => {
  event.preventDefault(); const file = $('#file').files[0];
  if (!file || !file.name.toLowerCase().endsWith('.csv')) return status('Choose a .csv file.', true);
  if (file.size > 10 * 1024 * 1024) return status('The CSV must be no larger than 10 MB.', true);
  const body = new FormData(); body.append('file', file);
  submit(event.currentTarget, '/api/bulk', {method:'POST', body});
});

function renderPage() {
  const total = batch.results.length; const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  currentPage = Math.min(Math.max(1, currentPage), pages);
  const start = (currentPage - 1) * PAGE_SIZE; const shown = batch.results.slice(start, start + PAGE_SIZE);
  $('#cards').replaceChildren();
  for (const result of shown) {
    const card = el('article', undefined, 'panel result-card');
    const head = el('div', undefined, 'result-head');
    head.append(el('strong', `${String(result.row).padStart(4,'0')} / ${result.product}`), el('span', result.label, `pill ${result.unusual ? 'warning' : ''}`));
    const reviewWhy = el('div', undefined, 'review-why-grid');
    const reviewColumn = el('section', undefined, 'review-column');
    reviewColumn.append(el('span', 'ANALYZED REVIEW', 'reason-kicker'), el('p', result.review, 'review-text'));
    const whyColumn = el('aside', undefined, `why-panel ${result.unusual ? 'warning' : ''}`);
    whyColumn.append(el('span', 'WHY THIS RESULT?', 'reason-kicker'), el('p', result.explanation || 'The model compares review patterns with its training baseline.', 'why-summary'));
    const contextSignals = result.context_signals || [];
    const signalList = el('ul', undefined, 'why-signals');
    for (const signal of contextSignals) signalList.append(el('li', signal));
    whyColumn.append(signalList, el('p', 'These are context clues from the supplied data, not proof that a review is fake or genuine.', 'why-note'));
    reviewWhy.append(reviewColumn, whyColumn);
    card.append(head, reviewWhy, el('p', `Anomaly score: ${result.score > 0 ? '+' : ''}${result.score.toFixed(4)} · Higher means more unusual against the training baseline; not a probability.`, 'score'));
    const details = el('details'); details.append(el('summary','Inspect supplied history and text signals'));
    const meta = [`Rating: ${result.rating ?? 'not supplied'}`, `Verified purchase: ${result.verified_purchase === null ? 'not supplied' : result.verified_purchase ? 'yes' : 'no'}`, `Helpful votes: ${result.helpful_votes ?? 'not supplied'}`, `Date: ${result.date ?? 'not supplied'}`];
    if (result.reviewer_repeats !== null) meta.push(`This reviewer appears ${result.reviewer_repeats} time(s) in this upload`);
    if (result.product_day_count !== null) meta.push(`${result.product_day_count} review(s) for this product share this date in the upload`);
    details.append(el('p', meta.join(' · ')));
    const indicators = result.indicators;
    details.append(el('p', `Repeated phrases (3+ occurrences): ${indicators.repeated_phrases.join('; ') || 'None found'}.`));
    details.append(el('p', `Promotional phrase matches: ${indicators.promotional_phrases.join('; ') || 'None found'}. Exclamation marks: ${indicators.exclamation_marks}. These are context signals, not proof.`));
    details.append(el('p', result.nearest_row ? `Closest wording in this upload: row ${result.nearest_row}; similarity ${result.similarity.toFixed(3)}. Similarity does not establish deception.` : 'Similarity needs at least two reviews in the upload.'));
    card.append(details); $('#cards').append(card);
  }
  $('#page-status').textContent = `Showing ${start + 1}–${Math.min(start + shown.length, total)} of ${total.toLocaleString()} reviews · page ${currentPage} of ${pages}`;
  $('#previous-page').disabled = currentPage <= 1; $('#next-page').disabled = currentPage >= pages;
}

function renderResults() {
  const summary = batch.summary; $('#empty').hidden = true; $('#results').hidden = false; $('#export').disabled = false;
  $('#total').textContent = summary.total.toLocaleString(); $('#flagged').textContent = summary.unusual.toLocaleString();
  $('#share').textContent = `${(100 * summary.unusual / summary.total).toFixed(1)}%`;
  $('#groups').replaceChildren();
  for (const group of summary.products) {
    const row = el('div', undefined, 'product-row');
    row.append(el('strong', group.product));
    const bar = el('progress'); bar.max = group.total; bar.value = group.unusual; bar.setAttribute('aria-label', `${group.product}: ${group.unusual} unusual patterns out of ${group.total}`);
    const context = [`${group.unusual}/${group.total} unusual · ${(group.proportion*100).toFixed(1)}%`];
    if (group.average_rating !== null) context.push(`avg rating ${group.average_rating}/5`);
    if (group.verified_share !== null) context.push(`${(group.verified_share*100).toFixed(0)}% verified`);
    row.append(bar, el('span', context.join(' · '))); $('#groups').append(row);
  }
  $('#result-pages').hidden = summary.total <= PAGE_SIZE; renderPage();
}
$('#previous-page').addEventListener('click', () => { currentPage--; renderPage(); window.scrollTo(0, $('#cards').offsetTop); });
$('#next-page').addEventListener('click', () => { currentPage++; renderPage(); window.scrollTo(0, $('#cards').offsetTop); });

function csvCell(value) { let text = String(value ?? ''); if (/^[\s]*[=+\-@\t\r]/.test(text)) text = "'" + text; return '"' + text.replaceAll('"','""') + '"'; }
$('#export').addEventListener('click', () => {
  if (!batch) return;
  const fields = ['row','product','review','pattern_label','why_this_result','context_signals','anomaly_score_not_probability','rating','date','verified_purchase','helpful_votes','reviewer_repeats_in_upload','product_day_review_count','nearest_row','text_similarity'];
  const rows = batch.results.map(r => [r.row,r.product,r.review,r.label,r.explanation,(r.context_signals || []).join(' | '),r.score,r.rating,r.date,r.verified_purchase,r.helpful_votes,r.reviewer_repeats,r.product_day_count,r.nearest_row,r.similarity]);
  const csv = '\ufeff' + [fields,...rows].map(row => row.map(csvCell).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], {type:'text/csv;charset=utf-8'})); const link = el('a'); link.href = url; link.download = 'reviewlens-pattern-results.csv'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('#clear').addEventListener('click', () => {
  batch = null; currentPage = 1; $('#cards').replaceChildren(); $('#groups').replaceChildren(); $('#results').hidden = true; $('#empty').hidden = false; $('#export').disabled = true;
  $('#review').value = ''; $('#product').value = ''; $('#file').value = ''; $('#review').dispatchEvent(new Event('input'));
});

let modelLoaded = false;
async function loadModel() {
  if (modelLoaded) return;
  try {
    const response = await fetch('/api/model'); const info = await response.json();
    if (!response.ok || !info.training_samples) throw new Error('Model details unavailable.');
    $('#eval-info').textContent = `${info.source_dataset}; trained on ${info.training_samples.toLocaleString()} sampled reviews after reading ${info.source_rows_seen.toLocaleString()} eligible rows. ${info.label_note}`;
    const row = el('tr'); row.append(el('td', info.algorithm), el('td', info.training_samples.toLocaleString()), el('td', 'Unlabeled source data'));
    $('#metrics').replaceChildren(row); $('#matrix').textContent = info.threshold_note; modelLoaded = true;
  } catch (error) { $('#eval-info').textContent = error.message; }
}
route(location.pathname);
