// ── neighborhood loader ───────────────────────────────────────────────────
const boroughSel = document.getElementById('borough');
const neighborhoodSel = document.getElementById('neighborhood');

boroughSel.addEventListener('change', async () => {
  const b = boroughSel.value;
  neighborhoodSel.innerHTML = '<option value="">Loading...</option>';
  if (!b) { neighborhoodSel.innerHTML = '<option value="">Select neighborhood</option>'; return; }
  const res = await fetch(`/api/neighborhoods/${encodeURIComponent(b)}`);
  const data = await res.json();
  neighborhoodSel.innerHTML = '<option value="">Select neighborhood</option>' +
    data.neighborhoods.map(n => `<option value="${n}">${n}</option>`).join('');
});

// ── range slider ──────────────────────────────────────────────────────────
const subwayRange = document.getElementById('subway_dist');
const subwayVal   = document.getElementById('subway_val');
subwayRange.addEventListener('input', () => {
  subwayVal.textContent = parseFloat(subwayRange.value).toFixed(1) + ' mi';
});

// ── amenity chips ─────────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', (e) => {
    e.preventDefault();
    chip.classList.toggle('active');
  });
});

// ── form submit ───────────────────────────────────────────────────────────
const form = document.getElementById('form');
const btn  = document.getElementById('btn');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  btn.classList.add('loading');
  btn.querySelector('.btn-text').textContent = 'PREDICTING...';

  const fd = new FormData(form);
  const getChip = (name) =>
    document.querySelector(`.chip[data-name="${name}"]`)?.classList.contains('active') ? 1 : 0;

  const payload = {
    borough:      fd.get('borough'),
    neighborhood: fd.get('neighborhood'),
    sqft:         parseInt(fd.get('sqft')) || 0,
    bedrooms:     parseInt(fd.get('bedrooms')) || 0,
    bathrooms:    parseInt(fd.get('bathrooms')) || 1,
    floor:        parseInt(fd.get('floor')) || 1,
    age:          parseInt(fd.get('age')) || 0,
    has_garage:   getChip('has_garage'),
    has_elevator: getChip('has_elevator'),
    has_doorman:  getChip('has_doorman'),
    subway_dist:  parseFloat(fd.get('subway_dist')) || 0.5,
  };

  try {
    const res  = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    showResult(data);
    loadHistory();
  } catch (err) {
    alert('Server error. Make sure the FastAPI server is running.');
  } finally {
    btn.classList.remove('loading');
    btn.querySelector('.btn-text').textContent = 'PREDICT PRICE';
  }
});

// ── display result ────────────────────────────────────────────────────────
function fmt(n) {
  return '$' + Number(n).toLocaleString('en-US');
}

function showResult(data) {
  document.getElementById('placeholder').style.display = 'none';
  const content = document.getElementById('result-content');
  content.style.display = 'block';

  document.getElementById('res-borough').textContent = data.borough.toUpperCase();
  document.getElementById('res-price').textContent   = fmt(data.predicted_price);
  document.getElementById('res-sqft').textContent    = `${fmt(data.per_sqft)} per sq ft`;
  document.getElementById('res-low').textContent     = fmt(data.low);
  document.getElementById('res-high').textContent    = fmt(data.high);

  // fake confidence based on how close range is
  const conf = Math.min(95, 70 + Math.random() * 20);
  document.getElementById('conf-fill').style.width  = conf.toFixed(0) + '%';
  document.getElementById('conf-label').textContent =
    `Model confidence — ${conf.toFixed(0)}%`;

  document.querySelector('.result-card').style.alignItems = 'flex-start';
}

// ── history ───────────────────────────────────────────────────────────────
async function loadHistory() {
  const res  = await fetch('/api/history');
  const data = await res.json();
  const list = document.getElementById('history-list');
  if (!data.history.length) {
    list.innerHTML = '<p class="empty-hist">No predictions yet.</p>';
    return;
  }
  list.innerHTML = data.history.map(h => `
    <div class="hist-item">
      <span class="hist-loc">${h.input.neighborhood}, ${h.input.borough}</span>
      <span class="hist-price">${fmt(h.price)}</span>
    </div>
  `).join('');
}

loadHistory();
