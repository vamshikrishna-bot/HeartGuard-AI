const form = document.querySelector('#assessment-form');
const result = document.querySelector('#result');
const age = document.querySelector('#age');
const ageValue = document.querySelector('#age-value');
const maxRate = document.querySelector('#max-rate');
const maxRateValue = document.querySelector('#max-rate-value');
const button = form.querySelector('button');

function updateRange(input, output, suffix) {
  output.textContent = `${input.value} ${suffix}`;
}

age.addEventListener('input', () => updateRange(age, ageValue, 'years'));
maxRate.addEventListener('input', () => updateRange(maxRate, maxRateValue, 'bpm'));

function showResult(prediction) {
  const elevated = prediction === 1;
  result.hidden = false;
  result.className = `result ${elevated ? 'high' : 'low'}`;
  result.innerHTML = `
    <div class="result-symbol">${elevated ? '!' : '✓'}</div>
    <div><span class="eyebrow">Assessment signal</span>
      <h2>${elevated ? 'Elevated risk signal detected' : 'Lower risk signal detected'}</h2>
      <p>${elevated ? 'The model indicates a higher screening risk from the information provided. Please arrange a proper evaluation with a qualified doctor.' : 'The model does not indicate an elevated screening risk from the information provided. Continue healthy habits and regular checkups.'}</p>
      <small>This result is not a diagnosis and should not replace professional medical advice.</small>
    </div>`;
  result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  button.querySelector('span').textContent = 'Analyzing...';
  result.hidden = true;

  const data = Object.fromEntries(new FormData(form).entries());
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || 'Unable to complete the assessment.');
    showResult(payload.prediction);
  } catch (error) {
    result.hidden = false;
    result.className = 'result error';
    result.innerHTML = `<div class="result-symbol">!</div><div><h2>Assessment unavailable</h2><p>${error.message}</p></div>`;
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Analyze heart risk';
  }
});
