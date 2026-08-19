// DOM Elements
const predictedPriceEl = document.getElementById('predicted-price');
const contribBaseEl = document.getElementById('contrib-base');
const contribEngineEl = document.getElementById('contrib-engine');
const contribDimsEl = document.getElementById('contrib-dims');
const contribBrandEl = document.getElementById('contrib-brand');
const contribMpgEl = document.getElementById('contrib-mpg');
const contribOtherEl = document.getElementById('contrib-other');

let currentPrice = 0;

// Initialize Web App
window.addEventListener('DOMContentLoaded', () => {
    populateDropdowns();
    setInitialFormValues();
    calculatePrice();
    
    // Bind all inputs to trigger real-time calculation
    const formInputs = document.querySelectorAll('#prediction-form input, #prediction-form select');
    formInputs.forEach(input => {
        input.addEventListener('input', calculatePrice);
        input.addEventListener('change', calculatePrice);
    });
});

// Populate Select Elements Dynamically from modelData
function populateDropdowns() {
    if (typeof modelData === 'undefined') {
        console.error("modelData is not loaded. Cannot populate dropdowns.");
        return;
    }

    const categories = modelData.categories;
    const categoricalCols = modelData.categorical_cols;

    categoricalCols.forEach(col => {
        const selectEl = document.getElementById(col);
        if (selectEl) {
            selectEl.innerHTML = ''; // clear
            
            const cats = categories[col];
            cats.forEach(cat => {
                const option = document.createElement('option');
                option.value = cat;
                
                // Format text to look cleaner
                let text = cat;
                if (typeof cat === 'string') {
                    text = cat.replace(/-/g, ' ');
                    text = text.charAt(0).toUpperCase() + text.slice(1);
                }
                option.textContent = text;
                
                // Set default matching DEFAULTS in model_data
                if (cat === getDefaultValueFor(col)) {
                    option.selected = true;
                }
                
                selectEl.appendChild(option);
            });
        }
    });
}

function getDefaultValueFor(col) {
    const defaults = {
        brand: 'toyota',
        carbody: 'sedan',
        drivewheel: 'fwd',
        enginelocation: 'front',
        fueltype: 'gas',
        aspiration: 'std',
        doornumber: 'four',
        cylindernumber: 'four',
        enginetype: 'ohc',
        fuelsystem: 'mpfi'
    };
    return defaults[col] || '';
}

// Prefill form values
function setInitialFormValues() {
    const defaults = {
        horsepower: 95,
        enginesize: 120,
        peakrpm: 5100,
        curbweight: 2422,
        carlength: 173.2,
        carwidth: 65.5,
        carheight: 54.1,
        wheelbase: 97.0,
        compressionratio: 9.0,
        boreratio: 3.32,
        stroke: 3.29,
        citympg: 24,
        highwaympg: 30,
        symboling: 1
    };

    Object.keys(defaults).forEach(key => {
        const el = document.getElementById(key);
        if (el) {
            el.value = defaults[key];
            const displayEl = document.getElementById(`val-${key}`);
            if (displayEl) {
                displayEl.textContent = defaults[key];
            }
        }
    });
}

// Switch between Main Tabs (Predictor vs Evaluation)
function switchMainTab(tabId) {
    // Hide all main tab content
    const contents = document.querySelectorAll('.container > .tab-content');
    contents.forEach(content => content.classList.remove('active'));

    // Deactivate all main tab buttons
    const buttons = document.querySelectorAll('.container > .tabs-nav > .tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));

    // Activate selected
    document.getElementById(tabId).classList.add('active');
    
    // Find matching button and activate
    const activeBtn = Array.from(buttons).find(btn => {
        const onclickAttr = btn.getAttribute('onclick');
        return onclickAttr && onclickAttr.includes(tabId);
    });
    if (activeBtn) activeBtn.classList.add('active');
}

// Switch between Form Step Tabs
function switchFormTab(event, tabId) {
    event.preventDefault();
    
    // Get form container's tab contents
    const formPanel = event.currentTarget.closest('.glass-panel');
    const contents = formPanel.querySelectorAll('form > .tab-content');
    contents.forEach(content => content.classList.remove('active'));

    // Deactivate current tab buttons
    const buttons = formPanel.querySelectorAll('.tabs-nav > .tab-btn');
    buttons.forEach(btn => btn.classList.remove('active'));

    // Activate selected content and button
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

// Update value displays for sliders
function updateValDisplay(element) {
    const displayEl = document.getElementById(`val-${element.id}`);
    if (displayEl) {
        displayEl.textContent = element.value;
    }
}

// Sync Highway MPG (hidden) dynamically
function syncHighwayMpg(cityMpgValue) {
    const cityMpg = parseFloat(cityMpgValue);
    const highwayMpgEl = document.getElementById('highwaympg');
    if (highwayMpgEl) {
        highwayMpgEl.value = (cityMpg * 1.25).toFixed(1);
    }
}

// Core Prediction calculation
function calculatePrice() {
    if (typeof modelData === 'undefined') return;

    // Create a map of feature coefficients
    const coefMap = {};
    for (let i = 0; i < modelData.feature_names.length; i++) {
        coefMap[modelData.feature_names[i]] = modelData.coefficients[i];
    }

    // Gather input data from elements
    const input = {};
    
    // Read numerical values
    modelData.numerical_cols.forEach(col => {
        const el = document.getElementById(col);
        if (el) {
            input[col] = parseFloat(el.value) || 0;
        }
    });

    // Read categorical values
    modelData.categorical_cols.forEach(col => {
        const el = document.getElementById(col);
        if (el) {
            input[col] = el.value;
        }
    });

    // Calculate Contributions
    let price = modelData.intercept;
    
    let engineContrib = 0;
    let dimsContrib = 0;
    let brandContrib = 0;
    let mpgContrib = 0;
    let otherContrib = 0;

    // 1. Process Numerical Columns
    modelData.numerical_cols.forEach((col, idx) => {
        const val = input[col];
        const mean = modelData.means[idx];
        const scale = modelData.scales[idx];
        const scaledVal = (val - mean) / scale;
        
        const coef = coefMap[col] || 0;
        const contrib = coef * scaledVal;
        
        price += contrib;

        // Group into contribution cards
        if (['horsepower', 'enginesize', 'peakrpm'].includes(col)) {
            engineContrib += contrib;
        } else if (['curbweight', 'carlength', 'carwidth', 'carheight', 'wheelbase', 'compressionratio', 'boreratio', 'stroke'].includes(col)) {
            dimsContrib += contrib;
        } else if (['citympg', 'highwaympg'].includes(col)) {
            mpgContrib += contrib;
        } else {
            otherContrib += contrib; // symboling, etc.
        }
    });

    // 2. Process Categorical Columns
    modelData.categorical_cols.forEach(col => {
        const val = input[col];
        const featureName = `${col}_${val}`;
        
        const coef = coefMap[featureName] || 0; // if dropped category, coef is 0
        price += coef;

        // Group into contribution cards
        if (col === 'brand') {
            brandContrib += coef;
        } else if (['fueltype', 'aspiration', 'cylindernumber', 'enginetype', 'fuelsystem'].includes(col)) {
            engineContrib += coef;
        } else if (['carbody', 'drivewheel', 'enginelocation', 'doornumber'].includes(col)) {
            dimsContrib += coef;
        } else {
            otherContrib += coef;
        }
    });

    // Enforce reasonable boundaries (a car price cannot drop below $3000 in this logic)
    if (price < 3000) price = 3000;

    // Render results
    animatePriceDisplay(price);
    renderContributions({
        base: modelData.intercept,
        engine: engineContrib,
        dims: dimsContrib,
        brand: brandContrib,
        mpg: mpgContrib,
        other: otherContrib
    });
}

// Smoothly animate the price digits count-up/count-down
function animatePriceDisplay(targetVal) {
    const startVal = currentPrice;
    currentPrice = targetVal;

    const duration = 250; // ms
    const startTime = performance.now();

    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function (easeOutQuad)
        const easeProgress = progress * (2 - progress);
        
        const currentVal = startVal + (targetVal - startVal) * easeProgress;
        predictedPriceEl.textContent = formatCurrency(currentVal);
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        } else {
            predictedPriceEl.textContent = formatCurrency(targetVal);
        }
    }
    requestAnimationFrame(updateCounter);
}

// Render formatted values into Contributions panel
function renderContributions(contribs) {
    contribBaseEl.textContent = formatCurrency(contribs.base);
    
    updateContribElement(contribEngineEl, contribs.engine);
    updateContribElement(contribDimsEl, contribs.dims);
    updateContribElement(contribBrandEl, contribs.brand);
    updateContribElement(contribMpgEl, contribs.mpg);
    updateContribElement(contribOtherEl, contribs.other);
}

function updateContribElement(element, value) {
    element.textContent = (value >= 0 ? '+' : '-') + ' ' + formatCurrency(Math.abs(value));
    element.className = 'contrib-value ' + (value > 50 ? 'positive' : (value < -50 ? 'negative' : 'neutral'));
}

function formatCurrency(val) {
    return '$' + val.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}
