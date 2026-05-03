// ── WASM Initialization ──────────────────────────────────────────────────────────
// Loaded via <script src="wordlebrain_wasm.js"> which sets window.wasm_bindgen

const {
    evaluate,
    solve_full,
    solve_step,
    get_hint,
    random_word,
    validate_word,
    word_count
} = wasm_bindgen;

let wasmInitialized = false;

async function initWasm() {
    try {
        await wasm_bindgen.default();
        const count = word_count();
        console.log('WordleBrain: Loaded ' + count + ' words');
        wasmInitialized = true;
        enableUI();
    } catch (err) {
        console.error('Failed to load WASM:', err);
        showMessage('play-message', 'Failed to load game. Refresh to retry.');
    }
}

function enableUI() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.disabled = false;
    });
}

// ── Constants ───────────────────────────────────────────────────────────────────

const MAX_GUESSES = 6;
const WORD_LENGTH = 5;
const KEYBOARD_ROWS = [
    ['Q','W','E','R','T','Y','U','I','O','P'],
    ['A','S','D','F','G','H','J','K','L'],
    ['ENTER','Z','X','C','V','B','N','M','⌫']
];

// ── Game State ──────────────────────────────────────────────────────────────────

let playSolution = '';
let playHistory = [];   // [{guess: "crane", pattern: "G_Y__"}]
let currentGuess = '';
let gameOver = false;

let aiSolution = '';
let aiSteps = [];      // [{guess, pattern, remaining, won}]
let aiCurrentStep = -1;
let aiRunning = false;
let aiTimer = null;

let stats = loadStats();

// ── Init ────────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    buildKeyboard('play-keyboard');
    buildGrid('play-grid');
    buildGrid('ai-grid');
    setupTabs();
    setupPlayInput();
    setupAIControls();
    setupStats();
    initWasm();
});

// ── Tab System ──────────────────────────────────────────────────────────────────

function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tab = btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById(tab + '-tab').classList.add('active');
            if (tab === 'stats') updateStatsUI();
        });
        btn.disabled = true;
    });
}

// ── Grid Building ──────────────────────────────────────────────────────────────

function buildGrid(containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    for (let r = 0; r < MAX_GUESSES; r++) {
        const row = document.createElement('div');
        row.className = 'grid-row';
        row.dataset.row = r;
        for (let c = 0; c < WORD_LENGTH; c++) {
            const tile = document.createElement('div');
            tile.className = 'tile';
            tile.dataset.row = r;
            tile.dataset.col = c;
            row.appendChild(tile);
        }
        container.appendChild(row);
    }
}

// ── Keyboard Building ──────────────────────────────────────────────────────────

function buildKeyboard(containerId) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';
    KEYBOARD_ROWS.forEach(row => {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'keyboard-row';
        row.forEach(key => {
            const btn = document.createElement('button');
            btn.className = 'key' + (key === 'ENTER' ? ' key-enter' : key === '⌫' ? ' key-backspace' : '');
            btn.textContent = key;
            btn.dataset.key = key;
            btn.addEventListener('click', () => handleKeyPress(key));
            rowDiv.appendChild(btn);
        });
        container.appendChild(rowDiv);
    });
}

// ── Play Mode ──────────────────────────────────────────────────────────────────

function setupPlayInput() {
    document.addEventListener('keydown', e => {
        if (!wasmInitialized) return;
        const activeTab = document.querySelector('.tab-content.active');
        if (activeTab && activeTab.id === 'play-tab') {
            if (e.key === 'Enter') handleKeyPress('ENTER');
            else if (e.key === 'Backspace') handleKeyPress('⌫');
            else if (/^[a-zA-Z]$/.test(e.key)) handleKeyPress(e.key.toUpperCase());
        }
    });

    document.getElementById('play-new-game-btn')?.addEventListener('click', startNewGame);

    startNewGame();
}

function startNewGame() {
    if (!wasmInitialized) return;
    playSolution = random_word();
    playHistory = [];
    currentGuess = '';
    gameOver = false;
    clearGrid('play-grid');
    clearKeyboard('play-keyboard');
    showMessage('play-message', '');
}

function handleKeyPress(key) {
    if (gameOver) return;

    if (key === 'ENTER') {
        if (currentGuess.length !== WORD_LENGTH) {
            shakeRow('play-grid', playHistory.length);
            return;
        }
        if (!validate_word(currentGuess.toLowerCase())) {
            showMessage('play-message', 'Not in word list');
            shakeRow('play-grid', playHistory.length);
            return;
        }
        submitGuess();
    } else if (key === '⌫') {
        if (currentGuess.length > 0) {
            currentGuess = currentGuess.slice(0, -1);
            updateGridRow('play-grid', playHistory.length, currentGuess, null);
        }
    } else if (currentGuess.length < WORD_LENGTH) {
        currentGuess += key.toLowerCase();
        updateGridRow('play-grid', playHistory.length, currentGuess, null);
    }
}

function submitGuess() {
    const pattern = evaluate(currentGuess.toLowerCase(), playSolution);
    playHistory.push({ guess: currentGuess, pattern });

    revealRow('play-grid', playHistory.length - 1, currentGuess, pattern);
    updateKeyboard('play-keyboard', currentGuess, pattern);

    if (pattern === 'GGGGG') {
        gameOver = true;
        showMessage('play-message', '🎉 Solved in ' + playHistory.length + '!');
        saveGameResult(playHistory.length, true);
    } else if (playHistory.length >= MAX_GUESSES) {
        gameOver = true;
        showMessage('play-message', '💀 The word was: ' + playSolution.toUpperCase());
        saveGameResult(playHistory.length, false);
    }

    currentGuess = '';
}

// ── AI Solve Mode ──────────────────────────────────────────────────────────────

function setupAIControls() {
    document.getElementById('ai-random-btn').addEventListener('click', () => {
        if (!wasmInitialized) return;
        document.getElementById('ai-solution-input').value = random_word().toUpperCase();
    });
    document.getElementById('ai-start-btn').addEventListener('click', startAI);
    document.getElementById('ai-step-btn').addEventListener('click', stepAI);
    document.getElementById('ai-run-btn').addEventListener('click', runAI);
    document.getElementById('ai-reset-btn').addEventListener('click', resetAI);
}

function startAI() {
    if (!wasmInitialized) return;
    const input = document.getElementById('ai-solution-input').value.trim().toLowerCase();
    if (input.length !== 5 || !validate_word(input)) {
        document.getElementById('ai-status').textContent = 'Enter a valid 5-letter word';
        return;
    }
    aiSolution = input;
    aiSteps = [];
    aiCurrentStep = -1;
    clearGrid('ai-grid');
    document.getElementById('ai-status').textContent = 'Solving: ' + aiSolution.toUpperCase();
    document.getElementById('ai-step-btn').disabled = false;
    document.getElementById('ai-run-btn').disabled = false;
    document.getElementById('ai-reset-btn').disabled = false;
    document.getElementById('ai-start-btn').disabled = true;
    document.getElementById('ai-solution-input').disabled = true;

    // Solve the entire game
    const result = solve_full(aiSolution);
    aiSteps = JSON.parse(result);
}

function stepAI() {
    if (aiSteps.length === 0) return;
    aiCurrentStep++;
    if (aiCurrentStep >= aiSteps.length) return;

    const step = aiSteps[aiCurrentStep];
    revealRow('ai-grid', aiCurrentStep, step.guess, step.pattern);

    if (step.pattern === 'GGGGG') {
        document.getElementById('ai-status').textContent =
            '✅ Solved in ' + (aiCurrentStep + 1) + ' guesses!';
        document.getElementById('ai-step-btn').disabled = true;
        document.getElementById('ai-run-btn').disabled = true;
    } else if (aiCurrentStep >= MAX_GUESSES - 1) {
        document.getElementById('ai-status').textContent = '❌ Failed!';
        document.getElementById('ai-step-btn').disabled = true;
        document.getElementById('ai-run-btn').disabled = true;
    }
}

function runAI() {
    document.getElementById('ai-step-btn').disabled = true;
    document.getElementById('ai-run-btn').disabled = true;
    aiRunning = true;

    function nextStep() {
        if (!aiRunning) return;
        stepAI();
        if (aiCurrentStep < aiSteps.length - 1 && aiSteps[aiCurrentStep].pattern !== 'GGGGG') {
            aiTimer = setTimeout(nextStep, 500);
        } else {
            aiRunning = false;
        }
    }
    nextStep();
}

function resetAI() {
    aiRunning = false;
    if (aiTimer) clearTimeout(aiTimer);
    aiSolution = '';
    aiSteps = [];
    aiCurrentStep = -1;
    clearGrid('ai-grid');
    document.getElementById('ai-status').textContent = '';
    document.getElementById('ai-step-btn').disabled = true;
    document.getElementById('ai-run-btn').disabled = true;
    document.getElementById('ai-reset-btn').disabled = true;
    document.getElementById('ai-start-btn').disabled = false;
    document.getElementById('ai-solution-input').disabled = false;
    document.getElementById('ai-solution-input').value = '';
}

// ── Stats ──────────────────────────────────────────────────────────────────────

function loadStats() {
    const stored = localStorage.getItem('wordlebrain_stats');
    if (stored) return JSON.parse(stored);
    return { played: 0, won: 0, currentStreak: 0, maxStreak: 0, distribution: [0,0,0,0,0,0] };
}

function saveStats() {
    localStorage.setItem('wordlebrain_stats', JSON.stringify(stats));
}

function saveGameResult(guesses, won) {
    stats.played++;
    if (won) {
        stats.won++;
        stats.currentStreak++;
        stats.maxStreak = Math.max(stats.maxStreak, stats.currentStreak);
        stats.distribution[guesses - 1]++;
    } else {
        stats.currentStreak = 0;
    }
    saveStats();
}

function setupStats() {
    document.getElementById('reset-stats-btn').addEventListener('click', () => {
        if (confirm('Reset all stats?')) {
            stats = { played: 0, won: 0, currentStreak: 0, maxStreak: 0, distribution: [0,0,0,0,0,0] };
            saveStats();
            updateStatsUI();
        }
    });
}

function updateStatsUI() {
    document.getElementById('stat-played').textContent = stats.played;
    document.getElementById('stat-win-percent').textContent =
        stats.played > 0 ? Math.round(100 * stats.won / stats.played) : 0;
    document.getElementById('stat-current-streak').textContent = stats.currentStreak;
    document.getElementById('stat-max-streak').textContent = stats.maxStreak;

    const maxVal = Math.max(...stats.distribution, 1);
    const barsDiv = document.getElementById('distribution-bars');
    barsDiv.innerHTML = '';
    for (let i = 0; i < 6; i++) {
        const row = document.createElement('div');
        row.className = 'distribution-row';
        const label = document.createElement('span');
        label.className = 'dist-label';
        label.textContent = (i + 1);
        const bar = document.createElement('div');
        bar.className = 'dist-bar';
        const pct = stats.distribution[i] / maxVal * 100;
        bar.style.width = Math.max(pct, 8) + '%';
        bar.textContent = stats.distribution[i];
        const num = document.createElement('span');
        num.className = 'dist-num';
        num.textContent = stats.distribution[i];
        row.appendChild(label);
        row.appendChild(bar);
        barsDiv.appendChild(row);
    }
}

// ── Grid & Keyboard Helpers ──────────────────────────────────────────────────

function clearGrid(containerId) {
    const container = document.getElementById(containerId);
    container.querySelectorAll('.tile').forEach(tile => {
        tile.textContent = '';
        tile.className = 'tile';
        tile.dataset.state = '';
    });
}

function clearKeyboard(containerId) {
    const container = document.getElementById(containerId);
    container.querySelectorAll('.key').forEach(key => {
        key.dataset.state = '';
        key.className = 'key';
    });
}

function updateGridRow(containerId, row, guess, pattern) {
    const container = document.getElementById(containerId);
    const tiles = container.querySelectorAll(`[data-row="${row}"] .tile`);
    for (let i = 0; i < WORD_LENGTH; i++) {
        if (i < guess.length) {
            tiles[i].textContent = guess[i].toUpperCase();
        } else {
            tiles[i].textContent = '';
        }
    }
}

function revealRow(containerId, row, guess, pattern) {
    const container = document.getElementById(containerId);
    const tiles = container.querySelectorAll(`[data-row="${row}"] .tile`);
    for (let i = 0; i < WORD_LENGTH; i++) {
        tiles[i].textContent = guess[i].toUpperCase();
        const state = pattern[i] === 'G' ? 'correct' : pattern[i] === 'Y' ? 'present' : 'absent';
        tiles[i].dataset.state = state;
        tiles[i].classList.add('flip');
    }
}

function shakeRow(containerId, row) {
    const container = document.getElementById(containerId);
    const rowEl = container.querySelectorAll('.grid-row')[row];
    if (rowEl) {
        rowEl.classList.add('shake');
        setTimeout(() => rowEl.classList.remove('shake'), 500);
    }
}

function updateKeyboard(containerId, guess, pattern) {
    const container = document.getElementById(containerId);
    for (let i = 0; i < WORD_LENGTH; i++) {
        const letter = guess[i].toUpperCase();
        const state = pattern[i] === 'G' ? 'correct' : pattern[i] === 'Y' ? 'present' : 'absent';
        const key = container.querySelector(`[data-key="${letter}"]`);
        if (key) {
            const currentState = key.dataset.state;
            // Priority: correct > present > absent
            if (state === 'correct' || (state === 'present' && currentState !== 'correct') || (state === 'absent' && !currentState)) {
                key.dataset.state = state;
            }
        }
    }
}

function showMessage(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}