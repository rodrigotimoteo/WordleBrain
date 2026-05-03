import init, {
    evaluate,
    solve_full,
    solve_step,
    get_hint,
    random_word,
    validate_word,
    word_count
} from './pkg/wordlebrain_wasm.js';

// ── WASM Initialization ──
let wasmInitialized = false;

async function initWasm() {
    try {
        await init();
        const count = word_count();
        console.log(`WordleBrain: Loaded ${count} words`);
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
    document.querySelector('[data-tab="play"]').click();
}

// ── Tab Navigation ──
function setupTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(`${tabId}-tab`).classList.add('active');
        });
    });
}

// ── Message Display ──
function showMessage(elementId, message, duration = 2000) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    if (duration > 0) {
        setTimeout(() => {
            if (el.textContent === message) el.textContent = '';
        }, duration);
    }
}

// ── Play Mode ──
let playSolution = null;
let playHistory = [];
let currentRow = 0;
let currentCol = 0;
let playGrid = [];
let gameActive = false;

function initPlayMode() {
    playSolution = random_word();
    playHistory = [];
    currentRow = 0;
    currentCol = 0;
    playGrid = Array(6).fill(null).map(() => Array(5).fill(''));
    gameActive = true;
    
    renderPlayGrid();
    renderKeyboard();
    showMessage('play-message', 'Guess the word!');
    console.log('Solution (for testing):', playSolution);
}

function renderPlayGrid() {
    const grid = document.getElementById('play-grid');
    grid.innerHTML = '';
    
    for (let row = 0; row < 6; row++) {
        const rowEl = document.createElement('div');
        rowEl.className = 'row';
        rowEl.dataset.row = row;
        
        for (let col = 0; col < 5; col++) {
            const tile = document.createElement('div');
            tile.className = 'tile';
            tile.dataset.row = row;
            tile.dataset.col = col;
            tile.textContent = playGrid[row][col];
            
            if (row === currentRow && col < currentCol) {
                tile.dataset.state = 'active';
            }
            
            rowEl.appendChild(tile);
        }
        
        grid.appendChild(rowEl);
    }
}

function updatePlayTile(row, col, letter) {
    playGrid[row][col] = letter;
    const tile = document.querySelector(`#play-grid .tile[data-row="${row}"][data-col="${col}"]`);
    if (tile) {
        tile.textContent = letter;
        if (letter) {
            tile.dataset.state = 'active';
            tile.classList.add('pop');
            setTimeout(() => tile.classList.remove('pop'), 100);
        } else {
            delete tile.dataset.state;
        }
    }
}

function handlePlayInput(letter) {
    if (!gameActive || currentCol >= 5) return;
    
    updatePlayTile(currentRow, currentCol, letter);
    currentCol++;
}

function handlePlayBackspace() {
    if (!gameActive || currentCol <= 0) return;
    
    currentCol--;
    updatePlayTile(currentRow, currentCol, '');
}

async function handlePlayEnter() {
    if (!gameActive || currentCol < 5) {
        if (currentCol < 5) {
            showMessage('play-message', 'Not enough letters');
            shakeRow('play-grid', currentRow);
        }
        return;
    }
    
    const guess = playGrid[currentRow].join('');
    
    if (!validate_word(guess)) {
        showMessage('play-message', 'Not in word list');
        shakeRow('play-grid', currentRow);
        return;
    }
    
    const pattern = evaluate(guess, playSolution);
    playHistory.push({ guess, pattern });
    
    await revealRow('play-grid', currentRow, guess, pattern);
    
    if (pattern === 'GGGGG') {
        gameActive = false;
        showMessage('play-message', 'Genius! 🎉', 0);
        bounceRow('play-grid', currentRow);
        updateStats(true, currentRow + 1);
        return;
    }
    
    if (currentRow >= 5) {
        gameActive = false;
        showMessage('play-message', `Game over! Word was: ${playSolution}`, 0);
        updateStats(false, 0);
        return;
    }
    
    currentRow++;
    currentCol = 0;
    updateKeyboard(playHistory);
}

async function revealRow(gridId, row, guess, pattern) {
    const grid = document.getElementById(gridId);
    const rowEl = grid.querySelector(`.row[data-row="${row}"]`);
    
    for (let i = 0; i < 5; i++) {
        await new Promise(resolve => setTimeout(resolve, 250));
        
        const tile = rowEl.querySelector(`.tile[data-col="${i}"]`);
        tile.classList.add('flip');
        
        setTimeout(() => {
            const state = pattern[i] === 'G' ? 'correct' : pattern[i] === 'Y' ? 'present' : 'absent';
            tile.dataset.state = state;
        }, 125);
        
        setTimeout(() => {
            tile.classList.remove('flip');
        }, 500);
    }
}

function shakeRow(gridId, row) {
    const grid = document.getElementById(gridId);
    const rowEl = grid.querySelector(`.row[data-row="${row}"]`);
    rowEl.classList.add('shake');
    setTimeout(() => rowEl.classList.remove('shake'), 500);
}

function bounceRow(gridId, row) {
    const grid = document.getElementById(gridId);
    const rowEl = grid.querySelector(`.row[data-row="${row}"]`);
    rowEl.classList.add('bounce');
    setTimeout(() => rowEl.classList.remove('bounce'), 500);
}

function renderKeyboard() {
    const keyboard = document.getElementById('play-keyboard');
    keyboard.innerHTML = '';
    
    const rows = [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['ENTER', 'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BACK']
    ];
    
    rows.forEach(rowKeys => {
        const rowEl = document.createElement('div');
        rowEl.className = 'keyboard-row';
        
        rowKeys.forEach(key => {
            const keyEl = document.createElement('button');
            keyEl.className = 'key';
            keyEl.textContent = key;
            keyEl.dataset.key = key;
            
            if (key === 'ENTER' || key === 'BACK') {
                keyEl.classList.add('large');
            }
            
            keyEl.addEventListener('click', () => handleKeyboardClick(key));
            rowEl.appendChild(keyEl);
        });
        
        keyboard.appendChild(rowEl);
    });
}

function updateKeyboard(history) {
    const keyStates = {};
    
    history.forEach(({ pattern }) => {
        for (let i = 0; i < 5; i++) {
            const letter = playGrid[history.indexOf({ pattern })]?.[i];
            if (!letter) continue;
            
            const currentState = keyStates[letter];
            const newState = pattern[i] === 'G' ? 'correct' : pattern[i] === 'Y' ? 'present' : 'absent';
            
            if (currentState === 'correct') continue;
            if (currentState === 'present' && newState === 'correct') continue;
            if (newState === 'absent' && currentState) continue;
            
            keyStates[letter] = newState;
        }
    });
    
    // Rebuild with correct letters from history
    history.forEach(({ guess, pattern }) => {
        for (let i = 0; i < 5; i++) {
            const letter = guess[i];
            const currentState = keyStates[letter];
            const newState = pattern[i] === 'G' ? 'correct' : pattern[i] === 'Y' ? 'present' : 'absent';
            
            if (!currentState || newState === 'correct' || (newState === 'present' && currentState !== 'correct')) {
                keyStates[letter] = newState;
            }
        }
    });
    
    document.querySelectorAll('#play-keyboard .key').forEach(keyEl => {
        const letter = keyEl.dataset.key;
        const state = keyStates[letter];
        if (state) {
            keyEl.dataset.state = state;
        } else {
            delete keyEl.dataset.state;
        }
    });
}

function handleKeyboardClick(key) {
    if (!gameActive) return;
    
    if (key === 'ENTER') {
        handlePlayEnter();
    } else if (key === 'BACK') {
        handlePlayBackspace();
    } else if (key.length === 1 && key >= 'A' && key <= 'Z') {
        handlePlayInput(key);
    }
}

function setupPlayKeyboard() {
    document.addEventListener('keydown', (e) => {
        if (!gameActive || document.getElementById('play-tab').classList.contains('active') === false) return;
        
        if (e.key === 'Enter') {
            handlePlayEnter();
        } else if (e.key === 'Backspace') {
            handlePlayBackspace();
        } else if (e.key.length === 1 && e.key >= 'a' && e.key <= 'z') {
            handlePlayInput(e.key.toUpperCase());
        }
    });
}

// ── AI Solve Mode ──
let aiSolution = null;
let aiSteps = [];
let aiCurrentStep = 0;
let aiRunning = false;
let aiRunInterval = null;

function initAISolve() {
    const grid = document.getElementById('ai-grid');
    grid.innerHTML = '';
    document.getElementById('ai-solution-input').value = '';
    document.getElementById('ai-status').textContent = '';
    aiSolution = null;
    aiSteps = [];
    aiCurrentStep = 0;
    aiRunning = false;
    if (aiRunInterval) {
        clearInterval(aiRunInterval);
        aiRunInterval = null;
    }
    
    document.getElementById('ai-step-btn').disabled = true;
    document.getElementById('ai-run-btn').disabled = true;
    document.getElementById('ai-reset-btn').disabled = true;
}

function renderAIGrid() {
    const grid = document.getElementById('ai-grid');
    grid.innerHTML = '';
    
    const maxRows = Math.max(6, aiSteps.length);
    
    for (let row = 0; row < maxRows; row++) {
        const rowEl = document.createElement('div');
        rowEl.className = 'row';
        rowEl.dataset.row = row;
        
        if (row < aiSteps.length) {
            const step = aiSteps[row];
            for (let col = 0; col < 5; col++) {
                const tile = document.createElement('div');
                tile.className = 'tile';
                tile.textContent = step.guess[col];
                
                const state = step.pattern[col] === 'G' ? 'correct' : step.pattern[col] === 'Y' ? 'present' : 'absent';
                tile.dataset.state = state;
                
                rowEl.appendChild(tile);
            }
        } else {
            for (let col = 0; col < 5; col++) {
                const tile = document.createElement('div');
                tile.className = 'tile';
                rowEl.appendChild(tile);
            }
        }
        
        grid.appendChild(rowEl);
    }
}

async function aiStart() {
    const input = document.getElementById('ai-solution-input');
    let solution = input.value.toUpperCase().trim();
    
    if (!solution) {
        solution = random_word();
        input.value = solution;
    }
    
    if (!validate_word(solution)) {
        document.getElementById('ai-status').textContent = 'Invalid word';
        return;
    }
    
    aiSolution = solution;
    document.getElementById('ai-status').textContent = `Solving for: ${solution}`;
    
    try {
        const result = solve_full(solution);
        aiSteps = JSON.parse(result);
        aiCurrentStep = 0;
        
        document.getElementById('ai-step-btn').disabled = false;
        document.getElementById('ai-run-btn').disabled = false;
        document.getElementById('ai-reset-btn').disabled = false;
        
        renderAIGrid();
    } catch (err) {
        document.getElementById('ai-status').textContent = `Error: ${err.message}`;
    }
}

async function aiStep() {
    if (aiCurrentStep >= aiSteps.length) {
        const won = aiSteps[aiSteps.length - 1]?.pattern === 'GGGGG';
        document.getElementById('ai-status').textContent = won 
            ? `Solved in ${aiSteps.length} guesses!` 
            : `Completed: ${aiSteps.length} guesses`;
        return;
    }
    
    const step = aiSteps[aiCurrentStep];
    document.getElementById('ai-status').textContent = `Step ${aiCurrentStep + 1}: ${step.guess} → ${step.pattern} (${step.remaining} remaining)`;
    
    aiCurrentStep++;
    renderAIGrid();
}

function aiRun() {
    if (aiRunning) {
        aiStop();
        return;
    }
    
    aiRunning = true;
    document.getElementById('ai-run-btn').textContent = 'Stop';
    
    aiRunInterval = setInterval(() => {
        if (aiCurrentStep >= aiSteps.length) {
            aiStop();
            const won = aiSteps[aiSteps.length - 1]?.pattern === 'GGGGG';
            document.getElementById('ai-status').textContent = won 
                ? `Solved in ${aiSteps.length} guesses!` 
                : `Completed: ${aiSteps.length} guesses`;
            return;
        }
        
        const step = aiSteps[aiCurrentStep];
        document.getElementById('ai-status').textContent = `Step ${aiCurrentStep + 1}: ${step.guess} → ${step.pattern}`;
        
        aiCurrentStep++;
        renderAIGrid();
    }, 500);
}

function aiStop() {
    aiRunning = false;
    document.getElementById('ai-run-btn').textContent = 'Run';
    if (aiRunInterval) {
        clearInterval(aiRunInterval);
        aiRunInterval = null;
    }
}

function setupAISolve() {
    document.getElementById('ai-random-btn').addEventListener('click', () => {
        document.getElementById('ai-solution-input').value = random_word();
    });
    
    document.getElementById('ai-start-btn').addEventListener('click', aiStart);
    document.getElementById('ai-step-btn').addEventListener('click', aiStep);
    document.getElementById('ai-run-btn').addEventListener('click', aiRun);
    document.getElementById('ai-reset-btn').addEventListener('click', initAISolve);
}

// ── Stats Mode ──
const STATS_KEY = 'wordlebrain_stats';

function loadStats() {
    const defaultStats = {
        played: 0,
        won: 0,
        currentStreak: 0,
        maxStreak: 0,
        distribution: [0, 0, 0, 0, 0, 0]
    };
    
    try {
        const stored = localStorage.getItem(STATS_KEY);
        return stored ? { ...defaultStats, ...JSON.parse(stored) } : defaultStats;
    } catch {
        return defaultStats;
    }
}

function saveStats(stats) {
    localStorage.setItem(STATS_KEY, JSON.stringify(stats));
}

function updateStats(won, guesses) {
    const stats = loadStats();
    stats.played++;
    
    if (won) {
        stats.won++;
        stats.currentStreak++;
        stats.maxStreak = Math.max(stats.maxStreak, stats.currentStreak);
        if (guesses >= 1 && guesses <= 6) {
            stats.distribution[guesses - 1]++;
        }
    } else {
        stats.currentStreak = 0;
    }
    
    saveStats(stats);
    renderStats();
}

function renderStats() {
    const stats = loadStats();
    
    document.getElementById('stat-played').textContent = stats.played;
    
    const winPercent = stats.played > 0 ? Math.round((stats.won / stats.played) * 100) : 0;
    document.getElementById('stat-win-percent').textContent = winPercent;
    
    document.getElementById('stat-current-streak').textContent = stats.currentStreak;
    document.getElementById('stat-max-streak').textContent = stats.maxStreak;
    
    const maxCount = Math.max(...stats.distribution, 1);
    const barsContainer = document.getElementById('distribution-bars');
    barsContainer.innerHTML = '';
    
    stats.distribution.forEach((count, i) => {
        const row = document.createElement('div');
        row.className = 'dist-row';
        
        const label = document.createElement('span');
        label.className = 'dist-label';
        label.textContent = i + 1;
        
        const barContainer = document.createElement('div');
        barContainer.className = 'dist-bar-container';
        
        const bar = document.createElement('div');
        bar.className = 'dist-bar';
        const width = Math.max((count / maxCount) * 100, 8);
        bar.style.width = `${width}%`;
        bar.textContent = count > 0 ? count : '';
        
        const countEl = document.createElement('span');
        countEl.className = 'dist-count';
        countEl.textContent = count;
        
        barContainer.appendChild(bar);
        row.appendChild(label);
        row.appendChild(barContainer);
        row.appendChild(countEl);
        barsContainer.appendChild(row);
    });
}

function setupStats() {
    renderStats();
    
    document.getElementById('reset-stats-btn').addEventListener('click', () => {
        if (confirm('Reset all stats?')) {
            localStorage.removeItem(STATS_KEY);
            renderStats();
        }
    });
}

// ── Initialization ──
function setupPlayMode() {
    document.getElementById('play-tab').addEventListener('click', () => {
        if (!playSolution) {
            initPlayMode();
        }
    });
    
    setupPlayKeyboard();
}

document.addEventListener('DOMContentLoaded', () => {
    setupTabs();
    setupPlayMode();
    setupAISolve();
    setupStats();
    initWasm();
});
