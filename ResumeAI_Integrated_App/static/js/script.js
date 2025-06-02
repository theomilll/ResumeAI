/**
 * ResumeAI Integrated Application Frontend Logic
 * Handles UI interactions and API communication with the backend that
 * integrates existing Speech-to-Text, Categorization, and Notion Export modules.
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('ResumeAI Integrated App initializing...');
    
    // Initialize each component
    initModels();
    initTabs();
    initTextInput();
    initSpeechToText();
    initNotionExport();
    
    console.log('ResumeAI Integrated App initialized successfully');
});

// DOM Elements
const elements = {
    // Tab elements
    tabButtons: document.querySelectorAll('.tab-btn'),
    tabPanels: document.querySelectorAll('.tab-panel'),
    
    // Text input elements
    textInput: document.getElementById('text-input'),
    modelSelect: document.getElementById('model-select'),
    processTextBtn: document.getElementById('process-text-btn'),
    clearTextBtn: document.getElementById('clear-text-btn'),
    
    // Speech-to-Text elements
    statusIndicator: document.getElementById('status-indicator'),
    statusText: document.getElementById('status-text'),
    toggleRecording: document.getElementById('toggle-recording'),
    copyTranscriptBtn: document.getElementById('copy-transcript'),
    transcriptText: document.getElementById('transcript-text'),
    speechModelSelect: document.getElementById('speech-model-select'),
    processTranscriptBtn: document.getElementById('process-transcript-btn'),
    
    // Results elements
    resultsContainer: document.getElementById('results-container'),
    noResults: document.getElementById('no-results'),
    categoryResult: document.getElementById('category-result'),
    confidenceBar: document.getElementById('confidence-bar'),
    confidenceValue: document.getElementById('confidence-value'),
    processedText: document.getElementById('processed-text'),
    
    // Notion Export elements
    notionToken: document.getElementById('notion-token'),
    destinationId: document.getElementById('destination-id'),
    exportType: document.getElementById('export-type'),
    exportTitle: document.getElementById('export-title'),
    exportBtn: document.getElementById('export-btn'),
    exportResult: document.getElementById('export-result'),
    exportStatus: document.getElementById('export-status')
};

// Global variables
let recognition = null;
let isRecording = false;
let currentTranscript = "";
let categoryData = null;
let availableModels = [];

/**
 * Initialize available models
 * Fetch the available models from the backend and update the UI
 */
function initModels() {
    fetch('/api/list-models')
        .then(response => response.json())
        .then(data => {
            availableModels = data.models;
            
            // Update model select dropdowns
            const textModelSelect = elements.modelSelect;
            const speechModelSelect = elements.speechModelSelect;
            
            // Clear existing options
            textModelSelect.innerHTML = '';
            speechModelSelect.innerHTML = '';
            
            // Add options for each available model
            availableModels.forEach(model => {
                const textOption = document.createElement('option');
                textOption.value = model.id;
                textOption.textContent = model.name;
                textOption.disabled = !model.available;
                textModelSelect.appendChild(textOption);
                
                const speechOption = document.createElement('option');
                speechOption.value = model.id;
                speechOption.textContent = model.name;
                speechOption.disabled = !model.available;
                speechModelSelect.appendChild(speechOption);
            });
            
            console.log('Models initialized:', availableModels);
        })
        .catch(error => {
            console.error('Error fetching models:', error);
        });
}

/**
 * Tab System
 * Handles switching between different tabs of the application
 */
function initTabs() {
    elements.tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const targetTab = button.dataset.tab;
            
            // Update active tab button
            elements.tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Show target tab panel, hide others
            elements.tabPanels.forEach(panel => {
                if (panel.id === targetTab) {
                    panel.classList.add('active');
                } else {
                    panel.classList.remove('active');
                }
            });
        });
    });
}

/**
 * Text Input Handlers
 * Manage text input area functionality
 */
function initTextInput() {
    elements.processTextBtn.addEventListener('click', async () => {
        const text = elements.textInput.value.trim();
        if (!text) {
            showNotification('Por favor, digite ou cole algum texto primeiro.', 'error');
            return;
        }
        
        const selectedModel = elements.modelSelect.value;
        await processContent('text', text, selectedModel);
    });
    
    elements.clearTextBtn.addEventListener('click', () => {
        elements.textInput.value = '';
    });
}

/**
 * Speech-to-Text Handlers
 * Manage speech recognition functionality using the browser's Web Speech API
 * and integration with our backend STT service
 */
function initSpeechToText() {
    // Check if browser supports Speech Recognition API
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        const speechTab = document.querySelector('[data-tab="speech-tab"]');
        speechTab.disabled = true;
        speechTab.title = 'Seu navegador não suporta reconhecimento de voz';
        showNotification('Seu navegador não suporta reconhecimento de voz', 'error');
        return;
    }
    
    // Set up Speech Recognition directly with simpler configuration
    // This is based on the original speech-to-text module approach
    function setupRecognition() {
        // Create a new instance directly
        recognition = new SpeechRecognition();
        
        // Set basic configuration (similar to original module)
        recognition.lang = 'pt-BR';
        recognition.continuous = true;
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;
        
        console.log('Speech recognition configured with direct settings');
        
        // Optional: Only fetch additional config if needed
        fetch('/api/speech-config')
            .then(response => response.json())
            .then(config => {
                // Only update if values are explicitly provided
                if (config.language) recognition.lang = config.language;
                if (config.hasOwnProperty('continuous')) recognition.continuous = config.continuous;
                if (config.hasOwnProperty('interimResults')) recognition.interimResults = config.interimResults;
                if (config.maxAlternatives) recognition.maxAlternatives = config.maxAlternatives;
                
                console.log('Speech recognition updated with server config:', config);
            })
            .catch(error => {
                console.error('Error fetching speech config, using defaults:', error);
                // Already using defaults, so no need to set them again
            });
        
        // Recognition event handlers
        recognition.onstart = () => {
            elements.statusIndicator.classList.add('recording');
            elements.statusText.textContent = 'Gravando...';
            elements.toggleRecording.innerHTML = '<span class="material-symbols-outlined">stop</span>Parar';
            elements.toggleRecording.classList.add('recording');
        };
        
        recognition.onresult = (event) => {
            let transcript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                transcript += event.results[i][0].transcript + ' ';
            }
            
            currentTranscript += transcript.trim() + '\n';
            elements.transcriptText.textContent = currentTranscript;
            elements.copyTranscriptBtn.disabled = false;
            elements.processTranscriptBtn.disabled = false;
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            elements.statusText.textContent = `Erro: ${event.error}`;
            stopRecognition();
        };
        
        recognition.onend = () => {
            console.log('Recognition ended, isRecording:', isRecording);
            if (isRecording) {
                // Try to restart if we're still supposed to be recording
                // Add a small delay to prevent rapid restart attempts
                setTimeout(() => {
                    try {
                        if (isRecording) {
                            recognition.start();
                            console.log('Recognition restarted automatically');
                        }
                    } catch (error) {
                        console.error('Error restarting recognition:', error);
                        isRecording = false;
                        resetRecordingUI();
                        showNotification('Erro ao continuar gravação. Por favor, tente novamente.', 'error');
                    }
                }, 300);
            } else {
                resetRecordingUI();
            }
        };
        
        recognition.onerror = (event) => {
            console.error('Recognition error:', event.error);
            elements.statusText.textContent = `Erro: ${event.error}`;
            
            // Handle specific errors
            if (event.error === 'not-allowed' || event.error === 'service-not-allowed') {
                showNotification('Permissão de microfone negada ou serviço não disponível. Verifique as permissões do navegador.', 'error');
                isRecording = false;
                resetRecordingUI();
            }
        };
    }
    
    // Define recording functions
    function startRecording() {
        // Always recreate the recognition object to avoid issues with reusing it
        setupRecognition();
        
        try {
            // Update UI first
            elements.statusIndicator.classList.add('recording');
            elements.statusText.textContent = 'Gravando...';
            elements.toggleRecording.innerHTML = '<span class="material-symbols-outlined">stop</span>Parar';
            elements.toggleRecording.classList.add('recording');
            
            // Clear any previous transcript if needed
            // currentTranscript = ""; // Uncomment to clear previous transcript
            
            // Start recognition after UI is updated
            recognition.start();
            isRecording = true;
            console.log('Speech recognition started');
        } catch (error) {
            console.error('Error starting speech recognition:', error);
            showNotification(`Erro ao iniciar gravação: ${error.message}`, 'error');
            
            // Reset UI state if recognition fails
            resetRecordingUI();
        }
    }
    
    function stopRecording() {
        try {
            if (recognition) {
                // Important: Set onend to null before stopping to prevent restart loops
                recognition.onend = null;
                recognition.stop();
                console.log('Speech recognition stopped');
            }
        } catch (error) {
            console.error('Error stopping recognition:', error);
        } finally {
            // Always update UI state even if an error occurs
            isRecording = false;
            resetRecordingUI();
        }
    }
    
    function resetRecordingUI() {
        elements.statusIndicator.classList.remove('recording');
        elements.statusText.textContent = 'Pronto para gravar';
        elements.toggleRecording.innerHTML = '<span class="material-symbols-outlined">mic</span>Gravar';
        elements.toggleRecording.classList.remove('recording');
    }
    
    // Event listeners
    elements.toggleRecording.addEventListener('click', () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
    
    elements.copyTranscriptBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(elements.transcriptText.textContent)
            .then(() => showNotification('Texto copiado para a área de transferência', 'success'))
            .catch(() => showNotification('Não foi possível copiar o texto', 'error'));
    });
    
    elements.processTranscriptBtn.addEventListener('click', async () => {
        const transcript = elements.transcriptText.textContent.trim();
        if (!transcript) {
            showNotification('A transcrição está vazia. Grave algum conteúdo primeiro.', 'error');
            return;
        }
        
        const selectedModel = elements.speechModelSelect.value;
        await processContent('transcript', transcript, selectedModel);
    });
}

/**
 * Process Content Handler
 * Send text or transcript for categorization and show results
 */
async function processContent(type, content, modelType = 'tfidf') {
    // Show loading state
    showLoadingState(true);
    
    try {
        // Send request to backend
        const response = await fetch('/api/categorize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text: content,
                model: modelType,
                include_confidence: true
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Erro desconhecido durante a categorização');
        }
        
        // Store data for potential Notion export
        categoryData = {
            category: data.category,
            confidence: data.confidence || 0.8,
            text: content,
            model: modelType,
            source_type: type === 'transcript' ? 'audio' : 'text'
        };
        
        // Display results
        displayResults(content, data.category, data.confidence || 0.8);
        
        // Switch to results tab
        document.querySelector('[data-tab="results-tab"]').click();
        
        // Enable export button
        elements.exportBtn.disabled = false;
        
    } catch (error) {
        console.error('Error processing content:', error);
        showNotification('Erro ao processar o conteúdo: ' + error.message, 'error');
    } finally {
        showLoadingState(false);
    }
}

/**
 * Display Results
 * Show categorization results in the UI
 */
function displayResults(text, category, confidence = 0.8) {
    elements.resultsContainer.classList.remove('hidden');
    elements.noResults.classList.add('hidden');
    
    // Update category and confidence display
    elements.categoryResult.textContent = category;
    
    const confidencePercent = Math.round(confidence * 100);
    elements.confidenceValue.textContent = `${confidencePercent}%`;
    elements.confidenceBar.style.width = `${confidencePercent}%`;
    
    // Add color coding to confidence bar
    if (confidencePercent >= 80) {
        elements.confidenceBar.style.backgroundColor = 'var(--secondary-color)';
    } else if (confidencePercent >= 50) {
        elements.confidenceBar.style.backgroundColor = '#FFC107'; // Amber
    } else {
        elements.confidenceBar.style.backgroundColor = 'var(--accent-color)';
    }
    
    // Display processed text
    elements.processedText.textContent = text;
}

/**
 * Notion Export Handlers
 * Manage exporting content to Notion
 */
function initNotionExport() {
    // Try to get token from localStorage if it exists (for convenience)
    const savedToken = localStorage.getItem('notion_token');
    if (savedToken) {
        elements.notionToken.value = savedToken;
    }
    
    const savedDestId = localStorage.getItem('notion_destination_id');
    if (savedDestId) {
        elements.destinationId.value = savedDestId;
    }
    
    elements.exportBtn.addEventListener('click', async () => {
        const token = elements.notionToken.value.trim();
        const destinationId = elements.destinationId.value.trim();
        
        if (!token || !destinationId) {
            showNotification('Por favor, preencha o token do Notion e o ID de destino.', 'error');
            return;
        }
        
        if (!categoryData || !elements.processedText.textContent.trim()) {
            showNotification('Não há conteúdo para exportar. Processe um texto primeiro.', 'error');
            return;
        }
        
        // Save token and destination ID for convenience
        localStorage.setItem('notion_token', token);
        localStorage.setItem('notion_destination_id', destinationId);
        
        // Show loading
        showLoadingState(true);
        elements.exportBtn.disabled = true;
        
        try {
            const title = elements.exportTitle.value.trim() || 
                       `ResumeAI: ${categoryData.category} - ${new Date().toLocaleDateString()}`;
            
            const response = await fetch('/api/export/notion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    token: token,
                    destination_id: destinationId,
                    export_type: elements.exportType.value,
                    title: title,
                    content: elements.processedText.textContent.trim(),
                    categories: [categoryData.category],
                    source_type: categoryData.source_type || 'text',
                    source_name: 'ResumeAI Integrated App',
                    language: 'pt-BR'
                })
            });
            
            const data = await response.json();
            
            // Display export result
            elements.exportResult.classList.remove('hidden');
            
            if (data.success) {
                elements.exportStatus.innerHTML = `
                    <div class="export-success">
                        <p>✓ Conteúdo exportado com sucesso para o Notion!</p>
                        ${data.notion_url ? `<p><a href="${data.notion_url}" target="_blank">Abrir no Notion</a></p>` : ''}
                    </div>
                `;
            } else {
                throw new Error(data.error || 'Erro desconhecido na exportação');
            }
            
        } catch (error) {
            console.error('Error exporting to Notion:', error);
            elements.exportResult.classList.remove('hidden');
            elements.exportStatus.innerHTML = `
                <div class="export-error">
                    <p>❌ Erro ao exportar para o Notion:</p>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            showLoadingState(false);
            elements.exportBtn.disabled = false;
        }
    });
}

/**
 * UI Utility Functions
 */
function showLoadingState(isLoading) {
    document.body.classList.toggle('loading', isLoading);
    
    // Disable buttons during loading
    elements.processTextBtn.disabled = isLoading;
    elements.processTranscriptBtn.disabled = isLoading || !elements.transcriptText.textContent.trim();
    elements.exportBtn.disabled = isLoading || !categoryData;
}

function showNotification(message, type = 'info') {
    // Simple alert for now, but could be replaced with a toast notification system
    alert(message);
}
