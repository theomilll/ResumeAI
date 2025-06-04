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
    initWhisperRecording();
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
    exportStatus: document.getElementById('export-status'),
    
    // Whisper Recording elements
    browserRecording: document.getElementById('browser-recording'),
    whisperRecording: document.getElementById('whisper-recording'),
    audioFileInput: document.getElementById('audio-file-input'),
    transcribeFileBtn: document.getElementById('transcribe-file-btn'),
    audioDeviceSelect: document.getElementById('audio-device-select'),
    refreshDevicesBtn: document.getElementById('refresh-devices-btn'),
    whisperTranscriptText: document.getElementById('whisper-transcript-text'),
    copyWhisperTranscriptBtn: document.getElementById('copy-whisper-transcript'),
    whisperModelSelect: document.getElementById('whisper-model-select'),
    processWhisperTranscriptBtn: document.getElementById('process-whisper-transcript-btn'),
    summarySection: document.getElementById('summary-section'),
    summaryText: document.getElementById('summary-text'),
    useSummaryBtn: document.getElementById('use-summary-btn'),
    
    // Summarization elements
    enableSummarization: document.getElementById('enable-summarization'),
    summaryResults: document.getElementById('summary-results'),
    enhancedSummary: document.getElementById('enhanced-summary'),
    mlSummary: document.getElementById('ml-summary')
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
        const useSummarization = elements.enableSummarization.checked;
        
        if (useSummarization) {
            await processContentWithSummary('text', text, selectedModel);
        } else {
            await processContent('text', text, selectedModel);
        }
    });
    
    elements.clearTextBtn.addEventListener('click', () => {
        elements.textInput.value = '';
        elements.enableSummarization.checked = false;
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
 * Process Content with Summarization
 * Full pipeline: ML Summary -> Gemini Enhancement -> Categorization
 */
async function processContentWithSummary(type, content, modelType = 'tfidf') {
    // Show loading state
    showLoadingState(true);
    
    // Show processing steps
    showProcessingSteps();
    
    try {
        updateProcessingStep(1, 'active'); // ML Summary step
        
        // Send request to backend for full pipeline
        const response = await fetch('/api/process-with-summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                text: content,
                model: modelType,
                export_to_notion: false
            })
        });
        
        if (!response.ok) {
            throw new Error(`Error: ${response.status}`);
        }
        
        const data = await response.json();
        
        if (!data.success) {
            throw new Error(data.error || 'Erro desconhecido durante o processamento');
        }
        
        updateProcessingStep(1, 'completed'); // ML Summary completed
        updateProcessingStep(2, 'completed'); // Gemini completed
        updateProcessingStep(3, 'completed'); // Categorization completed
        
        // Store data for potential Notion export
        categoryData = {
            category: data.category,
            confidence: data.confidence || 0.8,
            text: data.enhanced_summary, // Use enhanced summary for export
            original_text: data.original_text,
            ml_summary: data.ml_summary,
            enhanced_summary: data.enhanced_summary,
            model: modelType,
            source_type: type === 'transcript' ? 'audio' : 'text',
            has_summary: true
        };
        
        // Display results with summary information
        displayResultsWithSummary(data);
        
        // Switch to results tab
        document.querySelector('[data-tab="results-tab"]').click();
        
        // Enable export button
        elements.exportBtn.disabled = false;
        
        hideProcessingSteps();
        
    } catch (error) {
        console.error('Error processing content with summary:', error);
        showNotification('Erro ao processar o conteúdo com resumo: ' + error.message, 'error');
        hideProcessingSteps();
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
    elements.summaryResults.classList.add('hidden'); // Hide summary section
    
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
 * Display Results with Summary
 * Show categorization results including summary information
 */
function displayResultsWithSummary(data) {
    elements.resultsContainer.classList.remove('hidden');
    elements.noResults.classList.add('hidden');
    elements.summaryResults.classList.remove('hidden'); // Show summary section
    
    // Update category and confidence display
    elements.categoryResult.textContent = data.category;
    
    const confidencePercent = Math.round((data.confidence || 0.8) * 100);
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
    
    // Display summaries
    elements.enhancedSummary.textContent = data.enhanced_summary;
    elements.mlSummary.textContent = data.ml_summary;
    
    // Display original text
    elements.processedText.textContent = data.original_text;
}

/**
 * Processing Steps Functions
 */
function showProcessingSteps() {
    // Create processing steps UI if it doesn't exist
    let stepsContainer = document.getElementById('processing-steps');
    if (!stepsContainer) {
        stepsContainer = document.createElement('div');
        stepsContainer.id = 'processing-steps';
        stepsContainer.className = 'processing-steps';
        stepsContainer.innerHTML = `
            <h4>Processamento em Andamento:</h4>
            <div class="processing-step" id="step-1">
                <span class="step-icon pending"></span>
                <span>1. Gerando resumo com ML (BART)</span>
            </div>
            <div class="processing-step" id="step-2">
                <span class="step-icon pending"></span>
                <span>2. Aprimorando resumo com Gemini</span>
            </div>
            <div class="processing-step" id="step-3">
                <span class="step-icon pending"></span>
                <span>3. Categorizando resumo aprimorado</span>
            </div>
        `;
        
        // Insert before results container
        const resultsTab = document.getElementById('results-tab');
        resultsTab.insertBefore(stepsContainer, elements.noResults);
    }
    
    // Reset all steps to pending
    for (let i = 1; i <= 3; i++) {
        updateProcessingStep(i, 'pending');
    }
}

function updateProcessingStep(stepNumber, status) {
    const step = document.getElementById(`step-${stepNumber}`);
    const icon = step.querySelector('.step-icon');
    
    // Remove all status classes
    step.classList.remove('active', 'completed');
    icon.classList.remove('pending', 'active', 'completed');
    
    // Add new status
    step.classList.add(status);
    icon.classList.add(status);
}

function hideProcessingSteps() {
    const stepsContainer = document.getElementById('processing-steps');
    if (stepsContainer) {
        setTimeout(() => {
            stepsContainer.style.display = 'none';
        }, 2000); // Hide after 2 seconds
    }
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

/**
 * Initialize Whisper Recording functionality
 * Handles file uploads and device-based recording with Whisper transcription
 */
function initWhisperRecording() {
    // Recording mode toggle
    const recordingModeRadios = document.querySelectorAll('input[name="recording-mode"]');
    recordingModeRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if (e.target.value === 'browser') {
                elements.browserRecording.style.display = 'block';
                elements.whisperRecording.style.display = 'none';
            } else {
                elements.browserRecording.style.display = 'none';
                elements.whisperRecording.style.display = 'block';
                // Load audio devices when switching to Whisper mode
                loadAudioDevices();
            }
        });
    });
    
    // File upload handling
    elements.audioFileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        elements.transcribeFileBtn.disabled = !file;
    });
    
    elements.transcribeFileBtn.addEventListener('click', async () => {
        const file = elements.audioFileInput.files[0];
        if (!file) {
            showNotification('Por favor, selecione um arquivo de áudio primeiro.', 'error');
            return;
        }
        
        // Show loading state
        showLoadingState(true);
        elements.transcribeFileBtn.disabled = true;
        
        try {
            const formData = new FormData();
            formData.append('audio', file);
            
            const response = await fetch('/api/transcribe-audio', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                elements.whisperTranscriptText.textContent = data.transcription;
                elements.copyWhisperTranscriptBtn.disabled = false;
                elements.processWhisperTranscriptBtn.disabled = false;
                
                // Optionally generate summary
                await generateSummary(data.transcription);
            } else {
                throw new Error(data.error || 'Erro na transcrição');
            }
        } catch (error) {
            console.error('Error transcribing audio:', error);
            showNotification('Erro ao transcrever áudio: ' + error.message, 'error');
        } finally {
            showLoadingState(false);
            elements.transcribeFileBtn.disabled = false;
        }
    });
    
    // Copy whisper transcript
    elements.copyWhisperTranscriptBtn.addEventListener('click', () => {
        navigator.clipboard.writeText(elements.whisperTranscriptText.textContent)
            .then(() => showNotification('Texto copiado para a área de transferência', 'success'))
            .catch(() => showNotification('Não foi possível copiar o texto', 'error'));
    });
    
    // Process whisper transcript
    elements.processWhisperTranscriptBtn.addEventListener('click', async () => {
        const transcript = elements.whisperTranscriptText.textContent.trim();
        if (!transcript) {
            showNotification('A transcrição está vazia. Transcreva um áudio primeiro.', 'error');
            return;
        }
        
        const selectedModel = elements.whisperModelSelect.value;
        await processContent('whisper_transcript', transcript, selectedModel);
    });
    
    // Refresh audio devices
    elements.refreshDevicesBtn.addEventListener('click', () => {
        loadAudioDevices();
    });
    
    // Use summary button
    elements.useSummaryBtn.addEventListener('click', async () => {
        const summary = elements.summaryText.textContent.trim();
        if (!summary) {
            showNotification('Nenhum resumo disponível.', 'error');
            return;
        }
        
        const selectedModel = elements.whisperModelSelect.value;
        await processContent('summary', summary, selectedModel);
    });
}

/**
 * Load available audio devices
 */
async function loadAudioDevices() {
    try {
        const response = await fetch('/api/audio-devices');
        const data = await response.json();
        
        if (data.success) {
            elements.audioDeviceSelect.innerHTML = '';
            
            // Add default option
            const defaultOption = document.createElement('option');
            defaultOption.value = '';
            defaultOption.textContent = 'Selecione um dispositivo';
            elements.audioDeviceSelect.appendChild(defaultOption);
            
            // Add device options
            data.devices.forEach(device => {
                const option = document.createElement('option');
                option.value = device.id;
                option.textContent = `${device.name} (${device.channels} canais)`;
                if (device.id === data.suggested_id) {
                    option.textContent += ' (Recomendado)';
                    option.selected = true;
                }
                elements.audioDeviceSelect.appendChild(option);
            });
        } else {
            elements.audioDeviceSelect.innerHTML = '<option value="">Erro ao carregar dispositivos</option>';
        }
    } catch (error) {
        console.error('Error loading audio devices:', error);
        elements.audioDeviceSelect.innerHTML = '<option value="">Erro ao carregar dispositivos</option>';
    }
}

/**
 * Generate summary from transcription
 */
async function generateSummary(text) {
    try {
        const response = await fetch('/api/generate-summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        
        if (data.success) {
            elements.summaryText.textContent = data.final_summary;
            elements.summarySection.style.display = 'block';
        }
    } catch (error) {
        console.error('Error generating summary:', error);
        // Don't show error as summary is optional
    }
}
