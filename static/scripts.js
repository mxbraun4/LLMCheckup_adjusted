/**
 * LLMCheckup - Main JavaScript Functions
 * Handles all interactive functionality of the UI
 */

// DOM Element Selectors
const ttmInput = document.getElementById('user-input');
const ttmChat = document.getElementById('chat-messages');
const sendButton = document.getElementById('send-button');
const userInputs = [];
let userInputIndex = -1;

// Add click handler for send button
sendButton.addEventListener('click', function() {
    sendMessage();
});

// Handle Enter key
ttmInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Auto-resize textarea
ttmInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Handle arrow keys for message history
document.addEventListener('keydown', function(e) {
    if (e.keyCode === 38) { // up arrow
        if (userInputs.length === 0) return;
        userInputIndex = Math.max(0, userInputIndex - 1);
        ttmInput.value = userInputs[userInputIndex];
    } else if (e.keyCode === 40) { // down arrow
        if (userInputs.length === 0) return;
        userInputIndex = Math.min(userInputs.length - 1, userInputIndex + 1);
        ttmInput.value = userInputs[userInputIndex];
    }
});

function sendMessage() {
    const message = ttmInput.value.trim();
    
    if (message) {
        // Store message in history
        userInputs.push(message);
        userInputIndex = userInputs.length;
        
        // Add user message to chat
        addToChat("right", message, "");
        
        // Clear input
        ttmInput.value = '';
        ttmInput.style.height = 'auto';
        
        // Show generating message
        insertLoadingDots();
        
        // Send message to server
        const knowledgeLevel = document.querySelector('input[name="knowledge-level"]:checked');
        const promptType = document.querySelector('input[name="prompt-type"]:checked');
        
        const dataPackage = {
            userInput: message,
            custom_input: '0',
            qalevel: knowledgeLevel ? knowledgeLevel.value : 'beginner',
            prompt_type: promptType ? promptType.value : 'default'
        };
        
        botResponse(message, '0', dataPackage.qalevel, dataPackage.prompt_type);
    }
}

function addToChat(side, message, username) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${side}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = message;
    
    messageDiv.appendChild(contentDiv);
    ttmChat.appendChild(messageDiv);
    
    // Scroll to bottom
    ttmChat.scrollTop = ttmChat.scrollHeight;
}

function insertLoadingDots() {
    const messageDiv = document.createElement('div');
    messageDiv.id = 'generating-msg';
    messageDiv.className = 'message bot-message';
    messageDiv.innerHTML = `
        <div class="message-content">
            <div class="generating">
                <span>Generating response...</span>
                <div class="loading-dots">
                    <span class="dot"></span>
                    <span class="dot"></span>
                    <span class="dot"></span>
                </div>
                <button class="cancel-button" onclick="cancelGeneration()">Cancel</button>
            </div>
        </div>
    `;
    ttmChat.appendChild(messageDiv);
    ttmChat.scrollTop = ttmChat.scrollHeight;
}

function botResponse(msgText, custom_input, level, prompt) {
    const dataPackage = {
        userInput: msgText,
        custom_input: custom_input,
        qalevel: level,
        prompt_type: prompt
    };

    $.ajax({
        type: 'POST',
        url: '/get_response',
        data: JSON.stringify(dataPackage),
        contentType: 'application/json',
        processData: false,
        cache: false,
        success: function(data) {
            const generatingMsg = document.getElementById("generating-msg");
            if (generatingMsg) {
                generatingMsg.remove();
            }
            addToChat("left", data, "");
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
            const generatingMsg = document.getElementById("generating-msg");
            if (generatingMsg) {
                generatingMsg.remove();
            }
            addToChat("left", "An error occurred while generating the response. Please try again.", "");
        }
    });
}

/**
 * Utility function to get DOM elements
 * @param {string} selector - CSS selector for the element
 * @param {Document} root - Root element to search from (defaults to document)
 * @returns {Element} The found DOM element
 */
function get(selector, root = document) {
    return root.querySelector(selector);
}

/**
 * Cancels the current model generation
 * Sends request to cancel endpoint and updates UI
 */
function cancelGeneration() {
    fetch('/cancel_generation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === "cancelled") {
            const generatingMsg = document.getElementById("generating-msg");
            if (generatingMsg) {
                generatingMsg.remove();
            }
            addToChat("left", "Generation was cancelled. You can try again with a different prompt or question.", "");
        }
    })
    .catch(error => {
        console.error('Error cancelling generation:', error);
        const generatingMsg = document.getElementById("generating-msg");
        if (generatingMsg) {
            generatingMsg.remove();
        }
        addToChat("left", "Error cancelling generation. Please try again.", "");
    });
}

/**
 * Adds a message to the chat interface
 * @param {string} text - The message text to display
 * @param {string} side - The side to display the message on ("left" or "right")
 */
function addMessage(text, side) {
    const messageDiv = document.createElement("div");
    messageDiv.className = `msg ${side}-msg`;
    messageDiv.innerHTML = `
        <div class="msg-bubble">
            <div class="msg-text">${text}</div>
        </div>
    `;
    document.querySelector(".ttm-chat").appendChild(messageDiv);
}

// Knowledge Level Settings
function updateKnowledgeLevel(level) {
    // Update the model's knowledge level setting
    fetch('/update_knowledge_level', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ level: level })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Knowledge level updated successfully');
        } else {
            showNotification('Failed to update knowledge level', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error updating knowledge level', 'error');
    });
}

// Custom Input Functions
function sendCustomInput() {
    const claim = document.getElementById('custom-claim').value;
    const evidence = document.getElementById('custom-evidence').value;
    
    if (!claim || !evidence) {
        showNotification('Please provide both claim and evidence', 'error');
        return;
    }

    fetch('/custom_input', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            claim: claim,
            evidence: evidence
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Custom input sent successfully');
            clearCustomInput();
        } else {
            showNotification('Failed to send custom input', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error sending custom input', 'error');
    });
}

function clearCustomInput() {
    document.getElementById('custom-claim').value = '';
    document.getElementById('custom-evidence').value = '';
}

// Prompt Settings Functions
function refreshPrompt() {
    fetch('/get_current_prompt')
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            document.getElementById('current-prompt').textContent = data.prompt;
        } else {
            showNotification('Failed to refresh prompt', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error refreshing prompt', 'error');
    });
}

function updatePromptType(type) {
    fetch('/update_prompt_type', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ type: type })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Prompt type updated successfully');
            refreshPrompt();
        } else {
            showNotification('Failed to update prompt type', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error updating prompt type', 'error');
    });
}

function updateAdditionalPrompt() {
    const prompt = document.getElementById('additional-prompt').value;
    
    fetch('/update_additional_prompt', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: prompt })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showNotification('Additional prompt updated successfully');
            refreshPrompt();
        } else {
            showNotification('Failed to update additional prompt', 'error');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error updating additional prompt', 'error');
    });
}

// Export History Function
function exportHistory() {
    fetch('/export_history')
    .then(response => response.blob())
    .then(blob => {
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chat_history.json';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('Error exporting history', 'error');
    });
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Re-initialize DOM elements after page load (in case they weren't available before)
    const ttmInputEl = document.getElementById('user-input');
    const ttmChatEl = document.getElementById('chat-messages');
    const sendButtonEl = document.getElementById('send-button');
    
    if (ttmInputEl && sendButtonEl) {
        // Ensure the global variables are properly set
        if (typeof ttmInput === 'undefined' || !ttmInput) {
            window.ttmInput = ttmInputEl;
        }
        if (typeof ttmChat === 'undefined' || !ttmChat) {
            window.ttmChat = ttmChatEl;
        }
        if (typeof sendButton === 'undefined' || !sendButton) {
            window.sendButton = sendButtonEl;
        }
        
        // Re-attach event listeners to ensure they work
        sendButtonEl.addEventListener('click', function() {
            sendMessage();
        });
        
        ttmInputEl.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        ttmInputEl.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }

    // Knowledge Level Radio Buttons
    document.querySelectorAll('input[name="knowledge-level"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updateKnowledgeLevel(e.target.value);
        });
    });

    // Prompt Type Radio Buttons
    document.querySelectorAll('input[name="prompt-type"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            updatePromptType(e.target.value);
        });
    });

    // Additional Prompt Textarea
    const additionalPrompt = document.getElementById('additional-prompt');
    if (additionalPrompt) {
        additionalPrompt.addEventListener('change', () => {
            updateAdditionalPrompt();
        });
    }

    // Initial prompt refresh
    refreshPrompt();
});

/**
 * Shows a notification message to the user
 * @param {string} message - The message to display
 * @param {string} type - The type of notification ('success' or 'error')
 */
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Remove notification after 3 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 3000);
}

/**
 * Toggles the settings menu visibility
 */
function toggleSettings() {
    const settingsMenu = document.querySelector('.settings-menu');
    const mainContent = document.querySelector('.main-content');
    
    settingsMenu.classList.toggle('active');
    
    if (settingsMenu.classList.contains('active')) {
        mainContent.style.marginRight = 'var(--settings-width)';
    } else {
        mainContent.style.marginRight = '0';
    }
}

/**
 * Toggles the file upload interface
 */
function toggleFileUpload() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*,audio/*';
    fileInput.style.display = 'none';
    
    fileInput.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/upload_file', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showNotification('File uploaded successfully');
                    // Handle the uploaded file response
                    handleFileResponse(data);
                } else {
                    showNotification('Failed to upload file', 'error');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                showNotification('Error uploading file', 'error');
            });
        }
    };
    
    document.body.appendChild(fileInput);
    fileInput.click();
    document.body.removeChild(fileInput);
}

/**
 * Toggles the audio recording interface
 */
function toggleAudioRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        showNotification('Audio recording is not supported in your browser', 'error');
        return;
    }
    
    const isRecording = document.querySelector('.recording');
    if (isRecording) {
        // Stop recording
        stopRecording();
    } else {
        // Start recording
        startRecording();
    }
}

let mediaRecorder;
let audioChunks = [];

/**
 * Starts audio recording
 */
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            
            mediaRecorder.ondataavailable = (event) => {
                audioChunks.push(event.data);
            };
            
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const formData = new FormData();
                formData.append('audio', audioBlob);
                
                fetch('/process_audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showNotification('Audio processed successfully');
                        // Handle the audio response
                        handleAudioResponse(data);
                    } else {
                        showNotification('Failed to process audio', 'error');
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    showNotification('Error processing audio', 'error');
                });
            };
            
            mediaRecorder.start();
            document.querySelector('.tool-btn i.fa-microphone').parentElement.classList.add('recording');
            showNotification('Recording started...');
        })
        .catch(error => {
            console.error('Error:', error);
            showNotification('Error accessing microphone', 'error');
        });
}

/**
 * Stops audio recording
 */
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        document.querySelector('.tool-btn i.fa-microphone').parentElement.classList.remove('recording');
        showNotification('Recording stopped');
    }
}

/**
 * Handles file upload response
 */
function handleFileResponse(data) {
    if (data.extracted_text) {
        // If text was extracted from image/audio, add it to the input
        ttmInput.value += (ttmInput.value ? '\n' : '') + data.extracted_text;
        ttmInput.focus();
    }
}

/**
 * Handles audio processing response
 */
function handleAudioResponse(data) {
    if (data.transcribed_text) {
        // If audio was transcribed, add it to the input
        ttmInput.value += (ttmInput.value ? '\n' : '') + data.transcribed_text;
        ttmInput.focus();
    }
}
