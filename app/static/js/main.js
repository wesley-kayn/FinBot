document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const chatContainer = document.getElementById('chat-container');
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const uploadForm = document.getElementById('upload-form');
    const addDocumentForm = document.getElementById('add-document-form');
    
    // Bootstrap modals
    const uploadModal = new bootstrap.Modal(document.getElementById('uploadModal'));
    const addDocumentModal = new bootstrap.Modal(document.getElementById('addDocumentModal'));
    
    // Handle query submission
    queryForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        // Get query text
        const query = queryInput.value.trim();
        
        if (query === '') return;
        
        // Add user message to chat
        addMessageToChat('user', query);
        
        // Add loading indicator
        const loadingId = addLoadingIndicator();
        
        // Clear input
        queryInput.value = '';
        
        // Send query to backend
        fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ query: query })
        })
        .then(response => response.json())
        .then(data => {
            // Remove loading indicator
            removeLoadingIndicator(loadingId);
            
            if (data.error) {
                // Show error message
                addErrorMessage(data.error);
            } else {
                // Add assistant response to chat
                addMessageToChat('assistant', data.response, data.sources);
                
                // Handle special cases
                if (data.is_jailbreak) {
                    addSystemMessage('⚠️ Potential security issue detected. Please refrain from attempts to manipulate the system.');
                } else if (data.is_out_of_domain) {
                    addSystemMessage('ℹ️ Your question appears to be outside of banking topics. Please ask about Finbot services.');
                }
            }
        })
        .catch(error => {
            // Remove loading indicator
            removeLoadingIndicator(loadingId);
            
            // Show error message
            addErrorMessage('Error: Could not get a response. Please try again later.');
            console.error('Error:', error);
        });
    });
    
    // Handle file upload
    uploadForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const formData = new FormData(uploadForm);
        const fileInput = document.getElementById('file-upload');
        
        if (fileInput.files.length === 0) {
            showUploadMessage('Please select a file.', 'danger');
            return;
        }
        
        // Show uploading message
        showUploadMessage('Uploading file, please wait...', 'info');
        
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showUploadMessage(data.message, 'success');
                uploadForm.reset();
                
                // Close modal after 2 seconds
                setTimeout(() => {
                    uploadModal.hide();
                    addSystemMessage(`✅ New data added: ${data.message}`);
                }, 2000);
            } else {
                showUploadMessage(data.message, 'danger');
            }
        })
        .catch(error => {
            showUploadMessage('Error uploading file. Please try again.', 'danger');
            console.error('Error:', error);
        });
    });
    
    // Handle adding new document
    addDocumentForm.addEventListener('submit', function(event) {
        event.preventDefault();
        
        const category = document.getElementById('category').value.trim();
        const question = document.getElementById('question').value.trim();
        const answer = document.getElementById('answer').value.trim();
        
        if (!category || !question || !answer) {
            showAddDocumentMessage('All fields are required.', 'danger');
            return;
        }
        
        // Show adding message
        showAddDocumentMessage('Adding document, please wait...', 'info');
        
        fetch('/api/add-document', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                category: category,
                question: question,
                answer: answer
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showAddDocumentMessage(data.message, 'success');
                addDocumentForm.reset();
                
                // Close modal after 2 seconds
                setTimeout(() => {
                    addDocumentModal.hide();
                    addSystemMessage(`✅ New document added to category: ${category}`);
                }, 2000);
            } else {
                showAddDocumentMessage(data.message, 'danger');
            }
        })
        .catch(error => {
            showAddDocumentMessage('Error adding document. Please try again.', 'danger');
            console.error('Error:', error);
        });
    });
    
    // Helper functions
    function addMessageToChat(role, content, sources = []) {
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}-message`;
        
        // Create message content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        // Add paragraphs for text with line breaks
        content.split('\n').forEach(paragraph => {
            if (paragraph.trim()) {
                const p = document.createElement('p');
                p.textContent = paragraph;
                contentDiv.appendChild(p);
            }
        });
        
        messageDiv.appendChild(contentDiv);
        
        // Add sources if provided
        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement('div');
            sourcesDiv.className = 'sources';
            sourcesDiv.textContent = `Sources: ${sources.join(', ')}`;
            messageDiv.appendChild(sourcesDiv);
        }
        
        // Add to chat container
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function addSystemMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system-message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function addErrorMessage(content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message error-message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;
        
        messageDiv.appendChild(contentDiv);
        chatContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    
    function addLoadingIndicator() {
        const id = 'loading-' + Date.now();
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'loading';
        loadingDiv.id = id;
        
        const dotsDiv = document.createElement('div');
        dotsDiv.className = 'loading-dots';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('span');
            dotsDiv.appendChild(dot);
        }
        
        loadingDiv.appendChild(dotsDiv);
        chatContainer.appendChild(loadingDiv);
        
        // Scroll to bottom
        chatContainer.scrollTop = chatContainer.scrollHeight;
        
        return id;
    }
    
    function removeLoadingIndicator(id) {
        const loadingElement = document.getElementById(id);
        if (loadingElement) {
            loadingElement.remove();
        }
    }
    
    function showUploadMessage(message, type) {
        // Check if message element already exists, remove it if it does
        const existingMessage = document.querySelector('#upload-message');
        if (existingMessage) {
            existingMessage.remove();
        }
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.id = 'upload-message';
        messageDiv.className = `alert alert-${type} mt-3`;
        messageDiv.textContent = message;
        
        // Add to form
        const form = document.getElementById('upload-form');
        form.appendChild(messageDiv);
    }
    
    function showAddDocumentMessage(message, type) {
        // Check if message element already exists, remove it if it does
        const existingMessage = document.querySelector('#add-document-message');
        if (existingMessage) {
            existingMessage.remove();
        }
        
        // Create message element
        const messageDiv = document.createElement('div');
        messageDiv.id = 'add-document-message';
        messageDiv.className = `alert alert-${type} mt-3`;
        messageDiv.textContent = message;
        
        // Add to form
        const form = document.getElementById('add-document-form');
        form.appendChild(messageDiv);
    }
}); 