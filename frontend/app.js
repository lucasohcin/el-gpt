// El GPT 1.8 Ultra — ChatGPT Client, Live Code Sandbox, Voice & Training Logic
document.addEventListener("DOMContentLoaded", () => {
  // Chat Elements
  const chatContainer = document.getElementById("chatContainer");
  const heroWelcome = document.getElementById("heroWelcome");
  const messagesFlow = document.getElementById("messagesFlow");
  const chatInput = document.getElementById("chatInput");
  const sendBtn = document.getElementById("sendBtn");
  const stopBtn = document.getElementById("stopBtn");
  const newChatBtn = document.getElementById("newChatBtn");
  const historyList = document.getElementById("historyList");
  const sidebar = document.getElementById("sidebar");
  const sidebarToggleBtn = document.getElementById("sidebarToggleBtn");
  const mobileToggleBtn = document.getElementById("mobileToggleBtn");

  // Model Elements
  const modelSelectBtn = document.getElementById("modelSelectBtn");
  const modelDropdownMenu = document.getElementById("modelDropdownMenu");
  const currentModelName = document.getElementById("currentModelName");
  const currentModelTag = document.getElementById("currentModelTag");
  const heroTitle = document.getElementById("heroTitle");
  const heroSubtitle = document.querySelector(".hero-subtitle");
  const deviceLabel = document.getElementById("deviceLabel");

  // Toolbar & Input Extras
  const modeToggleBtn = document.getElementById("modeToggleBtn");
  const modeLabel = document.getElementById("modeLabel");
  const inputTokenCount = document.getElementById("inputTokenCount");
  const openPromptsBtn = document.getElementById("openPromptsBtn");
  const closePromptsBtn = document.getElementById("closePromptsBtn");
  const promptLibraryModal = document.getElementById("promptLibraryModal");
  const exportChatBtn = document.getElementById("exportChatBtn");

  // Live Sandbox Preview Elements
  const previewModal = document.getElementById("previewModal");
  const previewIframe = document.getElementById("previewIframe");
  const closePreviewBtn = document.getElementById("closePreviewBtn");
  const openExternalPreviewBtn = document.getElementById("openExternalPreviewBtn");

  // Split-Screen Canvas Workspace Elements
  const canvasWorkspace = document.getElementById("canvasWorkspace");
  const toggleCanvasBtn = document.getElementById("toggleCanvasBtn");
  const canvasTitle = document.getElementById("canvasTitle");
  const canvasFilename = document.getElementById("canvasFilename");
  const canvasRefreshBtn = document.getElementById("canvasRefreshBtn");
  const canvasDownloadBtn = document.getElementById("canvasDownloadBtn");
  const canvasCloseBtn = document.getElementById("canvasCloseBtn");
  const canvasFrameContainer = document.getElementById("canvasFrameContainer");
  const canvasIframe = document.getElementById("canvasIframe");
  const canvasCodeContainer = document.getElementById("canvasCodeContainer");
  const canvasCodeBlock = document.getElementById("canvasCodeBlock");
  const canvasStatusPill = document.getElementById("canvasStatusPill");
  const canvasConsoleDrawer = document.getElementById("canvasConsoleDrawer");
  const canvasConsoleLogs = document.getElementById("canvasConsoleLogs");
  const clearConsoleBtn = document.getElementById("clearConsoleBtn");
  const canvasLogCount = document.getElementById("canvasLogCount");
  const appLayout = document.querySelector(".app-layout");

  // Memory Elements
  const openMemoryBtn = document.getElementById("openMemoryBtn");
  const closeMemoryBtn = document.getElementById("closeMemoryBtn");
  const memoryModal = document.getElementById("memoryModal");
  const newMemoryInput = document.getElementById("newMemoryInput");
  const addMemoryBtn = document.getElementById("addMemoryBtn");
  const memoryCount = document.getElementById("memoryCount");
  const memoryCountSubtitle = document.getElementById("memoryCountSubtitle");
  const memoryItemsList = document.getElementById("memoryItemsList");
  const clearMemoriesBtn = document.getElementById("clearMemoriesBtn");

  // Studio Elements
  const openStudioBtn = document.getElementById("openStudioBtn");
  const quickTrainBtn = document.getElementById("quickTrainBtn");
  const closeStudioBtn = document.getElementById("closeStudioBtn");
  const studioModal = document.getElementById("studioModal");
  const statParams = document.getElementById("statParams");
  const statDevice = document.getElementById("statDevice");
  const statLoss = document.getElementById("statLoss");
  const statSpeed = document.getElementById("statSpeed");
  const chartStep = document.getElementById("chartStep");
  const lossCanvas = document.getElementById("lossCanvas");
  const inputScale = document.getElementById("inputScale");
  const inputDataset = document.getElementById("inputDataset");
  const inputEpochs = document.getElementById("inputEpochs");
  const inputBatchSize = document.getElementById("inputBatchSize");
  const inputLr = document.getElementById("inputLr");
  const startTrainBtn = document.getElementById("startTrainBtn");
  const stopTrainBtn = document.getElementById("stopTrainBtn");
  const trainSpinner = document.getElementById("trainSpinner");
  const trainBtnText = document.getElementById("trainBtnText");
  const trainingStatusBar = document.getElementById("trainingStatusBar");

  // Cloud Settings Elements
  const openCloudKeyBtn = document.getElementById("openCloudKeyBtn");
  const closeCloudKeyBtn = document.getElementById("closeCloudKeyBtn");
  const cloudKeyModal = document.getElementById("cloudKeyModal");
  const groqApiKeyInput = document.getElementById("groqApiKeyInput");
  const saveCloudKeyBtn = document.getElementById("saveCloudKeyBtn");
  const clearCloudKeyBtn = document.getElementById("clearCloudKeyBtn");
  const cloudKeyBtnText = document.getElementById("cloudKeyBtnText");
  const cloudStatusIndicator = document.getElementById("cloudStatusIndicator");
  const cloudStatusText = document.getElementById("cloudStatusText");
  const toggleApiKeyVisibilityBtn = document.getElementById("toggleApiKeyVisibilityBtn");

  // State
  let conversations = JSON.parse(localStorage.getItem("el_gpt_chats") || "[]");
  let activeChatId = null;
  let isGenerating = false;
  let abortController = null;
  let pollInterval = null;
  let canvasCtx = lossCanvas ? lossCanvas.getContext("2d") : null;
  let userGroqKey = localStorage.getItem("el_gpt_groq_key") || "";
  let currentModelId = localStorage.getItem("el_gpt_model_id") || "el-gpt-cloud-llama-70b";
  if (currentModelId === "el-gpt-pro" || currentModelId === "el-gpt-1-5-pro") {
    currentModelId = "el-gpt-cloud-llama-70b"; // Default to blazing-fast cloud model
  }
  let isDeepReasoning = true;
  let activeLiveCode = "";
  let activeSpeechBtn = null;
  let activeCanvasCode = "";
  let activeCanvasLang = "html";
  let activeCanvasFilename = "index.html";
  let activeCanvasTitle = "Interactive Preview";
  let canvasLogs = [];

  // Configure Markdown parser
  if (window.marked) {
    marked.setOptions({
      breaks: true,
      gfm: true,
      highlight: function (code, lang) {
        if (window.hljs && lang && hljs.getLanguage(lang)) {
          return hljs.highlight(code, { language: lang }).value;
        }
        return code;
      },
    });
  }

  // Initialize Application
  initApp();

  function initApp() {
    renderHistory();
    if (conversations.length > 0) {
      loadConversation(conversations[0].id);
    } else {
      startNewChat();
    }
    updateModelUI(currentModelId);
    fetchModelMetadata();
    initCloudSettings();
    loadMemories();
    setupEventListeners();
  }

  function updateModelUI(modelId) {
    currentModelId = modelId;
    localStorage.setItem("el_gpt_model_id", modelId);

    document.querySelectorAll(".model-option").forEach((opt) => {
      if (opt.getAttribute("data-model") === modelId) {
        opt.classList.add("active");
      } else {
        opt.classList.remove("active");
      }
    });

    if (modelId === "el-gpt-cloud-llama-70b") {
      currentModelName.textContent = "El GPT Cloud 70B";
      currentModelTag.textContent = "⚡ 450 tps Cloud";
      currentModelTag.className = "model-tag cloud-badge-lightning";
      if (heroTitle) heroTitle.textContent = "El GPT Cloud 70B";
      if (heroSubtitle) heroSubtitle.textContent = "Powered by Llama 3.3 70B in the cloud. Blazing speed (450+ tok/s), 0% Mac CPU, master-level coding, math & reasoning.";
    } else if (modelId === "el-gpt-cloud-deepseek-r1") {
      currentModelName.textContent = "El GPT DeepSeek R1";
      currentModelTag.textContent = "⚡ Reasoning Cloud";
      currentModelTag.className = "model-tag cloud-badge";
      if (heroTitle) heroTitle.textContent = "El GPT DeepSeek R1";
      if (heroSubtitle) heroSubtitle.textContent = "State-of-the-art Deep Reasoning Engine. Step-by-step chain-of-thought proofs, advanced logic, algorithms & coding.";
    } else if (modelId === "el-gpt-cloud-qwen-32b") {
      currentModelName.textContent = "El GPT Qwen 2.5 32B";
      currentModelTag.textContent = "⚡ 400 tps Cloud";
      currentModelTag.className = "model-tag cloud-badge";
      if (heroTitle) heroTitle.textContent = "El GPT Qwen 2.5 32B";
      if (heroSubtitle) heroSubtitle.textContent = "Elite Coding and Multilingual Engine. Superior HTML/CSS/JS web development & math.";
    } else if (modelId === "el-gpt-cloud-llama-8b") {
      currentModelName.textContent = "El GPT Cloud Instant";
      currentModelTag.textContent = "⚡ 750 tps Ultra-Speed";
      currentModelTag.className = "model-tag cloud-badge";
      if (heroTitle) heroTitle.textContent = "El GPT Cloud Instant";
      if (heroSubtitle) heroSubtitle.textContent = "Ultra-high speed 8B model. Near-instantaneous streaming (750+ tokens/sec) for rapid conversation.";
    } else if (modelId === "el-gpt-1-8-ultra") {
      currentModelName.textContent = "El GPT 1.8 Ultra";
      currentModelTag.textContent = "1B Local MPS";
      currentModelTag.className = "model-tag ultra-badge";
      if (heroTitle) heroTitle.textContent = "El GPT 1.8 Ultra";
      if (heroSubtitle) heroSubtitle.textContent = "1 Billion+ Parameter Flagship Local Neural Network. High RAM & compute load.";
    } else if (modelId === "el-gpt-1-5-flash") {
      currentModelName.textContent = "El GPT 1.5 Flash";
      currentModelTag.textContent = "500M Flash ⚡";
      currentModelTag.className = "model-tag flash-badge";
      if (heroTitle) heroTitle.textContent = "El GPT 1.5 Flash";
      if (heroSubtitle) heroSubtitle.textContent = "Ultra-Low Latency 500M Engine. Instant token streaming and rapid execution.";
    } else if (modelId === "el-gpt-1-5-pro") {
      currentModelName.textContent = "El GPT 1.5 Pro";
      currentModelTag.textContent = "500M Pro";
      currentModelTag.className = "model-tag pro-badge";
      if (heroTitle) heroTitle.textContent = "El GPT 1.5 Pro";
      if (heroSubtitle) heroSubtitle.textContent = "500 Million Parameter Neural Network with Persistent Memory, HTML/CSS Web Engine, & Math Reasoning.";
    } else if (modelId === "el-gpt-1-0-pro") {
      currentModelName.textContent = "El GPT 1.0 Pro";
      currentModelTag.textContent = "135M Compact";
      currentModelTag.className = "model-tag smart-badge";
      if (heroTitle) heroTitle.textContent = "El GPT 1.0 Pro";
      if (heroSubtitle) heroSubtitle.textContent = "Compact 135M Parameter Model pre-trained on 2T tokens for lightweight chat.";
    } else if (modelId === "el-gpt-scratch") {
      currentModelName.textContent = "El GPT Scratch";
      currentModelTag.textContent = "Custom 10M–1B";
      currentModelTag.className = "model-tag train-badge";
      if (heroTitle) heroTitle.textContent = "El GPT Scratch";
      if (heroSubtitle) heroSubtitle.textContent = "Your custom PyTorch Transformer decoder trained locally in the Training Studio.";
    }
  }

  function setupEventListeners() {
    // Model Selector Dropdown
    modelSelectBtn?.addEventListener("click", (e) => {
      e.stopPropagation();
      modelDropdownMenu.classList.toggle("hidden");
      modelSelectBtn.classList.toggle("open");
    });

    document.querySelectorAll(".model-option").forEach((opt) => {
      opt.addEventListener("click", (e) => {
        e.stopPropagation();
        const model = opt.getAttribute("data-model");
        if (model) {
          updateModelUI(model);
          modelDropdownMenu.classList.add("hidden");
          modelSelectBtn.classList.remove("open");
        }
      });
    });

    document.addEventListener("click", () => {
      modelDropdownMenu?.classList.add("hidden");
      modelSelectBtn?.classList.remove("open");
    });

    // Deep Reasoning / Fast Mode Toggle
    modeToggleBtn?.addEventListener("click", () => {
      isDeepReasoning = !isDeepReasoning;
      modeToggleBtn.classList.toggle("active", isDeepReasoning);
      if (isDeepReasoning) {
        modeLabel.textContent = "🧠 Deep Reasoning";
      } else {
        modeLabel.textContent = "⚡ Fast Stream";
      }
    });

    // Input resizing, stats & sending
    chatInput.addEventListener("input", () => {
      chatInput.style.height = "auto";
      chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + "px";
      const val = chatInput.value;
      sendBtn.disabled = !val.trim() || isGenerating;
      if (inputTokenCount) {
        const estTokens = Math.ceil(val.length / 4);
        inputTokenCount.textContent = `${val.length} chars (~${estTokens} tok)`;
      }
    });

    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) {
          submitMessage();
        }
      }
    });

    // Global Shortcut: ⌘K or Ctrl+K for New Chat
    document.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        startNewChat();
      }
    });

    sendBtn.addEventListener("click", submitMessage);
    stopBtn.addEventListener("click", stopGeneration);
    newChatBtn.addEventListener("click", startNewChat);

    // Sidebar toggles
    sidebarToggleBtn?.addEventListener("click", () => sidebar.classList.toggle("hidden"));
    mobileToggleBtn?.addEventListener("click", () => sidebar.classList.toggle("open"));

    // Suggestion pills
    document.querySelectorAll(".suggestion-card").forEach((card) => {
      card.addEventListener("click", () => {
        const prompt = card.getAttribute("data-prompt");
        if (prompt) {
          chatInput.value = prompt;
          chatInput.dispatchEvent(new Event("input"));
          submitMessage();
        }
      });
    });

    // Prompt Library Modal
    openPromptsBtn?.addEventListener("click", () => promptLibraryModal?.classList.remove("hidden"));
    closePromptsBtn?.addEventListener("click", () => promptLibraryModal?.classList.add("hidden"));
    promptLibraryModal?.addEventListener("click", (e) => {
      if (e.target === promptLibraryModal) promptLibraryModal.classList.add("hidden");
    });

    document.querySelectorAll(".prompt-template-card").forEach((card) => {
      card.addEventListener("click", () => {
        const prompt = card.getAttribute("data-prompt");
        if (prompt) {
          chatInput.value = prompt;
          chatInput.dispatchEvent(new Event("input"));
          promptLibraryModal?.classList.add("hidden");
          chatInput.focus();
        }
      });
    });

    // Export Conversation
    exportChatBtn?.addEventListener("click", exportCurrentChat);

    // Canvas Workspace Listeners
    toggleCanvasBtn?.addEventListener("click", () => {
      if (canvasWorkspace?.classList.contains("hidden")) {
        openCanvas(
          activeLiveCode ||
            `<!DOCTYPE html><html><body style="font-family:sans-serif;padding:40px;text-align:center;color:#475569;"><h2>Live Canvas Workspace</h2><p>El GPT live previews will appear here automatically when web or code artifacts are generated.</p></body></html>`,
          "html",
          "index.html",
          "Live Canvas Preview"
        );
      } else {
        closeCanvas();
      }
    });

    canvasCloseBtn?.addEventListener("click", () => closeCanvas());

    canvasRefreshBtn?.addEventListener("click", () => {
      if (canvasIframe && activeCanvasCode) {
        canvasIframe.srcdoc = activeCanvasCode;
        if (canvasStatusPill) {
          canvasStatusPill.textContent = "Refreshed";
          setTimeout(() => (canvasStatusPill.textContent = "Ready"), 1500);
        }
      }
    });

    canvasDownloadBtn?.addEventListener("click", () => {
      if (activeCanvasCode) {
        downloadCodeFile(activeCanvasCode, activeCanvasFilename);
      }
    });

    // Canvas Viewport Switcher
    document.querySelectorAll(".canvas-device-switcher .device-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".canvas-device-switcher .device-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const vp = btn.getAttribute("data-viewport");
        if (canvasFrameContainer) {
          canvasFrameContainer.setAttribute("data-viewport", vp);
        }
      });
    });

    // Canvas Footer Tabs (Preview, Code, Console)
    document.querySelectorAll(".canvas-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".canvas-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        const mode = tab.getAttribute("data-tab");
        if (mode === "preview") {
          canvasFrameContainer?.classList.remove("hidden");
          canvasCodeContainer?.classList.add("hidden");
          canvasConsoleDrawer?.classList.add("hidden");
        } else if (mode === "code") {
          canvasFrameContainer?.classList.add("hidden");
          canvasCodeContainer?.classList.remove("hidden");
          canvasConsoleDrawer?.classList.add("hidden");
        } else if (mode === "console") {
          canvasConsoleDrawer?.classList.toggle("hidden");
        }
      });
    });

    clearConsoleBtn?.addEventListener("click", () => {
      canvasLogs = [];
      if (canvasConsoleLogs) canvasConsoleLogs.innerHTML = "";
      if (canvasLogCount) canvasLogCount.textContent = "0";
    });

    // Listen to postMessage from iframe console
    window.addEventListener("message", (event) => {
      if (event.data && event.data.type === "el_gpt_console") {
        addCanvasLog(event.data.level, event.data.text);
      }
    });

    // Live Web Sandbox Modal
    closePreviewBtn?.addEventListener("click", () => previewModal?.classList.add("hidden"));
    previewModal?.addEventListener("click", (e) => {
      if (e.target === previewModal) previewModal.classList.add("hidden");
    });

    document.querySelectorAll(".device-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".device-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        const dev = btn.getAttribute("data-device");
        if (dev === "mobile") {
          previewIframe.style.width = "375px";
        } else if (dev === "tablet") {
          previewIframe.style.width = "768px";
        } else {
          previewIframe.style.width = "100%";
        }
      });
    });

    openExternalPreviewBtn?.addEventListener("click", () => {
      if (!activeLiveCode) return;
      const blob = new Blob([activeLiveCode], { type: "text/html" });
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank");
    });

    // Memory Modal
    openMemoryBtn?.addEventListener("click", () => openMemoryModal());
    closeMemoryBtn?.addEventListener("click", () => closeMemoryModal());
    memoryModal?.addEventListener("click", (e) => {
      if (e.target === memoryModal) closeMemoryModal();
    });
    addMemoryBtn?.addEventListener("click", () => addMemory());
    newMemoryInput?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        addMemory();
      }
    });
    // Cloud Settings Modal
    openCloudKeyBtn?.addEventListener("click", () => openCloudKeyModal());
    closeCloudKeyBtn?.addEventListener("click", () => closeCloudKeyModal());
    cloudKeyModal?.addEventListener("click", (e) => {
      if (e.target === cloudKeyModal) closeCloudKeyModal();
    });
    saveCloudKeyBtn?.addEventListener("click", () => saveCloudKey());
    clearCloudKeyBtn?.addEventListener("click", () => clearCloudKey());
    groqApiKeyInput?.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        saveCloudKey();
      }
    });
    toggleApiKeyVisibilityBtn?.addEventListener("click", () => {
      if (groqApiKeyInput) {
        groqApiKeyInput.type = groqApiKeyInput.type === "password" ? "text" : "password";
      }
    });

    // Studio Modal
    openStudioBtn?.addEventListener("click", () => openStudio());
    quickTrainBtn?.addEventListener("click", () => openStudio());
    closeStudioBtn?.addEventListener("click", () => closeStudio());
    studioModal?.addEventListener("click", (e) => {
      if (e.target === studioModal) closeStudio();
    });

    startTrainBtn?.addEventListener("click", launchTraining);
    stopTrainBtn?.addEventListener("click", haltTraining);
  }

  function startNewChat() {
    activeChatId = "chat_" + Date.now();
    const newChat = {
      id: activeChatId,
      title: "New Chat",
      messages: [],
      timestamp: Date.now(),
    };
    conversations.unshift(newChat);
    saveConversations();
    renderHistory();
    loadConversation(activeChatId);
    chatInput.focus();
  }

  function saveConversations() {
    localStorage.setItem("el_gpt_chats", JSON.stringify(conversations));
  }

  function renderHistory() {
    historyList.innerHTML = "";
    conversations.forEach((chat) => {
      const item = document.createElement("div");
      item.className = `history-item ${chat.id === activeChatId ? "active" : ""}`;
      item.textContent = chat.title || "New Chat";
      item.addEventListener("click", () => loadConversation(chat.id));
      historyList.appendChild(item);
    });
  }

  function loadConversation(chatId) {
    activeChatId = chatId;
    const chat = conversations.find((c) => c.id === chatId);
    messagesFlow.innerHTML = "";

    if (!chat || chat.messages.length === 0) {
      heroWelcome.classList.remove("hidden");
    } else {
      heroWelcome.classList.add("hidden");
      chat.messages.forEach((msg) => {
        const bubble = appendMessageBubble(msg.role, msg.content, false);
        if (msg.role === "assistant") {
          postProcessBubble(bubble, msg.content, msg.telemetry);
        }
      });
    }

    renderHistory();
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  function appendMessageBubble(role, initialContent = "", showCursor = false) {
    heroWelcome.classList.add("hidden");

    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    if (role === "assistant") {
      const avatar = document.createElement("div");
      avatar.className = "assistant-avatar";
      row.appendChild(avatar);
    }

    const wrapper = document.createElement("div");
    wrapper.className = "message-bubble-wrapper";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";

    if (role === "user") {
      bubble.textContent = initialContent;
    } else {
      bubble.innerHTML = formatMarkdownAndMath(initialContent);
      if (showCursor) {
        const cursor = document.createElement("span");
        cursor.className = "typing-cursor";
        bubble.appendChild(cursor);
      }
    }

    wrapper.appendChild(bubble);
    row.appendChild(wrapper);
    messagesFlow.appendChild(row);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return bubble;
  }

  function formatMarkdownAndMath(rawText) {
    if (!rawText) return "";
    let html = rawText;
    if (window.marked) {
      try {
        html = marked.parse(rawText);
      } catch (e) {
        html = rawText;
      }
    }
    return html;
  }

  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, function (m) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;",
      }[m];
    });
  }

  function downloadCodeFile(content, filename) {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  function openCanvas(code, lang = "html", filename = "index.html", title = "Interactive Preview") {
    activeCanvasCode = code;
    activeCanvasLang = lang;
    activeCanvasFilename = filename;
    activeCanvasTitle = title;
    activeLiveCode = code;

    if (canvasTitle) canvasTitle.textContent = title;
    if (canvasFilename) canvasFilename.textContent = filename;

    if (appLayout) appLayout.classList.add("canvas-open");
    if (canvasWorkspace) canvasWorkspace.classList.remove("hidden");

    if (canvasIframe) {
      canvasIframe.srcdoc = code;
    }
    if (canvasCodeBlock) {
      canvasCodeBlock.textContent = code;
      canvasCodeBlock.className = `language-${lang}`;
      if (window.hljs) hljs.highlightElement(canvasCodeBlock);
    }
    if (canvasStatusPill) canvasStatusPill.textContent = "Ready";
  }

  function closeCanvas() {
    if (appLayout) appLayout.classList.remove("canvas-open");
    if (canvasWorkspace) canvasWorkspace.classList.add("hidden");
  }

  function addCanvasLog(level, text) {
    canvasLogs.push({ level, text, time: new Date().toLocaleTimeString() });
    if (canvasLogCount) canvasLogCount.textContent = String(canvasLogs.length);
    if (!canvasConsoleLogs) return;

    const row = document.createElement("div");
    row.className = `console-log-row ${level}`;
    row.textContent = `[${level.toUpperCase()}] ${text}`;
    canvasConsoleLogs.appendChild(row);
    canvasConsoleLogs.scrollTop = canvasConsoleLogs.scrollHeight;
  }

  async function executeCodeSnippet(lang, code, outputEl, statusEl) {
    try {
      const res = await fetch("/api/run-code", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ language: lang, code: code, timeout: 6.0 }),
      });
      const data = await res.json();
      if (data.success) {
        outputEl.className = "terminal-output";
        outputEl.textContent = data.stdout || "(Code executed successfully with no output)";
        statusEl.textContent = `Completed (${data.execution_time_ms}ms)`;
      } else {
        outputEl.className = "terminal-output error";
        outputEl.textContent = data.stderr || data.stdout || `Error (exit code ${data.exit_code})`;
        statusEl.textContent = `Failed (exit ${data.exit_code})`;
      }
    } catch (e) {
      outputEl.className = "terminal-output error";
      outputEl.textContent = `Execution failed: ${e.message}`;
      statusEl.textContent = "Error";
    }
  }

  function assembleWebBundle(blocks) {
    let htmlPart = "";
    let cssPart = "";
    let jsPart = "";

    blocks.forEach((b) => {
      const l = (b.lang || "").toLowerCase();
      if (/html|xml/i.test(l) || /<!DOCTYPE html|<html|<body|<div/i.test(b.code)) {
        if (!htmlPart) htmlPart = b.code;
        else htmlPart += "\n" + b.code;
      } else if (l === "css") {
        cssPart += "\n" + b.code;
      } else if (/javascript|js/i.test(l)) {
        jsPart += "\n" + b.code;
      }
    });

    if (!htmlPart && !cssPart && !jsPart) return "";

    const consoleScript = `<script>
      (function() {
        var origLog = console.log, origWarn = console.warn, origErr = console.error;
        function send(type, args) {
          try {
            var str = Array.from(args).map(function(a) {
              return typeof a === 'object' ? JSON.stringify(a) : String(a);
            }).join(' ');
            window.parent.postMessage({ type: 'el_gpt_console', level: type, text: str }, '*');
          } catch(e) {}
        }
        console.log = function() { send('log', arguments); origLog.apply(console, arguments); };
        console.warn = function() { send('warn', arguments); origWarn.apply(console, arguments); };
        console.error = function() { send('error', arguments); origErr.apply(console, arguments); };
        window.onerror = function(msg, url, line) { send('error', [msg + ' (line ' + line + ')']); };
      })();
    <\/script>`;

    if (/<!DOCTYPE html|<html/i.test(htmlPart)) {
      let doc = htmlPart;
      if (cssPart) {
        if (doc.includes("</head>")) {
          doc = doc.replace("</head>", `<style>\n${cssPart}\n</style></head>`);
        } else {
          doc = `<style>\n${cssPart}\n</style>` + doc;
        }
      }
      if (jsPart) {
        if (doc.includes("</body>")) {
          doc = doc.replace("</body>", `<script>\n${jsPart}\n<\/script></body>`);
        } else {
          doc = doc + `<script>\n${jsPart}\n<\/script>`;
        }
      }
      if (doc.includes("<head>")) {
        doc = doc.replace("<head>", `<head>${consoleScript}`);
      } else {
        doc = consoleScript + doc;
      }
      return doc;
    }

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>El GPT Live Preview</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      padding: 16px;
      color: #1f2937;
      background: #ffffff;
      line-height: 1.5;
    }
    ${cssPart}
  </style>
  ${consoleScript}
</head>
<body>
  ${htmlPart || '<div style="padding:20px;text-align:center;color:#666;">Component loaded</div>'}
  <script>
    ${jsPart}
  <\/script>
</body>
</html>`;
  }

  function postProcessBubble(bubbleElement, rawText = "", telemetry = null) {
    const parentWrapper = bubbleElement.closest(".message-bubble-wrapper");

    // Gather all pre blocks first
    const preBlocks = Array.from(bubbleElement.querySelectorAll("pre"));
    const blocksData = preBlocks.map((pre) => {
      const codeEl = pre.querySelector("code");
      let lang = "code";
      if (codeEl) {
        codeEl.className.split(" ").forEach((c) => {
          if (c.startsWith("language-")) lang = c.replace("language-", "");
        });
      }
      const code = codeEl ? codeEl.textContent : pre.textContent;
      return { pre, codeEl, lang, code };
    });

    // Assemble unified web bundle if multiple web blocks exist
    const webBundle = assembleWebBundle(blocksData);

    blocksData.forEach((blockItem, idx) => {
      const { pre, codeEl, lang, code } = blockItem;
      if (pre.closest(".code-block-wrapper")) return;

      const normLang = (lang || "").toLowerCase();
      const isHTML = /html|xml/i.test(normLang) || /<!DOCTYPE html|<html|<body|<div/i.test(code);
      const isSVG = /svg/i.test(normLang) || /<svg[\s>]/i.test(code);
      const isMarkdown = /markdown|md/i.test(normLang);
      const isJSON = normLang === "json";
      const isPython = /python|py/i.test(normLang);
      const isJS = /javascript|js/i.test(normLang);
      const isCSS = normLang === "css";

      const ext = isPython ? "py" : isJS ? "js" : isHTML ? "html" : isCSS ? "css" : isJSON ? "json" : isSVG ? "svg" : isMarkdown ? "md" : "txt";
      const defaultFilename = `app_${idx + 1}.${ext}`;

      const wrapper = document.createElement("div");
      wrapper.className = "code-block-wrapper";

      const header = document.createElement("div");
      header.className = "code-header";

      // Segmented Tabs Header Left
      const headerLeft = document.createElement("div");
      headerLeft.className = "code-header-left";
      headerLeft.innerHTML = `<span class="code-lang-badge">${normLang || "CODE"}</span>`;

      const tabsContainer = document.createElement("div");
      tabsContainer.className = "code-tabs";

      const hasPreview = isHTML || isSVG || isMarkdown || isJSON;
      const isRunnable = isPython || isJS;
      const activeTab = hasPreview ? "preview" : "code";

      let previewTabBtn = null;
      let codeTabBtn = null;
      let runTabBtn = null;

      if (hasPreview) {
        previewTabBtn = document.createElement("button");
        previewTabBtn.className = "code-tab-btn active";
        previewTabBtn.innerHTML = `<span>👁️ Live Preview</span>`;
        tabsContainer.appendChild(previewTabBtn);
      }

      codeTabBtn = document.createElement("button");
      codeTabBtn.className = `code-tab-btn ${activeTab === "code" ? "active" : ""}`;
      codeTabBtn.innerHTML = `<span>💻 Code</span>`;
      tabsContainer.appendChild(codeTabBtn);

      if (isRunnable) {
        runTabBtn = document.createElement("button");
        runTabBtn.className = "code-tab-btn";
        runTabBtn.innerHTML = `<span>▶ Run & Output</span>`;
        tabsContainer.appendChild(runTabBtn);
      }

      headerLeft.appendChild(tabsContainer);
      header.appendChild(headerLeft);

      // Header actions (Copy, Canvas, Download)
      const actionsDiv = document.createElement("div");
      actionsDiv.className = "code-header-actions";

      // Copy Button
      const copyBtn = document.createElement("button");
      copyBtn.className = "code-action-btn copy-btn";
      copyBtn.innerHTML = `<span>Copy</span>`;
      copyBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(code);
        copyBtn.innerHTML = `<span>Copied!</span>`;
        setTimeout(() => (copyBtn.innerHTML = `<span>Copy</span>`), 2000);
      });
      actionsDiv.appendChild(copyBtn);

      // Canvas / Split-Screen Button
      const canvasBtn = document.createElement("button");
      canvasBtn.className = "code-action-btn canvas-btn";
      canvasBtn.innerHTML = `<span>↗ Canvas</span>`;
      canvasBtn.title = "Open side-by-side in Canvas Workspace";
      canvasBtn.addEventListener("click", () => {
        const previewCode = (isHTML || isCSS || isJS) && webBundle ? webBundle : code;
        openCanvas(previewCode, normLang, defaultFilename, isHTML ? "Web Application" : `${normLang.toUpperCase()} Preview`);
      });
      actionsDiv.appendChild(canvasBtn);

      // Download Button
      const downloadBtn = document.createElement("button");
      downloadBtn.className = "code-action-btn download-btn";
      downloadBtn.innerHTML = `<span>⬇ Download</span>`;
      downloadBtn.title = `Download ${defaultFilename}`;
      downloadBtn.addEventListener("click", () => {
        const contentToSave = (isHTML || isCSS || isJS) && webBundle && isHTML ? webBundle : code;
        downloadCodeFile(contentToSave, defaultFilename);
      });
      actionsDiv.appendChild(downloadBtn);

      header.appendChild(actionsDiv);
      wrapper.appendChild(header);

      // Panes:
      // 1. Preview Pane
      let previewPane = null;
      if (hasPreview) {
        previewPane = document.createElement("div");
        previewPane.className = `code-tab-content inline-preview-pane ${activeTab === "preview" ? "" : "hidden"}`;

        if (isHTML) {
          const iframe = document.createElement("iframe");
          iframe.className = "inline-preview-iframe";
          iframe.sandbox = "allow-scripts allow-modals allow-same-origin allow-forms";
          iframe.title = "Live HTML Preview";
          iframe.srcdoc = webBundle || code;
          previewPane.appendChild(iframe);
          activeLiveCode = webBundle || code;
        } else if (isSVG) {
          const svgPane = document.createElement("div");
          svgPane.className = "inline-svg-pane";
          svgPane.innerHTML = code;
          previewPane.appendChild(svgPane);
        } else if (isMarkdown) {
          const mdPane = document.createElement("div");
          mdPane.className = "inline-markdown-pane";
          mdPane.innerHTML = formatMarkdownAndMath(code);
          previewPane.appendChild(mdPane);
        } else if (isJSON) {
          const jsonPane = document.createElement("div");
          jsonPane.className = "inline-markdown-pane";
          try {
            const parsed = JSON.parse(code);
            jsonPane.innerHTML = `<pre style="margin:0;color:#6ee7b7;">${escapeHtml(JSON.stringify(parsed, null, 2))}</pre>`;
          } catch (e) {
            jsonPane.textContent = code;
          }
          previewPane.appendChild(jsonPane);
        }
        wrapper.appendChild(previewPane);
      }

      // 2. Code Pane
      const codePane = document.createElement("div");
      codePane.className = `code-tab-content code-tab-pane ${activeTab === "code" ? "" : "hidden"}`;
      // Replace pre in the DOM with our wrapper element
      if (pre.parentNode) {
        pre.parentNode.replaceChild(wrapper, pre);
      }
      codePane.appendChild(pre);
      wrapper.appendChild(codePane);

      // 3. Terminal Runner Pane (for Python & JS)
      let terminalPane = null;
      if (isRunnable) {
        terminalPane = document.createElement("div");
        terminalPane.className = "code-tab-content inline-terminal-pane hidden";

        const termHeader = document.createElement("div");
        termHeader.className = "terminal-header";
        termHeader.innerHTML = `
          <div style="display:flex;align-items:center;gap:6px;">
            <span class="terminal-meta-pill">${normLang.toUpperCase()} Terminal</span>
            <span class="terminal-meta-pill terminal-status">Ready</span>
          </div>
        `;
        const runBtn = document.createElement("button");
        runBtn.className = "terminal-run-btn";
        runBtn.innerHTML = `
          <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
          <span>Run</span>
        `;
        termHeader.appendChild(runBtn);
        terminalPane.appendChild(termHeader);

        const termOutput = document.createElement("div");
        termOutput.className = "terminal-output";
        termOutput.textContent = `Click "Run" to execute this ${normLang} code and see real-time output.`;
        terminalPane.appendChild(termOutput);

        const statusEl = termHeader.querySelector(".terminal-status");

        const executeAction = () => {
          runBtn.classList.add("running");
          runBtn.querySelector("span").textContent = "Running...";
          statusEl.textContent = "Executing...";
          termOutput.className = "terminal-output";
          termOutput.textContent = "$ Running code...\n";

          executeCodeSnippet(normLang, code, termOutput, statusEl).finally(() => {
            runBtn.classList.remove("running");
            runBtn.querySelector("span").textContent = "Run";
          });
        };

        runBtn.addEventListener("click", executeAction);
        wrapper.appendChild(terminalPane);

        // Run tab button switch
        runTabBtn?.addEventListener("click", () => {
          tabsContainer.querySelectorAll(".code-tab-btn").forEach((b) => b.classList.remove("active"));
          runTabBtn.classList.add("active");
          previewPane?.classList.add("hidden");
          codePane.classList.add("hidden");
          terminalPane.classList.remove("hidden");
          if (termOutput.textContent.includes('Click "Run"')) {
            executeAction();
          }
        });
      }

      // Tab click events
      previewTabBtn?.addEventListener("click", () => {
        tabsContainer.querySelectorAll(".code-tab-btn").forEach((b) => b.classList.remove("active"));
        previewTabBtn.classList.add("active");
        previewPane?.classList.remove("hidden");
        codePane.classList.add("hidden");
        terminalPane?.classList.add("hidden");
      });

      codeTabBtn.addEventListener("click", () => {
        tabsContainer.querySelectorAll(".code-tab-btn").forEach((b) => b.classList.remove("active"));
        codeTabBtn.classList.add("active");
        previewPane?.classList.add("hidden");
        codePane.classList.remove("hidden");
        terminalPane?.classList.add("hidden");
      });

      if (window.hljs && codeEl) {
        hljs.highlightElement(codeEl);
      }
    });

    // Render KaTeX math formulas
    if (window.renderMathInElement) {
      try {
        renderMathInElement(bubbleElement, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
          ],
          throwOnError: false,
        });
      } catch (e) {
        console.warn("KaTeX error:", e);
      }
    }

    // Add Assistant Message Actions Toolbar (Speech, Copy, Telemetry)
    if (parentWrapper && !parentWrapper.querySelector(".message-actions-bar")) {
      const actionsBar = document.createElement("div");
      actionsBar.className = "message-actions-bar";

      // 1. Voice Read Aloud button
      const speakBtn = document.createElement("button");
      speakBtn.className = "msg-action-btn speak-btn";
      speakBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
        <span>Read aloud</span>
      `;
      speakBtn.addEventListener("click", () => toggleSpeech(rawText || bubbleElement.textContent, speakBtn));
      actionsBar.appendChild(speakBtn);

      // 2. Copy Entire Message button
      const copyMsgBtn = document.createElement("button");
      copyMsgBtn.className = "msg-action-btn copy-msg-btn";
      copyMsgBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
          <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
        <span>Copy text</span>
      `;
      copyMsgBtn.addEventListener("click", () => {
        navigator.clipboard.writeText(rawText || bubbleElement.innerText);
        copyMsgBtn.querySelector("span").textContent = "Copied!";
        setTimeout(() => (copyMsgBtn.querySelector("span").textContent = "Copy text"), 2000);
      });
      actionsBar.appendChild(copyMsgBtn);

      // 3. Telemetry Badge
      if (telemetry) {
        const telePill = document.createElement("span");
        telePill.className = "telemetry-pill";
        telePill.textContent = `${telemetry.modelName || "1.54B"} • ${telemetry.speed || "48.5"} tok/s • ${telemetry.duration || "0.8"}s`;
        actionsBar.appendChild(telePill);
      }

      parentWrapper.appendChild(actionsBar);
    }
  }

  function openLivePreview(htmlContent) {
    activeLiveCode = htmlContent;
    if (!previewModal || !previewIframe) return;
    previewModal.classList.remove("hidden");
    previewIframe.srcdoc = htmlContent;
  }

  function toggleSpeech(text, btnElement) {
    if (!("speechSynthesis" in window)) {
      alert("Speech synthesis is not supported in this browser.");
      return;
    }

    if (window.speechSynthesis.speaking && activeSpeechBtn === btnElement) {
      window.speechSynthesis.cancel();
      btnElement.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
        <span>Read aloud</span>
      `;
      activeSpeechBtn = null;
      return;
    }

    window.speechSynthesis.cancel();
    if (activeSpeechBtn) {
      activeSpeechBtn.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
        <span>Read aloud</span>
      `;
    }

    // Strip code blocks and markdown syntax for clean spoken audio
    const cleanSpeech = text
      .replace(/```[\s\S]*?```/g, "Code block omitted.")
      .replace(/[#*_`$]/g, "")
      .trim();

    const utterance = new SpeechSynthesisUtterance(cleanSpeech);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;

    btnElement.innerHTML = `
      <div class="audio-wave">
        <span class="audio-bar"></span>
        <span class="audio-bar"></span>
        <span class="audio-bar"></span>
      </div>
      <span>Speaking...</span>
    `;
    activeSpeechBtn = btnElement;

    utterance.onend = () => {
      btnElement.innerHTML = `
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path>
        </svg>
        <span>Read aloud</span>
      `;
      activeSpeechBtn = null;
    };

    window.speechSynthesis.speak(utterance);
  }

  function exportCurrentChat() {
    const chat = conversations.find((c) => c.id === activeChatId);
    if (!chat || chat.messages.length === 0) {
      alert("No messages to export in this chat.");
      return;
    }

    let md = `# ${chat.title || "El GPT 1.8 Ultra Conversation"}\n\n`;
    md += `*Exported on ${new Date().toLocaleString()}*\n\n---\n\n`;

    chat.messages.forEach((msg) => {
      const speaker = msg.role === "user" ? "### 👤 User" : "### 🤖 El GPT 1.8 Ultra";
      md += `${speaker}\n\n${msg.content}\n\n---\n\n`;
    });

    const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(chat.title || "chat").replace(/[^a-zA-Z0-9_-]/g, "_")}.md`;
    a.click();
    URL.revokeObjectURL(url);
  }

  async function submitMessage() {
    const text = chatInput.value.trim();
    if (!text || isGenerating) return;

    chatInput.value = "";
    chatInput.style.height = "auto";
    sendBtn.disabled = true;
    if (inputTokenCount) inputTokenCount.textContent = "0 chars";

    // Retrieve active conversation
    let chat = conversations.find((c) => c.id === activeChatId);
    if (!chat) {
      startNewChat();
      chat = conversations.find((c) => c.id === activeChatId);
    }

    if (chat.messages.length === 0) {
      chat.title = text.slice(0, 32) + (text.length > 32 ? "..." : "");
      renderHistory();
    }

    // Add user message to UI & history
    chat.messages.push({ role: "user", content: text });
    appendMessageBubble("user", text);

    // Prepare assistant bubble
    isGenerating = true;
    sendBtn.classList.add("hidden");
    stopBtn.classList.remove("hidden");

    const assistantBubble = appendMessageBubble("assistant", "", true);
    let fullResponse = "";
    const genStartTime = performance.now();
    let tokenChunkCount = 0;

    abortController = new AbortController();

    const temp = currentModelId === "el-gpt-1-5-flash" ? 0.2 : isDeepReasoning ? 0.6 : 0.25;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: chat.messages,
          model_id: currentModelId,
          temperature: temp,
          top_k: currentModelId === "el-gpt-1-5-flash" ? 20 : 40,
          top_p: 0.9,
          max_new_tokens: 1024,
          api_key: userGroqKey || undefined,
        }),
        signal: abortController.signal,
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data: ")) {
            const dataStr = trimmed.slice(6);
            if (dataStr === "[DONE]") break;
            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.token) {
                fullResponse += parsed.token;
                tokenChunkCount++;
                assistantBubble.innerHTML = formatMarkdownAndMath(fullResponse);
                const cursor = document.createElement("span");
                cursor.className = "typing-cursor";
                assistantBubble.appendChild(cursor);
                chatContainer.scrollTop = chatContainer.scrollHeight;
              }
            } catch (err) {
              // Ignore partial JSON
            }
          }
        }
      }

      // Remove typing cursor
      const cursor = assistantBubble.querySelector(".typing-cursor");
      if (cursor) cursor.remove();

      const elapsedSec = ((performance.now() - genStartTime) / 1000).toFixed(1);
      const estTokens = Math.max(tokenChunkCount, Math.ceil(fullResponse.length / 4));
      const tokSpeed = (estTokens / Math.max(parseFloat(elapsedSec), 0.1)).toFixed(1);

      let displayModelName = "El GPT";
      if (currentModelId === "el-gpt-cloud-llama-70b") displayModelName = "Cloud 70B ⚡";
      else if (currentModelId === "el-gpt-cloud-deepseek-r1") displayModelName = "DeepSeek R1 ⚡";
      else if (currentModelId === "el-gpt-cloud-qwen-32b") displayModelName = "Qwen 32B ⚡";
      else if (currentModelId === "el-gpt-cloud-llama-8b") displayModelName = "Cloud Instant ⚡";
      else if (currentModelId === "el-gpt-1-8-ultra") displayModelName = "1.54B Ultra (Local)";
      else if (currentModelId === "el-gpt-1-5-pro") displayModelName = "500M Pro (Local)";
      else if (currentModelId === "el-gpt-1-5-flash") displayModelName = "500M Flash (Local)";
      else if (currentModelId === "el-gpt-1-0-pro") displayModelName = "135M Compact (Local)";
      else displayModelName = "Scratch (Local)";

      const telemetry = {
        modelName: displayModelName,
        speed: tokSpeed,
        duration: elapsedSec,
      };

      assistantBubble.innerHTML = formatMarkdownAndMath(fullResponse);
      try {
        postProcessBubble(assistantBubble, fullResponse, telemetry);
      } catch (procErr) {
        console.warn("Post processing error:", procErr);
      }

      chat.messages.push({ role: "assistant", content: fullResponse, telemetry });
      saveConversations();

      // Dynamically reload memories if user taught El GPT something
      if (/remember|prefer|preference|my name is/i.test(text)) {
        loadMemories();
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("Chat streaming error:", err);
        assistantBubble.innerHTML += `<br><span style="color:#f87171;">[Connection error: ${err.message}]</span>`;
      }
      const cursor = assistantBubble.querySelector(".typing-cursor");
      if (cursor) cursor.remove();
      if (fullResponse) {
        if (!chat.messages.length || chat.messages[chat.messages.length - 1].role !== "assistant") {
          chat.messages.push({ role: "assistant", content: fullResponse });
          saveConversations();
        }
        try {
          postProcessBubble(assistantBubble, fullResponse);
        } catch (procErr) {
          console.warn("Post processing error in catch:", procErr);
        }
      }
    } finally {
      isGenerating = false;
      sendBtn.classList.remove("hidden");
      stopBtn.classList.add("hidden");
      sendBtn.disabled = !chatInput.value.trim();
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  function stopGeneration() {
    if (abortController) {
      abortController.abort();
    }
  }

  async function fetchModelMetadata() {
    try {
      const res = await fetch("/api/models");
      if (res.ok) {
        const data = await res.json();
        const devStr = (data.device || "mps").toUpperCase();
        if (statDevice) statDevice.textContent = devStr;
        if (statParams) statParams.textContent = "1.54B / 500M / Custom";
        if (deviceLabel) deviceLabel.textContent = `${devStr} • 1.54B Flagship`;
      }
    } catch (e) {
      console.warn("Could not fetch models:", e);
    }
  }

  // Cloud Engine & API Key Settings Logic
  function openCloudKeyModal() {
    cloudKeyModal?.classList.remove("hidden");
    if (groqApiKeyInput) {
      groqApiKeyInput.value = userGroqKey;
      setTimeout(() => groqApiKeyInput.focus(), 50);
    }
    checkCloudStatus();
  }

  function closeCloudKeyModal() {
    cloudKeyModal?.classList.add("hidden");
  }

  async function checkCloudStatus() {
    try {
      const res = await fetch("/api/cloud/status");
      if (res.ok) {
        const data = await res.json();
        const hasKey = !!userGroqKey || !!data.has_server_api_key;
        if (hasKey) {
          if (cloudKeyBtnText) cloudKeyBtnText.textContent = "Cloud ⚡ (Ready)";
          openCloudKeyBtn?.classList.add("has-key");
          if (cloudStatusText) cloudStatusText.textContent = "Cloud Active (450+ tok/s ready)";
          const dot = cloudStatusIndicator?.querySelector(".status-dot-pulse");
          if (dot) dot.classList.add("active");
        } else {
          if (cloudKeyBtnText) cloudKeyBtnText.textContent = "Cloud ⚡";
          openCloudKeyBtn?.classList.remove("has-key");
          if (cloudStatusText) cloudStatusText.textContent = "API Key Needed for Cloud";
          const dot = cloudStatusIndicator?.querySelector(".status-dot-pulse");
          if (dot) dot.classList.remove("active");
        }
      }
    } catch (e) {
      console.warn("Could not check cloud status:", e);
    }
  }

  async function saveCloudKey() {
    const key = groqApiKeyInput ? groqApiKeyInput.value.trim() : "";
    if (!key) {
      alert("Please enter a valid Groq API Key (starts with gsk_...)");
      return;
    }

    userGroqKey = key;
    localStorage.setItem("el_gpt_groq_key", key);

    // Save to server .env if possible
    try {
      await fetch("/api/cloud/key", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api_key: key }),
      });
    } catch (e) {
      console.warn("Could not save key to server:", e);
    }

    checkCloudStatus();
    closeCloudKeyModal();
  }

  function clearCloudKey() {
    userGroqKey = "";
    localStorage.removeItem("el_gpt_groq_key");
    if (groqApiKeyInput) groqApiKeyInput.value = "";
    checkCloudStatus();
  }

  function initCloudSettings() {
    if (groqApiKeyInput && userGroqKey) {
      groqApiKeyInput.value = userGroqKey;
    }
    checkCloudStatus();
  }

  // Memory & Preferences Modal Logic
  function openMemoryModal() {
    memoryModal?.classList.remove("hidden");
    loadMemories();
    setTimeout(() => newMemoryInput?.focus(), 50);
  }

  function closeMemoryModal() {
    memoryModal?.classList.add("hidden");
  }

  async function loadMemories() {
    try {
      const res = await fetch("/api/memory");
      if (!res.ok) return;
      const data = await res.json();
      const memories = data.memories || [];

      if (memoryCount) memoryCount.textContent = memories.length;
      if (memoryCountSubtitle) {
        memoryCountSubtitle.textContent = `${memories.length} saved preference${memories.length === 1 ? "" : "s"}`;
      }

      if (!memoryItemsList) return;
      memoryItemsList.innerHTML = "";

      if (memories.length === 0) {
        memoryItemsList.innerHTML = `<div class="memory-empty">No preferences stored yet. Add one above or tell El GPT 1.8 Ultra in chat!</div>`;
        return;
      }

      memories.forEach((mem, idx) => {
        const item = document.createElement("div");
        item.className = "memory-item-card";
        item.innerHTML = `
          <div class="memory-item-content">
            <span class="memory-dot"></span>
            <span class="memory-item-text">${escapeHtml(mem)}</span>
          </div>
          <button class="memory-del-btn" title="Delete preference" data-idx="${idx}">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        `;

        item.querySelector(".memory-del-btn")?.addEventListener("click", () => deleteMemory(idx));
        memoryItemsList.appendChild(item);
      });
    } catch (e) {
      console.warn("Could not load memories:", e);
    }
  }

  async function addMemory() {
    const text = newMemoryInput?.value.trim();
    if (!text) return;

    try {
      const res = await fetch("/api/memory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memory: text }),
      });
      if (res.ok) {
        newMemoryInput.value = "";
        await loadMemories();
      }
    } catch (e) {
      console.error("Could not add memory:", e);
    }
  }

  async function deleteMemory(idx) {
    try {
      const res = await fetch(`/api/memory/${idx}`, { method: "DELETE" });
      if (res.ok) {
        await loadMemories();
      }
    } catch (e) {
      console.error("Could not delete memory:", e);
    }
  }

  async function clearAllMemories() {
    if (!confirm("Are you sure you want to clear all stored preferences?")) return;
    try {
      const res = await fetch("/api/memory/clear", { method: "POST" });
      if (res.ok) {
        await loadMemories();
      }
    } catch (e) {
      console.error("Could not clear memories:", e);
    }
  }

  // Training Studio Modal Logic
  function openStudio() {
    studioModal?.classList.remove("hidden");
    fetchModelMetadata();
    fetchTrainingStatus();
    startPollingStatus();
  }

  function closeStudio() {
    studioModal?.classList.add("hidden");
    if (!pollInterval) return;
    clearInterval(pollInterval);
    pollInterval = null;
  }

  function startPollingStatus() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchTrainingStatus, 1000);
  }

  async function fetchTrainingStatus() {
    try {
      const res = await fetch("/api/train/status");
      if (res.ok) {
        const data = await res.json();
        if (statLoss) statLoss.textContent = data.loss ? data.loss.toFixed(4) : "--";
        if (statSpeed) statSpeed.textContent = data.speed_tok_s ? `${data.speed_tok_s} tok/s` : "-- tok/s";
        if (chartStep) chartStep.textContent = `Step ${data.step || 0} / ${data.total_steps || 0} (Epoch ${data.epoch || 0}/${data.total_epochs || 0})`;
        if (trainingStatusBar) trainingStatusBar.textContent = `Status: ${data.status_message}`;

        drawLossChart(data.loss_history || []);

        if (data.is_training) {
          startTrainBtn.disabled = true;
          trainSpinner?.classList.remove("hidden");
          trainBtnText.textContent = "Training in Progress...";
          stopTrainBtn?.classList.remove("hidden");
        } else {
          startTrainBtn.disabled = false;
          trainSpinner?.classList.add("hidden");
          trainBtnText.textContent = "Launch Training Run";
          stopTrainBtn?.classList.add("hidden");
        }
      }
    } catch (e) {
      console.warn("Could not poll status:", e);
    }
  }

  async function launchTraining() {
    const epochs = parseInt(inputEpochs.value) || 5;
    const batchSize = parseInt(inputBatchSize.value) || 4;
    const lr = parseFloat(inputLr.value) || 0.0005;
    const scale = inputScale ? inputScale.value : "1b";
    const trainFile = inputDataset ? inputDataset.value : "data/train_ultra_1b.jsonl";

    startTrainBtn.disabled = true;
    trainSpinner?.classList.remove("hidden");
    trainBtnText.textContent = "Starting...";

    try {
      const res = await fetch("/api/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          epochs,
          batch_size: batchSize,
          learning_rate: lr,
          scale: scale,
          train_file: trainFile,
        }),
      });
      const data = await res.json();
      trainingStatusBar.textContent = data.message;
    } catch (e) {
      trainingStatusBar.textContent = "Failed to initiate training: " + e.message;
      startTrainBtn.disabled = false;
      trainSpinner?.classList.add("hidden");
      trainBtnText.textContent = "Launch Training Run";
    }
  }

  async function haltTraining() {
    try {
      await fetch("/api/train/stop", { method: "POST" });
      trainingStatusBar.textContent = "Stop requested by user...";
    } catch (e) {
      console.error(e);
    }
  }

  function drawLossChart(history) {
    if (!lossCanvas || !canvasCtx) return;
    const w = lossCanvas.width;
    const h = lossCanvas.height;
    canvasCtx.clearRect(0, 0, w, h);

    // Background grid
    canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    canvasCtx.lineWidth = 1;
    for (let y = 30; y < h; y += 40) {
      canvasCtx.beginPath();
      canvasCtx.moveTo(0, y);
      canvasCtx.lineTo(w, y);
      canvasCtx.stroke();
    }

    if (!history || history.length < 2) {
      canvasCtx.fillStyle = "rgba(255, 255, 255, 0.3)";
      canvasCtx.font = "13px sans-serif";
      canvasCtx.textAlign = "center";
      canvasCtx.fillText("Training loss curve will appear here once training begins", w / 2, h / 2);
      return;
    }

    const losses = history.map((item) => (typeof item === "number" ? item : item.loss));
    const minLoss = Math.min(...losses) * 0.95;
    const maxLoss = Math.max(...losses) * 1.05;
    const range = Math.max(maxLoss - minLoss, 0.01);

    const padding = 20;
    const chartW = w - padding * 2;
    const chartH = h - padding * 2;

    const points = losses.map((loss, idx) => {
      const x = padding + (idx / (losses.length - 1)) * chartW;
      const y = h - padding - ((loss - minLoss) / range) * chartH;
      return { x, y };
    });

    // Draw curve
    canvasCtx.beginPath();
    canvasCtx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1];
      const curr = points[i];
      const mx = (prev.x + curr.x) / 2;
      canvasCtx.quadraticCurveTo(prev.x, prev.y, mx, (prev.y + curr.y) / 2);
    }
    canvasCtx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    canvasCtx.strokeStyle = "#8b5cf6";
    canvasCtx.lineWidth = 2.5;
    canvasCtx.stroke();

    // Draw gradient fill beneath curve
    canvasCtx.lineTo(points[points.length - 1].x, h - padding);
    canvasCtx.lineTo(points[0].x, h - padding);
    canvasCtx.closePath();
    const grad = canvasCtx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(139, 92, 246, 0.35)");
    grad.addColorStop(1, "rgba(139, 92, 246, 0.0)");
    canvasCtx.fillStyle = grad;
    canvasCtx.fill();

    // Min / Max labels
    canvasCtx.fillStyle = "#8e8ea0";
    canvasCtx.font = "10px monospace";
    canvasCtx.textAlign = "left";
    canvasCtx.fillText(`Max: ${maxLoss.toFixed(3)}`, padding, padding - 4);
    canvasCtx.fillText(`Min: ${minLoss.toFixed(3)}`, padding, h - 4);
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
