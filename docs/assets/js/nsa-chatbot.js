/* NewSpace Analytics — Mini Chatbot Q&A para Leigos (100% Client-Side Assisted Q&A) */
(() => {
  'use strict';

  const NSA_CHATBOT_KNOWLEDGE = [
    {
      id: "origem-dados",
      question: "De onde vêm os dados deste painel?",
      answer: "O NewSpace Analytics utiliza mais de 122 mil registros de dados 100% reais e auditados de agências oficiais: NASA (bases JPL SBDB & NHATS), USSPACECOM NORAD e Harvard-Smithsonian. Todo esse volume é processado em alta velocidade pelo motor analítico Polars."
    },
    {
      id: "delta-v",
      question: "O que é Δv (Viabilidade Astrodinâmica)?",
      answer: "De forma simples, o Δv (Delta-v) mede o 'esforço de empurrão' ou a quantidade de combustível necessária para uma nave sair da Terra e alcançar o asteroide. Quanto menor o Δv e o tempo de viagem, mais barata e viável é a missão."
    },
    {
      id: "incerteza-occ",
      question: "O que significa Precisão Navegacional (OCC)?",
      answer: "OCC é o filtro de incerteza orbital. Ele mede o quão bem conhecemos a rota futura de um asteroide. O painel filtra apenas alvos com rotas altamente previsíveis e rastreáveis, evitando o risco de enviar uma sonda e 'perder' o alvo no espaço."
    },
    {
      id: "mineralogia-25m",
      question: "Por que buscar asteroides maiores que 25m?",
      answer: "Asteroides acima de 25 metros possuem volume útil para Aplicação Mineralógica. Neles é possível extrair água e voláteis (que servem para produzir combustível no próprio espaço) ou metais valiosos para uso industrial."
    },
    {
      id: "ranking-topsis",
      question: "Como funciona o Ranking TOPSIS (Top 20)?",
      answer: "O método matemático TOPSIS cruza múltiplos critérios em tempo real para ranquear os 20 melhores asteroides. Ele encontra o equilíbrio ideal entre menor gasto de energia (fácil de chegar) e maior volume de recursos (alto retorno)."
    },
    {
      id: "era-newspace",
      question: "O que é a Era NewSpace Exponencial (2015–2026)?",
      answer: "Na seção de Evolução da Era Espacial, mostramos como o setor mudou: antes dominado apenas por governos (era estatal clássica), entre 2015 e 2026 houve um salto exponencial no lançamento de satélites e objetos orbitais liderado por empresas privadas."
    },
    {
      id: "risco-atuarial",
      question: "Como funciona a Análise de Risco Atuarial e os Ratings?",
      answer: "Funciona como uma análise de seguro e investimento para missões espaciais. A matriz cruza um alvo (como o asteroide Apophis) com um foguete lançador (como o Falcon 9 ou Chang Zheng) e calcula uma nota de risco operacional (Rating A, BB, B, etc.) baseada no histórico real de sucesso."
    }
  ];

  const WELCOME_MESSAGE =
    "Olá! Sou o guia rápido do NewSpace Analytics. Clique em qualquer pergunta abaixo para entender os conceitos técnicos do painel de forma simples:";

  const askedIds = new Set();
  let showAllQuestions = false;
  let isOpen = false;
  let isInitialized = false;

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function getAvailableQuestions() {
    if (showAllQuestions || askedIds.size === 0 || askedIds.size >= NSA_CHATBOT_KNOWLEDGE.length) {
      return NSA_CHATBOT_KNOWLEDGE.slice();
    }
    return NSA_CHATBOT_KNOWLEDGE.filter(function (item) {
      return !askedIds.has(item.id);
    });
  }

  function renderChipsBlock(messagesEl) {
    if (!messagesEl) return;

    const existingBlock = document.getElementById('nsa-bot-chips-block');
    if (existingBlock && existingBlock.parentNode) {
      existingBlock.parentNode.removeChild(existingBlock);
    }

    const available = getAvailableQuestions();
    const allAnswered = askedIds.size >= NSA_CHATBOT_KNOWLEDGE.length;

    let labelText = 'Perguntas principais (clique para traduzir):';
    if (showAllQuestions && askedIds.size > 0) {
      labelText = 'Todas as perguntas do guia:';
    } else if (allAnswered) {
      labelText = 'Todos os tópicos explorados — clique para rever:';
    } else if (askedIds.size > 0) {
      labelText = 'Perguntas restantes (' + available.length + '):';
    }

    const block = document.createElement('div');
    block.className = 'nsa-bot-chips-block';
    block.id = 'nsa-bot-chips-block';

    const chipsHtml = available
      .map(function (item) {
        return (
          '<button type="button" class="nsa-bot-chip" data-question-id="' +
          escapeHtml(item.id) +
          '">' +
          '<span>' +
          escapeHtml(item.question) +
          '</span>' +
          '<span class="nsa-bot-chip-arrow" aria-hidden="true">&rarr;</span>' +
          '</button>'
        );
      })
      .join('');

    block.innerHTML =
      '<div class="nsa-bot-chips-label">' +
      escapeHtml(labelText) +
      '</div>' +
      '<div class="nsa-bot-chips-list">' +
      chipsHtml +
      '</div>';

    messagesEl.appendChild(block);

    const chipButtons = block.querySelectorAll('.nsa-bot-chip');
    chipButtons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        const qId = btn.getAttribute('data-question-id');
        askQuestionById(qId);
      });
    });
  }

  function appendMessage(messagesEl, role, text) {
    const msg = document.createElement('div');
    msg.className = 'nsa-bot-msg ' + (role === 'user' ? 'nsa-bot-msg-user' : 'nsa-bot-msg-bot');
    msg.textContent = text;
    messagesEl.appendChild(msg);
    return msg;
  }

  function renderInitialConversation() {
    const messagesEl = document.getElementById('nsa-bot-messages');
    if (!messagesEl) return;
    messagesEl.innerHTML = '';
    askedIds.clear();
    showAllQuestions = false;
    appendMessage(messagesEl, 'bot', WELCOME_MESSAGE);
    renderChipsBlock(messagesEl);
  }

  function askQuestionById(questionId) {
    const item = NSA_CHATBOT_KNOWLEDGE.find(function (entry) {
      return entry.id === questionId;
    });
    if (!item) return null;

    const messagesEl = document.getElementById('nsa-bot-messages');
    if (!messagesEl) return null;

    const existingBlock = document.getElementById('nsa-bot-chips-block');
    if (existingBlock && existingBlock.parentNode) {
      existingBlock.parentNode.removeChild(existingBlock);
    }

    askedIds.add(item.id);
    showAllQuestions = false;

    const userMsgEl = appendMessage(messagesEl, 'user', item.question);
    appendMessage(messagesEl, 'bot', item.answer);
    renderChipsBlock(messagesEl);

    if (typeof userMsgEl.scrollIntoView === 'function') {
      userMsgEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    return item;
  }

  function showAllTopics() {
    const messagesEl = document.getElementById('nsa-bot-messages');
    if (!messagesEl) return;
    showAllQuestions = true;
    renderChipsBlock(messagesEl);
    const block = document.getElementById('nsa-bot-chips-block');
    if (block && typeof block.scrollIntoView === 'function') {
      block.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else {
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }
  }

  function setWindowOpen(nextState) {
    isOpen = Boolean(nextState);
    const winEl = document.getElementById('nsa-bot-window');
    const launcherEl = document.getElementById('nsa-bot-launcher');
    if (winEl) {
      if (isOpen) {
        winEl.classList.add('nsa-bot-open');
      } else {
        winEl.classList.remove('nsa-bot-open');
      }
    }
    if (launcherEl) {
      launcherEl.setAttribute('aria-expanded', String(isOpen));
    }
  }

  function toggleWindow() {
    setWindowOpen(!isOpen);
  }

  function initChatbot() {
    if (typeof document === 'undefined' || !document.body) return;
    if (document.getElementById('nsa-bot-wrapper')) {
      isInitialized = true;
      return;
    }

    const wrapper = document.createElement('aside');
    wrapper.className = 'nsa-bot-wrapper';
    wrapper.id = 'nsa-bot-wrapper';
    wrapper.setAttribute('aria-label', 'Assistente de Perguntas e Respostas NewSpace');

    wrapper.innerHTML =
      '<div class="nsa-bot-window" id="nsa-bot-window" role="dialog" aria-modal="false" aria-labelledby="nsa-bot-title">' +
        '<div class="nsa-bot-header">' +
          '<div class="nsa-bot-header-info">' +
            '<h2 class="nsa-bot-title" id="nsa-bot-title">Assistente NewSpace (Guia para Leigos)</h2>' +
            '<p class="nsa-bot-subtitle">Tradução rápida das métricas do painel</p>' +
          '</div>' +
          '<button type="button" class="nsa-bot-close" id="nsa-bot-close" aria-label="Fechar assistente" title="Fechar">&times;</button>' +
        '</div>' +
        '<div class="nsa-bot-messages" id="nsa-bot-messages" aria-live="polite"></div>' +
        '<div class="nsa-bot-footer">' +
          '<button type="button" class="nsa-bot-footer-btn" id="nsa-bot-show-all">Ver todas as perguntas</button>' +
          '<button type="button" class="nsa-bot-footer-btn" id="nsa-bot-reset">Reiniciar guia</button>' +
        '</div>' +
      '</div>' +
      '<button type="button" class="nsa-bot-launcher" id="nsa-bot-launcher" aria-expanded="false" aria-controls="nsa-bot-window">' +
        '<span class="nsa-bot-launcher-dot" aria-hidden="true"></span>' +
        '<span class="nsa-bot-launcher-label">Guia Rápido • Dúvidas?</span>' +
      '</button>';

    document.body.appendChild(wrapper);
    renderInitialConversation();

    const launcherBtn = document.getElementById('nsa-bot-launcher');
    const closeBtn = document.getElementById('nsa-bot-close');
    const showAllBtn = document.getElementById('nsa-bot-show-all');
    const resetBtn = document.getElementById('nsa-bot-reset');

    if (launcherBtn) {
      launcherBtn.addEventListener('click', toggleWindow);
    }
    if (closeBtn) {
      closeBtn.addEventListener('click', function () {
        setWindowOpen(false);
      });
    }
    if (showAllBtn) {
      showAllBtn.addEventListener('click', showAllTopics);
    }
    if (resetBtn) {
      resetBtn.addEventListener('click', renderInitialConversation);
    }

    document.addEventListener('keydown', function (ev) {
      if (ev && ev.key === 'Escape' && isOpen) {
        setWindowOpen(false);
      }
    });

    isInitialized = true;
  }

  const globalRef = typeof window !== 'undefined' ? window : globalThis;
  globalRef.NSA_CHATBOT_KNOWLEDGE = NSA_CHATBOT_KNOWLEDGE;
  globalRef.NSAChatbot = {
    KNOWLEDGE: NSA_CHATBOT_KNOWLEDGE,
    WELCOME_MESSAGE: WELCOME_MESSAGE,
    init: initChatbot,
    open: function () { setWindowOpen(true); },
    close: function () { setWindowOpen(false); },
    toggle: toggleWindow,
    ask: askQuestionById,
    showAll: showAllTopics,
    reset: renderInitialConversation,
    getAvailableQuestions: getAvailableQuestions,
    isInitialized: function () { return isInitialized; }
  };

  if (typeof document !== 'undefined') {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', initChatbot);
    } else {
      initChatbot();
    }
  }
})();
