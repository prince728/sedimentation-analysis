(() => {
  "use strict";

  // ============================================================
  // API CONFIGURATION
  // ============================================================
  //
  // If frontend is opened using VS Code Live Server:
  //     http://127.0.0.1:5500
  //
  // send API requests to FastAPI:
  //     http://127.0.0.1:8000
  //
  // If frontend is served directly by FastAPI:
  //     http://127.0.0.1:8000
  //
  // then use relative URLs such as /health and /predict.
  // ============================================================

  const API_BASE_URL =
    window.location.port === "5500"
      ? "http://127.0.0.1:8000"
      : "";


  // ============================================================
  // EMOTION EMOJIS
  // ============================================================

  const EMOJI = {
    sadness: "😢",
    joy: "😄",
    love: "❤️",
    anger: "😠",
    fear: "😨",
    surprise: "😲",
  };


  // ============================================================
  // DOM ELEMENTS
  // ============================================================

  const el = {
    statusDot:
      document.getElementById("statusDot"),

    serverStatusText:
      document.getElementById("serverStatusText"),

    textInput:
      document.getElementById("textInput"),

    charCount:
      document.getElementById("charCount"),

    analyzeBtn:
      document.getElementById("analyzeBtn"),

    errorMsg:
      document.getElementById("errorMsg"),

    orb:
      document.getElementById("orb"),

    orbEmoji:
      document.getElementById("orbEmoji"),

    resultSection:
      document.getElementById("resultSection"),

    emotionWord:
      document.getElementById("emotionWord"),

    emotionEmoji:
      document.getElementById("emotionEmoji"),

    confidenceText:
      document.getElementById("confidenceText"),

    echoedText:
      document.getElementById("echoedText"),

    barsContainer:
      document.getElementById("barsContainer"),
  };


  // ============================================================
  // CHECK REQUIRED ELEMENTS
  // ============================================================

  const missingElements = Object.entries(el)
    .filter(([, element]) => !element)
    .map(([name]) => name);

  if (missingElements.length > 0) {
    console.error(
      "Moodline: Missing HTML elements:",
      missingElements
    );

    return;
  }


  // ============================================================
  // STATE
  // ============================================================

  let modelReady = false;
  let healthCheckTimer = null;


  // ============================================================
  // HEALTH CHECK
  // ============================================================

  async function checkHealth() {
    try {
      const response = await fetch(
        `${API_BASE_URL}/health`,
        {
          method: "GET",
          cache: "no-store",
        }
      );

      if (!response.ok) {
        throw new Error(
          `Health check failed (${response.status})`
        );
      }

      const data = await response.json();

      console.log("Health response:", data);

      modelReady = data.model_loaded === true;

      if (modelReady) {
        setStatus(
          "live",
          "model ready — say something"
        );

        clearHealthTimer();

      } else {
        setStatus(
          "warming",
          "waking the model up…"
        );

        scheduleHealthCheck(3000);
      }

    } catch (error) {
      console.error(
        "Health check failed:",
        error
      );

      modelReady = false;

      setStatus(
        "down",
        "can't reach the server"
      );

      scheduleHealthCheck(5000);
    }

    syncButtonState();
  }


  // ============================================================
  // HEALTH CHECK TIMER
  // ============================================================

  function scheduleHealthCheck(delay) {
    clearHealthTimer();

    healthCheckTimer = setTimeout(
      checkHealth,
      delay
    );
  }


  function clearHealthTimer() {
    if (healthCheckTimer !== null) {
      clearTimeout(healthCheckTimer);
      healthCheckTimer = null;
    }
  }


  // ============================================================
  // UPDATE SERVER STATUS
  // ============================================================

  function setStatus(kind, text) {
    el.statusDot.className =
      `brand-mark ${kind}`;

    el.serverStatusText.textContent =
      text;
  }


  // ============================================================
  // INPUT HANDLING
  // ============================================================

  el.textInput.addEventListener(
    "input",
    () => {
      const length =
        el.textInput.value.length;

      el.charCount.textContent =
        length;

      syncButtonState();

      if (length > 0) {
        hideError();
      }
    }
  );


  // ============================================================
  // CTRL + ENTER / CMD + ENTER
  // ============================================================

  el.textInput.addEventListener(
    "keydown",
    (event) => {
      if (
        (event.ctrlKey || event.metaKey) &&
        event.key === "Enter"
      ) {
        event.preventDefault();

        if (!el.analyzeBtn.disabled) {
          runAnalysis();
        }
      }
    }
  );


  // ============================================================
  // ENABLE / DISABLE ANALYZE BUTTON
  // ============================================================

  function syncButtonState() {
    const hasText =
      el.textInput.value.trim().length > 0;

    el.analyzeBtn.disabled =
      !hasText || !modelReady;
  }


  // ============================================================
  // ANALYZE BUTTON
  // ============================================================

  el.analyzeBtn.addEventListener(
    "click",
    runAnalysis
  );


  // ============================================================
  // ANALYSIS
  // ============================================================

  async function runAnalysis() {
    const text =
      el.textInput.value.trim();


    // ----------------------------------------------------------
    // EMPTY TEXT
    // ----------------------------------------------------------

    if (!text) {
      showError(
        "Please enter some text first."
      );

      return;
    }


    // ----------------------------------------------------------
    // MODEL NOT READY
    // ----------------------------------------------------------

    if (!modelReady) {
      showError(
        "The model is not ready yet. Please wait."
      );

      checkHealth();

      return;
    }


    // ----------------------------------------------------------
    // CLEAR PREVIOUS ERROR
    // ----------------------------------------------------------

    hideError();


    // ----------------------------------------------------------
    // SHOW LOADING STATE
    // ----------------------------------------------------------

    enterThinking();


    try {
      // --------------------------------------------------------
      // SEND REQUEST TO FASTAPI
      // --------------------------------------------------------

      console.log(
        "Sending prediction request..."
      );

      console.log(
        "API:",
        `${API_BASE_URL}/predict`
      );


      const response = await fetch(
        `${API_BASE_URL}/predict`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            text: text,
          }),
        }
      );


      // --------------------------------------------------------
      // READ RESPONSE
      // --------------------------------------------------------

      let data = {};

      try {
        data = await response.json();
      } catch (jsonError) {
        console.error(
          "Could not parse server response:",
          jsonError
        );
      }


      // --------------------------------------------------------
      // HANDLE HTTP ERROR
      // --------------------------------------------------------

      if (!response.ok) {
        let detail =
          `Request failed (${response.status}).`;

        if (data && data.detail) {
          if (
            typeof data.detail === "string"
          ) {
            detail = data.detail;
          } else {
            detail =
              "The model couldn't process that sentence.";
          }
        }

        throw new Error(detail);
      }


      // --------------------------------------------------------
      // LOG SERVER RESPONSE
      // --------------------------------------------------------

      console.log(
        "Prediction response:",
        data
      );


      // --------------------------------------------------------
      // VALIDATE RESPONSE
      // --------------------------------------------------------

      if (
        !data.predicted_emotion ||
        data.confidence === undefined ||
        !data.all_probabilities
      ) {
        throw new Error(
          "Invalid response received from the server."
        );
      }


      // --------------------------------------------------------
      // RENDER RESULT
      // --------------------------------------------------------

      renderResult(
        data,
        text
      );

    } catch (error) {
      console.error(
        "Prediction error:",
        error
      );

      exitThinking(false);

      showError(
        error.message ||
        "Something went wrong. Please try again."
      );
    }
  }


  // ============================================================
  // ENTER THINKING STATE
  // ============================================================

  function enterThinking() {
    el.analyzeBtn.classList.add(
      "loading"
    );

    const label =
      el.analyzeBtn.querySelector(
        ".btn-label"
      );

    if (label) {
      label.textContent =
        "Reading…";
    }

    el.analyzeBtn.disabled = true;

    el.orb.classList.remove(
      "settled"
    );

    el.orb.classList.add(
      "thinking"
    );

    el.orbEmoji.style.opacity =
      "0";
  }


  // ============================================================
  // EXIT THINKING STATE
  // ============================================================

  function exitThinking(success) {
    el.analyzeBtn.classList.remove(
      "loading"
    );

    const label =
      el.analyzeBtn.querySelector(
        ".btn-label"
      );

    if (label) {
      label.textContent =
        "Read the mood";
    }

    syncButtonState();

    el.orb.classList.remove(
      "thinking"
    );

    if (!success) {
      el.orbEmoji.textContent =
        "✎";

      el.orbEmoji.style.opacity =
        "1";
    }
  }


  // ============================================================
  // RENDER RESULT
  // ============================================================

  function renderResult(
    data,
    originalText
  ) {
    // ----------------------------------------------------------
    // GET EMOTION
    // ----------------------------------------------------------

    const emotion =
      data.predicted_emotion;


    // ----------------------------------------------------------
    // GET EMOJI
    // ----------------------------------------------------------

    const emoji =
      data.emoji ||
      EMOJI[emotion] ||
      "🙂";


    // ----------------------------------------------------------
    // UPDATE BODY EMOTION
    // ----------------------------------------------------------

    document.body.setAttribute(
      "data-emotion",
      emotion
    );


    // ----------------------------------------------------------
    // UPDATE ORB
    // ----------------------------------------------------------

    el.orb.classList.add(
      "settled"
    );

    el.orbEmoji.textContent =
      emoji;

    el.orbEmoji.style.opacity =
      "1";


    // ----------------------------------------------------------
    // EXIT LOADING STATE
    // ----------------------------------------------------------

    exitThinking(true);


    // ----------------------------------------------------------
    // UPDATE EMOTION TEXT
    // ----------------------------------------------------------

    el.emotionWord.textContent =
      capitalize(emotion);

    el.emotionEmoji.textContent =
      emoji;


    // ----------------------------------------------------------
    // UPDATE CONFIDENCE
    // ----------------------------------------------------------

    const confidence =
      Number(data.confidence);

    el.confidenceText.textContent =
      `${(confidence * 100).toFixed(1)}% confidence`;


    // ----------------------------------------------------------
    // DISPLAY ORIGINAL TEXT
    // ----------------------------------------------------------

    el.echoedText.textContent =
      `“${originalText}”`;


    // ----------------------------------------------------------
    // RENDER PROBABILITY BARS
    // ----------------------------------------------------------

    renderBars(
      data.all_probabilities
    );


    // ----------------------------------------------------------
    // SHOW RESULT SECTION
    // ----------------------------------------------------------

    el.resultSection.hidden =
      false;


    // Restart animation

    el.resultSection.classList.remove(
      "entering"
    );

    void el.resultSection.offsetWidth;

    el.resultSection.classList.add(
      "entering"
    );


    // ----------------------------------------------------------
    // SCROLL TO RESULT
    // ----------------------------------------------------------

    el.resultSection.scrollIntoView({
      behavior: "smooth",
      block: "nearest",
    });
  }


  // ============================================================
  // RENDER PROBABILITY BARS
  // ============================================================

  function renderBars(
    probabilities
  ) {
    // ----------------------------------------------------------
    // SAFETY CHECK
    // ----------------------------------------------------------

    if (
      !probabilities ||
      typeof probabilities !== "object"
    ) {
      console.error(
        "Invalid probabilities:",
        probabilities
      );

      return;
    }


    // ----------------------------------------------------------
    // CONVERT OBJECT TO ARRAY
    // ----------------------------------------------------------

    const entries =
      Object.entries(
        probabilities
      );


    // ----------------------------------------------------------
    // SORT HIGHEST PROBABILITY FIRST
    // ----------------------------------------------------------

    entries.sort(
      (a, b) =>
        Number(b[1]) -
        Number(a[1])
    );


    // ----------------------------------------------------------
    // CLEAR OLD BARS
    // ----------------------------------------------------------

    el.barsContainer.innerHTML =
      "";


    // ----------------------------------------------------------
    // CREATE BARS
    // ----------------------------------------------------------

    entries.forEach(
      ([label, value], index) => {
        const numericValue =
          Number(value);

        const percentage =
          Math.max(
            0,
            Math.min(
              100,
              numericValue * 100
            )
          );


        // ------------------------------------------------------
        // CREATE ROW
        // ------------------------------------------------------

        const row =
          document.createElement(
            "div"
          );

        row.className =
          `bar-row bar-${label}`;


        // ------------------------------------------------------
        // CREATE HTML
        // ------------------------------------------------------

        const labelElement =
          document.createElement(
            "span"
          );

        labelElement.className =
          "bar-label";

        labelElement.textContent =
          `${EMOJI[label] || ""} ${label}`;


        const track =
          document.createElement(
            "span"
          );

        track.className =
          "bar-track";


        const fill =
          document.createElement(
            "span"
          );

        fill.className =
          "bar-fill";


        const percentageElement =
          document.createElement(
            "span"
          );

        percentageElement.className =
          "bar-pct";

        percentageElement.textContent =
          `${percentage.toFixed(1)}%`;


        // ------------------------------------------------------
        // BUILD ROW
        // ------------------------------------------------------

        track.appendChild(fill);

        row.appendChild(
          labelElement
        );

        row.appendChild(
          track
        );

        row.appendChild(
          percentageElement
        );


        // ------------------------------------------------------
        // ADD ROW
        // ------------------------------------------------------

        el.barsContainer.appendChild(
          row
        );


        // ------------------------------------------------------
        // ANIMATE BAR
        // ------------------------------------------------------

        setTimeout(
          () => {
            fill.style.width =
              `${percentage}%`;
          },
          60 + index * 70
        );
      }
    );
  }


  // ============================================================
  // SHOW ERROR
  // ============================================================

  function showError(message) {
    el.errorMsg.textContent =
      message;

    el.errorMsg.hidden =
      false;
  }


  // ============================================================
  // HIDE ERROR
  // ============================================================

  function hideError() {
    el.errorMsg.hidden =
      true;

    el.errorMsg.textContent =
      "";
  }


  // ============================================================
  // CAPITALIZE EMOTION
  // ============================================================

  function capitalize(text) {
    if (!text) {
      return "";
    }

    return (
      text.charAt(0).toUpperCase() +
      text.slice(1)
    );
  }


  // ============================================================
  // INITIALIZE APPLICATION
  // ============================================================

  console.log(
    "Moodline initialized."
  );

  console.log(
    "API Base URL:",
    API_BASE_URL || "same origin"
  );

  checkHealth();

})();

