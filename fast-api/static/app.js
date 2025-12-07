// Helper: get checked radio value by group (name attribute)
function getCheckedValue(groupName) {
  const el = document.querySelector(`input[name="${groupName}"]:checked`);
  return el ? el.value : null;
}

// Helper: setup slider value label
function setupHumiditySlider(sliderId, labelId) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);

  if (!slider || !label) return;

  // Initial value
  label.textContent = slider.value;

  // Update on slide
  slider.addEventListener("input", () => {
    label.textContent = slider.value;
  });
}

// Handle form submit and send prediction request
async function handleFormSubmit(event) {
  event.preventDefault();

  const resultEl = document.getElementById("result");
  resultEl.textContent = "Loading...";

  // Date parsing: get Year / Month / Day from date input
  const dateInput = document.getElementById("ObsDate");
  let year, month, day;

  if (dateInput && dateInput.value) {
    const parts = dateInput.value.split("-"); // "YYYY-MM-DD"
    if (parts.length === 3) {
      year = parseInt(parts[0], 10);
      month = parseInt(parts[1], 10);
      day = parseInt(parts[2], 10);
    }
  }

  // Fallback to today if date is not set or invalid
  if (!year || !month || !day) {
    const now = new Date();
    year = now.getFullYear();
    month = now.getMonth() + 1; // 0-based
    day = now.getDate();
  }

  const payload = {
    model_name: getCheckedValue("model"),
    Location: getCheckedValue("location"),

    // For now wind directions are hardcoded; you can add UI later
    WindGustDir: "N",
    WindDir9am: "N",
    WindDir3pm: "N",

    MinTemp: parseFloat(document.getElementById("MinTemp").value),
    MaxTemp: parseFloat(document.getElementById("MaxTemp").value),
    Rainfall: parseFloat(document.getElementById("Rainfall").value),
    WindGustSpeed: parseFloat(document.getElementById("WindGustSpeed").value),

    Humidity9am: parseInt(document.getElementById("Humidity9am").value, 10),
    Humidity3pm: parseInt(document.getElementById("Humidity3pm").value, 10),

    Pressure9am: parseFloat(document.getElementById("Pressure9am").value),
    Pressure3pm: parseFloat(document.getElementById("Pressure3pm").value),

    Temp9am: parseFloat(document.getElementById("Temp9am").value),
    Temp3pm: parseFloat(document.getElementById("Temp3pm").value),

    WindSpeed9am: parseFloat(document.getElementById("WindSpeed9am").value),
    WindSpeed3pm: parseFloat(document.getElementById("WindSpeed3pm").value),

    RainToday: document.getElementById("RainToday").checked,

    Year: year,
    Month: month,
    Day: day
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (!res.ok) {
      resultEl.textContent = "Error: " + JSON.stringify(data, null, 2);
      return;
    }

    resultEl.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    resultEl.textContent = "Error: " + err;
  }
}

// Attach listeners when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  form.addEventListener("submit", handleFormSubmit);

  // Setup sliders labels
  setupHumiditySlider("Humidity9am", "Humidity9amValue");
  setupHumiditySlider("Humidity3pm", "Humidity3pmValue");
});
