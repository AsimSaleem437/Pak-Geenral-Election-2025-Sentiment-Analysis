async function predict() {
  const text = document.getElementById("inputText").value;

  // Validate input
  if (!text) {
      alert("Please enter some text.");
      return;
  }

  try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          headers: {
              "Content-Type": "application/json"
          },
          body: JSON.stringify({ text: text })
      });

      if (!response.ok) {
          throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      document.getElementById("partyResult").textContent = data.party;
      document.getElementById("sentimentResult").textContent = data.sentiment;

  } catch (error) {
      console.error("Error:", error);
      alert("There was an error processing your request.");
  }
}