document
  .getElementById("fetchResultsBtn")
  .addEventListener("click", async function () {
    try {
      document.getElementById("loadingIndicator").style.display = "block";
      document.getElementById("resultsDescription").style.display = "block"; // Show description

      const response = await fetch("/post-election-analysis/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });

      document.getElementById("loadingIndicator").style.display = "none";
      document.getElementById("resultsDescription").style.display = "none"; // Hide description

      if (!response.ok) {
        console.error("Response not OK:", response);
        throw new Error(`Error fetching data: ${response.statusText}`);
      }

      const data = await response.json();
      console.log("Fetched Data:", data);
      displayResults(data);

      // Reveal the results section
      document.getElementById("results").classList.remove("hidden");
    } catch (error) {
      document.getElementById("loadingIndicator").style.display = "none";
      document.getElementById("resultsDescription").style.display = "none"; // Hide description
      console.error("Fetch error:", error);
      alert(
        "An error occurred while fetching election analysis. Please try again later."
      );
    }
  });

function displayResults(data) {
  const resultsTable = document.getElementById("resultsTable");
  const plotsContainer = document.getElementById("plots");

  // Clear previous results
  resultsTable.innerHTML = "";
  plotsContainer.innerHTML = "";

  // Dynamic CSS for images
  const style = document.createElement("style");
  style.innerHTML = `
        .image-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .plot-image {
            border: 2px solid black;
            margin: 10px;
            max-width: 48%;
            height: auto;
        }
    `;
  document.head.appendChild(style);

  // Display tweets
  if (data.tweets && data.tweets.length > 0) {
    const headers = Object.keys(data.tweets[0]);
    const headerRow = document.createElement("tr");
    headers.forEach((header) => {
      const th = document.createElement("th");
      th.innerText = header;
      headerRow.appendChild(th);
    });
    resultsTable.appendChild(headerRow);

    data.tweets.forEach((tweet) => {
      const tweetRow = document.createElement("tr");
      headers.forEach((header) => {
        const td = document.createElement("td");
        td.innerText = tweet[header];
        tweetRow.appendChild(td);
      });
      resultsTable.appendChild(tweetRow);
    });
  } else {
    const noDataRow = document.createElement("tr");
    const noDataCell = document.createElement("td");
    noDataCell.colSpan = 3; // Adjust based on the number of headers
    noDataCell.innerText = "No tweets available.";
    noDataRow.appendChild(noDataCell);
    resultsTable.appendChild(noDataRow);
  }

  // Display plots
  if (data.plots) {
    Object.values(data.plots).forEach((imageBase64) => {
      const img = document.createElement("img");
      img.src = `data:image/png;base64,${imageBase64}`;
      img.className = "plot-image";
      plotsContainer.appendChild(img);
    });
  }
}

// Pre-post comparison
// Wait for the DOM to fully load
document.addEventListener("DOMContentLoaded", () => {
  // Load Plotly dynamically if not already loaded
  if (!window.Plotly) {
    const script = document.createElement("script");
    script.src = "https://cdn.plot.ly/plotly-latest.min.js";
    script.onload = initializeApp;
    document.head.appendChild(script);
  } else {
    initializeApp();
  }
});

function initializeApp() {
  const fetchSentimentComparison = async () => {
    const loadingMessage = document.getElementById("loading-indicator");
    const errorMessage = document.getElementById("main-error-container");
    const visualizationContainer = document.getElementById(
      "visualization-container"
    );

    // Clear previous content
    visualizationContainer.innerHTML = "";
    errorMessage.textContent = "";

    // Show loading message
    loadingMessage.style.display = "block";

    try {
      const response = await fetch("/compare-pre-post-election", {
        method: "POST",
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Create image elements and display side by side
      const imgData = [
        { id: "overall-sentiment", src: data.overall_sentiment },
        { id: "sentiment-by-party", src: data.sentiment_by_party },
        { id: "pre-trend", src: data.pre_trend },
        { id: "post-trend", src: data.post_trend },
      ];

      imgData.forEach((imgInfo) => {
        if (imgInfo.src) {
          const imgWrapper = document.createElement("div");
          imgWrapper.className = "img-wrapper";

          const img = document.createElement("img");
          img.src = `data:image/png;base64,${imgInfo.src}`;
          img.alt = `${imgInfo.id} visualization`;
          img.className = "visualization-image";

          imgWrapper.appendChild(img);
          visualizationContainer.appendChild(imgWrapper);
        }
      });

      // Ensure elements for Plotly plots are created
      const preSunburstDiv = document.createElement("div");
      preSunburstDiv.id = "pre-sunburst";
      visualizationContainer.appendChild(preSunburstDiv);

      const postSunburstDiv = document.createElement("div");
      postSunburstDiv.id = "post-sunburst";
      visualizationContainer.appendChild(postSunburstDiv);

      // Process and display the sunburst plots
      if (data.pre_sunburst) {
        const preData = JSON.parse(data.pre_sunburst);
        console.log("Pre-election sunburst data:", preData);
        Plotly.newPlot("pre-sunburst", preData.data, preData.layout);
      }

      if (data.post_sunburst) {
        const postData = JSON.parse(data.post_sunburst);
        console.log("Post-election sunburst data:", postData);
        Plotly.newPlot("post-sunburst", postData.data, postData.layout);
      }
    } catch (error) {
      console.error("Error fetching sentiment comparison data:", error);
      errorMessage.textContent = `Error: ${error.message}`;
    } finally {
      // Hide loading message
      loadingMessage.style.display = "none";
    }
  };
  // Expose the function globally to be called from HTML
  window.fetchSentimentComparison = fetchSentimentComparison;
}
//constituency level analysis
async function fetchConstituencyLevelAnalysis() {
  try {
    const response = await fetch("/constituency-level-analysis", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      // If you need to send any data with the request, include it in the body here
      // body: JSON.stringify({ /* your data */ })
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    displayAnalysisResults(data.plots);
  } catch (error) {
    console.error("Error fetching constituency level analysis:", error);
  }
}

function displayAnalysisResults(plots) {
  const gallery = document.getElementById("image-gallery");
  gallery.innerHTML = ""; // Clear the gallery

  plots.forEach((plot) => {
    const item = document.createElement("div");
    item.classList.add("image-item");

    const title = document.createElement("h3");
    title.textContent = `${plot.constituency} - ${plot.type.replace("_", " ")}`;

    const img = document.createElement("img");
    img.src = `data:image/png;base64,${plot.image}`;
    img.alt = `${plot.constituency} ${plot.type}`;

    item.appendChild(title);
    item.appendChild(img);
    gallery.appendChild(item);
  });
}

// Call the function to fetch and display analysis results
fetchConstituencyLevelAnalysis();

//prediction
document
  .getElementById("predictionForm")
  .addEventListener("submit", async (e) => {
    e.preventDefault(); // Prevent the default form submission behavior

    const text = document.getElementById("textInput").value; // Get the input text
    const resultContainer = document.getElementById("result"); // Get the element to display results

    try {
      // Send the text input to the FastAPI predict endpoint
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }), // Format the input text as JSON
      });

      if (!response.ok) {
        throw new Error("Prediction failed"); // Throw an error if response is not OK
      }

      const result = await response.json(); // Parse the JSON response
      const party = result.party; // Get the predicted party from the response
      const sentiment = result.sentiment; // Get the predicted sentiment from the response

      // Display the prediction result
      resultContainer.innerHTML = `
            <p>Predicted Party: ${party}</p>
            <p>Predicted Sentiment: ${sentiment}</p>
        `;
    } catch (error) {
      console.error("Error:", error); // Log the error to the console
      resultContainer.innerHTML = "<p>Error in prediction.</p>"; // Display an error message
    }
  });
