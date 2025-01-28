// Pre-post comparison
// Wait for the DOM to fully load
document.addEventListener("DOMContentLoaded", () => {
  // Load Plotly dynamically if not already loaded
  if (!window.Plotly) {
      const script = document.createElement('script');
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
      const visualizationContainer = document.getElementById("visualization-container");

      // Clear previous content
      visualizationContainer.innerHTML = "";
      errorMessage.textContent = "";

      // Show loading message
      loadingMessage.style.display = "block";

      try {
          const response = await fetch("/compare-pre-post-election", {
              method: "POST"
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
              { id: "post-trend", src: data.post_trend }
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