async function runSentimentAnalysis() {
  try {
    // Show loading indicators
    document.getElementById("sentimentAnalysisResults").innerHTML =
      "<p>Loading results...</p>";
    document.getElementById("sentimentPlotContainer").innerHTML =
      "<p>Loading overall sentiment distribution plot...</p>";
    document.getElementById("partySentimentPlotContainer").innerHTML =
      "<p>Loading sentiment distribution by party plot...</p>";
    document.getElementById("tweetVolumeContainer").innerHTML =
      "<p>Loading tweet volume over time plot...</p>";
    document.getElementById("trendOverTimeContainer").innerHTML =
      "<p>Loading sentiment trend over time plot...</p>";
    document.getElementById("stackedBarContainer").innerHTML =
      "<p>Loading stacked bar chart...</p>";
    document.getElementById("sunburstPlotContainer").innerHTML =
      "<p>Loading sunburst plot...</p>";
    document.getElementById("wordcloudContainer").innerHTML =
      "<p>Loading word clouds...</p>";

    const response = await fetch("/sentiment_analysis", { method: "POST" });

    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }

    const data = await response.json();

    // Clear loading indicators
    document.getElementById("sentimentAnalysisResults").innerHTML = "";
    document.getElementById("sentimentPlotContainer").innerHTML = "";
    document.getElementById("partySentimentPlotContainer").innerHTML = "";
    document.getElementById("tweetVolumeContainer").innerHTML = "";
    document.getElementById("trendOverTimeContainer").innerHTML = "";
    document.getElementById("stackedBarContainer").innerHTML = "";
    document.getElementById("sunburstPlotContainer").innerHTML = "";
    document.getElementById("wordcloudContainer").innerHTML = "";

    // Helper function to create image elements
    function createImageElement(src, altText, titleText) {
      const div = document.createElement("div");
      div.style.flex = "1"; // Allow the image to take equal space
      div.style.display = "flex"; // Use flexbox for layout
      div.style.flexDirection = "column"; // Stack title under the image
      div.style.alignItems = "center"; // Center align items
      div.style.margin = "10px"; // Add some margin between images

      const img = document.createElement("img");
      img.src = `data:image/png;base64,${src}`;
      img.alt = altText;
      img.style.border = "2px solid #007BFF"; // Add a border
      img.style.borderRadius = "5px"; // Round the corners
      img.style.maxWidth = "90%"; // Ensure images fit nicely
      img.style.height = "auto"; // Maintain aspect ratio

      const title = document.createElement("div");
      title.style.textAlign = "center"; // Center the title
      title.style.fontWeight = "bold"; // Make the title bold
      title.innerText = titleText;

      div.appendChild(img);
      div.appendChild(title);
      return div;
    }

    // Display each plot if available
    const containers = {
      overall_sentiment_distribution: "sentimentPlotContainer",
      sentiment_distribution_by_party: "partySentimentPlotContainer",
      tweet_volume_over_time: "tweetVolumeContainer",
      sentiment_trend_over_time: "trendOverTimeContainer",
      interactive_stacked_bar_chart: "stackedBarContainer",
      sunburst_chart: "sunburstPlotContainer",
    };

    // Create a container for image rows
    for (const [plotName, containerId] of Object.entries(containers)) {
      const container = document.getElementById(containerId);
      const imageRow = document.createElement("div");
      imageRow.style.display = "flex"; // Use flexbox for side-by-side layout
      imageRow.style.justifyContent = "space-around"; // Space out images evenly

      if (data[plotName]) {
        const plotElement = createImageElement(
          data[plotName],
          plotName.replace(/_/g, " "),
          plotName
            .replace(/_/g, " ")
            .replace(/\b\w/g, (char) => char.toUpperCase())
        );
        imageRow.appendChild(plotElement);
      } else {
        const noDataMessage = document.createElement("p");
        noDataMessage.innerText = `${plotName.replace(
          /_/g,
          " "
        )} not available.`;
        console.warn(`${plotName} data is missing or invalid.`);
        imageRow.appendChild(noDataMessage);
      }

      // Append image row to container
      container.appendChild(imageRow);
    }

    // Display word clouds if available
    if (data.party_wordclouds) {
      const wordcloudContainer = document.getElementById("wordcloudContainer");
      const wordcloudRow = document.createElement("div");
      wordcloudRow.style.display = "flex"; // Use flexbox for side-by-side layout
      wordcloudRow.style.justifyContent = "space-around"; // Space out word clouds evenly

      for (const [party, wordcloudBase64] of Object.entries(
        data.party_wordclouds
      )) {
        const wordcloudElement = createImageElement(
          wordcloudBase64,
          `Word Cloud for ${party}`,
          `Word Cloud for ${party}`
        );
        wordcloudRow.appendChild(wordcloudElement);
      }

      wordcloudContainer.appendChild(wordcloudRow);
    } else {
      document.getElementById("wordcloudContainer").innerHTML =
        "<p>Word clouds not available.</p>";
      console.warn("Party word clouds not available.");
    }

    // Display any backend error message if present
    if (data.error) {
      document.getElementById(
        "sentimentAnalysisResults"
      ).innerText = `Error: ${data.error}`;
    }
  } catch (error) {
    console.error("Error during sentiment analysis:", error);

    // Display error message and clear loading indicators
    document.getElementById("sentimentAnalysisResults").innerHTML = `
            <div class="error-message">
                Error: ${error.message}
            </div>
        `;
    document.getElementById("sentimentPlotContainer").innerHTML = "";
    document.getElementById("partySentimentPlotContainer").innerHTML = "";
    document.getElementById("tweetVolumeContainer").innerHTML = "";
    document.getElementById("trendOverTimeContainer").innerHTML = "";
    document.getElementById("stackedBarContainer").innerHTML = "";
    document.getElementById("sunburstPlotContainer").innerHTML = "";
    document.getElementById("wordcloudContainer").innerHTML = "";
  }
}
