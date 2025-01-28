// Party Support Analysis
async function analyzePartySupport() {
  try {
    // Show loading indicators
    document.getElementById("partySupportResults").innerHTML =
      "<p>Loading...</p>";
    document.getElementById("partySupportPlots").innerHTML =
      "<p>Loading charts...</p>";

    const response = await fetch("http://127.0.0.1:8000/party_support", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
    });

    // Check if the response is okay
    if (!response.ok) {
      throw new Error(`Network response was not ok: ${response.statusText}`);
    }

    const result = await response.json();

    // Clear loading indicators
    document.getElementById("partySupportResults").innerHTML = "";
    document.getElementById("partySupportPlots").innerHTML = "";

    // Verify if result has the required data
    if (
      !result.party_counts ||
      !result.party_percentages ||
      !result.support_plot ||
      !result.support_percentage_plot
    ) {
      throw new Error("Incomplete data received from the server");
    }

    // Format the counts and percentages for display in table format
    const formatDataToTable = (data, title) => {
      let tableHTML = `<table class="results-table"><thead><tr><th>${title}</th><th>Value</th></tr></thead><tbody>`;
      Object.entries(data).forEach(([party, value]) => {
        tableHTML += `<tr><td>${party}</td><td>${
          typeof value === "number" ? value.toFixed(2) : value
        }</td></tr>`;
      });
      tableHTML += "</tbody></table>";
      return tableHTML;
    };

    // Display results in a formatted table
    document.getElementById("partySupportResults").innerHTML = `
            <div class="results-container">
                <div class="results-section">
                    <h3>Party Support Counts</h3>
                    ${formatDataToTable(result.party_counts, "Party")}
                </div>
                <div class="results-section">
                    <h3>Party Support Percentages</h3>
                    ${formatDataToTable(result.party_percentages, "Party")}
                </div>
            </div>
        `;

    // Clear and update plots container
    const plotsContainer = document.getElementById("partySupportPlots");
    plotsContainer.innerHTML = "";

    // Create a container for the images
    const imageRow = document.createElement("div");
    imageRow.style.display = "flex"; // Use flexbox for side-by-side layout
    imageRow.style.flexWrap = "wrap"; // Allow wrapping for more than two images
    imageRow.style.justifyContent = "space-between"; // Space out images evenly

    // Create and append plots if they are available
    const plotSources = [
      {
        src: result.support_plot,
        alt: "Number of Tweets Supporting Each Party",
      },
      {
        src: result.support_percentage_plot,
        alt: "Percentage of Tweets Supporting Each Party",
      },
    ];

    plotSources.forEach((plot) => {
      if (plot.src) {
        const imgContainer = document.createElement("div");
        imgContainer.style.margin = "10px"; // Add margin between images
        imgContainer.style.flex = "1 1 calc(50% - 20px)"; // Set flex-basis for two images per row

        const img = document.createElement("img");
        img.src = plot.src;
        img.alt = plot.alt;
        img.className = "support-plot";
        img.style.border = "2px solid #007BFF"; // Add a border
        img.style.borderRadius = "5px"; // Round the corners
        img.style.maxWidth = "100%"; // Ensure images fit nicely
        img.style.height = "auto"; // Maintain aspect ratio

        imgContainer.appendChild(img);
        imageRow.appendChild(imgContainer);
      } else {
        console.warn(`Missing plot: ${plot.alt}`);
      }
    });

    // Append image row to plots container
    plotsContainer.appendChild(imageRow);
  } catch (error) {
    console.error("Error during party support analysis:", error);

    // Display error message and clear loading indicators
    document.getElementById("partySupportResults").innerHTML = `
            <div class="error-message">
                Error: ${error.message}
            </div>
        `;
    document.getElementById("partySupportPlots").innerHTML = "";
  }
}
