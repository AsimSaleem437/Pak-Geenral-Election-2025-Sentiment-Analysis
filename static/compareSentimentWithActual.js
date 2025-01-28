// Sentiment vs Actual Results
async function compareSentimentWithActual() {
  try {
    // Send a POST request to fetch the comparison data and plots
    const response = await fetch("http://127.0.0.1:8000/sentiment_vs_actual", {
      method: "POST",
    });

    if (!response.ok) {
      throw new Error("Failed to fetch sentiment vs actual comparison.");
    }

    // Parse the JSON response
    const result = await response.json();

    // Clear any previous results
    const resultContainer = document.getElementById("sentimentVsActualResults");
    resultContainer.innerHTML = "";

    // Display the comparison plot
    if (result.comparison_plot) {
      const comparisonImg = document.createElement("img");
      comparisonImg.src = `data:image/png;base64,${result.comparison_plot}`;
      comparisonImg.alt = "Comparison of Tweet Support and Actual Seats Won";
      comparisonImg.style.maxWidth = "100%";
      comparisonImg.style.border = "2px solid #007BFF"; // Add a border
      comparisonImg.style.borderRadius = "5px"; // Round the corners
      resultContainer.appendChild(comparisonImg);
    }

    // Display the percentage difference plot
    if (result.percentage_difference_plot) {
      const percentageDiffImg = document.createElement("img");
      percentageDiffImg.src = `data:image/png;base64,${result.percentage_difference_plot}`;
      percentageDiffImg.alt =
        "Percentage Difference between Tweet Support and Seats Won";
      percentageDiffImg.style.maxWidth = "100%";
      percentageDiffImg.style.border = "2px solid #007BFF"; // Add a border
      percentageDiffImg.style.borderRadius = "5px"; // Round the corners
      resultContainer.appendChild(percentageDiffImg);
    }

    // Display tweet counts in a table
    if (result.tweet_counts) {
      const tweetCountsTable = document.createElement("table");
      tweetCountsTable.className = "results-table";
      tweetCountsTable.innerHTML =
        "<thead><tr><th>Party</th><th>Tweet Count</th></tr></thead><tbody>";
      result.tweet_counts.forEach((count) => {
        tweetCountsTable.innerHTML += `<tr><td>${count.Party}</td><td>${count.Tweet_Count}</td></tr>`;
      });
      tweetCountsTable.innerHTML += "</tbody>";
      resultContainer.appendChild(tweetCountsTable);
    }

    // Display seat counts in a table
    if (result.seat_counts) {
      const seatCountsTable = document.createElement("table");
      seatCountsTable.className = "results-table";
      seatCountsTable.innerHTML =
        "<thead><tr><th>Party</th><th>Seats</th></tr></thead><tbody>";
      result.seat_counts.forEach((count) => {
        seatCountsTable.innerHTML += `<tr><td>${count.Party}</td><td>${count.Seats}</td></tr>`;
      });
      seatCountsTable.innerHTML += "</tbody>";
      resultContainer.appendChild(seatCountsTable);
    }

    // Display numerical differences in a table
    if (result.numerical_differences) {
      const numericalDiffTable = document.createElement("table");
      numericalDiffTable.className = "results-table";
      numericalDiffTable.innerHTML =
        "<thead><tr><th>Party</th><th>Numerical Difference</th></tr></thead><tbody>";
      result.numerical_differences.forEach((diff) => {
        numericalDiffTable.innerHTML += `<tr><td>${diff.Party}</td><td>${diff.Numerical_Difference}</td></tr>`;
      });
      numericalDiffTable.innerHTML += "</tbody>";
      resultContainer.appendChild(numericalDiffTable);
    }
  } catch (error) {
    console.error("Error:", error);
    document.getElementById("sentimentVsActualResults").innerText =
      "An error occurred while fetching comparison results.";
  }
}
