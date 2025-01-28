const apiBaseUrl = "http://127.0.0.1:8000"; // Define at the top level

async function annotateSentiment() {
  try {
    const response = await fetch(`${apiBaseUrl}/annotate_sentiment`, {
      method: "POST",
    });

    if (!response.ok) {
      throw new Error("Network response was not ok: " + response.statusText);
    }

    const result = await response.json();

    // Clear previous results
    const annotateSentimentResults = document.getElementById(
      "annotateSentimentResults"
    );
    annotateSentimentResults.innerHTML = ""; // Clear existing content

    // Display the message about annotation success
    const message = document.createElement("p");
    message.innerText = result.message;
    message.style.fontWeight = "bold";
    annotateSentimentResults.appendChild(message);

    // Display sentiment distribution in a table format
    const sentimentDistributionTable = document.createElement("table");
    sentimentDistributionTable.style.width = "100%";
    sentimentDistributionTable.style.borderCollapse = "collapse";
    sentimentDistributionTable.style.marginBottom = "20px";

    // Add table header for sentiment distribution
    const headerRow = document.createElement("tr");
    ["Sentiment", "Count"].forEach((headerText) => {
      const headerCell = document.createElement("th");
      headerCell.innerText = headerText;
      headerCell.style.border = "1px solid #ddd";
      headerCell.style.padding = "8px";
      headerCell.style.backgroundColor = "#f2f2f2";
      headerCell.style.fontWeight = "bold";
      headerCell.style.textAlign = "center";
      headerRow.appendChild(headerCell);
    });
    sentimentDistributionTable.appendChild(headerRow);

    // Populate sentiment distribution data rows
    Object.entries(result.sentiment_distribution).forEach(
      ([sentiment, count]) => {
        const row = document.createElement("tr");

        const sentimentCell = document.createElement("td");
        sentimentCell.innerText = sentiment;
        sentimentCell.style.border = "1px solid #ddd";
        sentimentCell.style.padding = "8px";
        sentimentCell.style.textAlign = "center";

        const countCell = document.createElement("td");
        countCell.innerText = count;
        countCell.style.border = "1px solid #ddd";
        countCell.style.padding = "8px";
        countCell.style.textAlign = "center";

        row.appendChild(sentimentCell);
        row.appendChild(countCell);
        sentimentDistributionTable.appendChild(row);
      }
    );

    // Append sentiment distribution table to results container
    annotateSentimentResults.appendChild(sentimentDistributionTable);

    // Display sample annotated tweets in a table format
    const sampleAnnotatedTable = document.createElement("table");
    sampleAnnotatedTable.style.width = "100%";
    sampleAnnotatedTable.style.borderCollapse = "collapse";

    // Add table header for sample annotated tweets
    const sampleHeaderRow = document.createElement("tr");
    ["Tweet", "Sentiment"].forEach((headerText) => {
      const headerCell = document.createElement("th");
      headerCell.innerText = headerText;
      headerCell.style.border = "1px solid #ddd";
      headerCell.style.padding = "8px";
      headerCell.style.backgroundColor = "#f2f2f2";
      headerCell.style.fontWeight = "bold";
      headerCell.style.textAlign = "center";
      sampleHeaderRow.appendChild(headerCell);
    });
    sampleAnnotatedTable.appendChild(sampleHeaderRow);

    // Populate sample annotated tweets data rows
    result.sample_annotated.forEach((item) => {
      const row = document.createElement("tr");

      const tweetCell = document.createElement("td");
      tweetCell.innerText = item.cleaned_tweet;
      tweetCell.style.border = "1px solid #ddd";
      tweetCell.style.padding = "8px";
      tweetCell.style.textAlign = "left";

      const sentimentCell = document.createElement("td");
      sentimentCell.innerText = item.Vader_Sentiment;
      sentimentCell.style.border = "1px solid #ddd";
      sentimentCell.style.padding = "8px";
      sentimentCell.style.textAlign = "center";

      row.appendChild(tweetCell);
      row.appendChild(sentimentCell);
      sampleAnnotatedTable.appendChild(row);
    });

    // Append sample annotated tweets table to results container
    annotateSentimentResults.appendChild(sampleAnnotatedTable);

    // Create download link if file path is provided
    // if (result.file_path) {
    //   const downloadLink = document.createElement("a");
    //   downloadLink.href = `${apiBaseUrl}/${result.file_path.replace(
    //     /\\/g,
    //     "/"
    //   )}`;
    //   downloadLink.download = "labeled_tweets_with_sentiments.xlsx";
    //   downloadLink.innerText = "Download Annotated Sentiment Data";
    //   downloadLink.style.display = "block";
    //   downloadLink.style.marginTop = "10px";

    //   // Clear any existing link and append the new link
    //   const outputFileDownloadLink = document.getElementById(
    //     "outputFileDownloadLink"
    //   );
    //   outputFileDownloadLink.innerHTML = ""; // Clear any previous content
    //   outputFileDownloadLink.appendChild(downloadLink);
    // }
  } catch (error) {
    console.error("Error annotating sentiment:", error);
    document.getElementById("annotateSentimentResults").innerText =
      "Error: " + error.message;
  }
}