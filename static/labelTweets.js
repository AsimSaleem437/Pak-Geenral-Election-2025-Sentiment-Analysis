// Label Tweets
const apiBaseUrl = "http://127.0.0.1:8000"; 

async function labelTweets() {
  try {
    const response = await fetch(`${apiBaseUrl}/label_tweets`, {
      method: "POST",
    });

    if (!response.ok) {
      throw new Error("Network response was not ok: " + response.statusText);
    }

    const result = await response.json();

    // Clear previous results
    const labelTweetsResults = document.getElementById("labelTweetsResults");
    labelTweetsResults.innerHTML = ""; // Clear existing content

    // Display the message about labeling success
    const message = document.createElement("p");
    message.innerText = result.message;
    message.style.fontWeight = "bold";
    labelTweetsResults.appendChild(message);

    // Display party distribution
    const partyDistributionHeader = document.createElement("h3");
    partyDistributionHeader.innerText = "Party Distribution";
    labelTweetsResults.appendChild(partyDistributionHeader);

    // Create a list for party distribution
    const partyDistributionList = document.createElement("ul");
    Object.entries(result.party_distribution).forEach(([party, count]) => {
      const listItem = document.createElement("li");
      listItem.innerText = `${party}: ${count} tweets`;
      partyDistributionList.appendChild(listItem);
    });
    labelTweetsResults.appendChild(partyDistributionList);

    // Display sample labeled tweets
    const sampleLabeledHeader = document.createElement("h3");
    sampleLabeledHeader.innerText = "Sample Labeled Tweets";
    labelTweetsResults.appendChild(sampleLabeledHeader);

    // Create a table for sample labeled tweets
    const sampleLabeledTable = document.createElement("table");
    sampleLabeledTable.style.width = "100%";
    sampleLabeledTable.style.borderCollapse = "collapse";

    // Add table header for sample labeled tweets
    const sampleHeaderRow = document.createElement("tr");
    ["Tweet", "Party"].forEach((headerText) => {
      const headerCell = document.createElement("th");
      headerCell.innerText = headerText;
      headerCell.style.border = "1px solid #ddd";
      headerCell.style.padding = "8px";
      headerCell.style.backgroundColor = "#f2f2f2";
      headerCell.style.fontWeight = "bold";
      headerCell.style.textAlign = "center";
      sampleHeaderRow.appendChild(headerCell);
    });
    sampleLabeledTable.appendChild(sampleHeaderRow);

    // Populate sample labeled tweets data rows
    result.sample_labeled.forEach((item) => {
      const row = document.createElement("tr");

      const tweetCell = document.createElement("td");
      tweetCell.innerText = item.cleaned_tweet || "No tweet content"; // Handle undefined
      tweetCell.style.border = "1px solid #ddd";
      tweetCell.style.padding = "8px";
      tweetCell.style.textAlign = "left";

      const partyCell = document.createElement("td");
      partyCell.innerText = item.Party || "No party assigned"; // Handle undefined
      partyCell.style.border = "1px solid #ddd";
      partyCell.style.padding = "8px";
      partyCell.style.textAlign = "center";

      row.appendChild(tweetCell);
      row.appendChild(partyCell);
      sampleLabeledTable.appendChild(row);
    });

    // Append sample labeled tweets table to results container
    labelTweetsResults.appendChild(sampleLabeledTable);

    //Set download link for labeled file if provided
    // if (result.file_saved_path) {
    //   const downloadLink = document.createElement("a");
    //   downloadLink.href = `${apiBaseUrl}/${result.file_saved_path}`;
    //   downloadLink.download = "labeled_tweets.xlsx";
    //   downloadLink.innerText = "Download Labeled Data";
    //   downloadLink.style.display = "block";
    //   downloadLink.style.marginTop = "10px";

    //   labelTweetsResults.appendChild(downloadLink);
    // }
  } catch (error) {
    console.error("Error labeling tweets:", error);
    document.getElementById("labelTweetsResults").innerText =
      "Error: " + error.message;
  }
}
