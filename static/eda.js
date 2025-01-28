// Run EDA and display results
async function runEDA() {
  try {
    const response = await fetch("http://127.0.0.1:8000/eda", {
      method: "POST",
    });

    if (!response.ok) {
      const error = await response.json();
      document.getElementById(
        "edaResults"
      ).innerText = `Error: ${error.detail}`;
      return;
    }

    const result = await response.json();

    // Clear previous results
    const edaResultsContainer = document.getElementById("edaResults");
    edaResultsContainer.innerHTML = "";

    // Display Shape
    const shapePara = document.createElement("p");
    shapePara.innerHTML = `<strong>Shape:</strong> ${result.shape[0]} rows, ${result.shape[1]} columns`;
    edaResultsContainer.appendChild(shapePara);

    // Display Columns
    const columnsPara = document.createElement("p");
    columnsPara.innerHTML = `<strong>Columns:</strong>`;
    edaResultsContainer.appendChild(columnsPara);

    const columnsList = document.createElement("ul");
    result.columns.forEach((col) => {
      const listItem = document.createElement("li");
      listItem.innerText = col;
      columnsList.appendChild(listItem);
    });
    edaResultsContainer.appendChild(columnsList);

    // Display First 5 Rows
    const firstRowsPara = document.createElement("p");
    firstRowsPara.innerHTML = `<strong>First 5 Rows:</strong>`;
    edaResultsContainer.appendChild(firstRowsPara);

    const rowsTable = document.createElement("table");
    rowsTable.style.width = "100%";
    rowsTable.style.borderCollapse = "collapse";

    // Add table header
    const headerRow = document.createElement("tr");
    result.columns.forEach((col) => {
      const th = document.createElement("th");
      th.innerText = col;
      th.style.border = "1px solid #ddd";
      th.style.padding = "8px";
      th.style.textAlign = "left";
      headerRow.appendChild(th);
    });
    rowsTable.appendChild(headerRow);

    // Add first 5 rows
    result.first_5_rows.forEach((row) => {
      const rowElement = document.createElement("tr");
      result.columns.forEach((col) => {
        const td = document.createElement("td");
        td.innerText = row[col]; // Accessing the value by column name
        td.style.border = "1px solid #ddd";
        td.style.padding = "8px";
        rowElement.appendChild(td);
      });
      rowsTable.appendChild(rowElement);
    });
    edaResultsContainer.appendChild(rowsTable);

    // Display Missing Values
    const missingValuesPara = document.createElement("p");
    missingValuesPara.innerHTML = `<strong>Missing Values:</strong>`;
    edaResultsContainer.appendChild(missingValuesPara);

    const missingTable = document.createElement("table");
    missingTable.style.width = "100%";
    missingTable.style.borderCollapse = "collapse";

    // Add missing values table header
    const missingHeaderRow = document.createElement("tr");
    ["Column", "Missing Count"].forEach((text) => {
      const th = document.createElement("th");
      th.innerText = text;
      th.style.border = "1px solid #ddd";
      th.style.padding = "8px";
      th.style.textAlign = "left";
      missingHeaderRow.appendChild(th);
    });
    missingTable.appendChild(missingHeaderRow);

    // Add missing values data
    Object.entries(result.missing_values).forEach(([col, count]) => {
      const row = document.createElement("tr");

      const colCell = document.createElement("td");
      colCell.innerText = col;
      colCell.style.border = "1px solid #ddd";
      colCell.style.padding = "8px";
      row.appendChild(colCell);

      const countCell = document.createElement("td");
      countCell.innerText = count;
      countCell.style.border = "1px solid #ddd";
      countCell.style.padding = "8px";
      row.appendChild(countCell);

      missingTable.appendChild(row);
    });
    edaResultsContainer.appendChild(missingTable);

    // Container for images displayed side by side
    const imageContainer = document.createElement("div");
    imageContainer.style.display = "grid";
    imageContainer.style.gridTemplateColumns =
      "repeat(auto-fill, minmax(300px, 1fr))";
    imageContainer.style.gap = "20px";
    imageContainer.style.marginTop = "20px";

    // Helper function to create an image element with title, border, and shadow
    const createImageElement = (title, base64Data) => {
      const imgWrapper = document.createElement("div");
      imgWrapper.style.width = "100%";
      imgWrapper.style.marginBottom = "10px";

      const imgTitle = document.createElement("h4");
      imgTitle.style.marginBottom = "5px";
      imgTitle.style.textAlign = "center";
      imgTitle.innerText = title;
      imgWrapper.appendChild(imgTitle);

      const img = document.createElement("img");
      img.src = `data:image/png;base64,${base64Data}`;
      img.alt = title;
      img.style.width = "100%";
      img.style.height = "auto";
      img.style.border = "2px solid #ccc";
      img.style.borderRadius = "8px";
      img.style.padding = "5px";
      img.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.1)";
      imgWrapper.appendChild(img);

      return imgWrapper;
    };

    // Add images with titles if available
    if (result.wordcloud) {
      imageContainer.appendChild(
        createImageElement("Word Cloud", result.wordcloud)
      );
    }
    if (result.tweet_length_distribution) {
      imageContainer.appendChild(
        createImageElement(
          "Tweet Length Distribution",
          result.tweet_length_distribution
        )
      );
    }
    if (result.hashtag_distribution) {
      imageContainer.appendChild(
        createImageElement("Hashtag Distribution", result.hashtag_distribution)
      );
    }
    // if (result.daily_tweet_counts) {
    //   imageContainer.appendChild(
    //     createImageElement("Daily Tweet Counts", result.daily_tweet_counts)
    //   );
    // }
    if (result.weekday_distribution) {
      imageContainer.appendChild(
        createImageElement(
          "Tweet Activity by Weekday",
          result.weekday_distribution
        )
      );
    }
    if (result.top_users) {
      imageContainer.appendChild(
        createImageElement("Top 10 Users by Tweet Volume", result.top_users)
      );
    }

    // Append the image container to the results container
    edaResultsContainer.appendChild(imageContainer);
  } catch (error) {
    console.error(error);
    document.getElementById("edaResults").innerText = `Error: ${error.message}`;
  }
}
