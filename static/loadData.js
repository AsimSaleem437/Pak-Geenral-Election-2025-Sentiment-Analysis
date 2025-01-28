async function loadData() {
  const statusElement = document.getElementById("loadDataStatus");

  try {
    // Call the API to load data
    const response = await fetch("http://127.0.0.1:8000/load_data", {
      method: "POST",
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error ${response.status}: ${errorText}`);
    }

    const result = await response.json();

    // Clear previous content
    statusElement.innerHTML = "";

    // Display dataset shape and column info
    const infoText = document.createElement("p");
    infoText.innerText = `Data loaded successfully!\nShape: ${result.shape[0]} rows, ${result.shape[1]} columns`;
    statusElement.appendChild(infoText);

    // Create a table to display column names, data types, and sample data
    const table = document.createElement("table");
    table.classList.add("data-table");

    // Create the header row
    const headerRow = document.createElement("tr");
    ["Column", "Data Type", "Sample Data"].forEach((headerText) => {
      const th = document.createElement("th");
      th.innerText = headerText;
      th.style.border = "1px solid #ddd";
      th.style.padding = "8px";
      th.style.textAlign = "left";
      headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    // Populate the table with data for each column
    result.columns.forEach((col) => {
      const row = document.createElement("tr");

      // Column name
      const colCell = document.createElement("td");
      colCell.innerText = col;
      colCell.style.border = "1px solid #ddd";
      colCell.style.padding = "8px";
      row.appendChild(colCell);

      // Data type
      const typeCell = document.createElement("td");
      typeCell.innerText = result.data_types[col];
      typeCell.style.border = "1px solid #ddd";
      typeCell.style.padding = "8px";
      row.appendChild(typeCell);

      // Sample data
      const sampleCell = document.createElement("td");
      const sampleList = document.createElement("ul");

      // Handle object-like sample data
      Object.values(result.sample_data[col])
        .slice(0, 3)
        .forEach((value) => {
          // Limit to 3 items
          const listItem = document.createElement("li");
          listItem.innerText = value;
          sampleList.appendChild(listItem);
        });

      sampleCell.appendChild(sampleList);
      sampleCell.style.border = "1px solid #ddd";
      sampleCell.style.padding = "8px";
      row.appendChild(sampleCell);

      table.appendChild(row);
    });

    statusElement.appendChild(table);
  } catch (error) {
    console.error(error);
    statusElement.innerText = `Error loading data: ${error.message}`;
    statusElement.style.color = "red"; // Highlight error messages
  }
}
