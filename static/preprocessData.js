async function preprocessData() {
  try {
    // Send POST request to the /preprocess endpoint
    const response = await fetch("http://127.0.0.1:8000/preprocess", {
      method: "POST",
    });

    // Check if the response is successful
    if (!response.ok) {
      const errorMessage = await response.text(); // Get the error message from the response
      throw new Error(`Error: ${response.status} - ${errorMessage}`);
    }

    // Parse the JSON result from the response
    const result = await response.json();

    // Log the result to inspect the structure
    console.log("Preprocessing result:", result);

    // Clear previous results
    const preprocessResultsContainer =
      document.getElementById("preprocessResults");
    preprocessResultsContainer.innerHTML = "";

    // Create a table for displaying preprocessing results
    const table = document.createElement("table");
    table.style.width = "100%";
    table.style.borderCollapse = "collapse";
    table.style.marginBottom = "20px";

    // Helper function to add rows to the table
    const addRow = (key, value) => {
      const row = document.createElement("tr");
    
      // Column heading cell (key)
      const cellKey = document.createElement("td");
      cellKey.style.border = "1px solid #ddd";
      cellKey.style.padding = "10px";
      cellKey.style.fontWeight = "bold";
      cellKey.style.backgroundColor = "#f2f2f2";
      cellKey.style.textAlign = "center";
      cellKey.innerText = key;
    
      // Data cell (value)
      const cellValue = document.createElement("td");
      cellValue.style.border = "1px solid #ddd";
      cellValue.style.padding = "10px";
      cellValue.style.textAlign = "left";
    
      // Check if the value is sample_processed
      if (key === "sample_processed") {
        if (Array.isArray(value) && value.length > 0) {
          const sampleTable = document.createElement("table");
          sampleTable.style.width = "100%";
          sampleTable.style.borderCollapse = "collapse";
          sampleTable.style.marginTop = "10px";
    
          // Create table headers for sample data
          const headerRow = document.createElement("tr");
          Object.keys(value[0]).forEach((col) => {
            const headerCell = document.createElement("th");
            headerCell.style.border = "1px solid #ddd";
            headerCell.style.padding = "5px";
            headerCell.style.fontWeight = "bold";
            headerCell.innerText = col;
            headerRow.appendChild(headerCell);
          });
          sampleTable.appendChild(headerRow);
    
          // Populate sample data rows
          value.forEach((sample) => {
            const sampleRow = document.createElement("tr");
            Object.values(sample).forEach((cellData) => {
              const sampleCell = document.createElement("td");
              sampleCell.style.border = "1px solid #ddd";
              sampleCell.style.padding = "5px";
              sampleCell.innerText = cellData !== null ? cellData : "N/A"; // Handle null values
              sampleRow.appendChild(sampleCell);
            });
            sampleTable.appendChild(sampleRow);
          });
    
          cellValue.appendChild(sampleTable);
        } else {
          console.warn("sample_processed is not an array or is empty:", value);
          cellValue.innerText = "No sample data available";
        }
      } else if (Array.isArray(value)) {
        // For simple arrays, display as comma-separated values
        cellValue.innerText = value.length > 0 ? value.join(", ") : "None";
      } else if (typeof value === "object") {
        // For plain objects, convert to key-value pairs
        const details = Object.entries(value)
          .map(([k, v]) => `${k}: ${v !== null ? v : "N/A"}`) 
          .join(", ");
        cellValue.innerText = details;
      } else {
        // For primitive values
        cellValue.innerText = value !== null ? value : "N/A";
      }
    
      row.appendChild(cellKey);
      row.appendChild(cellValue);
      table.appendChild(row);
    };

    // Loop through the result object and add each entry
    for (const [key, value] of Object.entries(result)) {
      if (key !== "file_saved_path") {
        addRow(key, value);
      }
    }

    // Append the table to the results container
    preprocessResultsContainer.appendChild(table);
  } catch (error) {
    // Display the error message
    document.getElementById(
      "preprocessResults"
    ).innerText = `Failed to preprocess data: ${error.message}`;
  }
}
