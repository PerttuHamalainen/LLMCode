import React from "react";
import readXlsxFile from "read-excel-file";
import Papa from "papaparse";

const FileUpload = ({ onUpload }) => {
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
  
    const fileExtension = file.name.split(".").pop().toLowerCase();
  
    const parseData = (rows, fileName) => {
      if (!rows || rows.length === 0) {
        alert("The file is empty or cannot be read.");
        return;
      }

      // Get header row and find column indices
      const header = rows[0].map((col) => (col ? col.toLowerCase() : ""));
      const idIndex = header.indexOf("id");
      const textIndex = header.indexOf("text");
      const depthIndex = header.indexOf("depth");
      const parentIdIndex = header.indexOf("parent_id");
    
      // Check if the required 'text' column is present
      if (textIndex === -1) {
        alert("The file must contain a 'text' column.");
        return;
      }
    
      const data = [];
      let idCounter = 0;
    
      // Parse each row (starting from the second row)
      for (let i = 1; i < rows.length; i++) {
        const row = rows[i];
        const text = row[textIndex] || "";
        if (!text) continue; // Skip rows without text
    
        const id = idIndex !== -1 ? row[idIndex] || null : `${idCounter++}`;
        const depth = depthIndex !== -1 ? parseInt(row[depthIndex], 10) : 0;
        const parentId = parentIdIndex !== -1 ? row[parentIdIndex] || null : null;
    
        // Create an object for the current row
        const item = {
          id,
          text,
          depth: isNaN(depth) ? 0 : depth, // Default depth to 0 if invalid
          parentId,
        };
    
        data.push(item);
      }
    
      // Call the onUpload callback with the parsed data
      onUpload({ fileName, data });
    };
  
    if (fileExtension === "csv") {
      // Parse CSV file using PapaParse
      Papa.parse(file, {
        header: true, // Use header row
        complete: (results) => {
          parseData(results.data, file.name);
        },
        error: (error) => {
          console.error("Error parsing CSV:", error);
          alert("Error parsing CSV:", error);
        },
      });
    } else if (fileExtension === "xlsx") {
      // Read Excel file using readXlsxFile
      readXlsxFile(file)
        .then((rows) => {
          parseData(rows, file.name);
        })
        .catch((error) => {
          console.error("Error reading Excel file:", error);
          alert("Error reading Excel file:", error);
        });
    } else {
      alert("Unsupported file type. Please upload a CSV or Excel file.");
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2px" }}>
      <input
        type="file"
        accept=".csv, .xlsx"
        onChange={handleFileUpload}
        style={{ marginBottom: "10px" }}
      />
      <p style={{ color: "#333", fontSize: "14px", lineHeight: 1.6 }}>Upload a CSV or Excel file containing one text per row. The texts to be coded should be in a column labelled 'text'. You may also provide an 'id' column.<br/><br/>Optionally, to display hierarchical data, you may include a column 'depth' containing a depth index for each text in the hierarchy. In this case, remember to also include the id of the parent text in the 'parent_id' column.</p>
    </div>
  );
};

export default FileUpload;