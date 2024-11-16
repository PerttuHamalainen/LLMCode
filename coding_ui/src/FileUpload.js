import React from "react";
import readXlsxFile from "read-excel-file";
import Papa from "papaparse";

const FileUpload = ({ onUpload }) => {
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;
  
    const fileExtension = file.name.split(".").pop().toLowerCase();
  
    const parseData = (rows, fileName) => {
      const header = rows[0];
      const textIndex = header.indexOf("text");
      const depthIndex = header.indexOf("depth");
  
      if (textIndex === -1) {
        alert("The file must contain a 'text' column.");
        return;
      }
  
      const texts = [];
      const depths = depthIndex !== -1 ? [] : null;
  
      for (let i = 1; i < rows.length; i++) {
        const row = rows[i];
        if (row[textIndex]) {
          texts.push(row[textIndex]);
          if (depths !== null) {
            const depthValue = parseInt(row[depthIndex], 10);
            depths.push(isNaN(depthValue) ? null : depthValue);
          }
        }
      }
  
      onUpload({ fileName, texts, depths });
    };
  
    if (fileExtension === "csv") {
      // Parse CSV file
      Papa.parse(file, {
        header: true, // Use header row
        complete: (results) => {
          parseData(results.data, file.name);
        },
        error: (error) => {
          console.error("Error parsing CSV:", error);
        },
      });
    } else if (fileExtension === "xlsx") {
      // Read Excel file
      readXlsxFile(file).then((rows) => {
        parseData(rows, file.name);
      }).catch((error) => {
        console.error("Error reading Excel file:", error);
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
      <p style={{ color: "#333", fontSize: "14px", lineHeight: 1.6 }}>Upload a CSV or Excel file containing one text per row. The texts to be coded should be in a column labelled 'text'.<br/><br/>Optionally, to display hierarchical data, you may include a column 'depth' containing a depth index for each text in the hierarchy.</p>
    </div>
  );
};

export default FileUpload;