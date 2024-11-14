import React from "react";
import readXlsxFile from "read-excel-file";
import Papa from "papaparse";

const FileUpload = ({ onUpload }) => {
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const fileExtension = file.name.split(".").pop().toLowerCase();

    if (fileExtension === "csv") {
      // Parse CSV file
      Papa.parse(file, {
        header: false,
        complete: (results) => {
          const texts = results.data.map((row) => row[0]).filter(Boolean); // Extract first column
          onUpload({ fileName: file.name, texts: texts });
        },
        error: (error) => {
          console.error("Error parsing CSV:", error);
        },
      });
    } else if (fileExtension === "xlsx") {
      // Read Excel file
      readXlsxFile(file).then((rows) => {
        const texts = rows.map((row) => row[0]).filter(Boolean); // Extract first column
        onUpload({ fileName: file.name, texts: texts });
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
      <p style={{ color: "#333", fontSize: "14px", lineHeight: 1.6 }}>Upload a single-column CSV or Excel file containing one text per row. Please ensure that the file does not contain a header row.</p>
    </div>
  );
};

export default FileUpload;