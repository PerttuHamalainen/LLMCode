import React, { useState, useRef, useEffect } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { formatTextWithHighlights } from "./helpers";
import MenuButton from "./components/MenuButton";

const FileManager = ({ texts, editLog, onDelete }) => {
  const [showPopup, setShowPopup] = useState(false);
  const popupRef = useRef(null);

  const handleClickOutside = (event) => {
    if (popupRef.current && !popupRef.current.contains(event.target)) {
      setShowPopup(false);
    }
  };

  useEffect(() => {
    if (showPopup) {
      document.addEventListener("mousedown", handleClickOutside);
    } else {
      document.removeEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showPopup]);

  const handleFileDownload = (fileType) => {
    const codedTexts = texts.map((item) => {
      return formatTextWithHighlights(item.text, item.highlights);
    });

    const data = texts.map((item, idx) => {
      const { parentId, ...rest } = item;
      return {
        ...rest,
        parent_id: parentId,
        coded_text: codedTexts[idx],
      };
    });

    if (fileType === "csv") {
      const csv = Papa.unparse(data);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "coded_texts.csv");
      link.click();
      URL.revokeObjectURL(url);
    } else if (fileType === "xlsx") {
      const worksheet = XLSX.utils.json_to_sheet(data);
      const workbook = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(workbook, worksheet, "Coded Texts");

      const excelBuffer = XLSX.write(workbook, {
        bookType: "xlsx",
        type: "array",
      });
      const blob = new Blob([excelBuffer], {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;charset=utf-8;",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "coded_texts.xlsx");
      link.click();
      URL.revokeObjectURL(url);
    } else {
      alert("Unsupported file type. Please choose 'csv' or 'xlsx'.");
    }
  };

  const handleLogDownload = () => {
    const jsonString = JSON.stringify(editLog, null, 2);
    const blob = new Blob([jsonString], { type: "application/json;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "coding_log.json");
    link.click();
    URL.revokeObjectURL(url);
  };

  const togglePopup = () => setShowPopup(!showPopup);

  return (
    <div style={{ position: "relative", fontSize: "24px", cursor: "pointer" }}>
      {/* Menu Icon */}
      <div onClick={togglePopup}>⋯</div>

      {/* Popup Menu */}
      {showPopup && (
        <div
          ref={popupRef}  
          style={{
            position: "absolute",
            top: "30px",
            left: "0px",
            width: "200px",
            backgroundColor: "#fff",
            boxShadow: "0 4px 8px rgba(0, 0, 0, 0.1)",
            borderRadius: "8px",
            padding: "10px",
            zIndex: 100,
          }}
        >
          <MenuButton onClick={() => handleFileDownload("csv")}>
            Download Coded Texts (CSV)
          </MenuButton>
          <MenuButton onClick={() => handleFileDownload("xlsx")}>
            Download Coded Texts (Excel)
          </MenuButton>
          <MenuButton onClick={handleLogDownload}>
            Download Coding Log
          </MenuButton>
          <MenuButton
            onClick={() => {
              const confirmDelete = window.confirm("Are you sure you want to delete this file?");
              if (confirmDelete) {
                onDelete();
              }
            }}
          >
            Delete File
          </MenuButton>
        </div>
      )}
    </div>
  );
};

export default FileManager;