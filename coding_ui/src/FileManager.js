import Papa from "papaparse";
import * as XLSX from "xlsx";

const DownloadButton = ({ text, onDownload }) => {
  return (
    <button
      onClick={onDownload}
      style={{
        padding: "5px 20px",
        color: "#333",
        cursor: "pointer",
      }}
    >
      {text}
    </button>
  );
};

const DeleteFileButton = ({ onDelete }) => {
  const handleDelete = () => {
    const userConfirmed = window.confirm("Are you sure you want to delete this file and all its annotations?");
    if (userConfirmed) {
      onDelete();
    }
  };

  return (
    <button
      onClick={handleDelete}
      style={{
        padding: "5px 20px",
        color: "#333",
        cursor: "pointer",
      }}
    >
      Delete file
    </button>
  );
};

const FileManager = ({ fileName, texts, highlights, editLog, onDelete }) => {
  const formatTextWithHighlights = (text, highlights) => {
    // Sort highlights by startIndex to process them in order
    const sortedHighlights = [...highlights].sort((a, b) => a.startIndex - b.startIndex);
  
    let result = "";
    let currentIndex = 0;
  
    sortedHighlights.forEach((hl) => {
      const { startIndex, endIndex, codes } = hl;
  
      // Append text before the current highlight
      if (currentIndex < startIndex) {
        result += text.slice(currentIndex, startIndex);
      }
  
      // Wrap the highlighted text with double asterisks and add the <sup> tag
      const highlightedText = text.slice(startIndex, endIndex);
      result += `**${highlightedText}**<sup>${codes}</sup>`;
  
      // Update the current index
      currentIndex = endIndex;
    });
  
    // Append any remaining text after the last highlight
    if (currentIndex < text.length) {
      result += text.slice(currentIndex);
    }
  
    return result;
  };

  const handleFileDownload = (fileType) => {
    const codedTexts = texts.map((item, idx) => {
      const textHighlights = highlights[idx];
      return formatTextWithHighlights(item.text, textHighlights);
    });
  
    // Prepare the data
    const data = texts.map((item, idx) => {
      const { parentId, ...rest } = item;
      return {
        ...rest,
        parent_id: parentId, // Rename 'parentId' to 'parent_id'
        coded_text: codedTexts[idx],
      };
    });
  
    if (fileType === "csv") {
      // CSV Download using PapaParse
      const csv = Papa.unparse(data);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
  
      // Trigger CSV download
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", "coded_texts.csv");
      link.click();
      URL.revokeObjectURL(url);
    } else if (fileType === "xlsx") {
      // Excel Download using xlsx
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
  
      // Trigger Excel download
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
    // Convert the `editLog` object to a JSON string
    const jsonString = JSON.stringify(editLog, null, 2);
  
    // Create a Blob with the JSON content and specify the MIME type
    const blob = new Blob([jsonString], { type: "application/json;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
  
    // Create a temporary link and trigger the download
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", "coding_log.json");
    link.click();
  
    // Clean up the URL object
    URL.revokeObjectURL(url);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        paddingBottom: "20px"
      }}
    >
      <p style={{
        whiteSpace: "normal",
        wordBreak: "break-word",
        overflowWrap: "break-word",
        color: "#333",
        fontSize: "14px",
        lineHeight: 1.6,
        margin: "0px 0px 15px 0px",
        padding: 0
      }}>
        {fileName}
      </p>

      <DownloadButton text="Download coded file" onDownload={() => handleFileDownload("xlsx")} />
      <DownloadButton text="Download logs" onDownload={handleLogDownload} />
      <DeleteFileButton onDelete={onDelete} />
    </div>
  );
};

export default FileManager;