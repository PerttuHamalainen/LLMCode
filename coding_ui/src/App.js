import React, { useState, useEffect, Fragment } from "react";
import CodingPane from './CodingPane';
import CodeList from "./CodeList";
import FileUpload from "./FileUpload";
import './App.css';
import FileManager from "./FileManager";

function App() {
  const [fileName, setFileName] = useState(() => {
    // Load file name from local storage if available
    const savedFileName = localStorage.getItem("fileName");
    return savedFileName ? JSON.parse(savedFileName) : "";
  });
  useEffect(() => {
    // Save file name to local storage whenever it changes
    localStorage.setItem("fileName", JSON.stringify(fileName));
  }, [fileName]);

  const [texts, setTexts] = useState(() => {
    // Load texts from local storage if available
    const savedTexts = localStorage.getItem("texts");
    return savedTexts ? JSON.parse(savedTexts) : [];
  });
  useEffect(() => {
    // Save texts to local storage whenever it changes
    localStorage.setItem("texts", JSON.stringify(texts));
  }, [texts]);

  const [highlights, setHighlights] = useState(() => {
    // Load highlights from local storage if available
    const savedHighlights = localStorage.getItem("highlights");
    return savedHighlights ? JSON.parse(savedHighlights) : [];
  });
  useEffect(() => {
    // Save highlights to local storage whenever it changes
    localStorage.setItem("highlights", JSON.stringify(highlights));
  }, [highlights]);

  const [focusedOnAny, setFocusedOnAny] = useState(false);
  useEffect(() => {
    setFocusedOnAny(highlights.flat().some((h) => h.focused));
  }, [highlights]);

  const [editLog, setEditLog] = useState(() => {
    // Load file name from local storage if available
    const savedEditLog = localStorage.getItem("editLog");
    return savedEditLog ? JSON.parse(savedEditLog) : [];
  });
  useEffect(() => {
    // Save log to local storage whenever it changes
    localStorage.setItem("editLog", JSON.stringify(editLog));
  }, [editLog]);

  const setHighlightsForIdx = (idx, updateFunc) => {
    setHighlights((prevHighlights) =>
      prevHighlights.map((hl, i) => (i === idx ? updateFunc(hl) : hl))
    );
  };

  const createLog = (logData) => {
    setEditLog((prevEntries) => {
      // Do not log highlights that never had any codes (that were instantly deleted)
      const prevLog = prevEntries.at(-1);
      if (prevLog && logData.highlight.id === prevLog.highlight.id && logData.event === "delete" && prevLog.event === "create") {
        return prevEntries.slice(0, -1);  // Delete previous log
      }

      // Do not store edit logs where there are no changes to the codes
      const prevLogForHighlight = prevEntries.findLast((prevLog) => logData.highlight.id === prevLog.highlight.id);
      if (prevLogForHighlight && logData.event === "edit" && logData.highlight.codes === prevLogForHighlight.highlight.codes) {
        return prevEntries;
      }
      
      return [
        ...prevEntries,
        {
          ...logData,
          date: new Date()
        },
      ];
    });
  };

  const handleFileUpload = (res) => {
    // Save file name and texts
    setFileName(res.fileName);
    setTexts(res.texts);
    // Initialise highlights with empty arrays for each text
    setHighlights(Array.from({ length: res.texts.length }, () => []));
  }

  const handleFileDelete = () => {
    // Restore default values
    setFileName("");
    setTexts([]);
    setHighlights([]);
    setEditLog([]);
  }

  return (
    <div 
      style={{
        height: "100%",
        display: "flex",
      }}
    >
      <div
        style={{
          position: "fixed",
          width: "230px",
          height: "100%",
          backgroundColor: "#f7f7f7",
          borderRight: "1px solid #ddd",
          padding: "50px 30px 50px 30px",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "2px",
            paddingBottom: "0px"
          }}
        >
          <h2>File</h2>
          { texts.length === 0 ? (
            <FileUpload onUpload={handleFileUpload} />
          ) : (
            <FileManager fileName={fileName} texts={texts} highlights={highlights} editLog={editLog} onDelete={handleFileDelete} />
          )}
        </div>
        
        { texts.length > 0 &&
          <CodeList highlights={highlights.flat()} focusedOnAny={focusedOnAny} />
        }
      </div>

      <div
        style={{
          marginLeft: "290px", // Prevents overlap with the fixed sidebar
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
          }}
        >
          {texts.map((text, idx) => {
            const textHighlights = highlights[idx] || [];
            return (
              <Fragment key={idx}>
                <CodingPane 
                  text={text} 
                  highlights={textHighlights}
                  setHighlights={(updateFunc) => setHighlightsForIdx(idx, updateFunc)}
                  focusedOnAny={focusedOnAny}
                  createLog={(logData) => createLog({ ...logData, textId: idx })}
                />
                {idx < texts.length - 1 && (
                  <div
                    style={{
                      flexGrow: 1, // Makes the line take up available space
                      borderBottom: "1px dashed",
                      borderColor: "#ddd",
                      borderWidth: "1px",
                      borderImage: "repeating-linear-gradient(to right, #ddd 0, #ddd 5px, transparent 5px, transparent 10px) 1",
                      marginLeft: "50px"
                    }}
                  />
                )}
              </Fragment>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default App;
