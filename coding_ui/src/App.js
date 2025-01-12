import React, { useState, useEffect } from "react";
import CodingPane from './CodingPane';
import CodeList from "./CodeList";
import FileUpload from "./FileUpload";
import './App.css';
import FileManager from "./FileManager";
import CodingStats from "./CodingStats";
import { initializeClient, queryLLM } from "./llmcode/LLM";

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

  const [apiKey, setApiKey] = useState(() => {
    const savedKey = localStorage.getItem("apiKey");
    initializeClient(savedKey);
    return savedKey ? savedKey : "";
  });
  useEffect(() => {
    initializeClient(apiKey);
    localStorage.setItem("apiKey", apiKey);
  }, [apiKey]);

  const codeWithLLM = async () => {
    const res = await queryLLM("Hello LLM!");
    console.log(res);
  }

  const setHighlightsForIdx = (idx, updateFunc) => {
    setHighlights((prevHighlights) =>
      prevHighlights.map((hl, i) => (i === idx ? updateFunc(hl) : hl))
    );
  };

  const createLog = (logData) => {
    console.log("New log: ", logData)

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
    // Save uploaded data to state
    setFileName(res.fileName);
    setTexts(res.data);
    setHighlights(res.highlights);
  }

  const handleFileDelete = () => {
    // Restore default values
    setFileName("");
    setTexts([]);
    setHighlights([]);
    setEditLog([]);
  }

  const setAnnotated = (idx, isAnnotated) => {
    setTexts((prevEntries) => {
      return prevEntries.map((item, itemIdx) => itemIdx == idx ? { ...item, isAnnotated: isAnnotated } : item )
    });
  }

  const setExample = (idx, isExample) => {
    setTexts((prevEntries) => {
      return prevEntries.map((item, itemIdx) => itemIdx == idx ? { ...item, isExample: isExample } : item )
    });
  }

  useEffect(() => {
    const handleBeforeUnload = (event) => {
      event.preventDefault();
      event.returnValue = "You may lose all progress on exiting the browser tab. Have you downloaded your codes and logs?";
    };

    window.addEventListener("beforeunload", handleBeforeUnload);

    // Cleanup the event listener on component unmount
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, []);

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
          overflowY: "auto",
          boxSizing: "border-box"
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

        <CodingStats texts={texts} minAnnotated={40} minExamples={3} onButtonClick={codeWithLLM} apiKey={apiKey} setApiKey={setApiKey} />
        
        { texts.length > 0 &&
          <CodeList highlights={highlights.flat()} focusedOnAny={focusedOnAny} />
        }
      </div>

      <div
        style={{
          marginLeft: "240px", // Prevents overlap with the fixed sidebar
          padding: "30px 0px"
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
          }}
        >
          {texts.map((item, idx) => {
            const textHighlights = highlights[idx] || [];
            return (
              <div style={{display: "flex", gap: "20px", marginLeft: "50px"}} key={idx}>
                {Array.from({ length: item.depth }).map((_, index) => (
                  <div
                    key={index}
                    style={{
                      width: "1px",
                      backgroundColor: "#ddd",
                      margin: "0px 0",
                    }}
                  />
                ))}

                <div>
                  <CodingPane 
                    text={item.text} 
                    isAnnotated={item.isAnnotated}
                    isExample={item.isExample}
                    highlights={textHighlights}
                    setHighlights={(updateFunc) => setHighlightsForIdx(idx, updateFunc)}
                    focusedOnAny={focusedOnAny}
                    createLog={(logData) => createLog({ ...logData, textId: item.id })}
                    setAnnotated={(isAnnotated) => setAnnotated(idx, isAnnotated)}
                    setExample={(isExample) => setExample(idx, isExample)}
                  />

                  {idx < texts.length - 1 && item.depth === texts[idx + 1].depth && (
                    <div
                      style={{
                        width: "50px",
                        height: "1px",
                        backgroundColor: "#ddd",
                      }}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default App;
