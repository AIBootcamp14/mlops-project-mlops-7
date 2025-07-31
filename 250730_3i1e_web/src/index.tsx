import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App"; // 또는 실제 루트 컴포넌트 경로에 맞게

const rootElement = document.getElementById("app");
if (rootElement) {
    const root = ReactDOM.createRoot(rootElement);
    root.render(
        <React.StrictMode>
            <App />
        </React.StrictMode>
    );
}
