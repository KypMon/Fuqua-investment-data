import '@fsb/fw-auth/dist/fw-auth.js';
import CssBaseline from "@mui/material/CssBaseline";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import axios from "axios";
import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App.jsx";

axios.defaults.withCredentials = true;

axios.interceptors.response.use(
  response => response,
  error => {
    if (error.response && error.response.status === 401) {
      // Force re-authentication if backend says 401
      console.log("REDIRECT from index.js");
      window.location.href = "https://go.fuqua.duke.edu/auth/onelink?service=" + encodeURIComponent(window.location.href);
    }
    return Promise.reject(error);
  }
);

const root = ReactDOM.createRoot(document.getElementById("root"));

const theme = createTheme();

root.render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <App />
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
);
