import React, { useEffect, useState } from "react";
import {
  Grid,
  TextField,
  Button,
  Typography,
  Box,
  IconButton,
  Autocomplete,
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import CloseIcon from "@mui/icons-material/Close";

const CACHE_KEY = "etfOptions";
let cachedOptions = null;

export default function EtfListInput({ etflist, setEtflist }) {
  const [options, setOptions] = useState([]);

  useEffect(() => {
    const loadOptions = async () => {
      if (cachedOptions) {
        setOptions(cachedOptions);
        return;
      }

      const stored = localStorage.getItem(CACHE_KEY);
      if (stored) {
        try {
          cachedOptions = JSON.parse(stored);
          setOptions(cachedOptions);
          return;
        } catch (_) {
          // fall through to fetch
        }
      }

      try {
        const res = await fetch(`${process.env.PUBLIC_URL}/etf-tickers.csv`);
        const text = await res.text();
        const lines = text.trim().split("\n").slice(1);
        cachedOptions = lines.map((line) => {
          const [ticker, name] = line.split(",");
          return `${ticker} - ${name}`;
        });
        setOptions(cachedOptions);
        localStorage.setItem(CACHE_KEY, JSON.stringify(cachedOptions));
      } catch (_) {
        setOptions([]);
      }
    };

    loadOptions();
  }, []);
  const handleEtfChange = (index, value) => {
    const updated = [...etflist];
    updated[index] = value;
    setEtflist(updated);
  };

  const addEtfField = () => {
    setEtflist([...etflist, ""]);
  };

  const removeEtfField = (index) => {
    if (etflist.length > 1) {
      const updated = [...etflist];
      updated.splice(index, 1);
      setEtflist(updated);
    }
  };

  return (
    <>
      <Typography variant="subtitle1" gutterBottom>
        ETF List
      </Typography>
      <Grid container spacing={12}>
        {etflist.map((etf, idx) => (
          <Grid item xs={12} sm={4} md={3} key={idx}>
            <Box position="relative">
              <Autocomplete
                freeSolo
                options={options}
                inputValue={etf}
                onInputChange={(e, value) => handleEtfChange(idx, value)}
                onChange={(e, value) => {
                  if (value) {
                    const ticker = value.split(" - ")[0];
                    handleEtfChange(idx, ticker);
                  }
                }}
                renderInput={(params) => (
                  <TextField {...params} label={`ETF ${idx + 1}`} fullWidth />
                )}
              />
              {etflist.length > 1 && (
                <IconButton
                  size="small"
                  onClick={() => removeEtfField(idx)}
                  sx={{
                    position: "absolute",
                    top: 4,
                    right: 4,
                    zIndex: 2,
                    backgroundColor: "#fff",
                  }}
                >
                  <CloseIcon fontSize="small" />
                </IconButton>
              )}
            </Box>
          </Grid>
        ))}
        <Grid item xs={6} sm={4} md={3}>
          <Button
            variant="outlined"
            startIcon={<AddIcon />}
            onClick={addEtfField}
            fullWidth
            sx={{ height: "100%" }}
          >
            Add ETF
          </Button>
        </Grid>
      </Grid>
    </>
  );
}
