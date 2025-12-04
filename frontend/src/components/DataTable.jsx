import React, { useMemo } from "react";
import { Box, Typography } from "@mui/material";
import MUIDataTable from "mui-datatables";
import { ensureCsvExtension } from "../utils/csv";

export default function DataTable({
  title,
  titleVariant = "subtitle1",
  data = [],
  columns = [],
  exportFileName = "table-data",
  tableOptions = {},
}) {
  const memoData = useMemo(() => (Array.isArray(data) ? data : []), [data]);

  const normalizedFileName = useMemo(() => {
    return ensureCsvExtension(exportFileName) ?? "table-data.csv";
  }, [exportFileName]);

  const normalizedColumns = useMemo(() => {
    return (columns || []).map((column, index) => {
      if (typeof column === "string") {
        return column;
      }

      const {
        accessorKey,
        header,
        label,
        name,
        muiTableHeadCellProps = {},
        muiTableBodyCellProps = {},
        Cell,
        options: columnOptions = {},
      } = column;

      const columnName = name ?? accessorKey ?? `column_${index}`;
      const columnLabel = label ?? header ?? columnName;
      const headAlign = muiTableHeadCellProps.align;
      const bodyAlign = muiTableBodyCellProps.align ?? headAlign;

      const normalizedOptions = {
        sort: false,
        filter: false,
        ...columnOptions,
      };

      if (headAlign) {
        const existingHeaderProps = normalizedOptions.setCellHeaderProps;
        normalizedOptions.setCellHeaderProps = (...args) => {
          const resolved = typeof existingHeaderProps === "function" ? existingHeaderProps(...args) || {} : {};
          return {
            ...resolved,
            style: {
              ...(resolved.style || {}),
              textAlign: headAlign,
            },
          };
        };
      }

      if (bodyAlign) {
        const existingCellProps = normalizedOptions.setCellProps;
        normalizedOptions.setCellProps = (...args) => {
          const resolved = typeof existingCellProps === "function" ? existingCellProps(...args) || {} : {};
          return {
            ...resolved,
            style: {
              ...(resolved.style || {}),
              textAlign: bodyAlign,
            },
          };
        };
      }

      if (typeof Cell === "function") {
        normalizedOptions.customBodyRender = (value, tableMeta) => {
          const row = memoData[tableMeta.rowIndex];
          const cell = {
            getValue: () => value,
          };
          return Cell({ cell, row, value, tableMeta });
        };
      }

      return {
        name: columnName,
        label: columnLabel,
        options: normalizedOptions,
      };
    });
  }, [columns, memoData]);

  const defaultOptions = useMemo(
    () => ({
      download: true,
      downloadOptions: {
        filename: normalizedFileName,
        separator: ",",
      },
      selectableRows: "none",
      filter: false,
      search: false,
      print: false,
      viewColumns: false,
      pagination: false,
      responsive: "standard",
      elevation: 0,
      tableBodyHeight: "auto",
      tableBodyMaxHeight: "",
      sort: false,
      setTableProps: () => ({
        size: "small",
      }),
      textLabels: {
        body: {
          noMatch: "No data",
        },
      },
    }),
    [normalizedFileName],
  );

  const mergedOptions = useMemo(() => {
    const downloadOptions = {
      ...defaultOptions.downloadOptions,
      ...(tableOptions.downloadOptions || {}),
      filename:
        ensureCsvExtension(tableOptions.downloadOptions?.filename) ?? normalizedFileName,
    };

    return {
      ...defaultOptions,
      ...tableOptions,
      downloadOptions,
    };
  }, [defaultOptions, tableOptions, normalizedFileName]);

  return (
    <Box sx={{ mb: 3 }}>
      <MUIDataTable
        title={title ? <Typography variant={titleVariant}>{title}</Typography> : null}
        data={memoData}
        columns={normalizedColumns}
        options={mergedOptions}
      />
    </Box>
  );
}
